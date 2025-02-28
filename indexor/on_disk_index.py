import os
from typing import OrderedDict
import marisa_trie
import mmap
import struct
import json
import glob
import time
import logging
import psutil
import numpy as np
import heapq

from numba import jit, uint16, uint32, int64, types
from concurrent.futures import ProcessPoolExecutor

from indexor.structures import Term, PostingList, IndexBase
from indexor.index_builder.constants import SIZE_KEY, READ_SIZE_KEY
from indexor.query import (
    BooleanQuery,
    FreeTextQuery,
    Query,
    TermQuery,
    AND,
    OR,
    NOT,
    PhraseQuery,
    ProximityQuery,
)
from utils.varint import decode_bytes_jit, decode_bytes

logging.basicConfig(
    format=f"%(asctime)s - %(name)s - {os.getpid()} - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@jit(int64(types.Array(types.uint8, 1, "A", readonly=True)), nopython=True, cache=True)
def from_bytes(byte_data):
    value = [b << i * 8 for i, b in enumerate(byte_data)]
    return sum(value)


@jit(
    types.Array(uint16, 1, "C")(types.Array(uint16, 1, "A", readonly=True)),
    nopython=True,
    fastmath=True,
    cache=True,
    boundscheck=False,
    nogil=True,
)
def fast_cumsum(deltas):
    result = np.empty_like(deltas)
    acc = 0
    for i in range(len(deltas)):
        acc += deltas[i]
        result[i] = acc

    return result


@jit(
    types.Tuple(
        (
            int64,
            types.Array(uint32, 1, "C"),
            types.Array(uint32, 1, "C"),
            types.Array(uint32, 1, "C"),
        )
    )(types.Array(types.uint8, 1, "A", readonly=True), int64, int64),
    nopython=True,
    fastmath=True,
    cache=True,
)
def process_posting_block(
    posting_data,
    limit,
    postings_count,
):
    limit = postings_count if limit == -1 else limit
    doc_ids = np.zeros(min(postings_count, limit), dtype=np.uint32)
    term_frequencies = np.zeros(min(postings_count, limit), dtype=np.uint32)
    position_counts = np.zeros(min(postings_count, limit), dtype=np.uint32)

    current_doc_id = 0
    offset = 0
    for i in range(min(postings_count, limit)):
        delta, offset = decode_bytes_jit(posting_data, offset)
        term_frequency, offset = decode_bytes_jit(posting_data, offset)
        position_count, offset = decode_bytes_jit(posting_data, offset)

        current_doc_id += delta

        doc_ids[i] = current_doc_id
        term_frequencies[i] = term_frequency
        position_counts[i] = position_count

    return offset, doc_ids, term_frequencies, position_counts


@jit(nopython=True, cache=True, fastmath=True)
def calculate_bm25_numba(
    scores,
    all_doc_ids_arr,
    doc_ids,
    term_frequencies,
    doc_freq,
    doc_lengths,
    doc_count,
    avg_doc_length,
    min_doc_id,
    id_range,
    k1,
    b,
):
    """
    Numba-accelerated function to update BM25 scores for a single term
    """
    # Check if any documents for this term are in our range
    in_range = False
    for i in range(len(doc_ids)):
        if min_doc_id <= doc_ids[i] <= min_doc_id + id_range - 1:
            in_range = True
            break

    if not in_range:
        return scores

    # Calculate IDF component
    doc_freq_float = float(doc_freq)
    doc_count_float = float(doc_count)
    idf = np.float32(
        np.log((doc_count_float - doc_freq_float + 0.5) / (doc_freq_float + 0.5))
    )

    # Create term frequency lookup array
    term_freq_lookup = np.zeros(id_range, dtype=np.float32)
    for i in range(len(doc_ids)):
        idx = doc_ids[i] - min_doc_id
        if 0 <= idx < id_range:
            term_freq_lookup[idx] = term_frequencies[i]

    # Calculate BM25 components for each document
    new_scores = np.zeros_like(scores)
    for i in range(len(all_doc_ids_arr)):
        doc_id = all_doc_ids_arr[i]
        idx = doc_id - min_doc_id
        if 0 <= idx < id_range:
            tf = term_freq_lookup[idx]
            if tf > 0:
                doc_length = doc_lengths[i]
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * (doc_length / avg_doc_length))
                bm25_score = idf * (numerator / denominator)
                new_scores[i] = scores[i] + bm25_score
            else:
                new_scores[i] = scores[i]
        else:
            new_scores[i] = scores[i]

    return new_scores


def _read_fst(
    fst,
    term: str | bytes,
    is_sharded=True,
    size_key=SIZE_KEY["pos_offset"],
    shard_size_key=SIZE_KEY["pos_offset_shard"],
):
    if isinstance(term, str):
        term_bytes = term.encode("utf-8")
    else:
        term_bytes = term

    encoded = fst.get(term_bytes)

    if is_sharded:
        if encoded is None:
            return

        for encoding in encoded:
            shard, *values = struct.unpack(shard_size_key, encoding)
            yield shard, *values
        return

    if encoded is None:
        yield -1, -1, -1
        return

    unpacked = [struct.unpack(size_key, x) for x in encoded]
    assert len(encoded) == 1, f"Term {term} not found in index {unpacked}"
    logger.info(f"Unpacked: {unpacked}")
    yield -1, *unpacked[0]


class MMappedFile:
    def __init__(self, path: str):
        self.path = path
        self.data = None

    def load(self):
        with open(self.path, "rb") as f:
            try:
                self.data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            except Exception as e:
                logger.error(f"Error loading mmap: {e}")

        return self

    def seek(self, position: int):
        if self.data is None:
            raise ValueError("File not loaded")

        self.data.seek(position)

    def read(self, size: int):
        if self.data is None:
            raise ValueError("File not loaded")

        return self.data.read(size)


class InMemoryFile:
    def __init__(self, path: str):
        self.path = path
        self.data = None

        self.position = 0

    def load(self):
        with open(self.path, "rb") as f:
            self.data = f.read()

        return self

    def seek(self, position: int):
        self.position = position

    def read(self, size: int):
        if self.data is None:
            raise ValueError("File not loaded")

        return self.data[self.position : self.position + size]

    def close(self):
        self.data = None


class MMappedReader:
    def __init__(
        self,
        path: str,
        glob_pattern="*",
        keep_in_memory: bool = False,
        is_sharded: bool = False,
        max_size=1024 * 1024 * 1024 * 3,
    ):
        self.path = path
        self.glob_pattern = os.path.join(self.path, glob_pattern)
        self.files = {}

        self.keep_in_memory = keep_in_memory
        self.is_sharded = is_sharded
        self.max_size = max_size

    def __del__(self):
        for item in self.files.values():
            item.close()

    def __getitem__(self, key):
        if key not in self.files:
            raise ValueError(
                f"{os.getpid()} Shard {key}({type(key)}) not found in {self.files}"
            )

        return self.files[key]

    def load(self, shard=-1):
        logger.debug(f"Loading shard {shard} from {self.glob_pattern}")
        if shard in self.files:
            return self

        for file in glob.glob(self.glob_pattern):
            if shard != -1 and f"shard_{shard}" not in file:
                continue

            filename = os.path.basename(file)
            shard_no = -1
            if self.is_sharded:
                shard_no = int(filename.split("_")[-1].split(".")[0])

            if self._is_space_available(file) and self.keep_in_memory:
                logger.debug(f"Loading {file} into memory")
                self.files[shard_no] = InMemoryFile(file).load()
            else:
                logger.debug(f"Loading {file} into mmap")
                self.files[shard_no] = MMappedFile(file).load()

            logger.info(f"{os.getpid()} Loaded {file} in {self.files}")

        return self

    def seek(self, shard: int, position: int):
        if shard not in self.files:
            raise ValueError(f"Shard {shard} not found")

        self.files[shard].seek(position)

    def read(self, shard: int = -1, size: int = 0):
        if shard not in self.files:
            raise ValueError(f"Shard {shard} not found")

        if size == 0:
            return self.files[shard].data

        return self.files[shard].read(size)

    def _is_space_available(self, filename):
        file_size = os.path.getsize(filename)
        available_memory = psutil.virtual_memory().available
        available_memory = min(available_memory, self.max_size)

        return file_size < available_memory


def _load_doc_id_bounds(load_path: str = ""):
    doc_id_bounds, doc_stats = {}, {}
    with open(os.path.join(load_path, "shard.meta"), "r") as f:
        shard_meta = json.load(f)
        for shard, item in shard_meta.items():
            doc_id_bounds[shard] = item.get("bounds", {})
            doc_stats[shard] = item.get("metadata", {})

    return doc_id_bounds, doc_stats


class TermCache:
    def __init__(self):
        self.cache = OrderedDict()
        self.freq_threshold = 10_000
        self.max_size = 1_000_000

    def get(self, term: str, positions=False):
        return self.cache.get((term, positions), None)

    def put(self, term: str, term_obj, positions=False):
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        if term_obj.document_frequency > self.freq_threshold:
            self.cache[(term, positions)] = term_obj


class DocLengthCache:
    def __init__(self):
        self.cache = OrderedDict()
        self.max_size = 1_000_000

    def get(self, doc_id: int):
        return self.cache.get(doc_id, None)

    def put(self, doc_id: int, doc_length: int):
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        self.cache[doc_id] = doc_length


class ShardWorker:
    def __init__(self, index_path: str = "", is_sharded: bool = False):
        self.index_path = index_path

        glob_pattern_common = "shard_*.index" if is_sharded else "postings.bin"
        glob_pattern_positions = (
            "shard_*.position" if is_sharded else "postings_positions.bin"
        )
        glob_pattern_docs = "doc_shard_*.index" if is_sharded else "docs.bin"
        glob_pattern_docs_offset = "doc_shard_*.offset" if is_sharded else "docs.offset"

        self.postings_mmaps = MMappedReader(
            index_path, glob_pattern_common, is_sharded=is_sharded
        )
        self.positions_mmaps = MMappedReader(
            index_path, glob_pattern_positions, is_sharded=is_sharded
        )
        self.doc_mmaps = MMappedReader(
            index_path, glob_pattern_docs, is_sharded=is_sharded
        )
        self.doc_offset_mmaps = MMappedReader(
            index_path, glob_pattern_docs_offset, is_sharded=is_sharded
        )

        self.term_fst = marisa_trie.BytesTrie()
        self.doc_fst = marisa_trie.BytesTrie()

        self.shard = -1

        self.min_doc_id = 0
        self.max_doc_id = 78_254_177
        self.doc_set_mask = np.ones(self.max_doc_id, dtype=bool)

        self.term_fst.load(os.path.join(index_path, "terms.fst"))
        self.doc_fst.load(os.path.join(index_path, "docs.fst"))
        

        self.all_doc_metadata_keys = [
            "doc_length",
            "metadatascore",
            "score",
            "viewcount",
            "owneruserid",
            "answercount",
            "commentcount",
            "favoritecount",
            "ownerdisplayname",
            "tags",
            "creationdate",
            "hasacceptedanswer",
            "title",
            "body",
        ]

        self.doc_length_cache = DocLengthCache()
        self.doc_bounds, self.doc_stats = _load_doc_id_bounds(self.index_path)
        self.doc_count = 0
        self.avg_doc_length = 0

    def __del__(self):
        self.positions_mmaps.__del__()
        self.postings_mmaps.__del__()
        self.doc_mmaps.__del__()

    def _load_files(self, shard: int):
        self.postings_mmaps.load(shard)
        self.positions_mmaps.load(shard)
        self.doc_mmaps.load(shard)

        self.shard = shard

        self.doc_count, sum_doc_length = self._get_document_count()
        if self.doc_count == 0:
            logger.error(f"Doc count is 0 in shard {shard}")
            return

        self.avg_doc_length = sum_doc_length / self.doc_count

        doc_ids_npy = os.path.join(self.index_path, f"doc_ids_{shard}.npy")
        doc_offsets_npy = os.path.join(self.index_path, f"doc_offsets_{shard}.npy")
        doc_lengths_npy = os.path.join(self.index_path, f"doc_lengths_{shard}.npy")
        if os.path.exists(doc_ids_npy) and os.path.exists(doc_offsets_npy):
            self.doc_ids = np.load(doc_ids_npy, mmap_mode="r")
            self.doc_offsets = np.load(doc_offsets_npy, mmap_mode="r")
            self.doc_lengths = np.load(doc_lengths_npy, mmap_mode="r")

        self.min_doc_id = self.doc_ids[0]
        self.max_doc_id = self.doc_ids[-1]
        range_doc_ids = self.max_doc_id - self.min_doc_id + 1
        doc_set_mask_file = os.path.join(self.index_path, "doc_set_mask.npy")
        if os.path.exists(doc_set_mask_file):
            os.remove(doc_set_mask_file)

        self.doc_set_mask = np.ones(range_doc_ids, dtype=bool)
        np.save(doc_set_mask_file, self.doc_set_mask)
        time.sleep(0.5)
        self.doc_set_mask = np.load(
            doc_set_mask_file,
            mmap_mode="r+",
            allow_pickle=True,
        )

        """
        for doc_id, value in self.doc_fst.items():
            if shard != self.shard:
                continue

            shard, offset = struct.unpack(SIZE_KEY["offset_shard"], value)
            if shard != self.shard:
                continue

            doc_length = self._get_document_metadata(
                doc_id, shard, offset, keys=["doc_length"]
            )["doc_length"]
            self.doc_length_cache.put(np.int32(doc_id), doc_length)
        """

    def _get_term(
        self,
        term: str,
        shard: int,
        offset: int = -1,
        pos_offset=-1,
        limit=10_000,
        positions=False,
    ):
        start = time.time()

        if offset == -1 or pos_offset == -1:
            for _shard, _offset, _pos_offset in _read_fst(
                self.term_fst, term, is_sharded=True
            ):
                if _shard != shard:
                    continue

                offset = _offset
                pos_offset = _pos_offset

        if offset == -1 or pos_offset == -1:
            return (
                shard,
                0,
                np.array([], dtype=np.uint32),
                np.array([], dtype=np.uint32),
                [],
            )

        posting_mmap = self.postings_mmaps.load(shard)[shard]
        posting_mmap.seek(offset)

        position_mmap = self.positions_mmaps.load(shard)[shard]
        position_mmap.seek(pos_offset)

        postings_count = decode_bytes(posting_mmap)

        doc_ids = []
        term_frequencies = []
        all_positions = []

        max_posting_size = (
            READ_SIZE_KEY[SIZE_KEY["deltaTF"]]
            + READ_SIZE_KEY[SIZE_KEY["position_count"]]
        ) * postings_count
        posting_data = posting_mmap.read(max_posting_size)
        posting_data = np.frombuffer(posting_data, dtype=np.uint8)
        _offset, doc_ids, term_frequencies, position_counts = process_posting_block(
            posting_data, limit, postings_count
        )
        posting_mmap.seek(offset + _offset)

        if offset != -1 and positions:
            for i in range(len(doc_ids)):
                position_count = position_counts[i].item()
                if position_count == 0:
                    all_positions.append(np.array([], dtype=np.uint16))
                    continue

                all_deltas = np.array(
                    [decode_bytes(position_mmap) for _ in range(position_count)],
                    dtype=np.uint16,
                )
                all_positions.append(fast_cumsum(all_deltas).tolist())

        logger.info(f"Time taken: {time.time() - start} in shard {shard} for {term}")

        return shard, postings_count, doc_ids, term_frequencies, all_positions

    def _search_doc_lengths_np(self, doc_id: int, shard: int):
        if not hasattr(self, "doc_ids") or not hasattr(self, "doc_lengths"):
            return None

        idx = np.searchsorted(self.doc_ids, doc_id)
        if idx < len(self.doc_ids) and self.doc_ids[idx] == doc_id:
            return self.doc_lengths[idx]

        return None

    def _get_document_length(self, doc_id: int, shard: int):
        doc_length = self._search_doc_lengths_np(doc_id, shard)
        if doc_length is not None:
            self.doc_length_cache.put(doc_id, doc_length)
            return doc_length

        doc_length = self._get_document_metadata(doc_id, shard, -1, keys=["doc_length"])
        self.doc_length_cache.put(doc_id, doc_length)
        return doc_length

    def _get_document_length_batch(self, doc_ids: list[int] | np.ndarray, shard: int):
        if not hasattr(self, "doc_ids") or not hasattr(self, "doc_lengths"):
            return None

        if isinstance(doc_ids, list):
            doc_ids_arr = np.array(doc_ids)
        else:
            doc_ids_arr = doc_ids

        indices = np.searchsorted(self.doc_ids, doc_ids_arr)

        valid_mask = (indices < len(self.doc_ids)) & (
            self.doc_ids[indices] == doc_ids_arr
        )
        result = np.zeros(len(doc_ids), dtype=np.uint16)
        result[valid_mask] = self.doc_lengths[indices[valid_mask]]
        for i in range(len(doc_ids)):
            if not valid_mask[i]:
                result[i] = self._get_document_metadata(
                    doc_ids[i], shard, -1, keys=["doc_length"]
                )
                self.doc_length_cache.put(doc_ids[i], result[i])

        return result

    def _get_scored_search(
        self,
        query: FreeTextQuery,
        shard: int,
        limit=1_000_000,
        k1=1.2,
        b=0.75,
    ):
        """
        We use a locally scored BM25
        This is not accurate since we are only considering the local-terms
        """
        scores = []

        if self.doc_count == 0 or self.avg_doc_length == 0:
            self.doc_count, sum_doc_length = self._get_document_count()
            self.avg_doc_length = sum_doc_length / self.doc_count

        all_doc_ids = set()
        term_info = {}
        for term in query.parsed_query:
            _, doc_freq, doc_ids, term_frequencies, _ = self._get_term(
                term, shard, -1, limit=limit
            )
            if len(doc_ids) > 0:
                term_info[term] = {
                    "doc_ids": doc_ids,
                    "term_frequencies": term_frequencies,
                    "doc_freq": doc_freq,
                }
            all_doc_ids.update(doc_ids)

        if not all_doc_ids:
            return []

        all_doc_ids_arr = np.array(list(all_doc_ids))
        min_doc_id = np.min(all_doc_ids_arr)
        max_doc_id = np.max(all_doc_ids_arr)
        id_range = max_doc_id - min_doc_id + 1

        doc_lengths = self._get_document_length_batch(all_doc_ids_arr, shard)
        scores = np.zeros(len(all_doc_ids_arr), dtype=np.float32)
        rarest_term_order = sorted(
            term_info.keys(), key=lambda x: term_info[x]["doc_freq"]
        )

        for term in rarest_term_order:
            info = term_info[term]
            doc_ids = np.array(info["doc_ids"], dtype=np.int64)
            doc_freq = int(info["doc_freq"])
            term_frequencies = np.array(info["term_frequencies"], dtype=np.float32)

            scores = calculate_bm25_numba(
                scores,
                all_doc_ids_arr,
                doc_ids,
                term_frequencies,
                doc_freq,
                doc_lengths,
                self.doc_count,
                self.avg_doc_length,
                min_doc_id,
                id_range,
                k1,
                b,
            )

        """
            doc_ids = info["doc_ids"]
            doc_freq = info["doc_freq"]

            in_range_mask = (doc_ids >= min_doc_id) & (doc_ids <= max_doc_id)
            if not np.any(in_range_mask):
                continue

            idf = np.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

            max_doc_id = np.max(np.concatenate([doc_ids, all_doc_ids_arr])) + 1
            term_freq_lookup = np.zeros(id_range, dtype=np.float32)
            term_freq_lookup[doc_ids - min_doc_id] = info["term_frequencies"]

            matched_term_freqs = term_freq_lookup[all_doc_ids_arr - min_doc_id]
            numerator = matched_term_freqs * (k1 + 1)
            denominator = matched_term_freqs + k1 * (
                1 - b + b * (doc_lengths / self.avg_doc_length)
            )
            tf = numerator / denominator

            scores += idf * tf
        """

        if len(scores) <= limit:
            return [(score, doc_id) for score, doc_id in zip(scores, all_doc_ids_arr)]

        top_k_indicies = np.argpartition(scores, -limit)[-limit:]
        top_k_docs = [(scores[i], all_doc_ids_arr[i]) for i in top_k_indicies]

        return top_k_docs

    def _search(self, query: Query, shard: int = 0, limit=100_000):
        def _search_helper(query):
            if isinstance(query, TermQuery):
                return self._get_term(query.parsed_query, shard, -1, limit=-1)
            elif isinstance(query, AND):
                left = _search_helper(query.left)
                right = _search_helper(query.right)
                return self._get_intersection([left, right], shard, [-1, -1])
            elif isinstance(query, OR):
                left = _search_helper(query.left)
                right = _search_helper(query.right)

                return self._get_union([left, right], shard, [-1, -1])
            elif isinstance(query, NOT):
                left = _search_helper(query.left)
                return self._get_complement(left)
            elif isinstance(query, PhraseQuery):
                return self._prox_search(
                    query, shard, -1, limit=limit, exact=True, absolute=False
                )
            elif isinstance(query, ProximityQuery):
                return self._prox_search(
                    query, shard, -1, limit=limit, exact=False, absolute=True
                )
            elif isinstance(query, FreeTextQuery):
                return self._get_scored_search(query, shard, limit=limit)
            else:
                raise ValueError(f"Query type {type(query)} not supported")

        if isinstance(query, BooleanQuery):
            return _search_helper(query.parse().pop())
        elif isinstance(query, FreeTextQuery):
            return _search_helper(query)

    def _prox_search(
        self,
        query: Query,
        shard: int,
        offset: int,
        limit=100_000,
        exact=True,
        absolute=False,
    ):
        terms = query.parsed_query
        distances = query.distances

        if len(terms) == 1:
            return self._get_term(terms[0], shard, offset, limit=-1, positions=True)

        _, doc_freq, doc_ids, term_frequencies, all_positions = self._get_intersection(
            terms, shard, [-1] * len(terms), positions=True
        )

        valid_doc_mask = np.zeros(len(doc_ids), dtype=np.bool)

        for doc_idx in range(len(doc_ids)):
            term_positions = all_positions[doc_idx]
            num_terms = len(terms)

            for term_i in range(num_terms - 1):
                min_dist = float("inf")
                term_i_positions = term_positions[term_i]
                term_j_positions = term_positions[term_i + 1]

                for pos_i in term_i_positions:
                    for pos_j in term_j_positions:
                        dist = pos_j - pos_i

                        if exact and not absolute and dist == distances[term_i]:
                            valid_doc_mask[doc_idx] = True
                            break
                        elif not exact and absolute and abs(dist) <= distances[term_i]:
                            valid_doc_mask[doc_idx] = True
                            break
                        elif not exact and not absolute:
                            raise ValueError(
                                f"Invalid proximity search type: {exact} {absolute}"
                            )

        doc_ids = doc_ids[valid_doc_mask]
        term_frequencies = term_frequencies[valid_doc_mask]
        positions = [
            y for i, x in enumerate(all_positions) if valid_doc_mask[i] for y in x[0]
        ]

        return 0, len(doc_ids), doc_ids, term_frequencies, positions

    def _get_complement(self, term: str | tuple[int, np.ndarray, np.ndarray]):
        if isinstance(term, str):
            _, doc_freq, doc_ids, term_frequencies, _ = self._get_term(
                term, -1, -1, False
            )
        else:
            doc_freq, doc_ids, term_frequencies = term[0], term[2], term[3]

        if len(doc_ids) == 0:
            return (
                -1,
                len(self.doc_set_mask),
                np.where(self.doc_set_mask)[0] + self.min_doc_id,
                np.zeros(len(self.doc_set_mask), dtype=np.uint),
                [],
            )

        min_doc_id = np.min(doc_ids)
        max_doc_id = np.max(doc_ids)
        id_range = max_doc_id - min_doc_id + 1

        temp_mask = np.ones(id_range, dtype=bool)
        indices = doc_ids - min_doc_id

        temp_mask[indices] = False
        complement_doc_ids = np.where(temp_mask)[0]
        complement_doc_ids += min_doc_id

        return (
            -1,
            len(complement_doc_ids),
            complement_doc_ids,
            np.zeros(len(complement_doc_ids), dtype=np.uint),
            [],
        )

    def _get_intersection(
        self,
        terms: list[str | tuple[int, np.ndarray, np.ndarray, list]],
        shard: int,
        offsets: list[int],
        positions=False,
    ):
        intersection_mask = None
        intersection_doc_ids = None
        intersection_term_frequencies = None
        intersection_all_positions = []
        for term_idx, (term, offset) in enumerate(zip(terms, offsets)):
            for i in range(len(intersection_all_positions)):
                if len(intersection_all_positions[i]) <= term_idx:
                    intersection_all_positions[i].append([])

            if term == "":
                continue

            if isinstance(term, str):
                _, doc_freq, doc_ids, term_frequencies, all_positions = self._get_term(
                    term, shard, offset, positions=positions, limit=-1
                )
            else:
                doc_freq, doc_ids, term_frequencies, all_positions = (
                    term[1],
                    term[2],
                    term[3],
                    term[4],
                )

            if intersection_mask is None or intersection_doc_ids is None:
                intersection_doc_ids = doc_ids
                intersection_mask = np.ones(len(doc_ids), dtype=bool)
                intersection_term_frequencies = term_frequencies
                # [Doc1[position1, position2, ...], Doc2[...], ...]
                intersection_all_positions = []
                if positions:
                    intersection_all_positions = [
                        [[] for _ in range(len(terms))] for _ in doc_ids
                    ]

                    for idx, doc_positions in enumerate(all_positions):
                        intersection_all_positions[idx][term_idx] = doc_positions
                continue

            current_doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
            common_doc_mask = np.array(
                [doc_id in current_doc_id_to_idx for doc_id in intersection_doc_ids]
            )
            new_doc_ids = intersection_doc_ids[common_doc_mask]
            new_term_frequencies = intersection_term_frequencies[common_doc_mask]

            if positions:
                new_positions = []
                for idx, (doc_id, keep) in enumerate(
                    zip(intersection_doc_ids, common_doc_mask)
                ):
                    if not keep:
                        continue

                    doc_positions = intersection_all_positions[idx]
                    current_idx = current_doc_id_to_idx[doc_id]
                    doc_positions[term_idx] = all_positions[current_idx]
                    new_positions.append(doc_positions)

                intersection_all_positions = new_positions

            intersection_doc_ids = new_doc_ids
            intersection_term_frequencies = new_term_frequencies

        return (
            0,
            len(intersection_doc_ids),
            intersection_doc_ids,
            intersection_term_frequencies,
            intersection_all_positions,
        )

    def _get_union(
        self,
        terms: list[str | tuple[int, np.ndarray, np.ndarray]],
        shard: int,
        offsets: list[int],
    ):
        union_mask = None
        union_doc_ids = None
        union_term_frequencies = None
        for term, offset in zip(terms, offsets):
            if isinstance(term, str):
                _, doc_freq, doc_ids, term_frequencies, _ = self._get_term(
                    term, shard, offset, False
                )
            else:
                doc_freq, doc_ids, term_frequencies = term[1], term[2], term[3]

            if union_mask is None:
                union_mask = doc_ids
                union_doc_ids = doc_ids
                union_term_frequencies = term_frequencies
                continue

            union_mask = np.union1d(union_mask, doc_ids)
            union_doc_ids = np.union1d(union_doc_ids, doc_ids)
            union_term_frequencies = np.union1d(
                union_term_frequencies, term_frequencies
            )

        return 0, len(union_doc_ids), union_doc_ids, union_term_frequencies, []

    def _get_document_metadata(
        self, doc_id: int, shard: int, offset: int, keys=["all"]
    ):
        if keys[0] == "doc_length":
            doc_length = self.doc_length_cache.get(doc_id)
            if doc_length is not None:
                return doc_length

        mmap = self.doc_mmaps.load(shard)[shard]
        doc_length = -1
        if offset == -1:
            for _shard, _offset, _doc_length in _read_fst(
                self.doc_fst,
                str(doc_id),
                is_sharded=True,
                size_key=SIZE_KEY["offset_doc_length"],
                shard_size_key=SIZE_KEY["offset_shard_doc_length"],
            ):
                if _shard != shard:
                    continue

                offset = _offset
                doc_length = _doc_length

        if offset != -1 and keys[0] == "doc_length" and len(keys) == 1:
            self.doc_length_cache.put(doc_id, doc_length)
            return doc_length

        mmap.seek(offset)
        if "all" in keys:
            keys = self.all_doc_metadata_keys

        if len(keys) == 0:
            return {}

        ret = {}
        for key in self.all_doc_metadata_keys:
            if key not in keys:
                continue

            if key in ["ownerdisplayname", "tags", "title", "body"]:
                size = struct.unpack(
                    SIZE_KEY[f"doc_{key}"],
                    mmap.read(READ_SIZE_KEY[SIZE_KEY[f"doc_{key}"]]),
                )[0]
                ret[key] = mmap.read(size).decode("utf-8")
                continue

            key_name = f"doc_{key}" if "doc" not in key else key
            ret[key] = struct.unpack(
                SIZE_KEY[key_name],
                mmap.read(READ_SIZE_KEY[SIZE_KEY[key_name]]),
            )[0]

        return ret

    def _get_document_count(self):
        doc_count = 0
        sum_doc_length = 0
        for shard, (start, end) in self.doc_bounds.items():
            shard = int(shard)
            if start == float("inf") or end == float("-inf"):
                continue

            mmap = self.doc_offset_mmaps.load(shard)[shard]
            mmap.seek(0)

            doc_count += struct.unpack(
                SIZE_KEY["doc_count"],
                mmap.read(READ_SIZE_KEY[SIZE_KEY["doc_count"]]),
            )[0]
            sum_doc_length += struct.unpack(
                SIZE_KEY["doc_length"],
                mmap.read(READ_SIZE_KEY[SIZE_KEY["doc_length"]]),
            )[0]

        return doc_count, sum_doc_length


_worker = None


def worker_init(*args, **kwargs):
    global _worker
    _worker = ShardWorker(*args, **kwargs)


def worker_get_term(*args, **kwargs):
    global _worker
    return _worker._get_term(*args, **kwargs)


def worker_search(*args, **kwargs):
    global _worker
    return _worker._search(*args, **kwargs)


def worker_load(*args, **kwargs):
    global _worker
    return _worker._load_files(*args, **kwargs)


def worker_get_document_metadata(*args, **kwargs):
    global _worker
    return _worker._get_document_metadata(*args, **kwargs)


def worker_get_document_count(*args, **kwargs):
    global _worker
    return _worker._get_document_count(*args, **kwargs)


class OnDiskIndex(IndexBase):
    """
    Stores the index on disk as a memory-mapped file.
    The index is stored in a trie, and the postings are stored in a memory-mapped file.
    The trie provides an offset to the postings in the memory-mapped file.

    Trie:
    - terms.fst: Trie of terms
    - pos_terms.fst: Trie of terms with positions

    Postings:
    - postings.bin: Postings lists
        - Begins with number of postings
        - Each posting list is stored as a sequence of (delta, term_frequency) pairs

    Positions:
    - positions.bin: Positions of terms in documents
         - Begins with number of postings
         - Each posting list then stores the sequence (delta, term_frequency and position count)
            - Followed by `position_count` number of position deltas

    """

    def __init__(self, load_path=None, index_builder_kwargs: dict = {}):
        self.load_path = load_path
        assert self.load_path is not None, "load_path must be provided"
        assert os.path.isdir(self.load_path), "load_path must be a directory"

        with open(os.path.join(self.load_path, "config.json"), "r") as f:
            config = json.load(f)
        self.is_sharded = config.get("is_sharded", False)

        self.term_fst = self._load_fst(os.path.join(self.load_path, "terms.fst"))
        self.doc_fst = self._load_fst(os.path.join(self.load_path, "docs.fst"))
        # self.doc_set = set(self.doc_fst.keys())  # enough mem??
        # self.doc_set = set()

        self.num_workers = min(
            index_builder_kwargs.get("num_workers", 4), os.cpu_count() - 1
        )
        self.doc_bounds, self.doc_stats = _load_doc_id_bounds(self.load_path)
        self.term_cache = TermCache()

        self.num_shards = len(self.doc_bounds)

        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=worker_init,
            initargs=(
                self.load_path,
                self.is_sharded,
            ),
        )
        self.worker_pool = {}
        worker_per_shard = self.num_workers // len(self.doc_bounds)
        load_result = []
        for shard in range(len(self.doc_bounds)):
            if shard not in self.worker_pool:
                self.worker_pool[shard] = ProcessPoolExecutor(
                    max_workers=worker_per_shard,
                    initializer=worker_init,
                    initargs=(
                        self.load_path,
                        self.is_sharded,
                    ),
                )
                for _ in range(worker_per_shard):
                    load_result.append(
                        self.worker_pool[shard].submit(worker_load, shard)
                    )

        # divide workers
        for future in load_result:
            future.result()

        logger.info(f"Initialized worker pool with {self.num_workers} workers")

    def __del__(self):
        for worker in self.worker_pool.values():
            worker.shutdown()

    def _load_fst(self, filename):
        fst = marisa_trie.BytesTrie()
        fst.load(filename)
        return fst

    def _build_term(self, term: Term, future, positions=False):
        shard, doc_freq, doc_ids, term_frequencies, position_list = future.result()
        start = time.time()
        term.update_with_term(
            doc_ids,
            term_frequencies,
            position_list,
            positions=positions,
            doc_freq=doc_freq,
        )
        logger.info(
            f"Shard {shard} updated term {term.term} with {len(doc_ids)} in {time.time() - start} seconds"
        )

        return term

    def get_term(self, term: str, limit=100_000, positions=False) -> Term:
        """
        Get a term from the index
        Args:
            term (str): The term to retrieve
            positions (bool): Whether to include positions in the term object
        """
        cached_term = self.term_cache.get(term, positions=positions)
        if cached_term is not None:
            return cached_term

        logger.info(f"Term {term} not found in cache")

        start = time.time()
        results = []
        _start = time.time()
        for shard, offset, pos_offset in _read_fst(
            self.term_fst, term, is_sharded=self.is_sharded
        ):
            if shard == -1 and offset == -1 and pos_offset == -1:
                continue

            future = self.worker_pool[shard].submit(
                worker_get_term,
                term,
                shard,
                offset,
                pos_offset,
                limit,
                positions=positions,
            )
            results.append(future)

        ret_term = Term(term)
        for future in results:
            ret_term = self._build_term(ret_term, future, positions=positions)

        self.term_cache.put(term, ret_term, positions=positions)
        logger.info(
            f"GET TERM: time taken: {time.time() - start} with positions={positions}"
        )

        return ret_term

    def scored_search(self, query: FreeTextQuery, limit=100_000):
        futures = []
        for shard in range(self.num_shards):
            futures.append(
                self.worker_pool[shard].submit(worker_search, query, shard, limit=limit)
            )

        scores = []
        for future in futures:
            scores.extend(future.result())

        # RESCORE??
        scores.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"Retrieved {len(scores)} results")
        scores = scores[:limit]

        return scores

    def search(self, query: Query, limit: int = -1):
        if isinstance(query, FreeTextQuery):
            return self.scored_search(query, limit)

        futures = []
        for shard in range(self.num_shards):
            futures.append(
                self.worker_pool[shard].submit(worker_search, query, shard, limit)
            )

        scores = []
        for future in futures:
            res = future.result()
            scores.extend([(0, x) for x in res[2]])

        return scores

    def get_term_by_prefix(self, prefix: str, positions=False) -> Term:
        """
        Get all terms that have the given prefix
        Args:
            prefix (str): The prefix to search for
            positions (bool): Whether to include positions in the term object
        """
        start = time.time()
        fst = self.pos_term_fst if positions else self.term_fst
        terms = list(fst.keys(prefix))

        return self.get_union(terms)

    def _get_shard_for_doc_id(self, doc_id: int):
        for shard, (start, end) in self.doc_bounds.items():
            if start == float("inf") or end == float("-inf"):
                continue

            if int(start) <= doc_id <= int(end):
                return int(shard)

        raise ValueError(f"Document {doc_id} not found in index")

    def get_document_frequency(self, term: str) -> int:
        return self.get_term(term).document_frequency

    def get_document_metadata(self, doc_id: int, keys=["all"]) -> dict | None:
        items = list(
            _read_fst(
                self.doc_fst,
                str(doc_id),
                is_sharded=self.is_sharded,
                size_key=SIZE_KEY["offset_doc_length"],
                shard_size_key=SIZE_KEY["offset_shard_doc_length"],
            )
        )
        if len(items) == 0:
            return None

        shard, offset, doc_length = items[0]
        if offset != -1 and keys[0] == "doc_length" and len(keys) == 1:
            return {"doc_length": doc_length}

        future = self.worker_pool[shard].submit(
            worker_get_document_metadata, doc_id, shard=shard, offset=offset, keys=keys
        )

        return future.result()

    def get_document_length(self, doc_id: int) -> int:
        shard, offset, doc_length = list(
            _read_fst(
                self.doc_fst,
                str(doc_id),
                is_sharded=self.is_sharded,
                size_key=SIZE_KEY["offset_doc_length"],
                shard_size_key=SIZE_KEY["offset_shard_doc_length"],
            )
        )[0]
        return doc_length

    def get_vocab(self, top_p=0.2) -> list[str]:
        """
        Retrieves the vocabulary of the index
        Args:
            top_p (float): The percentage of terms to return based on document frequency
        """
        ret_list = []
        capacity = int(len(self.term_fst) * top_p)
        for term in self.term_fst.keys():
            if isinstance(term, bytes):
                term = term.decode("utf-8")

            doc_freq = self.get_document_frequency(term)
            if len(ret_list) < capacity:
                heapq.heappush(ret_list, (doc_freq, term))
            else:
                heapq.heappushpop(ret_list, (doc_freq, term))

        return [x[1] for x in ret_list]

    def get_all_posting_lists(self, term: str, positions=False) -> list[PostingList]:
        return [
            x
            for x in self.get_term(term, positions=positions).posting_lists
            if x is not None
        ]

    def get_posting_list(
        self, term: str, doc_id: int, positions=False
    ) -> PostingList | None:
        target_shard = self._get_shard_for_doc_id(doc_id)
        logger.info(f"Target shard: {target_shard}")
        for shard, offset, pos_offset in _read_fst(
            self.term_fst, term, is_sharded=self.is_sharded
        ):
            if shard != target_shard:
                continue

            logger.info(f"Shard: {shard}, Offset: {offset}")
            ret = self._build_term(
                Term(term),
                self.worker_pool[shard].submit(
                    worker_get_term, term, shard, offset, pos_offset=pos_offset
                ),
                positions=positions,
            )
            return ret.get_posting_list(doc_id)

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        posting_list = self.get_posting_list(term, doc_id)
        if posting_list is None:
            return 0

        return posting_list.doc_term_frequency

    def get_all_documents(self):
        docs = OrderedDict()
        for doc in self.doc_fst.keys():
            docs[int(doc)] = OrderedDict()

        for term in self.term_fst.keys():
            if isinstance(term, bytes):
                term = term.decode("utf-8")
            for posting in self.get_all_posting_lists(term, positions=True):
                if posting.doc_id not in docs:
                    raise ValueError(
                        f"Doc ID {posting.doc_id} not found in index {docs}"
                    )

                if term not in docs[posting.doc_id]:
                    docs[posting.doc_id][term] = Term(term, 0, [])

                docs[posting.doc_id][term].update_with_postings(
                    posting.doc_id, posting.positions
                )

        return docs

    def get_document_count(self):
        return self.worker_pool[0].submit(worker_get_document_count).result()[0]

    def get_term_count(self):
        return len(self.term_fst)

    def write_index_to_txt(self, path: str):
        logger.info(f"Writing index to {path}")

        doc_meta_path = os.path.join(path, "doc_metadata.txt")

        index_path = os.path.join(path, "index.txt")
        with open(index_path, "w") as f:
            sorted_terms = sorted(self.term_fst.keys())
            for term in sorted_terms:
                if isinstance(term, bytes):
                    term = term.decode("utf-8")

                term_obj = self.get_term(term, positions=True)
                output = f"{term_obj.term}:{term_obj.document_frequency}\n"
                for posting in term_obj.posting_lists:
                    positions = ",".join([str(x) for x in posting.positions])
                    output += f"\t{posting.doc_id}:{positions}\n"
                f.write(output)

        logger.info(f"Index written to {index_path}")

        sorted_doc_ids = sorted([int(x) for x in self.doc_fst.keys()])
        with open(doc_meta_path, "w") as f:
            tags = [
                "creationdate",
                "hasacceptedanswer",
                "metadatascore",
                "score",
                "viewcount",
                "owneruserid",
                "answercount",
                "commentcount",
                "favoritecount",
                "ownerdisplayname",
                "tags",
                "title",
                "body",
            ]
            f.write(f'doc_id,{",".join(tags)}\n')
            for doc_id in sorted_doc_ids:
                if isinstance(doc_id, bytes):
                    doc_id = doc_id.decode("utf-8")

                doc_metadata = self.get_document_metadata(int(doc_id))
                out = ",".join([str(doc_metadata[tag]) for tag in tags])

                f.write(f"{doc_id},{out}\n")

        logger.info(f"Document metadata written to {doc_meta_path}")
        if not os.path.exists(os.path.dirname(path)):
            logger.info(f"Creating directory {os.path.dirname(path)}")
            os.makedirs(os.path.dirname(path))

        doc_terms = os.path.join(path, "document_terms.txt")
        docs = self.get_all_documents()
        logger.info(f"Writing document terms to {doc_terms}")
        with open(doc_terms, "w") as f:
            for doc_id in sorted_doc_ids:
                terms = docs[doc_id]
                f.write(f"{doc_id} : ")
                term_string = "\t".join(
                    [
                        f"{term}"
                        for term, term_obj in terms.items()
                        for _ in range(term_obj.document_frequency)
                    ]
                )
                f.write(f"{term_string}\n")
