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
            self.data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

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

            logger.debug(f"{os.getpid()} Loaded {file} in {self.files}")

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
    with open(os.path.join(load_path, "shard.meta"), "r") as f:
        doc_id_bounds = json.load(f)

    return doc_id_bounds


class TermCache:
    def __init__(self):
        self.cache = OrderedDict()
        self.freq_threshold = 10_000
        self.max_size = 1_000_000

    def get(self, term: str, positions=False):
        return self.cache.get((term, positions), None)

    def put(self, term: str, term_obj, positions=False):
        if len(self.cache) > 100:
            self.cache.popitem(last=False)

        if term_obj.document_frequency > self.freq_threshold:
            self.cache[(term, positions)] = term


class DocLengthCache:
    def __init__(self):
        self.cache = {}

        self.max_size = 10_000_000

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

        self.doc_set = {}
        self.shard = -1

        self.term_fst.load(os.path.join(index_path, "terms.fst"))
        self.doc_fst.load(os.path.join(index_path, "docs.fst"))

        self.all_doc_metadata_keys = [
            "doc_length",
            "score",
            "viewcount",
            "owneruserid",
            "answercount",
            "commentcount",
            "favoritecount",
            "ownerdisplayname",
            "tags",
            "creationdate",
        ]

        self.doc_length_cache = DocLengthCache()
        self.doc_bounds = _load_doc_id_bounds(self.index_path)
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
        self.avg_doc_length = sum_doc_length / self.doc_count

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

    def _get_term(
        self,
        term: str,
        shard: int,
        offset: int = -1,
        pos_offset=-1,
        limit=100_000,
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

        posting_mmap = self.postings_mmaps.load(shard)[shard]
        posting_mmap.seek(offset)

        position_mmap = self.positions_mmaps.load(shard)[shard]
        position_mmap.seek(pos_offset)

        postings_count = struct.unpack(
            SIZE_KEY["postings_count"],
            posting_mmap.read(READ_SIZE_KEY[SIZE_KEY["postings_count"]]),
        )[0]

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
        print(f"Time taken: {time.time() - start} in shard {shard} for {term}")

        return shard, doc_ids, term_frequencies, all_positions

    def _get_scored_search(
        self,
        query: FreeTextQuery,
        shard: int,
        limit=100_000,
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
            _, doc_ids, term_frequencies, _ = self._get_term(
                term, shard, -1, limit=limit
            )
            if len(doc_ids) > 0:
                term_info[term] = {
                    "doc_ids": doc_ids,
                    "term_frequencies": term_frequencies,
                    "doc_freq": len(doc_ids),
                }
            all_doc_ids.update(doc_ids)

        if not all_doc_ids:
            return []

        all_doc_ids_arr = np.array(list(all_doc_ids))
        doc_lengths = np.array(
            [
                self._get_document_metadata(doc_id, shard, -1, keys=["doc_length"])[
                    "doc_length"
                ]
                for doc_id in all_doc_ids
            ]
        )
        scores = np.zeros(len(all_doc_ids_arr), dtype=np.float32)
        for term, info in term_info.items():
            doc_ids = info["doc_ids"]
            doc_freq = info["doc_freq"]

            idf = np.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

            doc_mask = np.isin(all_doc_ids_arr, doc_ids)
            matching_positions = np.searchsorted(doc_ids, all_doc_ids_arr[doc_mask])

            term_frequencies = np.zeros(len(all_doc_ids_arr), dtype=np.int32)
            term_frequencies[doc_mask] = info["term_frequencies"][matching_positions]
            tf = (term_frequencies * (k1 + 1)) / (
                term_frequencies
                + k1 * (1 - b + b * (doc_lengths / self.avg_doc_length))
            )

            scores += idf * tf

        if len(scores) <= limit:
            return [(score, doc_id) for score, doc_id in zip(scores, all_doc_ids_arr)]

        top_k_indicies = np.argpartition(scores, -limit)[-limit:]
        top_k_docs = [(scores[i], all_doc_ids_arr[i]) for i in top_k_indicies]

        return top_k_docs

    def _search(self, query: Query, shard: int = 0, limit=100_000):
        def _search_helper(query):
            if isinstance(query, TermQuery):
                return self._get_term(query.query, shard, -1, limit=limit)
            elif isinstance(query, AND):
                left = _search_helper(query.left)
                right = _search_helper(query.right)

                return self._get_intersection([left, right], shard, [-1, -1])
            elif isinstance(query, OR):
                left = _search_helper(query.left)
                right = _search_helper(query.right)

                return self._get_union([left, right], shard, [-1, -1])
            elif isinstance(query, NOT):
                left = _search_helper(query.right)
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
            return self._get_term(terms[0], shard, offset, limit=limit, positions=True)

        _, doc_ids, term_frequencies, all_positions = self._get_intersection(
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

        return 0, doc_ids, term_frequencies, positions

    def _get_complement(self, term: str | tuple[int, np.ndarray, np.ndarray]):
        if isinstance(term, str):
            _, doc_ids, term_frequencies, _ = self._get_term(term, 0, -1, False)
        else:
            doc_ids, term_frequencies = term[1], term[2]

        doc_set_arr = np.array(list(self.doc_set))
        complement_mask = ~np.isin(doc_set_arr, doc_ids)

        complement_doc_ids = np.array(doc_set_arr)[complement_mask]
        complement_term_frequencies = np.zeros(len(complement_doc_ids), dtype=np.int32)

        return 0, complement_doc_ids, complement_term_frequencies, []

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
                _, doc_ids, term_frequencies, all_positions = self._get_term(
                    term, shard, offset, positions=positions
                )
            else:
                doc_ids, term_frequencies, all_positions = term[1], term[2], term[3]

            if intersection_mask is None or intersection_doc_ids is None:
                intersection_mask = doc_ids
                intersection_doc_ids = doc_ids
                intersection_term_frequencies = term_frequencies
                # [Doc1[position1, position2, ...], Doc2[...], ...]
                intersection_all_positions = []
                if positions:
                    intersection_all_positions = [[x] for x in all_positions]
                continue

            intersection_mask = np.isin(intersection_doc_ids, doc_ids)
            intersection_doc_ids = intersection_doc_ids[intersection_mask]
            intersection_term_frequencies = intersection_term_frequencies[
                intersection_mask
            ]

            if positions:
                for idx, doc_positions in enumerate(all_positions):
                    doc_id = doc_ids[idx]
                    if doc_id not in intersection_doc_ids:
                        continue

                    intersection_idx = np.where(intersection_doc_ids == doc_id)[0][0]
                    if len(intersection_all_positions[intersection_idx]) <= term_idx:
                        intersection_all_positions[intersection_idx].append([])

                    intersection_all_positions[intersection_idx][term_idx].extend(
                        doc_positions
                    )
                intersection_all_positions = [
                    x
                    for idx, x in enumerate(intersection_all_positions)
                    if intersection_mask[idx]
                ]

        return (
            0,
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
                _, doc_ids, term_frequencies, _ = self._get_term(
                    term, shard, offset, False
                )
            else:
                doc_ids, term_frequencies = term[1], term[2]

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

        return 0, union_doc_ids, union_term_frequencies, []

    def _get_document_metadata(
        self, doc_id: int, shard: int, offset: int, keys=["all"]
    ):
        if keys[0] == "doc_length":
            doc_length = self.doc_length_cache.get(doc_id)
            if doc_length is not None:
                return {"doc_length": doc_length}

        mmap = self.doc_mmaps.load(shard)[shard]
        if offset == -1:
            for _shard, _offset in _read_fst(
                self.doc_fst,
                str(doc_id),
                is_sharded=True,
                size_key=SIZE_KEY["offset"],
                shard_size_key=SIZE_KEY["offset_shard"],
            ):
                if _shard != shard:
                    continue

                offset = _offset

        mmap.seek(offset)
        if "all" in keys:
            keys = self.all_doc_metadata_keys

        if len(keys) == 0:
            return {}

        ret = {}
        for key in keys:
            if key in ["ownerdisplayname", "tags", "creationdate"]:
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
        self.doc_set = set(self.doc_fst.keys())  # enough mem??
        # self.doc_set = set()

        self.num_workers = min(
            index_builder_kwargs.get("num_workers", 24), os.cpu_count() - 1
        )
        self.doc_bounds = _load_doc_id_bounds(self.load_path)
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
        shard, doc_ids, term_frequencies, position_list = future.result()
        start = time.time()
        term.update_with_term(
            doc_ids, term_frequencies, position_list, positions=positions
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
        logger.info(f"Term {term} not found in cache")
        if cached_term is not None:
            return cached_term

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
        print(f"GET TERM: time taken: {time.time() - start} with positions={positions}")

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

        ret_term = Term(query.__str__())
        for future in futures:
            ret_term = self._build_term(ret_term, future)

        return ret_term

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

    def get_document_metadata(self, doc_id: int, keys=["all"]) -> dict:
        shard, offset = list(
            _read_fst(
                self.doc_fst,
                str(doc_id),
                is_sharded=self.is_sharded,
                size_key=SIZE_KEY["offset"],
                shard_size_key=SIZE_KEY["offset_shard"],
            )
        )[0]
        future = self.worker_pool[shard].submit(
            worker_get_document_metadata, doc_id, shard=shard, offset=offset, keys=keys
        )

        return future.result()

    def get_document_length(self, doc_id: int) -> int:
        shard, offset = list(
            _read_fst(
                self.doc_fst,
                str(doc_id),
                is_sharded=self.is_sharded,
                size_key=SIZE_KEY["offset"],
                shard_size_key=SIZE_KEY["offset_shard"],
            )
        )[0]
        future = self.worker_pool[shard].submit(
            worker_get_document_metadata,
            doc_id,
            shard=shard,
            offset=offset,
            keys=["doc_length"],
        )

        return future.result()["doc_length"]

    def get_vocab(self) -> list[str]:
        return list(self.term_fst.keys())

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
        return self.worker_pool[0].submit(worker_get_document_count).result()

    def get_term_count(self):
        return len(self.term_fst)

    def write_index_to_txt(self, path: str):
        logger.info(f"Writing index to {path}")
        if not os.path.exists(os.path.dirname(path)):
            logger.info(f"Creating directory {os.path.dirname(path)}")
            os.makedirs(os.path.dirname(path))

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

        doc_meta_path = os.path.join(path, "doc_metadata.txt")
        sorted_doc_ids = sorted([int(x) for x in self.doc_fst.keys()])
        with open(doc_meta_path, "w") as f:
            tags = [
                "creationdate",
                "score",
                "viewcount",
                "owneruserid",
                "answercount",
                "commentcount",
                "favoritecount",
                "ownerdisplayname",
                "tags",
            ]
            f.write(f'doc_id,{",".join(tags)}')
            for doc_id in sorted_doc_ids:
                if isinstance(doc_id, bytes):
                    doc_id = doc_id.decode("utf-8")

                doc_metadata = self.get_document_metadata(int(doc_id))
                out = ",".join([str(doc_metadata[tag]) for tag in tags])

                f.write(f"{doc_id},{out}\n")

        logger.info(f"Document metadata written to {doc_meta_path}")

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
