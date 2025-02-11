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

from numba import jit, int32, uint16, int64, types
from concurrent.futures import ProcessPoolExecutor

from indexor.structures import Term, PostingList, IndexBase
from indexor.index_builder.constants import SIZE_KEY, READ_SIZE_KEY

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
    types.UniTuple(types.Array(int32, 1, "C"), 3)(
        types.Array(types.uint8, 1, "A", readonly=True), int64, int64, int64, int64
    ),
    nopython=True,
    fastmath=True,
    cache=True,
)
def process_posting_block(
    posting_data,
    limit,
    postings_count,
    deltaTF_size,
    position_count_size,
):
    doc_ids = np.zeros(min(postings_count, limit), dtype=np.int32)
    term_frequencies = np.zeros(min(postings_count, limit), dtype=np.int32)
    position_counts = np.zeros(min(postings_count, limit), dtype=np.int32)

    current_doc_id = 0
    current_pos = 0

    for i in range(min(postings_count, limit)):
        delta = from_bytes(posting_data[current_pos : current_pos + 4])
        term_frequency = from_bytes(
            posting_data[current_pos + 4 : current_pos + 4 + 2],
        )
        position_count = from_bytes(
            posting_data[
                current_pos + deltaTF_size : current_pos
                + deltaTF_size
                + position_count_size
            ],
        )

        current_pos += position_count_size + deltaTF_size
        current_doc_id += delta

        doc_ids[i] = current_doc_id
        term_frequencies[i] = term_frequency
        position_counts[i] = position_count

    return doc_ids, term_frequencies, position_counts


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
        logger.info(f"Loading shard {shard} from {self.glob_pattern}")
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
                logger.info(f"Loading {file} into memory")
                self.files[shard_no] = InMemoryFile(file).load()
            else:
                logger.info(f"Loading {file} into mmap")
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


class ShardWorker:
    def __init__(self, index_path: str = "", is_sharded: bool = False):
        self.index_path = index_path

        glob_pattern_common = "shard_*.index" if is_sharded else "postings.bin"
        glob_pattern_positions = (
            "shard_*.position" if is_sharded else "postings_positions.bin"
        )
        glob_pattern_docs = "doc_shard_*.index" if is_sharded else "docs.bin"
        glob_pattern_docs_offset = (
            "doc_shard_*.offset" if is_sharded else "docs_offset.bin"
        )

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

        self.doc_bounds = _load_doc_id_bounds(self.index_path)

    def __del__(self):
        self.positions_mmaps.__del__()
        self.postings_mmaps.__del__()
        self.doc_mmaps.__del__()

    def _load_files(self, shard: int):
        self.postings_mmaps.load(shard)
        self.positions_mmaps.load(shard)
        self.doc_mmaps.load(shard)

    def _get_term(
        self,
        term: str,
        shard: int,
        offset: int,
        pos_offset=-1,
        limit=100_000,
        positions=False,
    ):
        start = time.time()

        posting_mmap = self.postings_mmaps.load(shard)[shard]
        posting_mmap.seek(offset)

        position_mmap = self.positions_mmaps.load(shard)[shard]
        position_mmap.seek(pos_offset)

        postings_count = struct.unpack(
            SIZE_KEY["postings_count"],
            posting_mmap.read(READ_SIZE_KEY[SIZE_KEY["postings_count"]]),
        )[0]
        current_doc_id = 0

        deltaTF_fmt = SIZE_KEY["deltaTF"]
        position_count_fmt = SIZE_KEY["position_count"]
        position_delta_fmt = SIZE_KEY["position_delta"]

        deltaTF_size = READ_SIZE_KEY[deltaTF_fmt]
        position_count_size = READ_SIZE_KEY[position_count_fmt]
        position_delta_size = READ_SIZE_KEY[position_delta_fmt]

        doc_ids = []
        term_frequencies = []
        all_positions = []

        posting_data = posting_mmap.read(
            (deltaTF_size + position_count_size) * postings_count
        )
        posting_data = np.frombuffer(posting_data, dtype=np.uint8)

        doc_ids, term_frequencies, position_counts = process_posting_block(
            posting_data,
            limit,
            postings_count,
            deltaTF_size,
            position_count_size,
        )

        if offset != -1 and positions:
            for i in range(len(doc_ids)):
                position_count = position_counts[i].item()
                if position_count == 0:
                    all_positions.append(np.array([], dtype=np.uint16))
                    continue

                position_data = position_mmap.read(position_count * position_delta_size)
                deltas = np.frombuffer(position_data, dtype=np.uint16)
                all_positions.append(fast_cumsum(deltas).tolist())

        logger.info(f"Time taken: {time.time() - start} in shard {shard} for {term}")
        print(f"Time taken: {time.time() - start} in shard {shard} for {term}")

        return shard, doc_ids, term_frequencies, all_positions

    def _get_intersection(
        self, terms: list[str | Term], shard: int, offsets: list[int]
    ):
        curr_doc_ids_dict = None
        for term, offset in zip(terms, offsets):
            if isinstance(term, str):
                shard, doc_ids, term_frequencies, all_positions = self._get_term(
                    term, shard, offset, False
                )
                doc_ids_dict = {
                    doc_id: term_frequency
                    for doc_id, term_frequency in zip(doc_ids, term_frequencies)
                }
            else:
                posting_lists = term.posting_lists
                doc_ids_dict = {
                    posting.doc_id: posting.doc_term_frequency
                    for posting in posting_lists
                }

            if curr_doc_ids_dict is None:
                curr_doc_ids_dict = doc_ids_dict
            else:
                curr_doc_ids_dict = {
                    doc_id: term_frequency + curr_doc_ids_dict[doc_id]
                    for doc_id, term_frequency in doc_ids_dict.items()
                    if doc_id in curr_doc_ids_dict
                }

        return curr_doc_ids_dict

    def _get_union(self, terms: list[str | Term], shard: int, offsets: list[int]):
        all_docs = None
        for term, offset in zip(terms, offsets):
            if isinstance(term, str):
                shard, doc_ids, term_frequencies, all_positions = self._get_term(
                    term, shard, offset, False
                )
                doc_ids_dict = {
                    doc_id: term_frequency
                    for doc_id, term_frequency in zip(doc_ids, term_frequencies)
                }
            else:
                posting_lists = term.posting_lists
                doc_ids_dict = {
                    posting.doc_id: posting.doc_term_frequency
                    for posting in posting_lists
                }

            if all_docs is None:
                all_docs = doc_ids_dict
            else:
                for doc_id, term_frequency in doc_ids_dict.items():
                    if doc_id not in all_docs:
                        all_docs[doc_id] = term_frequency
                    else:
                        all_docs[doc_id] += term_frequency

        return all_docs

    def _get_document_length(self, doc_id: int, shard: int, offset: int):
        mmap = self.doc_mmaps.load(shard)[shard]
        mmap.seek(offset)

        return struct.unpack(
            SIZE_KEY["doc_length"],
            mmap.read(READ_SIZE_KEY[SIZE_KEY["doc_length"]]),
        )[0]

    def _get_document_count(self):
        doc_count = 0
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

        return doc_count


_worker = None


def worker_init(*args, **kwargs):
    global _worker
    _worker = ShardWorker(*args, **kwargs)


def worker_get_term(*args, **kwargs):
    global _worker
    return _worker._get_term(*args, **kwargs)


def worker_get_intersection(*args, **kwargs):
    global _worker
    return _worker._get_intersection(*args, **kwargs)


def worker_get_union(*args, **kwargs):
    global _worker
    return _worker._get_union(*args, **kwargs)


def worker_load(*args, **kwargs):
    global _worker
    return _worker._load_files(*args, **kwargs)


def worker_get_document_length(*args, **kwargs):
    global _worker
    return _worker._get_document_length(*args, **kwargs)


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
        self.doc_set = set()

        self.num_workers = min(
            index_builder_kwargs.get("num_workers", 24), os.cpu_count() - 1
        )
        self.worker_pool = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=worker_init,
            initargs=(
                self.load_path,
                self.is_sharded,
            ),
        )
        for shard in range(self.num_workers):
            self.worker_pool.submit(worker_load, shard)

        logger.info(f"Initialized worker pool with {self.num_workers} workers")

        self.doc_bounds = _load_doc_id_bounds(self.load_path)

        self.term_cache = TermCache()

    def _load_fst(self, filename):
        fst = marisa_trie.BytesTrie()
        fst.load(filename)
        return fst

    def _read_fst(self, fst, term: str | bytes):
        if isinstance(term, str):
            term_bytes = term.encode("utf-8")
        else:
            term_bytes = term

        encoded = fst.get(term_bytes)

        if self.is_sharded:
            if encoded is None:
                raise ValueError(f"Term {term} not found in index")

            for encoding in encoded:
                shard, offset, pos_offset = struct.unpack(
                    SIZE_KEY["pos_offset_shard"], encoding
                )
                yield shard, offset, pos_offset
            return

        if encoded is None:
            yield -1, -1, -1
            return

        unpacked = [struct.unpack(SIZE_KEY["pos_offset"], x) for x in encoded]
        assert len(encoded) == 1, f"Term {term} not found in index {unpacked}"
        logger.info(f"Unpacked: {unpacked}")
        yield -1, unpacked[0][0], unpacked[0][1]

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
        for shard, offset, pos_offset in self._read_fst(self.term_fst, term):
            if shard == -1 and offset == -1 and pos_offset == -1:
                continue

            future = self.worker_pool.submit(
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

    def __del__(self):
        self.worker_pool.shutdown()

    def _get_shard_for_doc_id(self, doc_id: int):
        for shard, (start, end) in self.doc_bounds.items():
            if start == float("inf") or end == float("-inf"):
                continue

            if int(start) <= doc_id <= int(end):
                return int(shard)

        raise ValueError(f"Document {doc_id} not found in index")

    def get_document_frequency(self, term: str) -> int:
        return self.get_term(term).document_frequency

    def get_document_length(self, doc_id: int) -> int:
        value = self.doc_fst.get(str(doc_id))
        if value is None:
            try_again = self.doc_fst.get(str(doc_id)[:-2])
            final = self.doc_fst.get(str(doc_id)[:-1])
            raise ValueError(
                f"Document {doc_id} not found in index of length {len(self.doc_fst)} {try_again} {final} {list(self.doc_fst.keys())[:10]}"
            )

        shard, offset = struct.unpack(SIZE_KEY["offset_shard"], value[0])
        future = self.worker_pool.submit(
            worker_get_document_length, doc_id, shard=shard, offset=offset
        )

        return future.result()

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
        for shard, offset, pos_offset in self._read_fst(self.term_fst, term):
            if shard != target_shard:
                continue

            logger.info(f"Shard: {shard}, Offset: {offset}")
            ret = self._build_term(
                Term(term),
                self.worker_pool.submit(
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

    def get_intersection(self, terms: list[str | Term]) -> Term:
        """
        Get the intersection of the posting lists of the given terms
        """
        fst = self.term_fst

        results = []
        _start = time.time()

        all_term_offsets = {}
        for term in terms:
            for shard, offset, _ in self._read_fst(self.term_fst, term):
                if shard not in all_term_offsets:
                    all_term_offsets[shard] = []

                all_term_offsets[shard].append(offset)

        for shard, offsets in all_term_offsets.items():
            future = self.worker_pool.submit(
                worker_get_intersection, terms, shard, offsets
            )

            results.append(future)

        ret_term = Term(" AND ".join([str(term) for term in terms]))
        for future in results:
            doc_tf_dict = future.result()
            doc_ids = list(doc_tf_dict.keys())
            term_frequencies = list(doc_tf_dict.values())
            position_lists = [[] for _ in range(len(doc_ids))]

            ret_term.update_with_term(doc_ids, term_frequencies, position_lists)

        return ret_term

    def get_union(self, terms: list[str | Term]) -> Term:
        """
        Get the union of the posting lists of the given terms
        """
        fst = self.term_fst

        results = []
        _start = time.time()

        all_term_offsets = {}
        for term in terms:
            for shard, offset, _ in self._read_fst(self.term_fst, term):
                if shard not in all_term_offsets:
                    all_term_offsets[shard] = []

                all_term_offsets[shard].append(offset)

        for shard, offsets in all_term_offsets.items():
            future = self.worker_pool.submit(worker_get_union, terms, shard, offsets)

            results.append(future)

        ret_term = Term(" OR ".join([str(term) for term in terms]))
        for future in results:
            doc_tf_dict = future.result()
            doc_ids = list(doc_tf_dict.keys())
            term_frequencies = list(doc_tf_dict.values())
            position_lists = [[] for _ in range(len(doc_ids))]

            ret_term.update_with_term(doc_ids, term_frequencies, position_lists)

        return ret_term

    def get_complement(self, term: str | Term):
        """
        Get the complement of the posting list of the given term
        """
        if isinstance(term, str):
            posting_list = self.get_all_posting_lists(term, positions=False)
        else:
            posting_list = list(term.posting_lists.values())

        term_docs = set([posting.doc_id for posting in posting_list])

        complement = self.doc_set - term_docs
        return Term(f"NOT {term}", len(complement), complement)

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
        return self.worker_pool.submit(worker_get_document_count).result()

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
            for doc_id in sorted_doc_ids:
                if isinstance(doc_id, bytes):
                    doc_id = doc_id.decode("utf-8")

                doc_length = self.get_document_length(int(doc_id))
                f.write(f"{doc_id}: {doc_length}\n")

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
