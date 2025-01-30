import os
from typing import OrderedDict
import marisa_trie
import mmap
import struct
import copy

from indexor.structures import Term, PostingList, IndexBase
from indexor.index_builder.constants import SIZE_KEY, READ_SIZE_KEY


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

        self.term_fst = marisa_trie.BytesTrie()
        self.term_fst.load(os.path.join(self.load_path, "terms.fst"))

        self.pos_term_fst = marisa_trie.BytesTrie()
        self.pos_term_fst.load(os.path.join(self.load_path, "pos_terms.fst"))

        with open(os.path.join(self.load_path, "postings.bin"), "rb") as f:
            self.postings_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        with open(os.path.join(self.load_path, "positions.bin"), "rb") as f:
            self.positions_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def _get_term(self, term: str, positions=False):
        term_bytes = term.encode("utf-8")

        fst = self.pos_term_fst if positions else self.term_fst
        offset_data = fst.get(term_bytes)
        if offset_data is None:
            raise ValueError(f"Term {term} not found in index")

        offset = struct.unpack(SIZE_KEY["offset"], offset_data[0])[0]
        mmap = self.positions_mmap if positions else self.postings_mmap
        mmap.seek(offset)

        postings_count = struct.unpack(
            SIZE_KEY["postings_count"],
            mmap.read(READ_SIZE_KEY[SIZE_KEY["postings_count"]]),
        )[0]
        current_doc_id = 0

        yield term, postings_count, []

        for _ in range(postings_count):
            delta, term_frequency = struct.unpack(
                SIZE_KEY["deltaTF"],
                mmap.read(READ_SIZE_KEY[SIZE_KEY["deltaTF"]]),
            )
            current_doc_id += delta

            _positions = []
            if positions:
                curr_position = 0
                position_count = struct.unpack(
                    SIZE_KEY["position_count"],
                    mmap.read(READ_SIZE_KEY[SIZE_KEY["position_count"]]),
                )[0]
                for _ in range(position_count):
                    delta = struct.unpack(
                        SIZE_KEY["position_delta"],
                        mmap.read(READ_SIZE_KEY[SIZE_KEY["position_delta"]]),
                    )[0]
                    curr_position += delta
                    _positions.append(curr_position)

            yield current_doc_id, term_frequency, _positions

    def get_term(self, term: str, positions=False) -> Term:
        """
        Get a term from the index
        Args:
            term (str): The term to retrieve
            positions (bool): Whether to include positions in the term object
        """
        iterator = iter(self._get_term(term, positions))

        term, postings_count, _ = next(iterator)
        postings = {}
        for current_doc_id, term_frequency, _positions in iterator:
            postings[current_doc_id] = PostingList(
                current_doc_id, term_frequency, _positions
            )

        return Term(term, postings_count, postings)

    def get_document_frequency(self, term: str) -> int:
        return self.get_term(term).document_frequency

    def get_all_posting_lists(self, term: str, positions=True) -> list[PostingList]:
        return list(self.get_term(term, positions=positions).posting_lists.values())

    def _get_posting_list(self, doc_id, iterator):
        _, _, _ = next(iterator)
        for current_doc_id, term_frequency, _positions in iterator:
            if current_doc_id == doc_id:
                return PostingList(doc_id, term_frequency, _positions)

    def get_posting_list(self, term: str, doc_id: int) -> PostingList | None:
        iterator = iter(self._get_term(term, positions=True))
        return self._get_posting_list(doc_id, iterator)

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        posting_list = self.get_posting_list(term, doc_id)
        if posting_list is None:
            return 0

        return posting_list.doc_term_frequency

    def get_intersection(self, terms: list[str | Term]) -> Term:
        """
        Get the intersection of the posting lists of the given terms
        """
        if not terms:
            return Term("", 0)

        intersection = OrderedDict()
        if isinstance(terms[0], str):
            posting_lists: list[PostingList] = self.get_all_posting_lists(
                terms[0], positions=False
            )
        else:
            posting_lists: list[PostingList] = list(terms[0].posting_lists.values())
        postings_lists = sorted(posting_lists, key=lambda x: x.doc_term_frequency)
        for posting in postings_lists:
            intersection[posting.doc_id] = copy.deepcopy(posting)

        for term in terms[1:]:
            posting_lists = []
            if isinstance(term, str):
                posting_lists: list[PostingList] = self.get_all_posting_lists(
                    term, positions=False
                )
            else:
                posting_lists: list[PostingList] = list(term.posting_lists.values())

            posting_lists = sorted(posting_lists, key=lambda x: x.doc_term_frequency)
            i, j = 0, 0
            intersection_vals = list(intersection.values())
            while i < len(intersection_vals) and j < len(posting_lists):
                if intersection_vals[i].doc_id == posting_lists[j].doc_id:
                    intersection_vals[i].doc_term_frequency += posting_lists[
                        j
                    ].doc_term_frequency
                    i += 1
                    j += 1
                elif intersection_vals[i].doc_id < posting_lists[j].doc_id:
                    del intersection[
                        intersection_vals[i].doc_id
                    ]  # Remove the document from the intersection
                    i += 1
                else:
                    j += 1

        term_str = " AND ".join([str(term) for term in terms])
        return Term(term_str, len(intersection), intersection)

    def get_union(self, terms: list[str | Term]) -> Term:
        """
        Get the union of the posting lists of the given terms
        """
        if not terms:
            return Term("", 0)

        union = OrderedDict()
        for term in terms:
            posting_lists = []
            if isinstance(term, str):
                posting_lists: list[PostingList] = self.get_all_posting_lists(
                    term, positions=False
                )
            else:
                posting_lists: list[PostingList] = list(term.posting_lists.values())

            posting_lists = sorted(posting_lists, key=lambda x: x.doc_term_frequency)
            for posting in posting_lists:
                if posting.doc_id not in union:
                    posting = copy.deepcopy(posting)
                    posting.positions = []

                    union[posting.doc_id] = posting
                else:
                    union[
                        posting.doc_id
                    ].doc_term_frequency += posting.doc_term_frequency

        term_str = " OR ".join([str(term) for term in terms])
        return Term(term_str, len(union), union)

    def get_complement(self, term: str | Term):
        """
        Get the complement of the posting list of the given term
        """
        if isinstance(term, str):
            posting_list = self.get_all_posting_lists(term, positions=False)
        else:
            posting_list = list(term.posting_lists.values())

        all_docs = set(range(1, 10001))
        term_docs = set([posting.doc_id for posting in posting_list])

        complement = all_docs - term_docs
        return Term(f"NOT {term}", len(complement), complement)
