import os
import marisa_trie
import mmap
import struct

from indexor.structures import READ_SIZE_KEY, Term, PostingList, IndexBase, SIZE_KEY


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

    def get_term(self, term: str, positions=False) -> Term:
        """
        Get a term from the index
        Args:
            term (str): The term to retrieve
            positions (bool): Whether to include positions in the term object
        """
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
        postings = {}
        current_doc_id = 0

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

            postings[current_doc_id] = PostingList(
                current_doc_id, term_frequency, _positions
            )

        return Term(term, postings_count, postings)

    def get_document_frequency(self, term: str) -> int:
        return self.get_term(term).document_frequency

    def get_all_posting_lists(self, term: str) -> list[PostingList]:
        return list(self.get_term(term, positions=True).posting_lists.values())

    def get_posting_list(self, term: str, doc_id: int) -> PostingList | None:
        term_dict = self.get_term(term, positions=True)
        return term_dict.posting_lists.get(doc_id, None)

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        posting_list = self.get_posting_list(term, doc_id)
        if posting_list is None:
            return 0

        return posting_list.doc_term_frequency
