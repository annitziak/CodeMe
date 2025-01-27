import os
import marisa_trie
import mmap
import struct

from indexor.structures import Term, PostingList, IndexBase


class OnDiskIndex(IndexBase):
    def __init__(self, load_path=None, index_builder_kwargs: dict = {}):
        self.load_path = load_path
        assert self.load_path is not None, "load_path must be provided"
        assert os.path.isdir(self.load_path), "load_path must be a directory"

        self.term_fst = marisa_trie.BytesTrie()
        self.term_fst.load(os.path.join(self.load_path, "terms.fst"))

        with open(os.path.join(self.load_path, "postings.bin"), "rb") as f:
            self.postings_mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def get_term(self, term: str) -> Term:
        current_doc_id = 0

        term_bytes = term.encode("utf-8")
        offset_data = self.term_fst.get(term_bytes)
        if offset_data is None:
            raise ValueError(f"Term {term} not found in index")

        offset = struct.unpack("<Q", offset_data[0])[0]
        self.postings_mmap.seek(offset)

        postings_count = struct.unpack("<I", self.postings_mmap.read(4))[0]
        postings = {}
        for _ in range(postings_count):
            delta, term_frequency = struct.unpack("<IH", self.postings_mmap.read(6))
            current_doc_id += delta

            postings[current_doc_id] = PostingList(current_doc_id, term_frequency, [])

        return Term(term, postings_count, postings)

    def get_document_frequency(self, term: str) -> int:
        return self.get_term(term).document_frequency

    def get_all_posting_lists(self, term: str) -> list[PostingList]:
        return list(self.get_term(term).posting_lists.values())

    def get_posting_list(self, term: str, doc_id: int) -> PostingList | None:
        term_dict = self.get_term(term)
        return term_dict.posting_lists.get(doc_id, None)

    def get_term_frequency(self, term: str, doc_id: int) -> int:
        posting_list = self.get_posting_list(term, doc_id)
        if posting_list is None:
            return 0

        return posting_list.doc_term_frequency
