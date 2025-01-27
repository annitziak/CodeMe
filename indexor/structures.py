import pprint

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class PostingList:
    doc_id: int
    doc_term_frequency: int
    positions: list[int]

    def __str__(self):
        return f"{self.doc_id}:{self.doc_term_frequency}"

    def __repr__(self):
        return f"{self.doc_id}:{self.doc_term_frequency}"

    def sort(self):
        self.positions.sort()

    def update_position(self, position: int):
        self.positions.append(position)
        self.doc_term_frequency += 1

    def update_with_positions(self, positions: list[int]):
        for position in positions:
            self.positions.append(position)

        self.doc_term_frequency += len(positions)


@dataclass
class Term:
    term: str
    document_frequency: int = 0
    posting_lists: OrderedDict[int, PostingList] = field(default_factory=OrderedDict)

    def __str__(self):
        posting_list = [x for x in self.posting_lists.values()][:10]
        posting_list[-1] = "..."
        str_posting_list = pprint.pformat(posting_list)

        return f"{self.term}({self.document_frequency},{str_posting_list})"

    def __repr__(self):
        return f"{self.document_frequency}"

    def sort_posting_lists(self):
        for posting_list in self.posting_lists.values():
            posting_list.sort()

    def update_with_postings(self, doc_id: int, positions: list[int]):
        if doc_id in self.posting_lists:
            self.posting_lists[doc_id].update_with_positions(positions)
            return

        self.posting_lists[doc_id] = PostingList(doc_id, len(positions), positions)
        self.document_frequency += 1

        assert all(
            [
                self.posting_lists[doc_id].doc_id == doc_id
                for doc_id in sorted(list(self.posting_lists.keys()))
            ]
        ), f"Posting list for term {self.term} is not sorted by document id"

    def get_posting_list(self, doc_id: int) -> PostingList | None:
        return self.posting_lists.get(doc_id, None)


class IndexBase(ABC):
    @abstractmethod
    def get_term(self, term: str) -> Term:
        pass

    @abstractmethod
    def get_document_frequency(self, term: str) -> int:
        pass

    @abstractmethod
    def get_all_posting_lists(self, term: str) -> list[PostingList]:
        pass

    @abstractmethod
    def get_term_frequency(self, term: str, doc_id: int) -> int:
        pass

    @abstractmethod
    def get_posting_list(self, term: str, doc_id: int) -> PostingList | None:
        pass
