import pprint
import copy

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import OrderedDict

SIZE_KEY = {
    "offset": "<Q",
    "postings_count": "<I",
    "deltaTF": "<IH",
    "position_count": "<I",
    "position_delta": "<I",
    "df": "<I",
    "tf": "<H",
    "term_bytes": "<I",
}

READ_SIZE_KEY = {
    "<Q": 8,
    "<I": 4,
    "<H": 2,
    "<IH": 6,
}


@dataclass
class PostingList:
    doc_id: int
    doc_term_frequency: int
    positions: list[int]

    def __str__(self):
        positions = self.positions[:10] if len(self.positions) > 0 else ["."]
        positions[-1] = "..." if len(self.positions) > 10 else positions[-1]
        return f"{self.doc_id}:{self.doc_term_frequency}=>{positions}"

    def __repr__(self):
        return self.__str__()

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

    def __and__(self, other):
        return_term = Term(f"{self.term} AND {other.term}", 0, OrderedDict())
        for doc_id, posting_list in self.posting_lists.items():
            if doc_id in other.posting_lists:
                posting_list = copy.deepcopy(posting_list)
                posting_list.positions = []
                posting_list.doc_term_frequency = 0
                return_term.posting_lists[doc_id] = posting_list

        return return_term

    def __or__(self, other):
        return_term = Term(f"{self.term} OR {other.term}", 0, OrderedDict())
        for doc_id, posting_list in self.posting_lists.items():
            posting_list = copy.deepcopy(posting_list)
            posting_list.positions = []
            posting_list.doc_term_frequency = 0
            return_term.posting_lists[doc_id] = posting_list

        for doc_id, posting_list in other.posting_lists.items():
            if doc_id not in return_term.posting_lists:
                posting_list = copy.deepcopy(posting_list)
                posting_list.positions = []
                posting_list.doc_term_frequency = 0
                return_term.posting_lists[doc_id] = posting_list

        return return_term

    def __not__(self):
        raise NotImplementedError("NOT operator is not supported")

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
