import bisect
from collections import namedtuple
import pprint
import time
import copy
import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


PostingList = namedtuple("PostingList", ["doc_id", "doc_term_frequency", "positions"])


@dataclass
class MutablePostingList:
    doc_id: int
    doc_term_frequency: int
    positions: list[int] = field(default_factory=list)

    def update_doc_term_frequency(self, term_frequency: int):
        self.doc_term_frequency += term_frequency

    def update_with_positions(self, new_positions: list[int]):
        self.positions.extend(new_positions)


@dataclass
class Term:
    term: str
    document_frequency: int = 0
    # posting_lists: dict[int, PostingList | None] = field(default_factory=dict)
    posting_lists: list[PostingList] = field(default_factory=list)

    def __str__(self):
        posting_list = [x for x in self.posting_lists if x is not None][:10]
        posting_list[-1] = "..."
        str_posting_list = pprint.pformat(posting_list)

        return f"{self.term}({self.document_frequency},{str_posting_list})"

    def __repr__(self):
        return f"{self.document_frequency}"

    def __and__(self, other):
        return_term = Term(f"{self.term} AND {other.term}", 0)
        for doc_id, posting_list in self.posting_lists.items():
            if doc_id in other.posting_lists and posting_list is not None:
                posting_list = copy.deepcopy(posting_list)
                posting_list.positions = []
                posting_list.doc_term_frequency = 0
                return_term.posting_lists[doc_id] = posting_list

        return return_term

    def __or__(self, other):
        return_term = Term(f"{self.term} OR {other.term}", 0)
        for posting_list in self.posting_lists:
            doc_id = posting_list.doc_id
            if posting_list is None:
                continue

            posting_list = copy.deepcopy(posting_list)
            posting_list.positions = []
            posting_list.doc_term_frequency = 0
            return_term.posting_lists[doc_id] = posting_list

        for posting_list in other.posting_lists:
            doc_id = posting_list.doc_id
            if doc_id not in return_term.posting_lists and posting_list is not None:
                posting_list = copy.deepcopy(posting_list)
                posting_list.positions = []
                posting_list.doc_term_frequency = 0
                return_term.posting_lists[doc_id] = posting_list

        return return_term

    def __not__(self):
        raise NotImplementedError("NOT operator is not supported")

    def update(self, term, positions=False):
        self.document_frequency += term.document_frequency
        self.posting_lists.extend(term.posting_lists)

        """
        for doc_id, posting_list in term.posting_lists.items():
            if posting_list is not None:
                if positions:
                    self.update_with_postings(doc_id, posting_list.positions)
                else:
                    self.update_with_doc_term_frequency(
                        doc_id, posting_list.doc_term_frequency
                    )
        """

    def update_with_term(
        self,
        doc_ids: list[int],
        term_frequencies: list[int],
        position_lists: list[list[int]],
        positions=False,
    ):
        add_posting_lists = [
            PostingList(doc_ids[i], term_frequencies[i], position_lists[i])
            for i in range(len(doc_ids))
        ]
        self.posting_lists.extend(add_posting_lists)
        self.document_frequency += len(doc_ids)

        """
        assert all(
            [
                self.posting_lists[doc_id].doc_id == doc_id
                for doc_id in sorted(list(self.posting_lists.keys()))
            ]
        ), f"Posting list for term {self.term} is not sorted by document id"
        """

    def get_posting_list(self, doc_id: int) -> PostingList | None:
        if len(self.posting_lists) == 0 or doc_id < self.posting_lists[0].doc_id:
            if len(self.posting_lists) > 0:
                logger.info(
                    f"Doc ID {doc_id} is less than the first doc ID in the posting list {self.posting_lists[0].doc_id}"
                )
            return None

        if doc_id > self.posting_lists[-1].doc_id:
            logger.info(
                f"Doc ID {doc_id} is greater than the last doc ID in the posting list {self.posting_lists[-1].doc_id}"
            )
            return None

        idx = bisect.bisect_left([x.doc_id for x in self.posting_lists], doc_id)
        if idx < len(self.posting_lists) and self.posting_lists[idx].doc_id == doc_id:
            return self.posting_lists[idx]

        logger.info(f"Doc ID {doc_id} not found in the posting list")
        return None

    @staticmethod
    def build_term(
        term: str,
        sharded_values: list[tuple[int, list[int], list[int], list[list[int]]]],
        positions=False,
    ):
        start = time.time()
        ret_term = Term(term)
        if len(sharded_values) == 0:
            return

        if len(sharded_values) == 1:
            _, doc_ids, term_frequencies, position_list = sharded_values[0]
            for i in range(len(doc_ids)):
                if positions:
                    ret_term.update_with_postings(doc_ids[i], position_list[i])
                else:
                    ret_term.update_with_doc_term_frequency(
                        doc_ids[i], term_frequencies[i]
                    )
            return

        for _, doc_ids, term_frequencies, position_lists in sharded_values:
            for i in range(len(doc_ids)):
                if positions:
                    ret_term.update_with_postings(doc_ids[i], position_lists[i])
                else:
                    ret_term.update_with_doc_term_frequency(
                        doc_ids[i], term_frequencies[i]
                    )

        """
        iterators = {}
        for shard, doc_ids, term_frequencies, position_lists in sharded_values:
            iterators[shard] = iter(zip(doc_ids, term_frequencies, position_lists))

        heap = []
        for shard_id, iterator in iterators.items():
            try:
                doc_id, term_frequency, position_list = next(iterator)
                heap.append((doc_id, term_frequency, position_list, shard_id, iterator))
            except StopIteration:
                continue
        heapq.heapify(heap)

        while heap:
            doc_id, term_frequency, position_list, shard_id, iterator = heapq.heappop(
                heap
            )
            if positions:
                ret_term.update_with_postings(doc_id, position_list)
            else:
                ret_term.update_with_doc_term_frequency(doc_id, term_frequency)

            try:
                doc_id, term_frequency, position_list = next(iterator)
                heapq.heappush(
                    heap, (doc_id, term_frequency, position_list, shard_id, iterator)
                )
            except StopIteration:
                continue
        """

        logger.info(f"Built term {term} in {time.time() - start} seconds")
        return ret_term


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
