import bisect
from collections import namedtuple
import pprint
import time
import logging
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from numba import jit
from array import array

logger = logging.getLogger(__name__)


PostingList = namedtuple(
    "PostingList", ["doc_id", "doc_term_frequency", "position_idx"]
)


class PositionStore:
    def __init__(self):
        self.positions = []

    def add_positions(self, pos_list: list[int], position_idx=-1) -> None:
        if position_idx != -1:
            self.positions[position_idx].extend(pos_list)
            return

        pos_array = array("i", pos_list)
        self.positions.append(pos_array)
        return len(self.positions) - 1

    def get_positions(self, pos_id: int) -> list[int]:
        if pos_id == -1:
            return []

        return self.positions[pos_id]

    def __getitem__(self, pos_id: int) -> list[int]:
        return self.get_positions(pos_id)

    def __str__(self):
        return str(self.positions)


position_store = PositionStore()


@dataclass
class MutablePostingList:
    doc_id: int
    doc_term_frequency: int
    position_idx: int = -1

    def update_doc_term_frequency(self, term_frequency: int):
        self.doc_term_frequency += term_frequency

    def update_with_positions(self, new_positions: list[int]):
        position_store.add_positions(new_positions)


@jit(nopython=True)
def create_posting_lists_numba(
    doc_ids,
    term_frequencies,
    position_lists,
    offsets,
):
    result = []
    for i in range(len(doc_ids)):
        start, end = offsets[i], offsets[i + 1]
        position_list = position_lists[start:end]
        result.append(PostingList(doc_ids[i], term_frequencies[i], position_list))
    return result


@dataclass
class Term:
    term: str
    document_frequency: int = 0
    # posting_lists: dict[int, PostingList | None] = field(default_factory=dict)
    posting_lists: list[PostingList | MutablePostingList] = field(default_factory=list)

    def __str__(self):
        posting_list = [x for x in self.posting_lists if x is not None][:10]
        if len(self.posting_lists) > 10:
            posting_list[-1] = "..."
        str_posting_list = pprint.pformat(posting_list)

        return f"{self.term}({self.document_frequency},{str_posting_list})"

    def __repr__(self):
        return f"{self.document_frequency}"

    def update_with_doc_term_frequency(self, doc_id: int, term_frequency: int):
        posting_list = self.get_posting_list(doc_id)
        if posting_list is None:
            self.posting_lists.append(PostingList(doc_id, term_frequency, []))
        else:
            posting_list.doc_term_frequency += term_frequency

        self.document_frequency += 1

    def update_with_postings(self, doc_id: int, new_positions: list[int]):
        posting_list = self.get_posting_list(doc_id)
        if posting_list is None:
            self.posting_lists.append(
                PostingList(doc_id, len(new_positions), new_positions)
            )
        else:
            posting_list.doc_term_frequency += 1
            posting_list.positions.extend(new_positions)

        self.document_frequency += 1

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
        start = time.time()
        doc_ids = np.asarray(doc_ids, dtype=np.uint64)
        term_frequencies = np.asarray(term_frequencies, dtype=np.uint32)

        lengths = np.array([len(x) for x in position_lists], dtype=np.uint32)
        offsets = np.zeros(len(lengths) + 1, dtype=np.uint32)
        np.cumsum(lengths, out=offsets[1:])

        total_length = offsets[-1]
        concat = np.zeros(total_length, dtype=np.uint32)

        for i, position_list in enumerate(position_lists):
            concat[offsets[i] : offsets[i + 1]] = position_list

        """
        position_idxs = np.ones_like(doc_ids, dtype=np.int32) * -1
        if positions:
            for i in range(len(doc_ids)):
                position_idx = position_store.add_positions(position_lists[i])
                position_idxs[i] = position_idx
        """

        add_posting_lists = create_posting_lists_numba(
            doc_ids, term_frequencies, concat, offsets
        )
        end = time.time()

        print(
            "Time to construct posting lists:",
            end - start,
            len(add_posting_lists),
            doc_ids[:10],
            term_frequencies[:10],
            add_posting_lists[:10],
            position_lists[:10],
        )
        self.posting_lists.extend(add_posting_lists)
        self.document_frequency += len(doc_ids)

        print(
            "Result",
            [x.position_idx for x in self.posting_lists[:10]],
        )

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
            return None

        if doc_id > self.posting_lists[-1].doc_id:
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
