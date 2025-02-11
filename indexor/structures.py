import bisect
from collections import namedtuple
import pprint
import time
import logging
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from numba import jit

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


@jit(nopython=True, cache=True, fastmath=True)
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


@jit(nopython=True, cache=True, fastmath=True)
def concat_positions_numba(positions_list, lengths):
    total_lengths = np.sum(lengths)
    concat = np.zeros(total_lengths, dtype=np.uint32)

    start = 0
    for i, pos_list in enumerate(positions_list):
        if hasattr(pos_list, "__len__"):
            concat[start : start + lengths[i]] = pos_list
            start += lengths[i]

    return concat


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
        t_start = time.time()

        doc_ids = np.asarray(doc_ids, dtype=np.uint32)
        term_frequencies = np.asarray(term_frequencies, dtype=np.uint32)

        if len(position_lists) > 0 and isinstance(position_lists[0], list):
            lengths = np.fromiter(
                (len(x) for x in position_lists),
                dtype=np.uint32,
                count=len(position_lists),
            )
        else:
            lengths = np.zeros(len(position_lists), dtype=np.uint32)

        offsets = np.empty(len(lengths) + 1, dtype=np.uint32)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])

        # Create concatenated array and fill in one go if possible
        concat = np.zeros(offsets[-1], dtype=np.uint32)
        if position_lists:
            # If position_lists are already numpy arrays, use direct assignment
            if isinstance(position_lists[0], np.ndarray):
                np.concatenate(position_lists, out=concat)
            else:
                # For lists, use efficient slicing
                start = 0
                for pos_list in position_lists:
                    if hasattr(pos_list, "__len__"):
                        concat[start : start + len(pos_list)] = pos_list
                        start += len(pos_list)

        add_posting_lists = create_posting_lists_numba(
            doc_ids, term_frequencies, concat, offsets
        )
        t_end = time.time()

        self.posting_lists.extend(add_posting_lists)
        self.document_frequency += len(doc_ids)
        print(
            "Time to construct posting lists:",
            t_end - t_start,
        )

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
