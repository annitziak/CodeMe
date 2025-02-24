import bisect
from collections import namedtuple
import pprint
import time
import datetime
import logging
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from numba import jit

logger = logging.getLogger(__name__)


PostingList = namedtuple("PostingList", ["doc_id", "doc_term_frequency", "positions"])


@dataclass
class Stat:
    value: None | int = None
    min: int = float("inf")
    max: int = float("-inf")

    def __post_init__(self):
        if self.value and isinstance(self.value, int):
            if not self.min:
                self.min = self.value
            if not self.max:
                self.max = self.value

            self.min = min(self.min, self.value)
            self.max = max(self.max, self.value)

    def update(self, new_value: "Stat | int", reset=False):
        if reset:
            self.value = None
            self.min = float("inf")
            self.max = float("-inf")

        if self.value is None:
            if isinstance(new_value, Stat):
                self.value = new_value.value
            else:
                self.value = new_value
        if self.min is None:
            self.min = new_value
        if self.max is None:
            self.max = new_value

        if isinstance(new_value, str) or new_value is None:
            return

        if isinstance(new_value, Stat):
            if new_value.value is not None:
                self.value = new_value.value
            if new_value.min is not None:
                self.min = min(new_value.min, self.min)
            if new_value.max is not None:
                self.max = max(new_value.max, self.max)
            return

        if new_value > -float("inf") and new_value < float("inf"):
            self.min = min(self.min, new_value)
            self.max = max(self.max, new_value)
            self.value = new_value

    def get_value(self):
        if self.value is None:
            return 0
        if not isinstance(self.value, int):
            return 0

        return self.value


@dataclass
class DocMetadata:
    creationdate: Stat
    score: Stat
    viewcount: Stat
    owneruserid: Stat
    ownerdisplayname: str
    tags: str
    answercount: Stat
    commentcount: Stat
    favoritecount: Stat
    hasacceptedanswer: bool = False

    title: str = ""
    body: str = ""
    doc_length: Stat = field(default_factory=lambda: Stat(0))

    @staticmethod
    def default():
        return DocMetadata(
            Stat(), Stat(), Stat(), Stat(), "", "", Stat(), Stat(), Stat()
        )

    @staticmethod
    def from_json(json_data):
        def get_stat(json_data, key):
            min_value = json_data.get(f"{key}_min", np.inf)
            max_value = json_data.get(f"{key}_max", -1 * np.inf)
            value = None

            return Stat(value, min_value, max_value)

        return DocMetadata(
            get_stat(json_data, "creationdate"),
            get_stat(json_data, "score"),
            get_stat(json_data, "viewcount"),
            get_stat(json_data, "owneruserid"),
            json_data.get("ownerdisplayname", ""),
            json_data.get("tags", ""),
            get_stat(json_data, "answercount"),
            get_stat(json_data, "commentcount"),
            get_stat(json_data, "favoritecount"),
            get_stat(json_data, "doc_length"),
        )

    def __getitem__(self, key):
        if key == "creationdate":
            return self.creationdate
        if key == "score":
            return self.score
        if key == "viewcount":
            return self.viewcount
        if key == "owneruserid":
            return self.owneruserid
        if key == "ownerdisplayname":
            return self.ownerdisplayname
        if key == "tags":
            return self.tags
        if key == "answercount":
            return self.answercount
        if key == "commentcount":
            return self.commentcount
        if key == "favoritecount":
            return self.favoritecount
        if key == "doc_length":
            return self.doc_length
        if key == "title":
            return self.title
        if key == "body":
            return self.body

    def to_json(self):
        def get_stat(stat, key):
            return {
                key: stat.value,
                f"{key}_min": stat.min,
                f"{key}_max": stat.max,
            }

        return {
            **get_stat(self.creationdate, "creationdate"),
            **get_stat(self.score, "score"),
            **get_stat(self.viewcount, "viewcount"),
            **get_stat(self.owneruserid, "owneruserid"),
            **get_stat(self.answercount, "answercount"),
            **get_stat(self.commentcount, "commentcount"),
            **get_stat(self.favoritecount, "favoritecount"),
            **get_stat(self.doc_length, "doc_length"),
        }

    def __post_init__(self):
        if not self.ownerdisplayname:
            self.ownerdisplayname = ""
        if not self.tags:
            self.tags = ""
        if not self.creationdate:
            self.creationdate = Stat(0)
        elif isinstance(self.creationdate, str):
            self.creationdate = Stat(
                int(
                    time.mktime(
                        datetime.datetime.strptime(
                            self.creationdate, "%Y-%m-%dT%H:%M:%S.%f"
                        ).timetuple()
                    )
                )
            )
        elif isinstance(self.creationdate, datetime.datetime):
            self.creationdate = Stat(int(time.mktime(self.creationdate.timetuple())))
        if not self.score or isinstance(self.score, int):
            self.score = Stat(self.score)
        if not self.viewcount or isinstance(self.viewcount, int):
            self.viewcount = Stat(self.viewcount)
        if not self.owneruserid or isinstance(self.owneruserid, int):
            self.owneruserid = Stat(self.owneruserid)
        if not self.answercount or isinstance(self.answercount, int):
            self.answercount = Stat(self.answercount)
        if not self.commentcount or isinstance(self.commentcount, int):
            self.commentcount = Stat(self.commentcount)
        if not self.favoritecount or isinstance(self.favoritecount, int):
            self.favoritecount = Stat(self.favoritecount)
        if not self.doc_length or isinstance(self.doc_length, int):
            self.doc_length = Stat(self.doc_length)

        if self.owneruserid.value is not None:
            self.owneruserid.value = max(0, self.owneruserid.value)
        if self.score.value is not None:
            self.score.value = max(0, self.score.value)
        if self.viewcount.value is not None:
            self.viewcount.value = max(0, self.viewcount.value)
        if self.answercount.value is not None:
            self.answercount.value = max(0, self.answercount.value)
        if self.commentcount.value is not None:
            self.commentcount.value = max(0, self.commentcount.value)
        if self.favoritecount.value is not None:
            self.favoritecount.value = max(0, self.favoritecount.value)

    def __len__(self):
        return 10

    def update(self, other: "DocMetadata"):
        self.creationdate.update(other.creationdate)
        self.score.update(other.score)
        self.viewcount.update(other.viewcount)
        self.owneruserid.update(other.owneruserid)
        self.answercount.update(other.answercount)
        self.commentcount.update(other.commentcount)
        self.favoritecount.update(other.favoritecount)
        self.doc_length.update(other.doc_length)

        self.tags = other.tags
        self.title = other.title
        self.body = other.body
        self.hasacceptedanswer = other.hasacceptedanswer

        return self

    def update_with_raw(
        self,
        creationdate=None,
        score=None,
        viewcount=None,
        owneruserid=None,
        ownerdisplayname="",
        tags="",
        title="",
        body="",
        answercount=None,
        commentcount=None,
        favoritecount=None,
        doc_length=None,
        hasacceptedanswer=False,
    ):
        self.creationdate.update(creationdate)
        self.score.update(score)
        self.viewcount.update(viewcount)
        self.owneruserid.update(owneruserid)
        self.ownerdisplayname = ownerdisplayname
        self.title = title
        self.body = body
        self.tags = tags
        self.answercount.update(answercount)
        self.commentcount.update(commentcount)
        self.favoritecount.update(favoritecount)
        self.doc_length.update(doc_length)
        self.hasacceptedanswer = hasacceptedanswer

        return self


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

        return f"{self.term.__str__()}({self.document_frequency},{str_posting_list})"

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
                try:
                    np.concatenate(position_lists, out=concat)
                except ValueError as e:
                    print("Error concatenating arrays", e)
                    print(position_lists[: -len(position_lists) - 10])
                    print(len(position_lists))
                    print(position_lists[0].shape)
                    print(concat.shape)
                    print(offsets.shape)
                    print(offsets)
                    print(lengths)
                    raise ValueError()
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
