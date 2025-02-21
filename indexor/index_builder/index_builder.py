import os
import logging
import time
import struct
import json
import filelock

from collections import OrderedDict

from indexor.index_builder.constants import SIZE_KEY
from indexor.structures import DocMetadata, Term
from indexor.metadata_score import metadata_score
from utils.varint import encode

logger = logging.getLogger(__name__)


class DocumentShardedIndexBuilder:
    def __init__(self, index_path: str, shard: int = -1, shard_size: int = 50_000):
        """
        Initialises the index builder for a specific shard
        When a shard is set we have the identitiy of the shard.
        If no shard is set, we have a single shard index.

        `shard_size` is the number of documents to process before flushing the index to disk.
        As such, when the index flushes to disk, the index is split into multiple shards
        Work for merging these so-called `sub-shards` is performed by the `IndexMerger` class.

        We save the shard as `shard_{shard}_{sub_shard}.index` and `shard_{shard}_{sub_shard}.offset`
        Additionally, we save the positional index as `pos_shard_{shard}_{sub_shard}.index` and `pos_shard_{shard}_{sub_shard}.offset`

        `shard_0_0.offset`:
        - term_length: int (byte length of term)
        - term: bytes
        - offset: int (offset in `shard_0_0.index`)

        `shard_0_0.index`:
            - document_frequency: int
            - doc_id_delta: int
            - doc_term_frequency: int

        `pos_shard_0_0.index`:
            - document_frequency: int
            - doc_id_delta: int
            - doc_term_frequency: int
            - position_count: int
            - position_delta: int

        Note: we specify `doc_term_frequency` and `position_count` separately as not all terms in a document have valid positions (links/alt-text)
        """
        self.index_path = index_path
        self.shard = shard
        self.curr_shard = 0

        self.shard_size = shard_size
        self.term_map: dict[str, Term] = {}
        self.doc_map: dict[int, DocMetadata] = {}
        self.doc_stats_map: DocMetadata = DocMetadata.default()

        self.current_docs = 0
        self.doc_count = 0

        self.min_doc_id = float("inf")
        self.max_doc_id = float("-inf")
        self.start = time.time()

        if not os.path.exists(self.index_path):
            logger.info(f"Creating index directory: {self.index_path}")
            os.makedirs(self.index_path)

    def __str__(self):
        return ""

    def add_document(
        self,
        doc_id: int,
        doc_terms: dict[str, list[int]] = {},
        doc_metadata=DocMetadata.default(),
    ):
        """
        Add a document to the index.

        Args:
            doc_id: int
            doc_terms: dict
            doc_length: int
        """
        if doc_id < self.min_doc_id:
            self.min_doc_id = doc_id
        if doc_id > self.max_doc_id:
            self.max_doc_id = doc_id

        self.current_docs += 1
        self.doc_count += 1
        for term, positions in doc_terms.items():
            if term not in self.term_map:
                self.term_map[term] = Term(term)

            self.term_map[term].update_with_postings(doc_id, positions)

        self.doc_map[doc_id] = doc_metadata
        if self.current_docs >= self.shard_size:
            self.flush(self.shard, self.curr_shard)

            self.curr_shard += 1
            self.current_docs = 0

    def flush(self, shard: int = -1, sub_shard: int = 0):
        """
        By default, flushes both a non-positional and positional index to disk.

        sub-shard is only set when the shard is set as part of the class
            (i.e. where the index is not sharded)
        """
        if sub_shard == 0:
            sub_shard = self.curr_shard if self.shard != -1 else 0

        self._flush_shard(shard, sub_shard=sub_shard, shard_filename="shard")
        self._flush_doc_map(shard, sub_shard)

        del self.term_map
        self.term_map = {}
        self.doc_map = {}
        self.min_doc_id = float("inf")
        self.max_doc_id = float("-inf")
        self.start = time.time()

    def _flush_shard(
        self,
        shard: int = -1,
        sub_shard: int = 0,
        shard_filename="shard",
    ):
        """
        Flush the current term map to disk.
        Saves the term map to a file in the index directory.

        If `flush_positions` is True, then the positional index is flushed to disk.

        When the shard is set as part of the class (i.e. where the index is not sharded),
            then the shard is set to the current shard
        When the shard is not set, then the shard is set to the class shard (e.g. shard_0)
            and sub_shard is set to the current shard based on the number of documents processed
        """
        shard = self.curr_shard if shard == -1 else self.shard

        shard_str = str(shard)
        shard_str = f"{shard}_{sub_shard}"

        index_path = os.path.join(
            self.index_path, f"{shard_filename}_{shard_str}.index"
        )
        frequency_path = os.path.join(
            self.index_path, f"{shard_filename}_{shard_str}.index"
        )
        position_offset_path = os.path.join(
            self.index_path, f"{shard_filename}_{shard_str}.position"
        )
        offset_dict = OrderedDict()
        position_offset_dict = OrderedDict()
        skip_offset_dict = OrderedDict()

        logger.info(
            f"Flushing shard {shard_str} with {len(self.term_map)} terms to {frequency_path}. Documents processed: {self.current_docs} in {time.time()-self.start}. Min doc ID: {self.min_doc_id}. Max doc ID: {self.max_doc_id}"
        )

        skip_interval = 16
        minimum_for_skip_list = 32

        lock_flush_file = filelock.FileLock(frequency_path + ".lock")
        lock_position_file = filelock.FileLock(position_offset_path + ".lock")
        logger.debug(f"Locking {frequency_path}")
        with lock_flush_file, lock_position_file:
            with (
                open(frequency_path, "wb") as f,
                open(position_offset_path, "wb") as pos_f,
            ):
                # Sort the terms to ensure consistent ordering
                sorted_terms = sorted(self.term_map.keys())
                for term in sorted_terms:
                    term_obj = self.term_map[term]

                    # Save the offset of the term in the file in `offset_dict`
                    offset_dict[term] = f.tell()

                    assert (
                        term_obj.document_frequency == len(term_obj.posting_lists)
                    ), f"Document frequency {term_obj.document_frequency} does not match the number of posting lists {(term_obj.posting_lists)}"
                    f.write(encode(term_obj.document_frequency))

                    prev_doc_id = 0
                    position_offset_dict[term] = pos_f.tell()

                    for _, posting in enumerate(term_obj.posting_lists):
                        delta = posting.doc_id - prev_doc_id

                        f.write(encode(delta))
                        f.write(encode(posting.doc_term_frequency))
                        prev_doc_id = posting.doc_id

                        filter_positions = [p for p in posting.positions if p >= 0]
                        filter_positions = sorted(filter_positions)
                        f.write(encode(len(filter_positions)))

                        prev_position = 0
                        for position in filter_positions:
                            pos_delta = position - prev_position
                            pos_f.write(encode(pos_delta))
                            prev_position = position

        logger.debug(f"Unlocked {frequency_path}")
        logger.debug(f"Unlocked {position_offset_path}")

        offset_path = os.path.join(
            self.index_path, f"{shard_filename}_{shard_str}.offset"
        )
        logger.info(
            f"Flushing offset for shard {shard_str} with {len(self.term_map)} with filename {offset_path}"
        )

        lock_offset_file = filelock.FileLock(offset_path + ".lock")
        logger.debug(f"Locking {offset_path}")
        with lock_offset_file:
            with open(offset_path, "wb") as offset_f:
                for term, offset in offset_dict.items():
                    position_offset = position_offset_dict.get(term, -1)
                    assert (
                        position_offset != -1
                    ), f"Position offset for {term} not found {offset}"

                    term_bytes = term.encode("utf-8")

                    offset_f.write(struct.pack(SIZE_KEY["term_bytes"], len(term_bytes)))
                    offset_f.write(term_bytes)
                    offset_f.write(struct.pack(SIZE_KEY["offset"], offset))
                    offset_f.write(struct.pack(SIZE_KEY["offset"], position_offset))

        logger.debug(f"Unlocked {offset_path}")

    def _flush_doc_map(self, shard: int = -1, sub_shard: int = 0):
        """
        Flush the document map to disk.
        Saves the document map to a file in the index directory.

        When the shard is set as part of the class (i.e. where the index is not sharded),
            then the shard is set to the current shard
        When the shard is not set, then the shard is set to the class shard (e.g. shard_0)
            and sub_shard is set to the current shard based on the number of documents processed
        """
        shard = self.curr_shard if shard == -1 else self.shard

        shard_str = str(shard)
        shard_str = f"{shard}_{sub_shard}"

        doc_map_path = os.path.join(self.index_path, f"doc_shard_{shard_str}.index")
        doc_offset = os.path.join(self.index_path, f"doc_shard_{shard_str}.offset")

        logger.info(
            f"Flushing doc map for shard {shard_str} with {len(self.doc_map)} documents to {doc_map_path}. Documents processed: {self.current_docs} in {time.time()-self.start}. Min doc ID: {self.min_doc_id}. Max doc ID: {self.max_doc_id}"
        )

        doc_offset_dict = OrderedDict()

        lock_flush_file = filelock.FileLock(doc_map_path + ".lock")
        lock_offset_file = filelock.FileLock(doc_offset + ".lock")
        logger.debug(f"Locking {doc_map_path}")
        with lock_flush_file, lock_offset_file:
            sum_doc_length = 0
            with open(doc_map_path, "wb") as f:
                for doc_id, doc_metadata in self.doc_map.items():
                    doc_offset_dict[doc_id] = f.tell()
                    sum_doc_length += doc_metadata.doc_length.get_value()
                    f.write(
                        struct.pack(
                            SIZE_KEY["doc_length"], doc_metadata.doc_length.get_value()
                        )
                    )
                    f.write(
                        struct.pack(
                            SIZE_KEY["doc_metadatascore"],
                            metadata_score(
                                score=doc_metadata.score.get_value(),
                                viewcount=doc_metadata.viewcount.get_value(),
                                creationdate=doc_metadata.creationdate.get_value(),
                                reputationuser=doc_metadata.owneruserid.get_value(),
                                answercount=doc_metadata.answercount.get_value(),
                                commentcount=doc_metadata.commentcount.get_value(),
                                favoritecount=doc_metadata.favoritecount.get_value(),
                                minmax_dict={},
                            ),
                        )
                    )  # metadata_score
                    f.write(
                        struct.pack(
                            SIZE_KEY["doc_score"], doc_metadata.score.get_value()
                        )
                    )
                    f.write(
                        struct.pack(
                            SIZE_KEY["doc_viewcount"],
                            doc_metadata.viewcount.get_value(),
                        )
                    )
                    f.write(
                        struct.pack(
                            SIZE_KEY["doc_owneruserid"],
                            doc_metadata.owneruserid.get_value(),
                        )
                    )
                    f.write(
                        struct.pack(
                            SIZE_KEY["doc_answercount"],
                            doc_metadata.answercount.get_value(),
                        )
                    )
                    f.write(
                        struct.pack(
                            SIZE_KEY["doc_commentcount"],
                            doc_metadata.commentcount.get_value(),
                        )
                    )
                    f.write(
                        struct.pack(
                            SIZE_KEY["doc_favoritecount"],
                            doc_metadata.favoritecount.get_value(),
                        )
                    )
                    raw_bytes_display_name = doc_metadata.ownerdisplayname.encode(
                        "utf-8"
                    )
                    display_name_size = len(raw_bytes_display_name)
                    f.write(
                        struct.pack(SIZE_KEY["doc_ownerdisplayname"], display_name_size)
                    )
                    f.write(raw_bytes_display_name)

                    raw_bytes_tags = doc_metadata.tags.encode("utf-8")
                    tags_size = len(raw_bytes_tags)
                    f.write(struct.pack(SIZE_KEY["doc_tags"], tags_size))
                    f.write(raw_bytes_tags)

                    f.write(
                        struct.pack(
                            SIZE_KEY["doc_creationdate"],
                            doc_metadata.creationdate.get_value(),
                        )
                    )

                    self.doc_stats_map = self._update_doc_stats_map(
                        doc_metadata=doc_metadata
                    )

            with open(doc_offset, "wb") as f:
                f.write(struct.pack(SIZE_KEY["doc_count"], len(doc_offset_dict)))
                f.write(struct.pack(SIZE_KEY["offset"], sum_doc_length))
                for doc_id, offset in doc_offset_dict.items():
                    f.write(struct.pack(SIZE_KEY["doc_id"], doc_id))
                    f.write(struct.pack(SIZE_KEY["offset"], offset))

        doc_meta = os.path.join(self.index_path, "shard.meta")
        lock_doc_meta_file = filelock.FileLock(doc_meta + ".lock")
        logger.debug(f"Locking {doc_meta}")
        with lock_doc_meta_file:
            if not os.path.exists(doc_meta):
                shard_bounds = {}
            else:
                try:
                    with open(doc_meta) as f:
                        shard_bounds = json.load(f)
                except json.JSONDecodeError:
                    shard_bounds = {}
                    logger.error(f"Error reading {doc_meta}")

            doc_bounds = shard_bounds.get(str(shard), {}).get(
                "bounds", [float("inf"), float("-inf")]
            )
            curr_min_doc_id, curr_max_doc_id = doc_bounds

            doc_metadata = DocMetadata.from_json(
                shard_bounds.get(str(shard), {}).get("metadata", {})
            )
            doc_metadata = doc_metadata.update(self.doc_stats_map)

            shard_bounds[str(shard)] = {
                "bounds": [
                    min(curr_min_doc_id, self.min_doc_id),
                    max(curr_max_doc_id, self.max_doc_id),
                ],
                "metadata": doc_metadata.to_json(),
            }

            with open(doc_meta, "w") as f:
                json.dump(shard_bounds, f)

            logger.info(f"Written {shard_bounds} to {doc_meta}")

        logger.debug(f"Unlocked {doc_meta}")
        logger.info(
            f"Flushed doc map for shard {shard_str} with {len(self.doc_map)} documents to {doc_map_path}"
        )

    def _update_doc_stats_map(self, doc_metadata: DocMetadata):
        """
        Update the document stats map with the document metadata.
        """
        return self.doc_stats_map.update(doc_metadata)
