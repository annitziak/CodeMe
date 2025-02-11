import os
import logging
import glob
import heapq
import struct
import shutil
import filelock
import concurrent.futures
import marisa_trie
import json
import datrie

from collections import OrderedDict
from dataclasses import dataclass

from indexor.index_builder.constants import SIZE_KEY, READ_SIZE_KEY
from indexor.index_builder.shard_reader import ShardReader

logger = logging.getLogger(__name__)


@dataclass
class HeapItem:
    term: str
    offset: int
    pos_offset: int
    reader: ShardReader

    def __lt__(self, other):
        return self.term < other.term

    def __gt__(self, other):
        return self.term > other.term

    def __iter__(self):
        return iter((self.term, self.offset, self.pos_offset, self.reader))

    def __getitem__(self, key):
        return (self.term, self.offset, self.pos_offset, self.reader)[key]


class IndexMerger:
    def __init__(self, index_path: str, output_dir: str = ""):
        self.index_path = os.path.abspath(index_path)
        self.output_dir = output_dir if output_dir != "" else self.index_path

    def cleanup(self):
        if os.path.exists(self.output_dir):
            logger.info(f"Removing existing index directory: {self.output_dir}")

            for f in os.listdir(self.output_dir):
                if "shard" in f or "postings" in f or "terms" in f or "positions" in f:
                    if "finished" in f:
                        continue

                    logger.info(f"Removing {f}")
                    os.remove(os.path.join(self.output_dir, f))

    def post_merge_cleanup(self):
        logger.info("Removing temporary files and lock files")
        for f in os.listdir(self.output_dir):
            if not os.path.exists(os.path.join(self.output_dir, f)):
                continue

            if ".temp" in f:
                logger.info(f"Removing {f}")
                os.remove(os.path.join(self.output_dir, f))
            elif ".lock" in f:
                logger.info(f"Removing {f}")
                os.remove(os.path.join(self.output_dir, f))
            elif "terms_" in f and ".fst" in f:
                logger.info(f"Removing {f}")
                os.remove(os.path.join(self.output_dir, f))
            elif "docs_" in f and ".fst" in f:
                logger.info(f"Removing {f}")
                os.remove(os.path.join(self.output_dir, f))

    def build_datrie_from_generator(self, generator, chunk_size=100000):
        chunk = []
        min_char = "a"
        max_char = "z"

        for term, shard, offset in generator:
            min_char = min(min_char, min(term))
            max_char = max(max_char, max(term))

            chunk.append((term, shard, offset))
            if len(chunk) >= chunk_size:
                break

        trie = datrie.Trie(ranges=[(min_char, max_char)])

        for term, shard, offset in chunk:
            packed_value = struct.pack(SIZE_KEY["offset_shard"], shard, offset)
            trie[term] = packed_value

        for term, shard, offset in generator:
            packed_value = struct.pack(SIZE_KEY["offset_shard"], shard, offset)
            if term < min_char or term > max_char:
                raise ValueError(f"Invalid term {term}")
            trie[term] = packed_value

        return trie

    def build_all_term_fsts(
        self,
        term_offsets: OrderedDict | None = None,
        shards: int = 24,
        prefix="shard",
        save_filename="terms",
    ):
        items = []

        def gen_term_offset(term_offsets):
            if term_offsets is None:
                term_offsets = OrderedDict()
                for i in range(shards):
                    reader = ShardReader(
                        os.path.join(self.index_path, f"{prefix}_{i}.index"),
                        os.path.join(self.index_path, f"{prefix}_{i}.position"),
                        os.path.join(self.index_path, f"{prefix}_{i}.offset"),
                    )
                    while True:
                        term, offset, pos_offset = reader.next_term()
                        if term is None or offset is None or pos_offset is None:
                            break

                        yield term, i, offset, pos_offset

                    reader.close()

                    logger.info(f"Finished reading shard {i}")
                    yield None, None, None, None

            for term, shard, offset, pos_offset in term_offsets:
                yield term, shard, offset, pos_offset

        for term, shard, offset, pos_offset in gen_term_offset(term_offsets):
            if term is None or offset is None or shard is None:
                fst = marisa_trie.BytesTrie(items)
                fst.save(os.path.join(self.output_dir, f"{save_filename}.fst"))
                logger.info(f"Saved {save_filename}.fst")
                continue

            encoded_value = struct.pack(
                SIZE_KEY["pos_offset_shard"], shard, offset, pos_offset
            )
            items.append((term, encoded_value))

        fst = marisa_trie.BytesTrie(items)
        fst.save(os.path.join(self.output_dir, f"{save_filename}.fst"))
        # datrie = self.build_datrie_from_generator(gen_term_offset(term_offsets))
        # datrie.save(os.path.join(self.output_dir, f"{save_filename}.dat"))
        # return datrie

        return fst

    def build_all_doc_fsts(
        self,
        doc_offsets: OrderedDict | None = None,
        save_filename="docs",
        shards=24,
        prefix="doc_shard",
    ):
        items = []

        def gen_term_offset(term_offsets):
            if term_offsets is None:
                term_offsets = OrderedDict()
                for i in range(shards):
                    f_offset_filename = os.path.join(
                        self.index_path, f"{prefix}_{i}.offset"
                    )
                    with open(f_offset_filename, "rb") as f_offset:
                        doc_count = struct.unpack(
                            SIZE_KEY["doc_count"],
                            f_offset.read(READ_SIZE_KEY[SIZE_KEY["doc_count"]]),
                        )[0]

                        for _ in range(doc_count):
                            try:
                                doc_id = struct.unpack(
                                    SIZE_KEY["doc_id"],
                                    f_offset.read(READ_SIZE_KEY[SIZE_KEY["doc_id"]]),
                                )[0]
                                offset = struct.unpack(
                                    SIZE_KEY["offset"],
                                    f_offset.read(READ_SIZE_KEY[SIZE_KEY["offset"]]),
                                )[0]
                                yield doc_id, i, offset
                            except struct.error as e:
                                logger.error(
                                    f"Error reading {f_offset_filename} after length {len(term_offsets)} {e}"
                                )
                                break
            else:
                for item in term_offsets:
                    yield item

        for doc_id, shard, offset in gen_term_offset(doc_offsets):
            # print(doc_id, shard, offset)
            encoded_value = struct.pack(SIZE_KEY["offset_shard"], shard, offset)
            items.append((str(doc_id), encoded_value))

        print(len(items))

        fst = marisa_trie.BytesTrie(items)
        fst.save(os.path.join(self.output_dir, f"{save_filename}.fst"))
        # datrie = self.build_datrie_from_generator(gen_term_offset(term_offsets))
        # datrie.save(os.path.join(self.output_dir, f"{save_filename}.dat"))
        # return datrie

        return fst

    def merge(self, shards: int = 24, merge_subshards=False, force_merge=False):
        """
        Merges N (shards) into a single index
        We only merge the shards if there is more than two shards (unless `force_merge` is True)

        If `merge_subshards` is True, then we merge the sub-shards into a single shard
            (i.e. `shard_0_0.index`, `shard_0_1.index`, ..., `shard_0_N.index` => `shard_0.index`)
        If `merge_subshards` is False, then we merge all shards into a single index
            (i.e. `shard_0.index`, `shard_1.index`, ..., `shard_N.index` => `postings.bin`)

        If `force_merge` is True, then we merge all shards regardless of the number of files

        We call merge for both the positional and non-positional indexes
        """
        if shards == -1 or not merge_subshards:
            logger.info("Merging all shards")
            self._merge_all()
            return

        for i in range(shards):
            self._merge(
                shard_filename="shard",
                save_filename="postings",
                force_merge=force_merge,
                shard=i,
            )
            self._merge_docs(shard=i, force_merge=force_merge)

    def _merge_all(self):
        self._merge(
            shard_filename="shard",
            save_filename="postings",
            force_merge=True,
        )
        self._merge_docs(force_merge=True)

    def _merge(
        self,
        shard_filename="shard",
        save_filename="postings",
        force_merge=False,
        shard=-1,
    ):
        """
        Merges the shards or sub-shards into a single index

        `shard` determines the shard to be merged. If -1, then all shards are merged
        based on the glob pattern `shard_*.index` or `pos_shard_*.index`
        When set to not -1, then the glob pattern is `shard_{shard}_*.index`

        `shard_filename` is the prefix of the shard files (e.g. `shard` or `pos_shard`)
        `save_filename` is the name of the output file (e.g. `postings` or `positions`)
        `force_merge` determines if the merge should occur regardless of the number of files

        When shard is set to -1, we merge all shards into a single index and save as `postings.bin` or `positions.bin`
            Then we build the FST for the terms and save as `terms.fst` or `pos_terms.fst`

        When shard is not set to -1, we merge the sub-shards into a single shard and save as `shard_0.index` or `pos_shard_0.index`
            Then we build the offset file for the terms and save as `shard_0.offset` or `pos_shard_0.offset`

        Performs the merge by:
            1. Reading the first term and offset from each shard file
            2. Pushes the term, offset, and reader to a heap (first from each shard)
            3. Pops the smallest term from the heap
            4. Reads the postings for the term from the reader
            5. Merges the postings from the other readers with the same term
                Ensure that the term from the initial reader has been left unread
            6. Writes the merged postings to the output file

        Note: This assumes that the shard files are sorted by term
        """
        # Will be `shard_0.index` or `pos_shard_0.index`
        base_shard_files = glob.glob(
            os.path.join(self.index_path, f"{shard_filename}_{shard}.index")
        )
        base_position_files = glob.glob(
            os.path.join(self.index_path, f"{shard_filename}_{shard}.position")
        )
        base_offset_files = glob.glob(
            os.path.join(self.index_path, f"{shard_filename}_{shard}.offset")
        )

        # Will be `shard_0_0.index` or `pos_shard_0_0.index` if merging sub-shards
        # Will be `shard_*.index` or `pos_shard_*.index` if merging all shards
        sub_shard_name = f"{shard}_*" if shard != -1 else "*"
        shard_files = glob.glob(
            os.path.join(self.index_path, f"{shard_filename}_{sub_shard_name}.index")
        )
        position_files = glob.glob(
            os.path.join(self.index_path, f"{shard_filename}_{sub_shard_name}.position")
        )
        offset_files = glob.glob(
            os.path.join(self.index_path, f"{shard_filename}_{sub_shard_name}.offset")
        )

        if len(shard_files) < 2 and len(offset_files) < 2 and not force_merge:
            logger.debug(f"Skipping shard {shard} with {len(shard_files)} files")
            return

        if len(shard_files) != len(offset_files):
            return None, None, None

        if len(base_shard_files) > 0 and len(base_offset_files) > 0:
            shard_files.insert(0, base_shard_files[0])
            position_files.insert(0, base_position_files[0])
            offset_files.insert(0, base_offset_files[0])

        def extract_shard(x):
            final_split = x.split("/")[-1]
            start_index = final_split.index("shard")
            final_split = final_split[start_index:]
            possible_nums = final_split.split("_")
            if len(possible_nums) == 2:
                # "shard_0.index" or "pos_shard_0.index"
                return (int(possible_nums[1].split(".")[0]), 0)
            elif len(possible_nums) == 3:
                # "shard_0_0.index" or "pos_shard_0_0.index
                return (int(possible_nums[1]), int(possible_nums[2].split(".")[0]))
            else:
                raise ValueError(f"Invalid shard file: {x} {possible_nums}")

        if len(shard_files) <= 1:
            # Only base shard file is present skip merging even if force_merge
            assert len(offset_files) == len(shard_files) and len(position_files) == len(
                shard_files
            )
            logger.info(f"Skipping shard with files less than 2: {shard_files}")
            if len(shard_files) == 1:
                basename = os.path.basename(shard_files[0])
                if basename.count("_") == 2:
                    shutil.move(
                        shard_files[0],
                        os.path.join(
                            self.index_path, f"{shard_filename}_{shard}.index"
                        ),
                    )
                    shutil.move(
                        position_files[0],
                        os.path.join(
                            self.index_path, f"{shard_filename}_{shard}.position"
                        ),
                    )
                    shutil.move(
                        offset_files[0],
                        os.path.join(
                            self.index_path, f"{shard_filename}_{shard}.offset"
                        ),
                    )
            return None, None, None

        shard_files = sorted(shard_files, key=lambda x: extract_shard(x))
        position_files = sorted(position_files, key=lambda x: extract_shard(x))
        offset_files = sorted(offset_files, key=lambda x: extract_shard(x))

        postings_file = os.path.join(self.output_dir, f"{save_filename}.bin")
        positions_file = os.path.join(self.output_dir, f"{save_filename}_positions.bin")
        if shard != -1:
            # Merging sub-shards so do not save as `postings.bin`
            # Save as `shard_0.index` or `pos_shard_0.index`
            # We use a temporary file to avoid file locks with the base shard
            postings_file = os.path.join(
                self.output_dir, f"{shard_filename}_{shard}.temp.index"
            )
            positions_file = os.path.join(
                self.output_dir, f"{shard_filename}_{shard}.temp.position"
            )

        term_offsets = OrderedDict()

        logger.info(
            f"Merging files {shard_files} and {offset_files}. force_merge={force_merge}. Saving to {postings_file} ..."
        )

        lock_postings_file = filelock.FileLock(postings_file + ".lock")
        lock_positions_file = filelock.FileLock(positions_file + ".lock")
        logger.debug(f"Locking {postings_file}")
        with lock_postings_file, lock_positions_file:
            with (
                open(postings_file, "wb") as postings_f,
                open(positions_file, "wb") as positions_f,
            ):
                heap = []

                for shard_file, position_file, offset_file in zip(
                    shard_files, position_files, offset_files
                ):
                    reader = ShardReader(shard_file, position_file, offset_file)
                    items = reader.next_term()
                    if all(x is not None for x in items):
                        heap.append(HeapItem(*items, reader))
                    else:
                        reader.close()

                heapq.heapify(heap)
                while heap:
                    heap_item = heapq.heappop(heap)

                    assert (heap_item.term, shard) not in term_offsets
                    term_offsets[(heap_item.term, shard)] = (
                        shard,
                        postings_f.tell(),
                        positions_f.tell(),
                    )

                    all_postings = heap_item.reader.read_postings(
                        heap_item.offset,
                        pos_offset=heap_item.pos_offset,
                    )

                    while heap and heap[0].term == heap_item.term:
                        other_heap_item = heapq.heappop(heap)
                        other_postings = other_heap_item.reader.read_postings(
                            other_heap_item.offset,
                            pos_offset=other_heap_item.pos_offset,
                        )
                        all_postings.extend(other_postings)

                        items = other_heap_item.reader.next_term()

                        if all(x is not None for x in items):
                            heapq.heappush(
                                heap, HeapItem(*items, other_heap_item.reader)
                            )
                        else:
                            other_heap_item.reader.close()

                    all_postings.sort(key=lambda x: x.doc_id)

                    self._write_merged_postings(
                        postings_f,
                        postings=all_postings,
                        positions_file=positions_f,
                    )

                    item = heap_item.reader.next_term()
                    if all(x is not None for x in item):
                        heapq.heappush(heap, HeapItem(*item, heap_item.reader))
                    else:
                        heap_item.reader.close()

        logger.debug(f"Unlocked {postings_file}")
        shutil.move(postings_file, postings_file.replace(".temp", ""))
        shutil.move(positions_file, positions_file.replace(".temp", ""))

        fst_save_filename = "terms" if shard == -1 else f"terms_{shard}"
        self._build_term_fst(term_offsets, save_filename=fst_save_filename)

        if shard != -1:
            offset_filename = os.path.join(
                self.output_dir, f"{shard_filename}_{shard}.temp.offset"
            )
            lock_offset_file = filelock.FileLock(offset_filename + ".lock")
            logger.debug(f"Locking {offset_filename}")
            with lock_offset_file:
                with open(offset_filename, "wb") as term_offsets_f:
                    self._write_merged_offset(term_offsets_f, term_offsets, shard)
            logger.debug(f"Unlocked {offset_filename}")

            shutil.move(offset_filename, offset_filename.replace(".temp", ""))

        self._remove_sub_shard_files(shard_files, base_shard_files, shard)
        self._remove_sub_shard_files(position_files, base_position_files, shard)
        self._remove_sub_shard_files(offset_files, base_offset_files, shard)

        logger.info(f"Merged {shard_filename} {shard} to {postings_file}")

        return None, None, None

    def _remove_sub_shard_files(self, shard_files, base_shard_files, shard=-1):
        for shard_file in shard_files:
            if (
                len(base_shard_files) > 0
                and shard_file == base_shard_files[0]
                and shard != -1
            ):
                continue

            os.remove(shard_file)

    def _build_doc_fst(self, doc_offsets, save_filename="docs", shard=-1):
        logger.info(f"Building FST for {len(doc_offsets)} docs")
        if os.path.exists(os.path.join(self.output_dir, f"{save_filename}.fst")):
            logger.info(f"Loading {save_filename}.fst")
            fst = marisa_trie.BytesTrie()
            fst.load(os.path.join(self.output_dir, f"{save_filename}.fst"))
            logger.info(f"Loaded {save_filename}.fst")

            for doc_id, values in fst.items():
                if doc_id in doc_offsets:
                    continue

                doc_offsets[int(doc_id)] = struct.unpack(
                    SIZE_KEY["offset_shard"], values
                )[0]

        items = []
        for doc_id, offset in doc_offsets.items():
            items.append(
                (str(doc_id), struct.pack(SIZE_KEY["offset_shard"], shard, offset))
            )

        fst = marisa_trie.BytesTrie(items)
        fst.save(os.path.join(self.output_dir, f"{save_filename}.fst"))

    def _build_term_fst(
        self,
        term_offsets,
        save_filename="terms",
        load_filename="terms",
        shard=-1,
    ):
        """
        Builds the FST for the terms and saves as `terms.fst` or `pos_terms.fst`
        If `shard` is -1 then we save each term as [term, offset] in the FST
        If `shard` is not -1 then we save each term with reference to the shard. As [term, shard, offset]
        """
        logger.info(f"Building FST for {len(term_offsets)} terms")
        if os.path.exists(os.path.join(self.output_dir, f"{load_filename}.fst")):
            logger.info(f"Loading {load_filename}.fst")
            fst = marisa_trie.BytesTrie()
            fst.load(os.path.join(self.output_dir, f"{load_filename}.fst"))
            logger.info(f"Loaded {load_filename}.fst")

            for term, values in fst.items():
                shard, offset, pos_offset = struct.unpack(
                    SIZE_KEY["pos_offset_shard"], values
                )
                if (term, shard) in term_offsets:
                    continue

                term_offsets[(term, shard)] = (shard, offset, pos_offset)

        items = []
        for (term, shard), (_, offset, pos_offset) in term_offsets.items():
            if shard == -1:
                offset_encoding = struct.pack(
                    SIZE_KEY["pos_offset"], offset, pos_offset
                )
                value = (term, offset_encoding)
            else:
                shard_offset_encoding = struct.pack(
                    SIZE_KEY["pos_offset_shard"], shard, offset, pos_offset
                )
                value = (term, shard_offset_encoding)
            items.append(value)

        fst = marisa_trie.BytesTrie(items)
        fst.save(os.path.join(self.output_dir, f"{save_filename}.fst"))

    def _write_merged_offset(self, f, term_offsets, shard):
        logger.info(f"Writing offsets for shard {shard}")
        for (term, _), (_, offset, pos_offset) in term_offsets.items():
            term_bytes = term.encode("utf-8")
            f.write(struct.pack(SIZE_KEY["term_bytes"], len(term_bytes)))
            f.write(term_bytes)
            f.write(struct.pack(SIZE_KEY["offset"], offset))
            f.write(struct.pack(SIZE_KEY["offset"], pos_offset))

    def _write_merged_postings(self, f, postings=[], positions_file=None):
        f.write(struct.pack(SIZE_KEY["postings_count"], len(postings)))
        prev_doc_id = 0
        for posting in postings:
            delta = posting.doc_id - prev_doc_id
            try:
                f.write(
                    struct.pack(SIZE_KEY["deltaTF"], delta, posting.doc_term_frequency)
                )
            except struct.error:
                print(posting.doc_id, prev_doc_id)
                print(posting.doc_term_frequency)
                print(delta)
                raise

            prev_doc_id = posting.doc_id

            prev_position = 0
            filter_positions = [p for p in posting.positions if p >= 0]
            f.write(struct.pack(SIZE_KEY["position_count"], len(filter_positions)))

            if positions_file is not None:
                for position in filter_positions:
                    delta = position - prev_position
                    positions_file.write(struct.pack(SIZE_KEY["position_delta"], delta))
                    prev_position = position

    def _merge_docs(self, shard=-1, sub_shard=0, force_merge=False):
        """
        Merges the documents from the shards into a single file
        If `shard` is -1, then we merge all shards into a single file
        If `shard` is not -1, then we merge the sub-shards into a single file

        We read the document from each shard and write to the output file
        """
        base_shard_files = glob.glob(
            os.path.join(self.index_path, f"doc_shard_{shard}.index")
        )
        base_offset_files = glob.glob(
            os.path.join(self.index_path, f"doc_shard_{shard}.offset")
        )

        sub_shard_name = f"{shard}_*" if shard != -1 else "*"
        shard_files = sorted(
            glob.glob(
                os.path.join(self.index_path, f"doc_shard_{sub_shard_name}.index")
            ),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        offset_files = sorted(
            glob.glob(
                os.path.join(self.index_path, f"doc_shard_{sub_shard_name}.offset")
            ),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )

        if len(shard_files) < 2 and not force_merge:
            logger.debug(f"Skipping shard {shard} with {len(shard_files)} files")
            return

        if len(base_shard_files) > 0:
            shard_files.insert(0, base_shard_files[0])
        if len(base_offset_files) > 0:
            offset_files.insert(0, base_offset_files[0])

        docs_file = os.path.join(self.output_dir, "docs.bin")
        offset_file = os.path.join(self.output_dir, "docs.offset")
        if shard != -1:
            docs_file = os.path.join(self.output_dir, f"doc_shard_{shard}.temp.index")
            offset_file = os.path.join(
                self.output_dir, f"doc_shard_{shard}.temp.offset"
            )

        logger.info(
            f"Merging files {shard_files}. force_merge={force_merge}. Saving to {docs_file} ..."
        )

        docs_offset = OrderedDict()

        lock_docs_file = filelock.FileLock(docs_file + ".lock")
        lock_offset_file = filelock.FileLock(offset_file + ".lock")
        logger.debug(f"Locking {docs_file}")
        with lock_docs_file, lock_offset_file:
            doc_count = 0
            for shard_file, offset_filename in zip(shard_files, offset_files):
                with (
                    open(offset_filename, "rb") as roffset_f,
                ):
                    doc_count += struct.unpack(
                        SIZE_KEY["doc_count"],
                        roffset_f.read(READ_SIZE_KEY[SIZE_KEY["doc_count"]]),
                    )[0]

            with open(docs_file, "wb") as docs_f, open(offset_file, "wb") as offset_f:
                offset_f.seek(0)
                offset_f.write(struct.pack(SIZE_KEY["doc_count"], doc_count))

                for shard_file, offset_filename in zip(shard_files, offset_files):
                    with (
                        open(shard_file, "rb") as shard_f,
                        open(offset_filename, "rb") as roffset_f,
                    ):
                        try:
                            local_doc_count = struct.unpack(
                                SIZE_KEY["doc_count"],
                                roffset_f.read(READ_SIZE_KEY[SIZE_KEY["doc_count"]]),
                            )[0]

                            for _ in range(local_doc_count):
                                doc_id = struct.unpack(
                                    SIZE_KEY["doc_id"],
                                    roffset_f.read(READ_SIZE_KEY[SIZE_KEY["doc_id"]]),
                                )[0]
                                offset = struct.unpack(
                                    SIZE_KEY["offset"],
                                    roffset_f.read(READ_SIZE_KEY[SIZE_KEY["offset"]]),
                                )[0]

                                shard_f.seek(offset)
                                doc_length = struct.unpack(
                                    SIZE_KEY["doc_length"],
                                    shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_length"]]),
                                )[0]

                                docs_offset[int(doc_id)] = docs_f.tell()
                                docs_f.write(
                                    struct.pack(SIZE_KEY["doc_length"], doc_length)
                                )
                        except struct.error as e:
                            logger.error(
                                f"Error reading {offset_filename} after length {len(docs_offset)} {e}"
                            )
                            break

                for doc_id, offset in docs_offset.items():
                    offset_f.write(struct.pack(SIZE_KEY["doc_id"], doc_id))
                    offset_f.write(struct.pack(SIZE_KEY["offset"], offset))

        logger.debug(f"Unlocked {docs_file}")
        shutil.move(docs_file, docs_file.replace(".temp", ""))
        shutil.move(offset_file, offset_file.replace(".temp", ""))

        min_doc_id = min(docs_offset.keys()) if len(docs_offset) > 0 else None
        max_doc_id = max(docs_offset.keys()) if len(docs_offset) > 0 else None

        docs_filename = "docs" if shard == -1 else f"docs_{shard}"
        self._build_doc_fst(docs_offset, save_filename=docs_filename, shard=shard)

        self._remove_sub_shard_files(shard_files, base_shard_files, shard)
        self._remove_sub_shard_files(offset_files, base_offset_files, shard)

        logger.info(f"Merged doc_shard {shard} to {docs_file}")

        return shard if min_doc_id is not None else None, min_doc_id, max_doc_id

    def merge_shards_to_size(self, shards=25, shard_size=1000):
        """
        Merges the shards into N shards
        """

        shard_files = sorted(
            glob.glob(os.path.join(self.index_path, "shard_*.index")),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )

        file_size = 0
        curr_files_in_shard = []
        shards = []
        for i in range(len(shard_files)):
            file_size += os.path.getsize(shard_files[i])
            filename = f"shard_{len(shards)}_{len(curr_files_in_shard)}"

            shutil.move(
                shard_files[i], os.path.join(self.index_path, f"{filename}.index")
            )
            shutil.move(
                os.path.join(self.index_path, f"shard_{i}.position"),
                os.path.join(self.index_path, f"{filename}.position"),
            )
            shutil.move(
                os.path.join(self.index_path, f"shard_{i}.offset"),
                os.path.join(self.index_path, f"{filename}.offset"),
            )
            shutil.move(
                os.path.join(self.index_path, f"doc_shard_{i}.index"),
                os.path.join(self.index_path, f"doc_{filename}.index"),
            )
            shutil.move(
                os.path.join(self.index_path, f"doc_shard_{i}.offset"),
                os.path.join(self.index_path, f"doc_{filename}.offset"),
            )

            logger.info(f"Moved {shard_files[i]} to {filename}")

            curr_files_in_shard.append(filename)

            if file_size > shard_size:
                shards.append(curr_files_in_shard)
                logger.info(
                    f"Created shard {len(shards)} with {len(curr_files_in_shard)} files of size {file_size}"
                )
                curr_files_in_shard = []
                file_size = 0

        shards.append(curr_files_in_shard)
        logger.info(
            f"Created shard {len(shards)} with {len(curr_files_in_shard)} files of size {file_size}"
        )

        futures = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(shards)
        ) as executor:
            for shard, _ in enumerate(shards):
                futures.append(
                    executor.submit(self._merge, shard=shard, force_merge=True)
                )
                futures.append(
                    executor.submit(self._merge_docs, shard=shard, force_merge=True)
                )

        bounds = {}
        for future in concurrent.futures.as_completed(futures):
            shard, low_doc_id, high_doc_id = future.result()
            if shard is None:
                continue

            bounds[shard] = (low_doc_id, high_doc_id)

        logger.info("Merged all shards")
        self.post_merge_cleanup()

        doc_meta = os.path.join(self.index_path, "shard.temp.meta")
        lock_doc_meta = filelock.FileLock(doc_meta + ".lock")
        with lock_doc_meta:
            with open(doc_meta, "w") as f:
                json.dump(bounds, f)
        logger.info(f"Saved document bounds {bounds} to {doc_meta}")

        shutil.move(doc_meta, doc_meta.replace(".temp", ""))

        return len(shards)
