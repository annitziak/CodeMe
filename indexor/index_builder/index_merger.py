import os
import logging
import glob
import heapq
import struct
import shutil
import filelock
import marisa_trie
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
    reader: ShardReader

    def __lt__(self, other):
        return self.term < other.term

    def __gt__(self, other):
        return self.term > other.term

    def __iter__(self):
        return iter((self.term, self.offset, self.reader))

    def __getitem__(self, key):
        return (self.term, self.offset, self.reader)[key]


class IndexMerger:
    def __init__(self, index_path: str, output_dir: str = ""):
        self.index_path = os.path.abspath(index_path)
        self.output_dir = output_dir if output_dir != "" else self.index_path

    def cleanup(self):
        if os.path.exists(self.output_dir):
            logger.info(f"Removing existing index directory: {self.output_dir}")
            confirm = input("Are you sure you want to remove the directory? (y/(n)): ")
            if confirm.lower() != "y":
                logger.info("Exiting cleanup")
                exit(0)

            for f in os.listdir(self.output_dir):
                if "shard" in f or "postings" in f or "terms" in f or "positions" in f:
                    logger.info(f"Removing {f}")
                    os.remove(os.path.join(self.output_dir, f))

    def post_merge_cleanup(self):
        logger.info("Removing temporary files and lock files")
        for f in os.listdir(self.output_dir):
            if ".temp" in f:
                logger.info(f"Removing {f}")
                os.remove(os.path.join(self.output_dir, f))
            if ".lock" in f:
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
        save_filename="terms",
    ):
        items = []

        def gen_term_offset(term_offsets):
            if term_offsets is None:
                term_offsets = OrderedDict()
                for i in range(shards):
                    reader = ShardReader(
                        os.path.join(self.index_path, f"shard_{i}.index"),
                        os.path.join(self.index_path, f"shard_{i}.offset"),
                    )
                    while True:
                        term, offset = reader.next_term()
                        if term is None or offset is None:
                            break

                        yield term, i, offset

                    reader.close()

                    logger.info(f"Finished reading shard {i}")
                    yield None, None, None

        for term, shard, offset in gen_term_offset(term_offsets):
            if term is None or offset is None or shard is None:
                fst = marisa_trie.BytesTrie(items)
                fst.save(os.path.join(self.output_dir, f"{save_filename}.fst"))
                logger.info(f"Saved {save_filename}.fst")
                continue

            encoded_value = struct.pack(SIZE_KEY["offset_shard"], shard, offset)
            items.append((term, encoded_value))

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
                merge_positions=False,
                force_merge=force_merge,
                shard=i,
            )
            self._merge(
                shard_filename="pos_shard",
                save_filename="positions",
                merge_positions=True,
                force_merge=force_merge,
                shard=i,
            )
            self._merge_docs(shard=i, force_merge=force_merge)

    def _merge_all(self):
        self._merge(
            shard_filename="shard",
            save_filename="postings",
            merge_positions=False,
            force_merge=True,
        )
        self._merge(
            shard_filename="pos_shard",
            save_filename="positions",
            merge_positions=True,
            force_merge=True,
        )

    def _merge(
        self,
        shard_filename="shard",
        save_filename="postings",
        merge_positions=False,
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
        `merge_positions` determines if the positional index is merged
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
        base_offset_files = glob.glob(
            os.path.join(self.index_path, f"{shard_filename}_{shard}.offset")
        )

        # Will be `shard_0_0.index` or `pos_shard_0_0.index` if merging sub-shards
        # Will be `shard_*.index` or `pos_shard_*.index` if merging all shards
        sub_shard_name = f"{shard}_*" if shard != -1 else "*"
        shard_files = glob.glob(
            os.path.join(self.index_path, f"{shard_filename}_{sub_shard_name}.index")
        )
        offset_files = glob.glob(
            os.path.join(self.index_path, f"{shard_filename}_{sub_shard_name}.offset")
        )

        if len(shard_files) < 2 and len(offset_files) < 2 and not force_merge:
            logger.debug(f"Skipping shard {shard} with {len(shard_files)} files")
            return

        if len(shard_files) != len(offset_files):
            return

        if len(base_shard_files) > 0 and len(base_offset_files) > 0:
            shard_files.insert(0, base_shard_files[0])
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

        shard_files = sorted(shard_files, key=lambda x: extract_shard(x))
        offset_files = sorted(offset_files, key=lambda x: extract_shard(x))

        postings_file = os.path.join(self.output_dir, f"{save_filename}.bin")
        if shard != -1:
            # Merging sub-shards so do not save as `postings.bin`
            # Save as `shard_0.index` or `pos_shard_0.index`
            # We use a temporary file to avoid file locks with the base shard
            postings_file = os.path.join(
                self.output_dir, f"{shard_filename}_{shard}.temp.index"
            )

        term_offsets = OrderedDict()

        logger.info(
            f"Merging files {shard_files} and {offset_files}. force_merge={force_merge}. Saving to {postings_file} ..."
        )

        lock_postings_file = filelock.FileLock(postings_file + ".lock")
        logger.debug(f"Locking {postings_file}")
        with lock_postings_file:
            with open(postings_file, "wb") as postings_f:
                heap = []

                for shard_file, offset_file in zip(shard_files, offset_files):
                    reader = ShardReader(shard_file, offset_file)
                    term, offset = reader.next_term()
                    if term is not None and offset is not None:
                        heap.append(HeapItem(term, offset, reader))
                    else:
                        reader.close()

                heapq.heapify(heap)
                while heap:
                    term, offset, reader = heapq.heappop(heap)

                    assert term not in term_offsets
                    term_offsets[term] = (shard, postings_f.tell())

                    all_postings = reader.read_postings(
                        offset, read_positions=merge_positions
                    )

                    while heap and heap[0].term == term:
                        _, offset, other_reader = heapq.heappop(heap)
                        other_postings = other_reader.read_postings(
                            offset, read_positions=merge_positions
                        )
                        all_postings.extend(other_postings)

                        next_term, next_offset = other_reader.next_term()

                        if next_term is not None and next_offset is not None:
                            heapq.heappush(
                                heap, HeapItem(next_term, next_offset, other_reader)
                            )
                        else:
                            other_reader.close()

                    all_postings.sort(key=lambda x: x.doc_id)

                    self._write_merged_postings(
                        postings_f, all_postings, merge_positions=merge_positions
                    )

                    next_term, next_offset = reader.next_term()
                    if next_term is not None and next_offset is not None:
                        heapq.heappush(heap, HeapItem(next_term, next_offset, reader))
                    else:
                        reader.close()

        logger.debug(f"Unlocked {postings_file}")
        shutil.move(postings_file, postings_file.replace(".temp", ""))

        if shard == -1:
            fst_save_filename = "pos_terms" if merge_positions else "terms"
            # Builds terms.fst with (-1, offset) i.e. shard, offset
            self._build_term_fst(term_offsets, save_filename=fst_save_filename)
        else:
            fst_save_filename = (
                f"pos_terms_{shard}" if merge_positions else f"terms_{shard}"
            )
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

        for shard_file in shard_files:
            if (
                len(base_shard_files) > 0
                and shard_file == base_shard_files[0]
                and shard != -1
            ):
                continue

            os.remove(shard_file)

        for offset_file in offset_files:
            if (
                len(base_offset_files) > 0
                and offset_file == base_offset_files[0]
                and shard != -1
            ):
                continue

            os.remove(offset_file)

        logger.info(f"Merged {shard_filename} {shard} to {postings_file}")

    def _build_term_fst(
        self, term_offsets, save_filename="terms", load_filename="terms"
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
            items = [(term, value) for term, values in fst.items() for value in values]
        else:
            items = []

        for term, (shard, offset) in term_offsets.items():
            if shard == -1:
                offset_encoding = struct.pack(SIZE_KEY["offset"], offset)
                value = (term, offset_encoding)
            else:
                shard_offset_encoding = struct.pack(
                    SIZE_KEY["offset_shard"], shard, offset
                )
                value = (term, shard_offset_encoding)
            items.append((term, value))

        fst = marisa_trie.BytesTrie(items)
        fst.save(os.path.join(self.output_dir, f"{save_filename}.fst"))

    def _write_merged_offset(self, f, term_offsets, shard):
        logger.info(f"Writing offsets for shard {shard}")
        for term, offset in term_offsets.items():
            term_bytes = term.encode("utf-8")
            f.write(struct.pack(SIZE_KEY["term_bytes"], len(term_bytes)))
            f.write(term_bytes)
            f.write(struct.pack(SIZE_KEY["offset"], offset))

    def _write_merged_postings(self, f, postings, merge_positions=False):
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

            if merge_positions:
                prev_position = 0
                filter_positions = [p for p in posting.positions if p >= 0]
                f.write(struct.pack(SIZE_KEY["position_count"], len(filter_positions)))
                for position in filter_positions:
                    delta = position - prev_position
                    f.write(struct.pack(SIZE_KEY["position_delta"], delta))
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

        sub_shard_name = f"{shard}_*" if shard != -1 else "*"
        shard_files = glob.glob(
            os.path.join(self.index_path, f"doc_shard_{sub_shard_name}.index")
        )

        if len(shard_files) < 2 and not force_merge:
            logger.debug(f"Skipping shard {shard} with {len(shard_files)} files")
            return

        if len(base_shard_files) > 0:
            shard_files.insert(0, base_shard_files[0])

        docs_file = os.path.join(self.output_dir, "docs.bin")
        if shard != -1:
            docs_file = os.path.join(self.output_dir, f"doc_shard_{shard}.temp.index")

        logger.info(
            f"Merging files {shard_files}. force_merge={force_merge}. Saving to {docs_file} ..."
        )

        lock_docs_file = filelock.FileLock(docs_file + ".lock")
        logger.debug(f"Locking {docs_file}")
        with lock_docs_file:
            with open(docs_file, "wb") as docs_f:
                for shard_file in shard_files:
                    with open(shard_file, "rb") as shard_f:
                        shutil.copyfileobj(shard_f, docs_f)

        logger.debug(f"Unlocked {docs_file}")
        shutil.move(docs_file, docs_file.replace(".temp", ""))

        for shard_file in shard_files:
            if len(base_shard_files) > 0 and shard_file == base_shard_files[0]:
                continue

            os.remove(shard_file)

        logger.info(f"Merged doc_shard {shard} to {docs_file}")

    def write_index(self, f):
        logger.info("Writing index to file")

        postings_file = os.path.join(self.output_dir, "positions.bin")
        term_fst = marisa_trie.BytesTrie()
        term_fst.load(os.path.join(self.output_dir, "pos_terms.fst"))

        with open(postings_file, "rb") as pf:
            for term, value in term_fst.items():
                curr_doc_id = 0
                offset = struct.unpack(SIZE_KEY["offset"], value)[0]
                pf.seek(offset)
                posting_count = struct.unpack(
                    SIZE_KEY["postings_count"],
                    pf.read(READ_SIZE_KEY[SIZE_KEY["postings_count"]]),
                )[0]

                f.write(f"{term}\t{posting_count}\n")

                for _ in range(posting_count):
                    delta, _ = struct.unpack(
                        SIZE_KEY["deltaTF"], pf.read(READ_SIZE_KEY[SIZE_KEY["deltaTF"]])
                    )
                    curr_doc_id += delta

                    positions = []
                    position_count = struct.unpack(
                        SIZE_KEY["position_count"],
                        pf.read(READ_SIZE_KEY[SIZE_KEY["position_count"]]),
                    )[0]
                    curr_position = 0
                    for _ in range(position_count):
                        delta = struct.unpack(
                            SIZE_KEY["position_delta"],
                            pf.read(READ_SIZE_KEY[SIZE_KEY["position_delta"]]),
                        )[0]
                        curr_position += delta
                        positions.append(str(curr_position))

                    positions = ",".join(positions)
                    f.write(f"\t{curr_doc_id}\t{positions}\n")
