import logging
import os
import concurrent.futures
import struct
import glob
import heapq
import marisa_trie
import time

from dataclasses import dataclass

from indexor.structures import Term, PostingList, SIZE_KEY, READ_SIZE_KEY

from preprocessing.preprocessor import Preprocessor
from utils.db_connection import DBConnection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEV_SHARDS = 1


class DocumentShardedIndexBuilder:
    def __init__(self, index_path: str, shard: int = -1, shard_size: int = 50_000):
        """
        Used to generate a sharded index where each shard is a separate index file for a subset of the documents.
        If `shard` is -1, then DocumentShardedIndexBuilder will generate shards based on the `shard_size`.
        If `shard` is not -1, then DocumentShardedIndexBuilder will generate a single shard with the specified `shard` number.
        """
        self.index_path = index_path
        self.shard = shard
        self.curr_shard = 0

        self.shard_size = shard_size
        self.term_map: dict[str, Term] = {}

        self.current_docs = 0

        self.min_doc_id = float("inf")
        self.max_doc_id = float("-inf")
        self.start = time.time()

        if not os.path.exists(self.index_path):
            logger.info(f"Creating index directory: {self.index_path}")
            os.makedirs(self.index_path)

    def __str__(self):
        return ""

    def add_document(
        self, doc_id: int, doc_terms: dict[str, list[int]] = {}, doc_length=0
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
        for term, positions in doc_terms.items():
            if term not in self.term_map:
                self.term_map[term] = Term(term, 0, {})

            self.term_map[term].update_with_postings(doc_id, positions)

            if self.current_docs >= self.shard_size:
                self.flush(self.shard, self.curr_shard)

                self.curr_shard += 1
                self.current_docs = 0

    def flush(self, shard: int = -1, sub_shard: int = 0):
        """
        By default, flushes both a non-positional and positional index to disk.
        """
        self._flush_shard(
            shard, sub_shard=sub_shard, shard_filename="shard", flush_positions=False
        )
        self._flush_shard(
            shard, sub_shard=sub_shard, shard_filename="pos_shard", flush_positions=True
        )

        self.term_map = {}
        self.min_doc_id = float("inf")
        self.max_doc_id = float("-inf")
        self.start = time.time()

    def _flush_shard(
        self,
        shard: int = -1,
        sub_shard: int = 0,
        shard_filename="shard",
        flush_positions=False,
    ):
        """
        Flush the current term map to disk.
        Saves the term map to a file in the index directory.

        In 'shard_{shard}.index', the posting lists are written in the following format:
        - document_frequency: int

        In 'shard_{shard}.offset', the term offsets are written in the following format:
        - term_length: int (byte length of term)
        - term: bytes
        - offset: int (offset in 'shard_{shard}.index')

        Args:
            shard: int - the shard number to flush to
        """
        shard = self.curr_shard if shard == -1 else shard
        shard = self.shard if self.shard != -1 else shard

        shard_str = str(shard)
        shard_str = f"{shard}_{sub_shard}"

        shard_path = os.path.join(
            self.index_path, f"{shard_filename}_{shard_str}.index"
        )
        offset_dict = {}

        logger.info(
            f"Flushing shard {shard_str} with {len(self.term_map)} terms to {shard_path}. Documents processed: {self.current_docs} in {time.time()-self.start}. Min doc ID: {self.min_doc_id}. Max doc ID: {self.max_doc_id}"
        )

        with open(shard_path, "wb") as f:
            for term, term_obj in self.term_map.items():
                # term_obj.sort_posting_lists()

                offset_dict[term] = f.tell()
                f.write(
                    struct.pack(SIZE_KEY["postings_count"], term_obj.document_frequency)
                )

                prev_doc_id = 0
                for posting in term_obj.posting_lists.values():
                    delta = posting.doc_id - prev_doc_id

                    f.write(
                        struct.pack(
                            SIZE_KEY["deltaTF"], delta, posting.doc_term_frequency
                        )
                    )
                    prev_doc_id = posting.doc_id

                    if flush_positions:
                        filter_positions = [p for p in posting.positions if p >= 0]
                        filter_positions = sorted(filter_positions)
                        f.write(
                            struct.pack(
                                SIZE_KEY["position_count"], len(filter_positions)
                            )
                        )

                        prev_position = 0
                        for position in filter_positions:
                            delta = position - prev_position
                            f.write(struct.pack(SIZE_KEY["position_delta"], delta))
                            prev_position = position

        offset_path = os.path.join(
            self.index_path, f"{shard_filename}_{shard_str}.offset"
        )
        logger.info(
            f"Flushing offset for shard {shard_str} with filename {offset_path}"
        )

        with open(offset_path, "wb") as offset_f:
            for term, offset in offset_dict.items():
                term_bytes = term.encode("utf-8")

                offset_f.write(struct.pack(SIZE_KEY["term_bytes"], len(term_bytes)))
                offset_f.write(term_bytes)
                offset_f.write(struct.pack(SIZE_KEY["offset"], offset))


class ShardReader:
    def __init__(self, shard_path: str, offset_path: str):
        self.shard_path = shard_path
        self.offset_path = offset_path

        self.f_shard = open(shard_path, "rb")
        self.f_offset = open(offset_path, "rb")

    def next_term(self):
        try:
            term_length = struct.unpack(
                SIZE_KEY["term_bytes"],
                self.f_offset.read(READ_SIZE_KEY[SIZE_KEY["term_bytes"]]),
            )[0]
            term = self.f_offset.read(term_length).decode("utf-8")
            offset = struct.unpack(
                SIZE_KEY["offset"],
                self.f_offset.read(READ_SIZE_KEY[SIZE_KEY["offset"]]),
            )[0]
            return term, offset
        except struct.error:
            return None, None

    def read_postings(self, offset: int, read_positions=False) -> list[PostingList]:
        self.f_shard.seek(offset)

        count = struct.unpack(
            SIZE_KEY["postings_count"],
            self.f_shard.read(READ_SIZE_KEY[SIZE_KEY["postings_count"]]),
        )[0]
        postings = []
        curr_doc_id = 0

        for _ in range(count):
            doc_id_delta, doc_term_frequency = struct.unpack(
                SIZE_KEY["deltaTF"],
                self.f_shard.read(READ_SIZE_KEY[SIZE_KEY["deltaTF"]]),
            )
            curr_doc_id += doc_id_delta

            positions = []
            if read_positions:
                curr_position = 0
                position_count = struct.unpack(
                    SIZE_KEY["position_count"],
                    self.f_shard.read(READ_SIZE_KEY[SIZE_KEY["position_count"]]),
                )[0]
                for _ in range(position_count):
                    position_delta = struct.unpack(
                        SIZE_KEY["position_delta"],
                        self.f_shard.read(READ_SIZE_KEY[SIZE_KEY["position_delta"]]),
                    )[0]
                    curr_position += position_delta
                    positions.append(curr_position)

            postings.append(
                PostingList(curr_doc_id, doc_term_frequency, sorted(positions))
            )

        return postings


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
        self.output_dir = index_path if output_dir == "" else output_dir

        self.cleanup()

    def cleanup(self):
        if os.path.exists(self.output_dir):
            logger.info(f"Removing existing index directory: {self.output_dir}")
            for f in os.listdir(self.output_dir):
                if "shard" in f:
                    logger.info(f"Removing {f}")
                    os.remove(os.path.join(self.output_dir, f))

    def merge(self, shards: int = 24, merge_subshards=False):
        """
        Merges the shards in the index directory into a single index file if shards is -1.
        Otherwise, merges the subshards of a single shard. There are `shards` number of shards.
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
                shard=i,
            )
            self._merge(
                shard_filename="pos_shard",
                save_filename="positions",
                merge_positions=True,
                shard=i,
            )

    def _merge_all(self):
        self._merge(
            shard_filename="shard", save_filename="postings", merge_positions=False
        )
        self._merge(
            shard_filename="pos_shard", save_filename="positions", merge_positions=True
        )

    def _merge(
        self,
        shard_filename="shard",
        save_filename="postings",
        merge_positions=False,
        shard=-1,
    ):
        if shard == -1:
            shard_files = glob.glob(
                os.path.join(self.index_path, f"{shard_filename}_*.index")
            )
            offset_files = glob.glob(
                os.path.join(self.index_path, f"{shard_filename}_*.offset")
            )
        else:
            shard_files = [
                os.path.join(self.index_path, f"{shard_filename}_{shard}_*.index")
            ]
            offset_files = [
                os.path.join(self.index_path, f"{shard_filename}_{shard}_*.offset")
            ]

        extract_shard = lambda x, y: int(x.split("_")[y].split(".")[0])
        shard_files = sorted(
            shard_files, key=lambda x: (extract_shard(x, -2), extract_shard(x, -1))
        )
        offset_files = sorted(
            offset_files, key=lambda x: (extract_shard(x, -2), extract_shard(x, -1))
        )

        postings_file = os.path.join(self.output_dir, f"{save_filename}.bin")
        term_offsets = {}

        logger.info(
            f"Merging files {shard_files} and {offset_files}. Saving to {postings_file} ..."
        )

        with open(postings_file, "wb") as f:
            heap = []

            for shard_file, offset_file in zip(shard_files, offset_files):
                reader = ShardReader(shard_file, offset_file)
                term, offset = reader.next_term()
                if term is not None and offset is not None:
                    heap.append(HeapItem(term, offset, reader))

            heapq.heapify(heap)
            while heap:
                term, offset, reader = heapq.heappop(heap)
                term_offsets[term] = f.tell()

                all_postings = reader.read_postings(
                    offset, read_positions=merge_positions
                )
                while heap and heap[0].term == term:
                    _, offset, other_reader = heapq.heappop(heap)
                    all_postings.extend(
                        other_reader.read_postings(
                            offset, read_positions=merge_positions
                        )
                    )

                    next_term, next_offset = other_reader.next_term()
                    if next_term is not None and next_offset is not None:
                        heapq.heappush(heap, HeapItem(next_term, offset, other_reader))

                all_postings.sort(key=lambda x: x.doc_id)
                self._write_merged_postings(
                    f, all_postings, merge_positions=merge_positions
                )

                next_term, next_offset = reader.next_term()
                if next_term is not None and next_offset is not None:
                    heapq.heappush(heap, HeapItem(next_term, next_offset, reader))

        self._build_term_fst(
            term_offsets, save_filename="pos_terms" if merge_positions else "terms"
        )

        for shard_file in shard_files:
            os.remove(shard_file)

        for offset_file in offset_files:
            os.remove(offset_file)

    def _build_term_fst(self, term_offsets, save_filename="terms"):
        logger.info(f"Building FST for {len(term_offsets)} terms")

        items = []
        for term, offset in term_offsets.items():
            value = struct.pack("<Q", offset)
            items.append((term, value))

        fst = marisa_trie.BytesTrie(items)
        fst.save(os.path.join(self.output_dir, f"{save_filename}.fst"))

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


class IndexBuilder:
    def __init__(
        self,
        db_params: dict,
        index_path: str,
        batch_size: int = 1000,
        num_shards: int = 24,
        num_workers: int = 24,
        write_txt: bool = False,
        debug=False,
    ) -> None:
        self.db_params = db_params
        self.db_connection = DBConnection(db_params)

        self.index_path = index_path
        self.doc_metadata = {}

        self.batch_size = batch_size
        self.num_shards = num_shards if not debug else DEV_SHARDS
        self.num_workers = min(min(num_workers, os.cpu_count() - 1), self.num_shards)

        self.title_preprocessor = Preprocessor(parser_kwargs={"parser_type": "raw"})
        self.body_preprocessor = Preprocessor(parser_kwargs={"parser_type": "html"})

        self.debug = debug

        self.write_txt = write_txt

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["db_connection"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.db_connection = DBConnection(state["db_params"])

    def _init_write_file(self):
        if not self.write_txt:
            return

        with open(os.path.join(self.index_path, "index.txt"), "w"):
            pass

    def _append_index_file(self):
        if not self.write_txt:
            return

        index_merger = IndexMerger(self.index_path)
        with open(os.path.join(self.index_path, "index.txt"), "a") as f:
            index_merger.write_index(f)

    def process_posts(self):
        with self.db_connection as conn:
            min_id, max_id, num_posts = self._get_id_stats(conn)
            logger.info(
                f"Retrieved post stats: min_id={min_id}, max_id={max_id}, num_posts={num_posts}"
            )
            max_id = max_id if not self.debug else min_id + 1000

            partitions = self._calculate_partitions(min_id, max_id, num_posts, conn)
            logger.info(f"Calculated partitions: {partitions}")

        assert (
            partitions[-1][1] == max_id and partitions[0][0] == min_id
        ), f"{partitions[0]}!={min_id} or {partitions[-1]}!={max_id}"

        logger.info(f"Processing {num_posts} posts in {self.num_shards} shards")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = [
                executor.submit(
                    self._process_posts_shard,
                    shard=i,
                    start=partitions[i][0],
                    end=partitions[i][1],
                    db_params=self.db_connection.db_params,
                )
                for i in range(self.num_shards)
            ]
            merge_future = executor.submit(self._merge_sub_shards)

            for future in concurrent.futures.as_completed(futures):
                _ = future.result()

            merge_future.cancel()  # stop sub-shards from merging

        self._append_index_file()
        self._merge_shards()

    def _merge_sub_shards(self):
        index_merger = IndexMerger(self.index_path)

        logger.info("Merging sub-shards")
        while True:
            index_merger.merge(shards=self.num_shards, merge_subshards=True)
            time.sleep(30)

    def _merge_shards(self):
        index_merger = IndexMerger(self.index_path)
        index_merger.merge()

    def _process_posts_shard(self, shard: int, start: int, end: int, db_params: dict):
        """
        Process a shard of posts and builds the inverted index.
        Each shard is responsible for processing a range of post IDs.
        Each shard opens a new connection to the database.

        Args:
            shard: int
            start: int
            end: int
            db_params: dict

        Returns:
            term_docs: dict
        """

        index_builder = DocumentShardedIndexBuilder(self.index_path, shard)

        proc_conn = DBConnection(db_params)
        min_id = float("inf")
        max_id = float("-inf")
        with proc_conn as conn:
            cur = conn.get_cursor(name=f"index_builder_{shard}")
            # include tags???
            select_query = f"SELECT id, title, body FROM posts WHERE id >= {start} AND id <= {end} ORDER BY id ASC"
            cur.execute(select_query)
            logger.info(
                "Processing shard %d: %d-%d with query %s",
                shard,
                start,
                end,
                select_query,
            )

            while True:
                batch = cur.fetchmany(self.batch_size)
                if not batch:
                    logger.info(
                        f"Finished processing shard {shard} with {str(index_builder)} and min_id={min_id}, max_id={max_id}"
                    )
                    break

                for post_id, doc_terms, doc_length in self._process_posts_batch(batch):
                    if post_id < min_id:
                        min_id = post_id
                    if post_id > max_id:
                        max_id = post_id

                    index_builder.add_document(post_id, doc_terms, doc_length)

        logger.info(f"Processed shard {shard}: {start}-{end} => {str(index_builder)}")
        index_builder.flush(shard)
        return shard

    def _process_posts_batch(self, rows: tuple[list]):
        """
        A generator that processes a batch of posts and yields the post ID and document terms for each post
        This is called by _process_posts_shard to process a batch of posts for each shard

        TODO Implement char_offset for blocks (currently only within a single block) perhaps when rebuilding we can use it

        Args:
            rows: list of tuples

        Yields:
            post_id: int
            doc_terms: dict
        """
        for row in rows:
            post_id, title, body = row

            doc_terms = {}
            position_offset = 0

            for field, text in [("title", title), ("body", body)]:
                if text is None:
                    continue

                blocks = self._tokenize(text, field=field)
                for block in blocks:
                    for word in block.words:
                        if word.term not in doc_terms.keys():
                            doc_terms[word.term] = [word.position + position_offset]
                        else:
                            doc_terms[word.term].append(word.position + position_offset)

                    position_offset += block.block_length

            yield post_id, doc_terms, position_offset

    def _tokenize(self, text: str, field: str = "body"):
        if field == "title":
            return self.title_preprocessor(text)
        elif field == "body":
            return self.body_preprocessor(text)
        else:
            raise ValueError(f"Invalid field: {field}")

    def _get_id_stats(self, conn):
        """
        Get the min and max post IDs and the total number of posts in the database.

        Args:
            conn: DBConnection object

        Returns:
            min_id: int
            max_id: int
            num_posts: int
        """
        conn.execute("SELECT min(id) as minid, max(id) as maxid FROM posts")
        min_id, max_id = conn.cur.fetchone()

        # estimate the number of posts (COUNT(*) is slow)
        conn.execute(
            "SELECT reltuples AS estimate FROM pg_class WHERE relname = 'posts';"
        )
        num_posts = conn.cur.fetchone()[0]

        return min_id, max_id, num_posts

    def _calculate_partitions(self, min_id: int, max_id: int, num_rows: int, conn=None):
        rows_per_shard = num_rows // self.num_shards

        if conn is None:
            return []

        partition_precomputed = [
            (4, 3151437),
            (3151438, 6120881),
            (6120882, 9103209),
            (9103210, 12120967),
            (12120968, 15197897),
            (15197899, 18345721),
            (18345722, 21501458),
            (21501459, 24696756),
            (24696757, 27929211),
            (27929213, 31113102),
            (31113103, 34294425),
            (34294426, 37514246),
            (37514248, 40772152),
            (40772154, 44040056),
            (44040057, 47344058),
            (47344062, 50695276),
            (50695277, 54025254),
            (54025256, 57348340),
            (57348342, 60697390),
            (60697392, 64144975),
            (64144977, 67700460),
            (67700461, 71226625),
            (71226626, 74816231),
            (74816232, 78253176),
        ]

        if len(partition_precomputed) == self.num_shards:
            return partition_precomputed

        partition_query = f"""
            WITH rows AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY id) as row_num
                FROM posts
            )
            SELECT MIN(id) as shard_start, MAX(id) as shard_end
            FROM (
                SELECT id, SUM(CASE WHEN row_num % {rows_per_shard} = 1 THEN 1 ELSE 0 END) OVER (ORDER BY id) as shard_num
                FROM rows
            ) as t
            GROUP BY shard_num
            ORDER BY shard_num
        """

        conn.execute(partition_query)
        return conn.fetchall()


if __name__ == "__main__":
    import argparse

    from constants import DB_PARAMS
    from indexor.index import Index

    DB_PARAMS["database"] = "stack_overflow"

    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=str, default=".cache/index")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-shards", type=int, default=24)
    parser.add_argument("--write-debug-index", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    builder = IndexBuilder(
        DB_PARAMS,
        args.index_path,
        debug=args.debug,
        batch_size=args.batch_size,
        num_shards=args.num_shards,
        write_txt=args.write_debug_index,
    )
    builder.process_posts()

    index = Index(load_path=args.index_path)
    print(index.get_term("python", positions=True))
    print(index.get_term("java", positions=True))
    print(index.get_term("javascript", positions=True))
