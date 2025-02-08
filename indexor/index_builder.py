import logging
import json
import os
import concurrent.futures
import time

from filelock import FileLock

from indexor.index_builder.index_builder import DocumentShardedIndexBuilder
from indexor.index_builder.index_merger import IndexMerger

from preprocessing.preprocessor import Preprocessor
from utils.db_connection import DBConnection

logging.basicConfig(
    level=logging.DEBUG,
    format=f"%(asctime)s - %(name)s - {os.getpid()} - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEV_SHARDS = 1


class IndexBuilder:
    def __init__(
        self,
        db_params: dict,
        index_path: str,
        batch_size: int = 1000,
        num_workers: int = 24,
        shard_size=1000,
        debug=False,
        action: str = "build",
        is_sharded=False,
    ) -> None:
        """
        Args:
            db_params: dict
            index_path: str
            batch_size: int
                Decides size of batch to fetch on each database access
            num_workers: int
                Number of workers to use for processing posts. One worker is reserved for merging therefore we require at least 2 workers
            debug: bool
                If True, only process a subset of posts for debugging purposes
        """
        self.db_params = db_params
        self.db_connection = DBConnection(db_params)

        self.index_path = index_path
        self.temp_index_path = index_path  # ".cache/temp"

        self.batch_size = batch_size
        self.num_workers = min(num_workers, os.cpu_count() - 1)
        self.num_shards = self.num_workers - 1  # leave one worker for merging

        self.shard_size = shard_size

        self.action = action
        self.is_sharded = is_sharded

        self.title_preprocessor = Preprocessor(parser_kwargs={"parser_type": "raw"})
        self.body_preprocessor = Preprocessor(parser_kwargs={"parser_type": "html"})

        self.debug = debug
        self.config = {
            "batch_size": self.batch_size,
            "num_shards": self.num_shards,
            "is_sharded": self.is_sharded,
        }

        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)

        with open(os.path.join(self.index_path, "config.json"), "w") as f:
            json.dump(self.config, f)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["db_connection"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.db_connection = DBConnection(state["db_params"])

    def process_posts(self):
        """
        Main entry point for processing posts and building the inverted index.

        Initialises the processes for each shard and merges the subsequent shards to build the final index
        Each process generates sub-shards (e.g. shard_0_0, shard_0_1, shard_0_2, etc.)
        These are then merged to form a shard (e.g. shard_0)

        The final shards are then merged to form the final index in the specified index_path (e.g. postings.bin and terms.fst)

        Note:
        - The number of shards is equal to the number of workers minus 1 (one worker is reserved for merging)
        - The number of workers is capped at the number of available CPUs minus 1
        - Coordination between the main process and the workers is done via a file .cache/shards_finished. Each worker writes to this file when it finishes processing its shard so that we can kill the sub-shard merging process
        """
        if self.action == "merge":
            logger.info("Merging shards")
            return self._merge_shards()
        elif self.action == "build-fst":
            logger.info("Building FST index")
            index_merger = IndexMerger(self.temp_index_path, output_dir=self.index_path)
            return index_merger.build_all_term_fsts(shards=self.num_shards)

        assert self.action == "build", f"Invalid action: {self.action}"
        logger.info("Building index")

        with self.db_connection as conn:
            min_id, max_id, num_posts = self._get_id_stats(conn)
            logger.info(
                f"Retrieved post stats: min_id={min_id}, max_id={max_id}, num_posts={num_posts}"
            )
            max_id = max_id if not self.debug else min_id + 1000

            partitions = self._calculate_partitions(min_id, max_id, num_posts, conn)
            partitions[-1][1] = max(max_id, partitions[-1][1])
            logger.info(f"Calculated partitions: {partitions}")

        shards_finished = os.path.join(self.index_path, "shards_finished")
        with open(shards_finished, "w") as f:
            f.write("")

        """
        assert (
            partitions[-1][1] == max_id and partitions[0][0] == min_id
        ), f"{partitions[0]}!={min_id} or {partitions[-1]}!={max_id}"
        """

        logger.info(f"Processing {num_posts} posts in {self.num_shards} shards")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers,
        ) as executor:
            index_merger = IndexMerger(self.temp_index_path)
            index_merger.cleanup()
            del index_merger

            merge_future = executor.submit(self._merge_sub_shards)
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

            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                _ = future.result()
                logger.info(f"Finished processing shard {idx}")

                file_lock = FileLock(shards_finished + ".lock")
                with file_lock:
                    with open(shards_finished, "a") as f:
                        f.write("FINISHED\n")

            time.sleep(1)
            merge_future.result()  # stop sub-shards from merging

        self._merge_shards()

    def _merge_sub_shards(self):
        """
        Called by the main process to merge the sub-shards generated by each worker
        Iterates through all the sub-shards and merges them to form the final shard
            e.g. (shard_0_0, shard_0_1, shard_0_2, etc.) => shard_0

        Also, deletes all prior shards in the index folder specified
        """
        index_merger = IndexMerger(self.temp_index_path)

        value = 0
        shards_finished = os.path.join(self.index_path, "shards_finished")
        file_lock = FileLock(shards_finished + ".lock")

        logger.info("Merging sub-shards")
        while value < self.num_shards:
            index_merger.merge(
                shards=self.num_shards, merge_subshards=True, force_merge=False
            )
            time.sleep(1)

            with file_lock:
                with open(shards_finished, "r") as f:
                    value = len(f.readlines())

        logger.info("All sub-shards finished. Performing final merge")
        index_merger.merge(
            shards=self.num_shards, merge_subshards=True, force_merge=True
        )
        logger.info("Finished merging sub-shards")

    def _merge_shards(self):
        index_merger = IndexMerger(self.temp_index_path, output_dir=self.index_path)

        if self.is_sharded:
            logger.info("Skipping final merge")
            index_merger.post_merge_cleanup()
            index_merger.build_all_term_fsts(shards=self.num_shards)
            index_merger.build_all_term_fsts(
                shards=self.num_shards, prefix="pos_shard", save_filename="pos_terms"
            )
            return

        logger.info("Performing final merge")
        index_merger.merge()
        index_merger.post_merge_cleanup()

    def _process_posts_shard(self, shard: int, start: int, end: int, db_params: dict):
        """
        Processes a shard of posts. Each batch is pre-processed and given to the DocumentShardedIndexBuilder to build the inverted index for the shard
        Once the shard id completed, the DocumentShardedIndexBuilder is flushed to disk via Obj.flush(shard)

        Args:
            shard: int
            start: int
            end: int
            db_params: dict

        Returns:
            shard: int - the shard number that was processed
        """

        index_builder = DocumentShardedIndexBuilder(self.temp_index_path, shard)

        proc_conn = DBConnection(db_params)
        min_id = float("inf")
        max_id = float("-inf")
        with proc_conn as conn:
            cur = conn.get_cursor(name=f"index_builder_{shard}")
            # include tags???
            debug = "LIMIT 100" if self.debug else ""
            select_query = f"SELECT id, title, body FROM posts WHERE id >= {start} AND id <= {end} ORDER BY id ASC {debug}"
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

        return index_builder.doc_count

    def _process_posts_batch(self, rows: tuple[list]):
        """
        A generator that processes a batch of posts and yields the post ID and document terms for each post
        This is called by _process_posts_shard to process a batch of posts for each shard

        Args:
            rows: list of tuples

        Yields:
            post_id: int
            doc_terms: dict
            postition_offset: int
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
        if self.num_shards == 1:
            return [(min_id, max_id)]

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
            (64144977, 71226625),
            # (67700461, 71226625),
            (71226626, 78253176),
            # (64144977, 67700460),
            # (67700461, 71226625),
            # (71226626, 78253176),
            #  (74816232, 78253176),
        ]

        if len(partition_precomputed) == self.num_shards:
            return partition_precomputed

        logger.info(
            f"Calculating partitions since {len(partition_precomputed)} != {self.num_shards}"
        )

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=str, default=".cache/index", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--db-name", type=str, default="stack_overflow_10k")
    parser.add_argument(
        "--action",
        choices=["build", "merge", "build-fst"],
        help="build: build the index and merge shards, merge: merge the shards to form the final index, build-fst: build the FST index from the existing index shards (shard_0, shard_1, etc.) => terms.fst",
    )
    parser.add_argument(
        "--is-sharded",
        action="store_true",
        help="Should we merge the shards shard_0...shard_n to form the final index postings.bin and terms.fst",
    )
    parser.add_argument("--write-index-to-txt", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    DB_PARAMS["database"] = args.db_name

    builder = IndexBuilder(
        DB_PARAMS,
        args.index_path,
        debug=args.debug,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        action=args.action,
        is_sharded=args.is_sharded,
    )
    builder.process_posts()
    index = Index(load_path=args.index_path)

    if args.write_index_to_txt:
        index.write_index_to_txt(os.path.join(args.index_path, "index.txt"))

    print(index.get_term("!", positions=False))
    print(index.get_term("python", positions=False))
    print(index.get_term("java", positions=False))
    print(index.get_term("javascript", positions=False))

    print(index.get_term("!", positions=True))
    print(index.get_term("python", positions=True))
    print(index.get_term("java", positions=True))
    print(index.get_term("javascript", positions=True))
