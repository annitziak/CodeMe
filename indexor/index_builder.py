import logging
import os
import concurrent.futures


from preprocessing.preprocessor import Preprocessor
from utils.db_connection import DBConnection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEV_SHARDS = 1


class IndexBuilder:
    def __init__(
        self,
        db_params: dict,
        index_path: str,
        batch_size: int = 1000,
        num_shards: int = 128,
        debug=False,
    ) -> None:
        self.db_params = db_params
        self.db_connection = DBConnection(db_params)

        self.index_path = index_path

        self.batch_size = batch_size
        self.num_shards = num_shards if not debug else DEV_SHARDS

        self.title_preprocessor = Preprocessor(parser_kwargs={"parser_type": "raw"})
        self.body_preprocessor = Preprocessor(parser_kwargs={"parser_type": "html"})

        self.debug = debug

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["db_connection"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.db_connection = DBConnection(state["db_params"])

    def process_posts(self):
        with self.db_connection as conn:
            min_id, max_id, num_posts = self._get_id_stats(conn)
            logger.info(
                f"Retrieved post stats: min_id={min_id}, max_id={max_id}, num_posts={num_posts}"
            )
            max_id = max_id if not self.debug else min_id + 1000

        chunk_size = (max_id - min_id) // self.num_shards
        partitions = [
            (i * chunk_size, (i + 1) * chunk_size) for i in range(self.num_shards)
        ]
        logger.info(f"Processing {num_posts} posts in {self.num_shards} shards")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(os.cpu_count() - 1, self.num_shards)
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

            for future in concurrent.futures.as_completed(futures):
                terms = future.result()
                print(terms)
                with open("index.txt", "w") as file:
                    file.write("")
                words = sorted(list(terms.keys()))
                for word in words:
                    doc_no = sorted(list(terms[word].keys()))
                    with open("index.txt", "a") as file:
                        try:
                            file.write(f"{word}:{len(doc_no)}\n")
                        except Exception:
                            file.write(f"bad_encoding:{len(doc_no)}\n")
                    for no in doc_no:
                        posn_list = str(terms[word][no]).replace(" ", "")[1:-1]
                        with open("index.txt", "a") as file:
                            file.write(f"\t{str(no)}: {posn_list}\n")

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

        proc_conn = DBConnection(db_params)
        with proc_conn as conn:
            cur = conn.get_cursor(name="index_builder")
            # include tags???
            cur.execute(
                """
                SELECT id, title, body
                FROM posts
                -- WHERE id >= %s AND id < %s
                """,
                (start, end),
            )
            logger.info("Processing shard %d: %d-%d", shard, start, end)

            term_docs = {}

            while True:
                batch = cur.fetchmany(self.batch_size)
                if not batch:
                    logger.info(
                        f"Finished processing shard {shard} with {len(term_docs)} terms"
                    )
                    break

                for post_id, doc_terms in self._process_posts_batch(batch):
                    for term, posn_list in doc_terms.items():
                        if term not in term_docs:
                            term_docs[term] = {}

                        term_docs[term][post_id] = posn_list

        logger.info(f"Processed shard {shard}: {start}-{end} => {len(term_docs)} terms")
        return term_docs

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

            yield post_id, doc_terms

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


if __name__ == "__main__":
    import argparse

    from constants import DB_PARAMS

    DB_PARAMS["database"] = "stack_overflow_small"

    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=str, default=".cache/index")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-shards", type=int, default=24)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    builder = IndexBuilder(
        DB_PARAMS,
        args.index_path,
        debug=args.debug,
        batch_size=args.batch_size,
        num_shards=args.num_shards,
    )
    builder.process_posts()
