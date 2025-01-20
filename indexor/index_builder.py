import logging
import concurrent.futures

from collections import defaultdict
from utils.db_connection import DBConnection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IndexBuilder:
    def __init__(
        self,
        db_params: dict,
        index_path: str,
        batch_size: int = 1000,
        num_shards: int = 128,
    ) -> None:
        self.db_params = db_params
        self.db_connection = DBConnection(db_params)

        self.index_path = index_path

        self.batch_size = batch_size
        self.num_shards = num_shards

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

        chunk_size = (max_id - min_id) // self.num_shards
        partitions = [
            (i * chunk_size, (i + 1) * chunk_size) for i in range(self.num_shards)
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
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
                future.result()

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
        logger.info("Processing shard %d: %d-%d", shard, start, end)

        proc_conn = DBConnection(db_params)
        with proc_conn as conn:
            cur = conn.get_cursor(name="index_builder")
            cur.execute(
                """
                SELECT id, title, body, tags
                FROM posts
                WHERE id >= %s AND id < %s
                """,
                (start, end),
            )
            logger.info("Processing shard %d: %d-%d", shard, start, end)

            term_docs = defaultdict(lambda: defaultdict(float))

            while True:
                batch = cur.fetchmany(self.batch_size)
                if not batch:
                    break

                for post_id, doc_terms in self._process_posts_batch(batch):
                    for term, tf in doc_terms.items():
                        term_docs[term][post_id] = tf

        logger.info(f"Processed shard {shard}: {start}-{end} => {len(term_docs)} terms")
        return term_docs

    def _process_posts_batch(self, rows: tuple[list]):
        """
        A generator that processes a batch of posts and yields the post ID and document terms for each post
        This is called by _process_posts_shard to process a batch of posts for each shard

        Args:
            rows: list of tuples

        Yields:
            post_id: int
            doc_terms: dict
        """
        for row in rows:
            post_id, title, body, tags = row

            doc_terms = defaultdict(float)

            for field, text in [("title", title), ("body", body), ("tags", tags)]:
                terms = self._tokenize(text)
                for term in terms:
                    doc_terms[term] += 1.0

            yield post_id, doc_terms

    def _tokenize(self, text: str):
        return text.lower().split()

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

        conn.execute("SELECT count(*) FROM posts")
        num_posts = conn.cur.fetchone()[0]

        return min_id, max_id, num_posts


if __name__ == "__main__":
    import os

    db_params = {
        "dbname": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
    }
    index_path = ".cache/index"

    builder = IndexBuilder(db_params, index_path)
    builder.process_posts()
