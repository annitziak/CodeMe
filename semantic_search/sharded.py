import numpy as np
import os
import torch
import time
from tqdm.auto import tqdm
import h5py
import pickle
from concurrent.futures import ThreadPoolExecutor
import logging
from transformers import AutoTokenizer

from preprocessing.parser import DefaultParserInterface, HTMLParserInterface

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64,
        storage_path="./embeddings",
        max_documents=None,
    ):
        """
        Initialize the embedding pipeline with the specified model and parameters.

        Args:
            model_name: The name of the embedding model to use
            batch_size: Number of documents to process in a single batch
            storage_path: Where to store the computed embeddings
            max_documents: Maximum number of documents to process (None for all)
        """
        self.model_name = model_name
        self.batch_size = batch_size

        self.storage_path = storage_path
        self.max_documents = max_documents

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)

        # Initialize model lazily (only when needed)
        self._model = None
        self._tokenizer = None
        self._metadata = {
            "total_documents": 0,
            "embedding_dim": None,
            "model_name": model_name,
            "batch_size": batch_size,
            "document_ids": [],
        }

    @property
    def model(self):
        """Lazy-load the model only when needed"""
        if self._model is None:
            logger.info(f"Loading model: {self.model_name}")
            start_time = time.time()

            # Import here to avoid loading these libraries unless needed
            from transformers import AutoModel
            from sentence_transformers import SentenceTransformer

            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self._model = SentenceTransformer(
                model_name_or_path=self.model_name,
                device=self.device,
            )

            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.model

        return self._tokenizer

    def fetch_documents(self, document_source, prefetch_size=10000):
        """
        Generator that yields batches of documents from the source.
        Implements prefetching to reduce I/O bottlenecks.

        Args:
            document_source: Iterable source of documents (e.g., database cursor)
            prefetch_size: Number of documents to prefetch at once

        Yields:
            Batches of documents as lists
        """
        logger.info("Starting document fetching process")
        document_buffer = []
        document_count = 0

        # Helper function to get a batch of documents from the source
        def get_next_batch():
            batch = []
            try:
                for _ in range(prefetch_size):
                    if self.max_documents and document_count >= self.max_documents:
                        break
                    doc = next(document_source)
                    batch.append(doc)
            except StopIteration:
                pass
            return batch

        # Start with an initial fetch
        document_buffer = get_next_batch()
        logger.info(f"Initially fetched {len(document_buffer)} documents")

        while document_buffer:
            # Yield a batch of documents
            batch_size = min(self.batch_size, len(document_buffer))
            batch = document_buffer[:batch_size]
            document_buffer = document_buffer[batch_size:]
            document_count += len(batch)

            # If the buffer is getting low, fetch more documents in the background
            if len(document_buffer) < self.batch_size * 2 and (
                self.max_documents is None or document_count < self.max_documents
            ):
                with ThreadPoolExecutor(
                    max_workers=min(20, os.cpu_count() - 2)
                ) as executor:
                    future = executor.submit(get_next_batch)
                    yield batch
                    # Get the results when ready
                    document_buffer.extend(future.result())
            else:
                yield batch

            if self.max_documents and document_count >= self.max_documents:
                logger.info(f"Reached maximum document count: {self.max_documents}")
                break

        logger.info(f"Finished fetching {document_count} documents")

    def compute_embeddings(self, documents, document_ids=None):
        """
        Compute embeddings for a batch of documents

        Args:
            documents: List of document texts
            document_ids: Optional list of document IDs

        Returns:
            Numpy array of embeddings
        """
        if not documents:
            return np.array([])

        if hasattr(self.model, "encode"):
            # Sentence Transformers models
            embeddings = self.model.encode(
                documents, convert_to_tensor=False, show_progress_bar=False
            )
        else:
            inputs = self.tokenizer(
                documents, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            embeddings = (
                self.model(**inputs).last_hidden_state[:, 0, :].detach().cpu().numpy()
            )

        # Update metadata if this is the first batch
        if self._metadata["embedding_dim"] is None:
            self._metadata["embedding_dim"] = embeddings.shape[1]

        return embeddings

    def process_all(self, document_iterator, doc_id_func=None):
        """
        Process all documents from the iterator, computing embeddings and saving them

        Args:
            document_iterator: Iterator yielding documents
            doc_id_func: Function to extract document ID from a document
        """
        # Initialize storage
        embedding_file = h5py.File(
            os.path.join(self.storage_path, "embeddings.h5"), "w"
        )
        embeddings_dataset = None
        document_ids = []

        total_docs = 0
        total_batches = 0
        start_time = time.time()
        checkpoint_interval = 100_000

        logger.info(f"Starting embedding computation with batch size {self.batch_size}")

        # Process documents in batches
        for batch in tqdm(self.fetch_documents(document_iterator)):
            # Extract text content and IDs if needed
            if doc_id_func:
                ids = [doc_id_func(doc) for doc in batch]
                document_ids.extend(ids)

            # Get the actual text content to embed
            texts = [self._extract_text(doc) for doc in batch]

            # Compute embeddings
            batch_embeddings = self.compute_embeddings(texts)

            # Initialize dataset if this is the first batch
            if embeddings_dataset is None and len(batch_embeddings) > 0:
                embedding_dim = batch_embeddings.shape[1]
                # Create a resizable dataset
                embeddings_dataset = embedding_file.create_dataset(
                    "embeddings",
                    shape=(0, embedding_dim),
                    maxshape=(None, embedding_dim),
                    dtype="float32",
                    chunks=(min(1000, self.batch_size), embedding_dim),
                    compression="gzip",
                )

            # Skip empty batches
            if len(batch_embeddings) == 0:
                continue

            # Resize dataset and add new embeddings
            current_size = embeddings_dataset.shape[0]
            embeddings_dataset.resize(current_size + len(batch_embeddings), axis=0)
            embeddings_dataset[current_size:] = batch_embeddings

            total_docs += len(batch)
            total_batches += 1

            # Update metadata periodically
            if total_docs % checkpoint_interval == 0:
                self._metadata["total_documents"] = total_docs
                self._metadata["document_ids"] = document_ids
                self._metadata["last_updated"] = time.time()

                # Save metadata snapshot
                metadata_path = os.path.join(self.storage_path, "metadata.pkl")
                with open(metadata_path, "wb") as f:
                    pickle.dump(self._metadata, f)

                logger.info(f"Saved metadata checkpoint at {total_docs} documents")

            # Periodic logging
            if total_batches % 10 == 0:
                elapsed = time.time() - start_time
                docs_per_sec = total_docs / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Processed {total_docs} documents in {elapsed:.2f}s ({docs_per_sec:.2f} docs/sec)"
                )

        # Save metadata
        self._metadata["total_documents"] = total_docs
        self._metadata["document_ids"] = document_ids

        with open(os.path.join(self.storage_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self._metadata, f)

        embedding_file.close()

        # Final statistics
        elapsed = time.time() - start_time
        logger.info(
            f"Finished processing {total_docs} documents in {elapsed:.2f} seconds"
        )
        logger.info(f"Average processing speed: {total_docs / elapsed:.2f} docs/sec")

        return total_docs

    def _extract_text(self, document):
        """
        Extract text content from a document object.
        Override this method based on your document format.

        Args:
            document: The document object

        Returns:
            Text content as a string
        """
        # Default implementation assumes document is already text or has a 'text' attribute
        if isinstance(document, str):
            return document
        elif hasattr(document, "text"):
            return document.text
        elif hasattr(document, "get"):
            return document.get("text", str(document))
        else:
            return str(document)


def setup_doc_generator():
    import argparse
    from utils.db_connection import DBConnection
    from constants.db import DB_PARAMS

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--db_name", type=str, default="stack_overflow")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default=".cache")
    parser.add_argument("--index-dir", type=str, default=".cache/index")
    args = parser.parse_args()

    DB_PARAMS["database"] = args.db_name
    db_connection = DBConnection(DB_PARAMS)
    top_p = 0.2

    parser = HTMLParserInterface()
    title_parser = DefaultParserInterface()
    batch_size = 64

    '''
    def document_generator(workers=4):
        """
        Generator that fetches and parses documents in paginated chunks.

        Args:
            conn: Database connection object.
            limit: Total number of documents to fetch.
            top_p: Percentage of posts to sample.
            num_posts: Total number of posts in the database.
            title_parser, parser: Parsers for title and body.
            batch_size: Number of rows to fetch per page.
            workers: Number of threads for parallel parsing.

        Yields:
            Parsed documents as dictionaries.
        """
        with db_connection as conn:
            conn.execute(
                "SELECT reltuples AS estimate FROM pg_class WHERE relname = 'posts';"
            )
            num_posts = conn.fetchone()[0]
            limit = min(args.limit, num_posts) if args.limit > 0 else num_posts

            total_docs = min(limit * top_p, num_posts * top_p)
            logger.info(
                f"Fetching up to {total_docs} documents in pages of {batch_size}"
            )

            def parse_row(row):
                """Parse title and body into text."""
                title_text = [x.text for x in title_parser.parse(row[1]) if x.text]
                body_text = [x.text for x in parser.parse(row[2]) if x.text]
                text = " ".join(title_text + body_text)
                return {"id": row[0], "text": text}

            offset = 0
            last_offset
            with ThreadPoolExecutor(max_workers=workers) as executor:
                while offset < total_docs:
                    # Build paginated SQL query
                    sql_query = f"""
                        SELECT id, title, body
                        FROM posts

                        LIMIT {batch_size} OFFSET {offset};
                    """
                    logger.info(f"Running query: {sql_query}")

                    cursor = conn.get_cursor(name="document_generator")
                    cursor.itersize = batch_size
                    cursor.execute(sql_query)

                    batch = cursor.fetchmany(batch_size)
                    if not batch:
                        break

                    # Process rows in parallel
                    future_to_row = {
                        executor.submit(parse_row, row): row for row in batch
                    }

                    for future in as_completed(future_to_row):
                        yield future.result()

                    offset += len(batch)

                    # Exit early if we hit the document limit
                    if offset >= total_docs:
                        logger.info(f"Fetched {offset}/{total_docs} documents")
                        break

                    cursor.close()

            logger.info("Finished generating documents")

    '''

    def document_generator():
        with db_connection as conn:
            conn.execute(
                "SELECT reltuples AS estimate FROM pg_class WHERE relname = 'posts';"
            )
            num_posts = conn.fetchone()[0]
            limit = min(args.limit, num_posts) if args.limit > 0 else num_posts

            sql_query = f"""
                SELECT id,title,body
                FROM posts
                WHERE posttypeid=1
                LIMIT {min(limit * top_p, num_posts * top_p)};
            """
            logger.info(f"Running query: {sql_query}")
            #  AND score > 5 AND viewcount > 1000
            # AND answercount > 1
            #  {int(limit * top_p)}
            cursor = conn.get_cursor(name="document_generator")
            cursor.itersize = 20000
            cursor.execute(sql_query)

            while True:
                batch = cursor.fetchmany(64)
                if not batch or len(batch) == 0:
                    break

                for row in batch:
                    title_text = [x.text for x in title_parser.parse(row[1]) if x.text]
                    body_text = [x.text for x in parser.parse(row[2]) if x.text]
                    text = " ".join(title_text + body_text)

                    yield {
                        "id": row[0],
                        "text": text,
                    }

    return document_generator


# Example usage:
if __name__ == "__main__":
    # Initialize and run the pipeline
    document_generator = setup_doc_generator()

    pipeline = EmbeddingPipeline(
        # model_name="microsoft/codebert-base",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # model_name="sentence-transformers/multi-qa-MiniLM-L6-dot-v1",
        # model_name="krlvi/sentence-t5-base-nlpl-code_search_net",
        batch_size=64,
        # max_documents=2000,  # Limit for testing
    )

    pipeline.process_all(
        document_generator(),
        doc_id_func=lambda doc: doc["id"] if isinstance(doc, dict) else str(id(doc)),
    )
