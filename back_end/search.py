import time
import os
import logging


from preprocessing.preprocessor import Preprocessor
from indexor.index import Index
from indexor.query import FreeTextQuery, BooleanQuery
from retrieval_models.retrieval_functions import (
    query_expansion,
    reorder_as_date,
    reorder_as_tag,
)
from retrieval_models.query_expansion import EmbeddingModel

from back_end.modeling_outputs import SearchResult
from back_end.mock_search import MockSearch
from back_end.reranker import Reranker

logger = logging.getLogger(__name__)

BOOSTED_TERMS = {
    "python",
    "java",
    "c",
    "javascript",
    "typescript",
    "rust",
    "golang",
    "swift",
    "php",
    "r",
    "matlab",
    "sql",
    "nosql",
    "html",
    "css",
    "ruby",
    "array",
    "list",
    "tree",
    "graph",
    "heap",
    "hashmap",
    "queue",
    "stack",
}
WORD_LIMIT_SEARCH = 20


def load_backend(
    index_path,
    embedding_path="retrieval_models/data/embedding2.pkl",
    reranker_path="/media/seanleishman/Disk/embeddings_v2",
):
    if index_path is None or not os.path.exists(index_path):
        logger.error(f"Index path {index_path} does not exist. Using mock data")
        return Search.mock()

    index = Index(load_path=index_path)
    preprocessor = Preprocessor(parser_kwargs={"parser_type": "raw"})
    embedding_model = EmbeddingModel(
        vocab=None,
        vocab_fn=index.get_vocab,
        save_path=embedding_path,
    )
    reranker = Reranker(load_dir=reranker_path)

    return Search(index, preprocessor, embedding_model, reranker)


def to_py(item):
    if hasattr(item, "item"):
        return item.item()

    return item


def reorder_as_per_filter(result, selected_clusters=None, reorder_date=False):
    if reorder_date:
        logger.info("Reordering results by date")
        result = reorder_as_date(result)  # Reorder by date

    if selected_clusters is not None:
        logger.info(f"Filtering results by selected clusters: {selected_clusters}")
        result = reorder_as_tag(
            result, selected_clusters
        )  # Reorder by selected cluster names

    return result  # Return reordered results


class SearchCache:
    def __init__(self, max_size=100_000):
        self.cache = {}
        self.max_size = max_size

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, key):
        return self.cache[key]

    def __setitem__(self, key, value):
        if len(self.cache) >= self.max_size:
            self.cache.popitem()
        self.cache[key] = value

    def get(self, key, default=None):
        return self.cache.get(key, default)


class Search:
    def __init__(
        self,
        index: Index,
        preprocessor: Preprocessor,
        embedding_model,
        reranker: Reranker,
    ):
        self.index = index
        self.embedding_model = embedding_model
        self.reranker = reranker

        self.preprocessor = preprocessor
        boosted_terms = [
            self.preprocessor.preprocess(term, return_words=True)
            for term in BOOSTED_TERMS
        ]
        self.boosted_terms = {k[0] for k in boosted_terms if len(k) == 1}
        logger.info(
            f"Search Engine initialized with the following boosted terms: {self.boosted_terms}"
        )

        self.cache = SearchCache()

    def _pre_search(self, query, expansion=False, boost_terms=True, k=10):
        logger.info(f"Preprocessing query: {query}")
        tokens = self.preprocessor.preprocess(query, return_words=True)

        if expansion and self.embedding_model:
            tokens = query_expansion(tokens, self.embedding_model, top_k=k)
        if boost_terms:
            tokens.extend([token for token in tokens if token in self.boosted_terms])

        query = self.preprocessor.preprocess(" ".join(tokens), return_words=True)

        logger.info(f"Query after preprocessing: {query}")
        return query

    def _post_search(
        self,
        results,
        query="",
        rerank_metadata=True,
        rerank_lm=False,
        page=0,
        page_size=20,
        selected_clusters=None,
        reorder_date=False,
    ):
        total_results = len(results)
        return_result = SearchResult(
            results=results, time_taken=0, total_results=total_results, query=query
        )

        start_idx = page * page_size
        end_idx = start_idx + 500

        results = results[start_idx:end_idx]
        if end_idx < total_results:
            return_result.has_next = True
        if start_idx > 0:
            return_result.has_prev = True

        return_result.results = self.format_results(results)
        return_result.query = query

        logger.info(f"Result from index after clipping: {return_result}")
        return_result.results = reorder_as_per_filter(
            return_result.results,
            selected_clusters=selected_clusters,
            reorder_date=reorder_date,
        )

        if rerank_metadata:
            return_result.results = self.reranker.rerank_metadata(return_result.results)
            logger.info(f"Reranked with metadata: {return_result}")
        if rerank_lm:
            return_result.results = self.reranker.rerank_lm(
                return_result.results, query
            )
            logger.info(f"Reranked with LM: {return_result}")

        return_result.results = self.reranker.fuse_scores(return_result.results)
        return_result.results = return_result.results[:page_size]

        return_result.results = reorder_as_per_filter(
            return_result.results,
            reorder_date=reorder_date,
        )

        return return_result

    def search(
        self,
        query,
        expansion=False,
        boost_terms=True,
        rerank_metadata=True,
        rerank_lm=True,
        page=0,
        page_size=20,
        k_word_expansion=10,
        selected_clusters=None,
        reorder_date=False,
        word_limit=WORD_LIMIT_SEARCH,
    ) -> SearchResult:
        start = time.time()

        tokens = self._pre_search(query, expansion, boost_terms, k=k_word_expansion)

        query = FreeTextQuery(tokens, word_limit=word_limit)
        if query in self.cache:
            results = self.cache[query]
        else:
            results = self.index.search(query)
            self.cache[query] = results

        results += self._backfill_results(page_size - len(results))

        ret = self._post_search(
            results,
            rerank_metadata=rerank_metadata,
            rerank_lm=rerank_lm,
            page=page,
            page_size=page_size,
            selected_clusters=selected_clusters,
            reorder_date=reorder_date,
        )

        ret.time_taken = time.time() - start
        logger.info(
            f"Search took {ret.time_taken} seconds to return {ret.total_results} results."
        )

        return ret

    def advanced_search(
        self,
        query,
        expansion=False,
        boost_terms=True,
        k=10,
        page=0,
        page_size=20,
        rerank_metadata=True,
        selected_clusters=None,
        reorder_date=False,
    ):
        start = time.time()
        query = BooleanQuery(query, preprocessor=self.preprocessor)
        query.parse()
        logger.info(f"Query after preprocessing: \n{query._ppformat()}")

        if query in self.cache:
            results = self.cache[query]
        else:
            results = self.index.search(query)
            self.cache[query] = results

        results += self._backfill_results(page_size - len(results))
        ret = self._post_search(
            results,
            page=page,
            page_size=page_size,
            rerank_metadata=rerank_metadata,
            rerank_lm=False,
            selected_clusters=selected_clusters,
            reorder_date=reorder_date,
        )

        ret.time_taken = time.time() - start
        logger.info(
            f"Search took {ret.time_taken} seconds to return {ret.total_results} results."
        )

        return ret

    def format_results(self, results):
        ret = []
        for doc_result in results:
            metadata = self.index.get_document_metadata(doc_result[1])
            if metadata is None:
                continue

            ret.append(
                {
                    "doc_id": to_py(doc_result[1]),
                    "bm25_score": to_py(doc_result[0]),
                    "score": to_py(metadata["score"]),
                    "tags": metadata["tags"],
                    "ownerdisplayname": metadata["ownerdisplayname"],
                    "creation_date": to_py(metadata["creationdate"]),
                    "view_count": to_py(metadata["viewcount"]),
                    "answer_count": to_py(metadata["answercount"]),
                    "comment_count": to_py(metadata["commentcount"]),
                    "favorite_count": to_py(metadata["favoritecount"]),
                    "metadata_score": to_py(metadata["metadatascore"]),
                    "title": metadata["title"],
                    "body": metadata["body"],
                }
            )

        return ret

    def _backfill_results(self, num_results, query_type=FreeTextQuery):
        if num_results <= 0:
            return []

        tokens = self._pre_search("python")
        query = query_type(tokens, preprocessor=self.preprocessor)
        results = self.index.search(query)
        return results[:num_results]

    @staticmethod
    def default():
        return load_backend(".cache/index-1m-metadata")

    @staticmethod
    def mock():
        return MockSearch()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search Engine")
    parser.add_argument("--index-path", type=str, help="Path to load index")
    parser.add_argument(
        "--embedding-path",
        type=str,
        help="Path to load embeddings",
        default="retrieval_models/data/embedding2.pkl",
    )
    parser.add_argument(
        "--reranker-path",
        type=str,
        help="Path to load reranker embeddings",
        default="/media/seanleishman/Disk/embeddings_v2",
    )
    args = parser.parse_args()

    search = load_backend(args.index_path, args.embedding_path, args.reranker_path)

    while True:
        query = input("Enter query: ")

        results = search.search(query, expansion=False, boost_terms=True)
        print(results)

        results = search.advanced_search(query, expansion=False, boost_terms=True)

    search.search("hello world in python", expansion=False, boost_terms=True)
    search.search(
        "how good is python as a programming langauge. Does it do well for FindMax queries",
        expansion=False,
    )
