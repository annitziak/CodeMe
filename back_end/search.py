import time
import os
import logging


from preprocessing.preprocessor import Preprocessor
from indexor.index import Index
from indexor.query import FreeTextQuery, BooleanQuery
from retrieval_models.retrieval_functions import query_expansion
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


def load_backend(index_path):
    if index_path is None or not os.path.exists(index_path):
        logger.error(f"Index path {index_path} does not exist. Using mock data")
        return Search.mock()

    index = Index(load_path=index_path)
    preprocessor = Preprocessor(parser_kwargs={"parser_type": "raw"})
    embedding_model = EmbeddingModel(
        vocab=index.get_vocab(), save_path="retrieval_models/data/embedding.pkl"
    )
    reranker = Reranker()

    return Search(index, preprocessor, embedding_model, reranker)


def to_py(item):
    if hasattr(item, "item"):
        return item.item()

    return item


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
        self, results, query="", rerank_metadata=True, rerank_lm=True, k=10
    ):
        total_results = len(results)
        return_result = SearchResult(
            results=results, time_taken=0, total_results=total_results, query=query
        )

        results = results[:k]
        return_result.results = self.format_results(results)
        return_result.query = query

        logger.info(f"Result from index after clipping: {return_result}")

        if rerank_metadata:
            return_result.results = self.reranker.rerank_metadata(return_result.results)
            logger.info(f"Reranked with metadata: {return_result}")
        if rerank_lm:
            return_result.results = self.reranker.rerank_lm(
                return_result.results, query
            )
            logger.info(f"Reranked with LM: {return_result}")

        return return_result

    def search(self, query, expansion=False, boost_terms=True, k=10) -> SearchResult:
        start = time.time()

        tokens = self._pre_search(query, expansion, boost_terms, k)
        query = FreeTextQuery(tokens)
        results = self.index.search(query)
        ret = self._post_search(results, rerank_metadata=True, rerank_lm=False)

        ret.time_taken = time.time() - start
        logger.info(
            f"Search took {ret.time_taken} seconds to return {ret.total_results} results."
        )

        return ret

    def advanced_search(self, query, expansion=False, boost_terms=True, k=10):
        start = time.time()
        tokens = self.preprocessor.preprocess(query, return_words=True)

        if expansion and self.embedding_model:
            tokens = query_expansion(tokens, self.embedding_model, top_k=k)
        if boost_terms:
            tokens.extend([token for token in tokens if token in self.boosted_terms])

        # Encompasses BooleanQuery, PhraseQuery and ProximityQuery
        tokens = self._pre_search(query, expansion, boost_terms, k)
        query = BooleanQuery(tokens)
        results = self.index.search(query)
        ret = self._post_search(results)

        ret.time_taken = time.time() - start
        logger.info(
            f"Search took {ret.time_taken} seconds to return {ret.total_results} results."
        )

        return ret

    def format_results(self, results):
        ret = []
        for doc_result in results:
            metadata = self.index.get_document_metadata(doc_result[1])
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
    args = parser.parse_args()

    search = load_backend(args.index_path)
    search.search("hello world in python", expansion=False, boost_terms=True, k=10)
    search.search(
        "how good is python as a programming langauge. Does it do well for FindMax queries",
        expansion=False,
        k=10,
    )

    while True:
        query = input("Enter query: ")
        results = search.search(query, expansion=False, boost_terms=True, k=10)
        print(results)
