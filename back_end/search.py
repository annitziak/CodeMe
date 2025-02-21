import time
import os
import logging

from preprocessing.preprocessor import Preprocessor
from indexor.index import Index
from indexor.query import FreeTextQuery, BooleanQuery
from retrieval_models.retrieval_functions import query_expansion
from retrieval_models.query_expansion import EmbeddingModel
from back_end.mock_search import MockSearch

logger = logging.getLogger(__name__)


def load_backend(index_path):
    if not os.path.exists(index_path):
        logger.error(f"Index path {index_path} does not exist. Using mock data")
        return Search.mock()

    index = Index(load_path=index_path)
    preprocessor = Preprocessor(parser_kwargs={"parser_type": "raw"})
    embedding_model = EmbeddingModel(
        vocab=index.get_vocab(), save_path="retrieval_models/data/embedding.pkl"
    )
    # reranker = Reranker()

    return Search(index, preprocessor, embedding_model)


def to_py(item):
    if hasattr(item, "item"):
        return item.item()

    return item


class Search:
    def __init__(self, index: Index, preprocessor: Preprocessor, embedding_model):
        self.index = index
        self.embedding_model = embedding_model

        self.preprocessor = preprocessor
        self.boosted_terms = {
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

    def search(self, query, expansion=False, boost_terms=True, k=10):
        start = time.time()
        tokens = self.preprocessor.preprocess(query, return_words=True)

        if expansion and self.embedding_model:
            tokens = query_expansion(tokens, self.embedding_model, top_k=k)
        if boost_terms:
            tokens.extend([token for token in tokens if token in self.boosted_terms])

        query = FreeTextQuery(tokens)
        results = self.index.search(query)

        results = results[:k]  # Reranking here

        ret = self.format_results(results)
        end = time.time()
        print(f"Search took {end-start} seconds to return {len(results)} results.")

        return ret

    def advanced_search(self, query, expansion=False, boost_terms=True, k=10):
        start = time.time()
        tokens = self.preprocessor.preprocess(query, return_words=True)

        if expansion and self.embedding_model:
            tokens = query_expansion(tokens, self.embedding_model, top_k=k)
        if boost_terms:
            tokens.extend([token for token in tokens if token in self.boosted_terms])

        # Encompasses BooleanQuery, PhraseQuery and ProximityQuery
        query = BooleanQuery(tokens)
        results = self.index.search(query)

        results = results[:k]
        ret = self.format_results(results)

        end = time.time()
        print(f"Search took {end-start} seconds to return {len(results)} results.")

        return ret

    def format_results(self, results):
        ret = []
        for doc_result in results:
            metadata = self.index.get_document_metadata(doc_result[1])
            ret.append(
                {
                    "doc_id": to_py(doc_result[1]),
                    "score": to_py(doc_result[0]),
                    "tags": metadata["tags"],
                    "ownerdisplayname": metadata["ownerdisplayname"],
                    "creation_date": to_py(metadata["creationdate"]),
                    "view_count": to_py(metadata["viewcount"]),
                    "answer_count": to_py(metadata["answercount"]),
                    "comment_count": to_py(metadata["commentcount"]),
                    "favorite_count": to_py(metadata["favoritecount"]),
                    "title": "TO BE ADDED",
                    "body": "TO BE ADDED",
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
        print(results[:10])
