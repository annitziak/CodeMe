import time

from preprocessing.preprocessor import Preprocessor
from indexor.index import Index
from indexor.query import FreeTextQuery
from retrieval_models.retrieval_functions import query_expansion
from retrieval_models.query_expansion import EmbeddingModel


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

    def search(self, query, index, expansion=False, boost_terms=True, k=10):
        start = time.time()
        tokens = self.preprocessor.preprocess(query, return_words=True)

        if expansion and self.embedding_model:
            tokens = query_expansion(tokens, self.embedding_model, top_k=k)
        if boost_terms:
            tokens.extend([token for token in tokens if token in self.boosted_terms])

        query = FreeTextQuery(tokens)
        results = index.search(query)

        end = time.time()
        print(f"Search took {end-start} seconds to return {len(results)} results.")

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search Engine")
    parser.add_argument("--index-path", type=str, help="Path to load index")
    args = parser.parse_args()

    index = Index(load_path=args.index_path)
    preprocessor = Preprocessor(parser_kwargs={"parser_type": "raw"})
    embedding_model = EmbeddingModel(
        vocab=index.get_vocab(), save_path="data/embeddings.pkl"
    )
    # reranker = Reranker()

    search = Search(index, preprocessor, embedding_model)
    search.search(
        "hello world in python", index, expansion=True, boost_terms=True, k=10
    )
    search.search(
        "how good is python as a programming langauge. Does it do well for FindMax queries",
        index,
        expansion=False,
        k=10,
    )

    while True:
        query = input("Enter query: ")
        results = search.search(query, index, expansion=False, boost_terms=True, k=10)
        print(results[:10])
