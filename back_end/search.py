import time
import os
import logging
import multiprocessing


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

    multiprocessing.set_start_method("spawn", force=True)

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
        self.post_cache = SearchCache()

    def _pre_search(self, query, expansion=False, boost_terms=True, k=3):
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
        end_idx = (page + 1) * page_size
        apply_reranking = (rerank_metadata or rerank_lm) and start_idx < 200

        return_result.total_results = total_results
        return_result.query = query

        if end_idx < total_results:
            if len(return_result.results) < page_size:
                return_result.has_next = False
            else:
                return_result.has_next = True
        if page > 0:
            return_result.has_prev = True

        if apply_reranking:
            if query in self.post_cache:
                logger.info(f"Result from cache {query}: {return_result}")
                return_result.results = self.post_cache[query]
            else:
                return_result.results = []
                for i in range(0, len(results), 200):
                    rerank_subset = results[i : i + 200]
                    return_result.results += self.format_results(rerank_subset)

                    if len(return_result.results) >= 200:
                        break

                logger.info(
                    f"Reranking top 200 results for query {len(return_result.results)} : {return_result}"
                )

                return_result.results = reorder_as_per_filter(
                    return_result.results,
                    selected_clusters=selected_clusters,
                    reorder_date=reorder_date,
                )

                logger.info(
                    f"Reordered results: {return_result} to {len(return_result.results)}"
                )

                if rerank_metadata:
                    return_result.results = self.reranker.rerank_metadata(
                        return_result.results
                    )
                    logger.info(
                        f"Reranked with metadata: {return_result} to {len(return_result.results)}"
                    )
                if rerank_lm:
                    return_result.results = self.reranker.rerank_lm(
                        return_result.results, query
                    )
                    logger.info(
                        f"Reranked with LM: {return_result} to {len(return_result.results)}"
                    )

                return_result.results = self.reranker.fuse_scores(return_result.results)
                logger.info(
                    f"Fused scores: {return_result} to {len(return_result.results)}"
                )

                self.post_cache[query] = return_result.results

            if end_idx < 200:
                logger.info("Entire page of results is in top 200")
                return_result.results = return_result.results[start_idx:end_idx]
            else:
                logger.info("Page of results is not in top 200")
                if start_idx < 200:
                    first_part = return_result.results[start_idx:200]
                    remainder_start = 200
                    remainder_end = end_idx
                    second_part = self.format_results(
                        results[remainder_start:remainder_end]
                    )
                    return_result.results = first_part + second_part
                else:
                    return_result.results = self.format_results(
                        results[start_idx:end_idx]
                    )
        else:
            logger.info("No reranking applied")
            return_result.results = self.format_results(results[start_idx:end_idx])

        return_result.results = reorder_as_per_filter(
            return_result.results,
            reorder_date=reorder_date,
        )
        logger.info(
            f"Reordered results: {return_result} to {len(return_result.results)}"
        )

        return return_result

    def search(
        self,
        query,
        expansion=True,
        boost_terms=True,
        rerank_metadata=True,
        rerank_lm=True,
        page=0,
        page_size=20,
        k_word_expansion=3,
        selected_clusters=None,
        reorder_date=False,
        use_semantic=False,
    ) -> SearchResult:
        start = time.time()

        tokens = self._pre_search(query, expansion, boost_terms, k=k_word_expansion)

        if use_semantic:
            logger.info("Using semantic search")
            results = self.reranker.semantic_search(query, top_k=(page + 1) * page_size)
            logger.info(f"Semantic search results: {results}")
            return_result = SearchResult(
                results=results,
                query=query,
                time_taken=time.time() - start,
                total_results=len(results),
            )
            return_result.results = self.format_results(results)
            return_result.results = reorder_as_per_filter(
                return_result.results,
                selected_clusters=selected_clusters,
                reorder_date=reorder_date,
            )
            return_result.results = return_result.results[
                page * page_size : (page + 1) * page_size
            ]
            return return_result

        free_text_query = FreeTextQuery(tokens)
        if query in self.cache:
            results = self.cache[query]
        else:
            results = self.index.search(free_text_query)
            self.cache[query] = results

        ret = self._post_search(
            results,
            query=query,
            rerank_metadata=rerank_metadata,
            rerank_lm=rerank_lm,
            page=page,
            page_size=page_size,
            selected_clusters=selected_clusters,
            reorder_date=reorder_date,
        )
        ret = self._backfill_results(ret, page_size - len(results))
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
        k_word_expansion=10,
        page=0,
        page_size=20,
        rerank_metadata=True,
        selected_clusters=None,
        reorder_date=False,
    ):
        start = time.time()
        boolean_query = BooleanQuery(query, preprocessor=self.preprocessor)
        boolean_query.parse()
        logger.info(f"Query after preprocessing: \n{boolean_query._ppformat()}")

        if query in self.cache:
            logger.info(f"Result from cache {query} of length {len(self.cache[query])}")
            results = self.cache[query]
        else:
            results = self.index.search(boolean_query)
            self.cache[query] = results

        ret = self._post_search(
            results,
            query=query,
            page=page,
            page_size=page_size,
            rerank_metadata=rerank_metadata,
            rerank_lm=False,
            selected_clusters=selected_clusters,
            reorder_date=reorder_date,
        )
        logger.info(f"Result from index: {ret}")
        # DO NOT BACKFILL FOR ADVANCED SEARCH
        # ret = self._backfill_results(ret, page_size - len(results))
        ret.time_taken = time.time() - start

        logger.info(
            f"Search took {ret.time_taken} seconds to return {ret.total_results} results."
        )

        return ret

    def format_results(self, results):
        ret = []
        for doc_result in results:
            lm_score = 0
            bm25_score = 0
            id = 0
            if isinstance(doc_result, tuple):
                id = doc_result[1]
                bm25_score = doc_result[0]
            elif isinstance(doc_result, dict):
                id = doc_result["id"]
                lm_score = doc_result.get("lm_score", 0)
            else:
                raise ValueError(f"Unknown type for doc_result: {type(doc_result)}")

            metadata = self.index.get_document_metadata(id)
            if metadata is None:
                continue

            ret.append(
                {
                    "doc_id": to_py(id),
                    "bm25_score": to_py(bm25_score),
                    "lm_score": to_py(lm_score),
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

    def _backfill_results(
        self,
        ret,
        num_results,
        query_type=FreeTextQuery,
        selected_clusters=None,
        reorder_date=False,
    ):
        if num_results <= 0:
            return ret

        # Do semantic search first
        try:
            results = self.reranker.semantic_search(ret.query, top_k=num_results * 50)
            if len(results) < num_results * 50:
                tokens = self._pre_search("python")
                query = query_type(tokens, preprocessor=self.preprocessor)
                results += self.index.search(query)[: num_results * 50]
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            tokens = self._pre_search("python and machine learning")
            query = query_type(tokens, preprocessor=self.preprocessor)
            results = self.index.search(query)[: num_results * 50]

        results = self.format_results(results)
        results = reorder_as_per_filter(
            results,
            selected_clusters=selected_clusters,
            reorder_date=reorder_date,
        )
        results = results[:num_results]

        ret.results += results
        ret.total_results += len(results)

        print("Backfilled results", ret)
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
