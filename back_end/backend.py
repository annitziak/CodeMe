import logging

from flask import Flask, request, jsonify
from back_end.search import load_backend

app = Flask(__name__)
logger = logging.getLogger(__name__)

IDX_TO_ITEM = {
    0: "doc_id",
    1: "score",
    2: "view_count",
    3: "owneruserid",
    4: "answer_count",
    5: "comment_count",
    6: "favorite_count",
    7: "ownerdisplayname",
    8: "tags",
    9: "creation_date",
    10: "snippet",
    11: "title",
    "doc_id": 0,
    "score": 1,
    "view_count": 2,
    "owneruserid": 3,
    "answer_count": 4,
    "comment_count": 5,
    "favorite_count": 6,
    "ownerdisplayname": 7,
    "tags": 8,
    "creation_date": 9,
    "snippet": 10,
    "title": 11,
}


@app.route("/", methods=["GET"])
def hello_world():
    return "<h1>backend is running!!!</h1>"


def extract_search_args(request):
    if request.method == "POST":
        data = request.get_json()
        logger.info(f"Data: {data}")
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        query = data.get("query", "")
        page = int(data.get("page", 0))
        page_size = int(data.get("page_size", 20))
        filters = data.get("filters", {})
        options = data.get("options", {})

        selected_clusters = filters.get("tags", None)
        reorder_date = bool(filters.get("date", False))

        rerank_metadata = bool(options.get("rerank_metadata", True))
        rerank_lm = bool(options.get("rerank_lm", True))

        use_semantic = bool(options.get("use_semantic", False))
        boost_terms = bool(options.get("boost_terms", True))
        expansion = bool(options.get("expansion", True))
        k_word_expansion = int(options.get("k_word_expansion", 3))
    else:
        query = request.args.get("query")  # Extract query
        filters = request.args.getlist("filters")  # Extract multiple filter values
        page = int(request.args.get("page", 0))  # Extract page number
        page_size = int(request.args.get("page_size", 20))  # Extract page size

        rerank_metadata = bool(request.args.get("rerank_metadata", True))
        rerank_lm = bool(request.args.get("rerank_lm", True))

        selected_clusters = request.args.get("tags", None)
        reorder_date = request.args.get("date", False)
        use_semantic = bool(request.args.get("use_semantic", False))
        boost_terms = bool(request.args.get("boost_terms", True))
        expansion = bool(request.args.get("expansion", True))
        k_word_expansion = int(request.args.get("k_word_expansion", 3))

    if (
        selected_clusters is not None
        and isinstance(selected_clusters, list)
        and len(selected_clusters) == 0
    ):
        selected_clusters = None

    return {
        "query": query,
        "page": page,
        "page_size": page_size,
        "rerank_metadata": rerank_metadata,
        "rerank_lm": rerank_lm,
        "selected_clusters": selected_clusters,
        "reorder_date": reorder_date,
        "use_semantic": use_semantic,
        "boost_terms": boost_terms,
        "expansion": expansion,
        "k_word_expansion": k_word_expansion,
    }


@app.route("/search", methods=["GET", "POST"])
def search():
    """
    Structure of the request:
        query: str
        page: int
        page_size: int
        # BELOW IS TENTATIVE TO SUPPORT HERE
        filters: dict[
            "date": "reorder": bool
            "tag": {
                "tags": list[str],
            }
        ]
        options: dict[
            expansion: bool
            boost_terms: bool
            rerank_metadata: bool [default: False]
            rerank_lm: bool [default: False]
            use_semantic: bool [default: False]
        ]
    Structure of the response:
        results: list[
            dict[
                doc_id: int
                score: float
                view_count: int
                owneruserid: int
                answer_count: int
                comment_count: int
                favorite_count: int
                ownerdisplayname: str
                tags: str
                creation_date: str
                body: str ??? unsupported (ONLY A SNIPPET)
                title: str ??? unsupported
            ]
        ]
        page: int
        page_size: int
        total_results: int # Total number from the search (not on the page)
        has_next: bool
        has_prev: bool
        time_taken: float

    Error Codes:
        200: OK
        400: Bad Request
        500: Internal Server Error
    """
    args = extract_search_args(request)
    result = search_module.search(
        args["query"],
        page=args["page"],
        page_size=args["page_size"],
        rerank_lm=args["rerank_lm"],
        rerank_metadata=args["rerank_metadata"],
        selected_clusters=args["selected_clusters"],
        reorder_date=args["reorder_date"],
        use_semantic=args["use_semantic"],
        boost_terms=args["boost_terms"],
        expansion=args["expansion"],
        k_word_expansion=args["k_word_expansion"],
    )

    return jsonify(
        {
            "result": result.results,
            "page": args["page"],
            "page_size": args["page_size"],
            "has_next": result.has_next,
            "has_prev": result.has_prev,
            "total_results": result.total_results,
            "time_taken": result.time_taken,
        }
    ), 200


@app.route("/advanced_search", methods=["GET", "POST"])
def advanced_search():
    """
    Structure of the request:
        query: str
        page: int
        page_size: int
        # BELOW IS TENTATIVE TO SUPPORT HERE
        filters: dict[
            "date": bool
            "tags": list[str],
        ]
        options: dict[
            expansion: bool
            boost_terms: bool
            rerank_metadata: bool [default: False]
            rerank_lm: bool [default: False]
        ]
    Structure of the response:
        results: list[
            dict[
                doc_id: int
                score: float
                view_count: int
                owneruserid: int
                answer_count: int
                comment_count: int
                favorite_count: int
                ownerdisplayname: str
                tags: str
                creation_date: str
                body: str ??? unsupported (ONLY A SNIPPET)
                title: str ??? unsupported
            ]
        ]
        page: int
        page_size: int
        total_results: int # Total number from the search (not on the page)
        has_next: bool
        has_prev: bool

    Error Codes:
        200: OK
        400: Bad Request
        500: Internal Server Error
    """
    args = extract_search_args(request)
    result = search_module.advanced_search(
        args["query"],
        page=args["page"],
        page_size=args["page_size"],
        rerank_metadata=args["rerank_metadata"],
        selected_clusters=args["selected_clusters"],
        reorder_date=args["reorder_date"],
        boost_terms=args["boost_terms"],
        expansion=args["expansion"],
        k_word_expansion=args["k_word_expansion"],
    )

    return jsonify(
        {
            "result": result.results,
            "page": args["page"],
            "page_size": args["page_size"],
            "has_next": result.has_next,
            "has_prev": result.has_prev,
            "total_results": result.total_results,
            "time_taken": result.time_taken,
        }
    ), 200


def format_result(result):
    new_result = []
    for doc_result in result:
        new_result.append({})
        for idx, value in enumerate(doc_result):
            if hasattr(value, "item"):
                value = value.item()
            new_result[-1][IDX_TO_ITEM[idx]] = value

    return new_result


def retreival_function(query):
    return ["doc1", "doc2", "doc3"]


# main driver function
if __name__ == "__main__":
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description="Search Engine")
    parser.add_argument(
        "--index-path",
        type=str,
        help="Path to load index",
        default=".cache/index-doc-title-body-v2",
    )
    parser.add_argument(
        "--embedding-path",
        type=str,
        help="Path to load embeddings",
        default=".cache/embedding2.pkl",
    )
    parser.add_argument(
        "--reranker-path",
        type=str,
        help="Path to load reranker embeddings",
        default=".cache/embeddings_v2",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server")
    args = parser.parse_args()

    # ENABLE ON WINDOWS IF USING MULTIPROCESSING
    multiprocessing.freeze_support()
    search_module = load_backend(
        args.index_path, args.embedding_path, args.reranker_path
    )

    app.run(host="0.0.0.0", port=args.port)
