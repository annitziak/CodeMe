from flask import Flask, request, jsonify
from retrieval_models.retrieval_functions import *

app = Flask(__name__)


@app.route("/search", methods=["GET"])
def search():
    """
    Structure of the request:
        query: str
        page: int
        page_size: int
        # BELOW IS TENTATIVE TO SUPPORT HERE
        filters: dict[ 
            from_date: str
            to_date: str
            tags: list[str]
        ]
        options: dict[
            expansion: bool
            boost_terms: bool
        ]
    Structure of the response:
        results: list[dict[
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
            snippet: str ??? unsupported
            title: str ??? unsupported
        ]]

    Error Codes:
        200: OK
        400: Bad Request
        500: Internal Server Error
    """
    query = request.form.get("query")  # Extract query
    filters = request.form.getlist("filters")  # Extract multiple filter values

    result = reorder_as_per_filter(query, filters)  # Apply filters

    return jsonify({"result": result}), 200

@app.route("/advanced_search", methods=["GET"])
def search():
    """
    Structure of the request:
        query: str
        page: int
        page_size: int
        # BELOW IS TENTATIVE TO SUPPORT HERE
        filters: dict[ 
            from_date: str
            to_date: str
            tags: list[str]
        ]
        options: dict[
            expansion: bool
            boost_terms: bool
        ]
    Structure of the response:
        results: list[dict[
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
            snippet: str ??? unsupported
            title: str ??? unsupported
        ]]

    Error Codes:
        200: OK
        400: Bad Request
        500: Internal Server Error
    """
    query = request.form.get("query")  # Extract query
    filters = request.form.getlist("filters")  # Extract multiple filter values

    result = reorder_as_per_filter(query, filters)  # Apply filters

    return jsonify({"result": result}), 200

def retreival_function(query):
    return ["doc1", "doc2", "doc3"]


def reorder_as_per_filter(query, filters):
    doc_list = retreival_function(query)  # Retrieve documents

    if "date" in filters:
        doc_list = reorder_as_date(doc_list)  # Reorder by date

    if "tag" in filters:
        doc_list = reorder_as_tag(doc_list)  # Reorder by tag

    return doc_list  # Return reordered results


@app.route("/Queryresponse/<result>")
def QueryResult(result):
    return "%s" % result


# main driver function
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
