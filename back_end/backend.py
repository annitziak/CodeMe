from flask import Flask, request, jsonify
from retrieval_models.retrieval_functions import *

app = Flask(__name__)


@app.route("/Queryrequest", methods=["POST"])
def Queryrequest():
    query = request.form.get("query")  # Extract query
    filters = request.form.getlist("filters")  # Extract multiple filter values

    result = reorder_as_per_filter(query, filters)  # Apply filters

    # Instead of redirecting, directly call Queryresponse function
    return Queryresponse(result)


@app.route("/Queryresponse", methods=["GET"])
def Queryresponse():
    result = request.args.getlist("result")  # Extract result from query params
    return jsonify({"result": result})  # Return response in JSON format


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
