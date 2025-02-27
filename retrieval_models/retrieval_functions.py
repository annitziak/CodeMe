import math
from collections import defaultdict
import re
from textblob import TextBlob
import random
from indexor.index import Index


def preprocess_query(query: str) -> list:
    """
    Preprocess the input query (placeholder depending on what we preprocessed on index) - returns a list
    """
    # here we actually need to enter the preprocessing done on index
    return query.lower().split()


def query_expansion(preprocessed_query: list, embedding_model, top_k=1) -> list:
    """
    Expand the query with synonyms or related terms based on precomputed embeddings.
    """
    if not embedding_model:
        return preprocessed_query  # No expansion if model is missing

    max_extra_words = 10

    expanded_terms = set(preprocessed_query)  # Ensure uniqueness
    extra_words_added = 0

    for word in preprocessed_query:
        if extra_words_added >= max_extra_words:
            break  # Stop if we have added enough words to not loose computational efficiency
        new_words = embedding_model.find_similar_words(
            word, top_k=random.choice([1, 2, 3])
        )
        expanded_terms.update(new_words)
        extra_words_added += len(new_words)

    return list(expanded_terms)


def compute_bm25(token, index, k1=1.5, b=0.75):
    """
    Compute BM25 score for each token over all documents in which it appears.
    """
    inverted_index = index if isinstance(index, dict) else index.get_index()
    doc_lengths = index.doc_lengths if hasattr(index, "doc_lengths") else {}

    if token not in inverted_index:
        # maybe here we can use a more advanced spell checker
        blob = TextBlob(token)
        corrected = str(blob.correct())

        # Use corrected token if its in index otherwise none
        token = corrected if corrected in inverted_index else None
        if not token:
            return {}

    total_docs = len(doc_lengths)
    avg_doc_length = sum(doc_lengths.values()) / total_docs if total_docs > 0 else 0
    df = inverted_index[token]["df"]  # Document frequency

    idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)  # IDF calculation
    scores = {}

    for docno, positions in inverted_index[token]["doc_info"]:
        freq = len(positions)
        doc_length = doc_lengths.get(docno, 1)

        # BM25 scoring formula
        term_freq_component = (freq * (k1 + 1)) / (
            freq + k1 * (1 - b + b * (doc_length / avg_doc_length))
        )
        scores[docno] = idf * term_freq_component

    return scores


def boolean_search(query: str, index):
    """
    Handles Boolean, phrase, and proximity search queries.
    """

    def apply_operator(op, left, right=None):
        if op == "AND":
            return left & right if left and right else set()
        if op == "OR":
            return left | right
        if op == "NOT":
            all_docs = {
                doc[0] for term_data in index.values() for doc in term_data["doc_info"]
            }
            return all_docs - right if left is None else left - right
        return set()

    precedence = {"NOT": 3, "AND": 2, "OR": 1}
    operators, operands = [], []

    phrase_matches = re.findall(r'"([^"]+)"', query)
    proximity_matches = re.findall(r"#(\d+)\(([^)]+)\)", query)

    # if phrase search
    for phrase in phrase_matches:
        terms = preprocess_query(phrase)

        # FIX: Fetch document IDs correctly based on `compute_bm25` structure
        docs = [
            set(index[term]["doc_info"]) if term in index else set() for term in terms
        ]

        phrase_result = set.intersection(*docs) if docs else set()
        operands.append(phrase_result)
        query = query.replace(f'"{phrase}"', "", 1)

    # Proximity search processing
    for proximity in proximity_matches:
        distance, terms = int(proximity[0]), preprocess_query(proximity[1])

        # FIX: Ensure correct document retrieval based on indexing format
        docs = [
            set(index[term]["doc_info"]) if term in index else set() for term in terms
        ]

        proximity_result = set.intersection(*docs) if docs else set()
        operands.append(proximity_result)
        query = query.replace(f"#{proximity[0]}({proximity[1]})", "", 1)

    # Boolean search logic
    tokens = preprocess_query(query)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in precedence:
            while operators and precedence[operators[-1]] >= precedence[token]:
                op = operators.pop()
                right = operands.pop()
                left = operands.pop() if operands else None
                operands.append(apply_operator(op, left, right))
            operators.append(token)
        else:
            # FIX: Retrieve doc IDs directly from `index`
            operands.append(set(index[token]["doc_info"]) if token in index else set())
        i += 1

    while operators:
        op = operators.pop()
        right = operands.pop()
        left = operands.pop() if operands else None
        operands.append(apply_operator(op, left, right))

    return sorted(operands.pop()) if operands else []


def reorder_as_date(result):
    # Extract metadata and store with original results
    results_with_date = []

    for doc_result in result:
        metadata = Index.get_document_metadata(
            doc_result[1]
        )  # Assuming metadata returns a dict
        creation_date = metadata.get(
            "creationdate", 0
        )  # Default to 0 if key doesn't exist
        results_with_date.append((creation_date, doc_result))

    # Sort by creation date
    results_with_date.sort(key=lambda x: x[0])

    # Extract the sorted results
    sorted_results = [doc[1] for doc in results_with_date]

    return sorted_results


# Mapping of cluster IDs to cluster names
CLUSTER_MAPPINGS = {
    1: "Programming & Development Fundamentals",
    2: "Software Engineering & System Design",
    3: "Advanced Computing & Algorithms",
    4: "Technologies & Frameworks",
    5: "Other",
}


def parse_clusters_from_file(file_path):
    cluster_mapping = {
        cluster_name: set() for cluster_name in CLUSTER_MAPPINGS.values()
    }
    current_cluster = None

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Check if the line starts with a cluster header
            cluster_match = re.match(r"Cluster (\d+) \(\d+ tags\): (.+)", line)
            if cluster_match:
                cluster_id = int(cluster_match.group(1))
                cluster_name = CLUSTER_MAPPINGS.get(cluster_id)

                if cluster_name:
                    current_cluster = cluster_name
                else:
                    current_cluster = None  # Skip if the cluster ID is invalid
            elif current_cluster:
                # Add tags to the current cluster (split by commas)
                tags = [tag.strip() for tag in line.split(",") if tag.strip()]
                cluster_mapping[current_cluster].update(tags)

    return cluster_mapping


file_path = "./retrieval_models/data/clustering_results.txt"
CLUSTER_MAPPING = parse_clusters_from_file(file_path)


def reorder_as_tag(result, selected_clusters):
    # Create a set of allowed tags based on selected clusters
    allowed_tags = set()
    for cluster_name in selected_clusters:
        allowed_tags.update(CLUSTER_MAPPING.get(cluster_name, set()))

    results_with_tags = []

    for doc_result in result:
        metadata = Index.get_document_metadata(doc_result[1])
        tags = metadata.get("tags", "").split(",")  # Assuming tags are comma-separated

        # Check if any tag in the document belongs to the allowed set
        if any(tag in allowed_tags for tag in tags):
            results_with_tags.append(doc_result)

    return results_with_tags


def retrieval_function(query, index, embedding_model=None, expansion=False, k=50):
    """
    Determines if a Boolean search is needed; otherwise, falls back to BM25.
    """
    if any(op in query for op in ["AND", "OR", "NOT", '"', "#"]):
        # this also assumes that they will be given in the correct format - handle edge cases with regex
        print("Starting boolean search")
        bool_results = boolean_search(query, index)
        if bool_results:
            return bool_results
        return []

    # if this is not a boolean_search then do ranked search using bm25
    print("Starting ranked search")
    tokens = preprocess_query(query)
    # do query expansion

    if expansion and embedding_model and len(tokens) <= 10:
        # make sure they are unique
        tokens = list(set(tokens + query_expansion(tokens, embedding_model)))

    boosted_terms = {
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

    # Boost important terms
    tokens.extend([token for token in tokens if token in boosted_terms])

    aggregated_scores = defaultdict(float)

    # in case they place a very large query - limit to 20 tokens
    if len(tokens) > 20:
        # random ones or most important ones
        tokens = tokens[:20]

    for token in tokens:
        for docno, score in compute_bm25(token, index).items():
            aggregated_scores[docno] += score
    # sorted or if no results then return a list of random documents - this can be the most recent documents lets see
    if aggregated_scores:
        sorted_docs = sorted(aggregated_scores, key=aggregated_scores.get, reverse=True)
        if len(sorted_docs) < k:
            # needs testing
            return sorted_docs + random.choices(
                index.get_all_documents(), k=k - len(sorted_docs)
            )
        return sorted_docs[:k]

    # return just the most popular ones or random ones - you choose.
    return [
        "818020",
        "816834",
        "1731441",
        "1477365",
    ]  # list of most popular documents - populate with 50 documents
