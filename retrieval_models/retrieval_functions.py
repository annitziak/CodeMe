import math
from collections import defaultdict
import tqdm
from retrieval_models.query_expansion import EmbeddingModel

def preprocess_query(query: str) -> list:
    """
    Preprocess the input query (placeholder depending on what we preprocessed on index)
    """
    return query.lower().split()

def query_expansion(preprocessed_query: list, embedding_model: EmbeddingModel, top_k=1) -> list:
    """
    Expand the query with synonyms or related terms based on precomputed embeddings.
    """
    extra_terms = []
    for word in preprocessed_query:
        extra_terms.append(embedding_model.find_similar_words(word, top_k))
    return list(set(extra_terms))  # Ensure unique terms


def compute_bm25(token, index, k1=1.5, b=0.75):
    """
    Compute BM25 score for each token over all documents in which it appears.
    """
    inverted_index = index.get_index()  # Retrieve the inverted index
    doc_lengths = index.doc_lengths  # Retrieve document lengths

    if token not in inverted_index:
        return {}  # Token not in the index

    total_docs = len(doc_lengths)  # Total number of documents
    avg_doc_length = sum(doc_lengths.values()) / total_docs if total_docs > 0 else 0
    df = inverted_index[token]['df']  # Document frequency

    idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)  # IDF calculation
    scores = {}

    for docno, positions in inverted_index[token]['doc_info']:
        freq = len(positions)  # Frequency of term in the document
        doc_length = doc_lengths.get(docno, 1)  # Get doc length, defaulting to 1

        # BM25 scoring formula
        term_freq_component = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (doc_length / avg_doc_length)))
        scores[docno] = idf * term_freq_component

    return scores


def retrieval_function(query, index, embedding_model=None, expansion=False, k=10):
    """
    Retrieve documents based on a query using BM25, with optional query expansion.

    Args:
        query (str): The input search query.
        index (Index): The precomputed index.
        embedding_model (object, optional): The model used for query expansion.
        expansion (bool, optional): Whether to apply query expansion.

    Returns:
        list: Sorted list of top document IDs based on retrieval scores, or a message if no results exist.
    """
    tokens = preprocess_query(query)

    # Apply query expansion if requested
    if expansion and embedding_model:
        tokens += query_expansion(tokens, embedding_model)

    # Score aggregation for BM25 retrieval
    aggregated_scores = defaultdict(float)
    found_any_results = False  # Track if at least one term retrieves documents

    for token in tokens:
        token_scores = compute_bm25(token, index)
        if token_scores:  # If this token has results, we mark it
            found_any_results = True
        for docno, score in token_scores.items():
            aggregated_scores[docno] += score

    # If no results were found for any token, return a message - we might have to just return some basic well-read documents here
    if not found_any_results:
        return "No relevant documents found for this query."

    # Sort documents by score and return top 10 results
    ranked_results = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    # Return only document IDs, ignoring scores
    return [doc[0] for doc in ranked_results]
