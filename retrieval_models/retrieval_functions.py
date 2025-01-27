import math
from collections import defaultdict

from query_expansion import EmbeddingModel
from data_loaders import Index

def preprocess_query(query: str) -> list:
    """
    Preprocess the input query (placeholder depending on what we preprocessed on index)

    """
    # Convert to lowercase and tokenize
    return query.lower().split()

def query_expansion(preprocessed_query: list, embedding_model:EmbeddingModel) -> list:
    """
    Expand the query with synonyms or related terms based on precomputed embeddings.
    This is a placeholder for more advanced query expansion techniques.
    
    """
    extra_terms = []
    for word in preprocessed_query:
        extra_terms.append(embedding_model.find_similar_words(word, top_k=1))

    # TO DO: maybe some advanced processing - not the same number of terms per query word?
    # need to experiment with optimal amount of terms
    #return set in case the same words were returned
    return list(set(extra_terms))

def compute_bm25(token, index, k1=1.5, b=0.75):
    """
    Compute BM25 score for each token over all documents in which it appears.

    Args:
        token (str): token
        index (dict): inverted index
        k1 (float): term saturation.
        b (float): length normalization.

    Returns:
        dict: A dictionary with document numbers as keys and BM25 scores as values.
    """
    #compute avg doc length needed for calculation
    avg_doc_length = sum(index.doc_lengths.values()) / len(index.doc_lengths) if len(index.doc_lengths) > 0 else 0
    if token not in index:
        return {}  # Token not in the index

    df = index[token]['df']  # doc frequency
    idf = math.log((index.total_docs - df + 0.5) / (df + 0.5) + 1)  # idf calculation

    scores = {}
    for docno, positions in index[token]['doc_info'].items():
        freq = len(positions)  # frequency is len of positions list
        doc_length = index.total_docs.get(docno, 0) #get the length of doc
        term_freq_component = freq * (k1 + 1) / (freq + k1 * (1 - b + b * (doc_length / avg_doc_length))) #the second component of formula
        scores[docno] = idf * term_freq_component

    return scores

def retrieval_function(query, index, embedding_model, expansion=False):
    """
    Retrieve documents based on a query using the BM25 scoring function, with optional query expansion.

    Args:
        query (str): The user query as a string.
        index (Index): The inverted index structure.
        embedding_model (EmbeddingModel): The embedding model instance.
        expansion (bool): Whether to use query expansion.

    Returns:
        dict: A dictionary containing results with and without expansion.
    """
    tokens = preprocess_query(query)
    results = {"with_expansion": [], "without_expansion": []}

    # Without expansion
    aggregated_scores_no_expansion = defaultdict(float)
    for token in tokens:
        token_scores = compute_bm25(token, index)
        for docno, score in token_scores.items():
            aggregated_scores_no_expansion[docno] += score
    results["without_expansion"] = sorted(aggregated_scores_no_expansion.items(), key=lambda x: x[1], reverse=True)[:100]

    # With expansion
    if expansion:
        expanded_tokens = tokens + query_expansion(tokens, embedding_model)
        aggregated_scores_with_expansion = defaultdict(float)
        for token in expanded_tokens:
            token_scores = compute_bm25(token, index)
            for docno, score in token_scores.items():
                aggregated_scores_with_expansion[docno] += score
        results["with_expansion"] = sorted(aggregated_scores_with_expansion.items(), key=lambda x: x[1], reverse=True)[:100]

    return results

# Example usage
if __name__ == "__main__":
    inverted_index_positions = Index()  
    vocab = inverted_index_positions.vocab
    embedding_model = EmbeddingModel(vocab)  # need to precompute embeddings - we can store this somewhere else

    # Example query
    query = "code for a simple web scraper"
    results = retrieval_function(query, inverted_index_positions, embedding_model, expansion=True)

    # Write results to a file
    with open("results.txt", "w") as f:
        f.write("Without Expansion:\n")
        for doc, score in results["without_expansion"]:
            f.write(f"Document: {doc}, Score: {score}\n")

        f.write("\nWith Expansion:\n")
        for doc, score in results["with_expansion"]:
            f.write(f"Document: {doc}, Score: {score}\n")