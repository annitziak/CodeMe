from data_loaders import Index
from retrieval_functions import retrieval_function
from collections import Counter
import math
import numpy as np

# Clarity Score which measures how focused the retrieved documents are compared to the overall collection (corpus) - A high Clarity Score means your search engine is pulling together results that are highly focused and distinct from the overall corpus.
def compute_clarity_score(retrieved_docs, corpus_vocab):
    """
    Compute Clarity Score for a set of retrieved documents.
    Args:
        retrieved_docs (list of list): Tokenized retrieved documents.
        corpus_vocab (Counter): Corpus-level term distribution.
    Returns:
        float: Clarity Score.
    """
    # Combine all terms from the retrieved documents
    retrieved_vocab = Counter()
    for doc in retrieved_docs:
        retrieved_vocab.update(doc)
    
    # Calculate the probability distribution of terms in the retrieved set
    total_retrieved_terms = sum(retrieved_vocab.values())
    retrieved_probs = {term: count / total_retrieved_terms for term, count in retrieved_vocab.items()}
    
    # Calculate the Clarity Score using KL divergence
    clarity_score = 0.0
    for term, p_retrieved in retrieved_probs.items():
        p_corpus = corpus_vocab.get(term, 1e-12)  # Avoid division by zero
        clarity_score += p_retrieved * math.log(p_retrieved / p_corpus)
    return clarity_score

# Collection Overlap- measures how much the terms in the query are present in the retrieved documents.
def compute_collection_overlap(query_tokens, retrieved_docs):
    """
    Compute Collection Overlap between query tokens and retrieved document terms.
    Args:
        query_tokens (list): List of tokens in the query.
        retrieved_docs (list of list): List of tokenized retrieved documents.
    Returns:
        float: Overlap score (proportion of query tokens in retrieved documents).
    """
    # Extract all unique terms from the retrieved documents
    retrieved_vocab = set()
    for doc in retrieved_docs:
        retrieved_vocab.update(doc)
    
    # Calculate the proportion of query terms present in the retrieved documents
    overlap = len(set(query_tokens) & retrieved_vocab) / len(query_tokens)
    return overlap

# nDCG (this is supervised!!!!- If you want to include ranking evaluation like nDCG, you’ll need to simulate relevance scores or generate them using LLMs. (is it worth it ?)
def compute_ndcg(retrieved_docs, engagement_data, k=10):
    """
    Compute nDCG for a ranked list of documents.
    Args:
        retrieved_docs (list): Ranked list of document IDs.
        engagement_data (dict): Dictionary with document IDs as keys and engagement scores as values.
        k (int): Rank cutoff for nDCG.
    Returns:
        float: nDCG score.
    """
    def dcg(scores):
        # Compute Discounted Cumulative Gain
        return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(scores))

    # Get relevance scores for the retrieved documents
    relevance_scores = [engagement_data.get(doc, 0) for doc in retrieved_docs[:k]]
    
    # Sort the relevance scores to compute the ideal ranking
    ideal_scores = sorted(relevance_scores, reverse=True)
    
    # Compute nDCG by normalizing DCG with the ideal DCG
    return dcg(relevance_scores) / (dcg(ideal_scores) + 1e-12)

# Other metrics suggested, but not quite sure if useful in our case? Will look more into it


# Mean Reciprocal Rank (needs labelled data)- is a ranking quality metric. Evaluates how quickly a ranking system can show the first relevant item in the top-K results
# Coverage- the proportion of relevant documents within a collection that are retrieved by a search query

# Evaluate Retrieval Models
def evaluate(query, index, embedding_model=None, expansion=False, engagement_data=None):
    """
    Evaluate the retrieval models on a single query.
    Args:
        query (str): User query.
        index (Index): Inverted index.
        embedding_model (EmbeddingModel): Optional embedding model for query expansion.
        expansion (bool): Whether to use query expansion.
        engagement_data (dict): Optional engagement data for weak supervision.
    """
    # Retrieve documents using the specified retrieval function
    results = retrieval_function(query, index, embedding_model, expansion)
    
    # Build the corpus vocabulary by aggregating term frequencies across all documents
    corpus_vocab = Counter()
    for term, data in index.inverted_index_positions.items():
        for _, positions in data['doc_info']:
            corpus_vocab[term] += len(positions)
    
    # Extract retrieved documents (top-ranked ones)
    retrieved_docs = [doc for doc, _ in results["without_expansion"]]
    top_docs = retrieved_docs[:10]  # Consider only the top 10 documents for evaluation
    
    # Tokenize the content of the top documents for metrics calculation
    tokenized_docs = [index.inverted_index_positions[doc]['doc_info'] for doc in top_docs]
    
    # Calculate metrics
    query_tokens = query.lower().split()  # Tokenize the query
    clarity = compute_clarity_score(tokenized_docs, corpus_vocab)  # Clarity Score
    overlap = compute_collection_overlap(query_tokens, tokenized_docs)  # Collection Overlap
    ndcg = compute_ndcg(retrieved_docs, engagement_data) if engagement_data else None  # nDCG (optional)

    # Print results
    print(f"Query: {query}")
    print(f"Clarity Score: {clarity:.4f}")
    print(f"Collection Overlap: {overlap:.4f}")
    if ndcg is not None:
        print(f"nDCG: {ndcg:.4f}")

# Example Usage
if __name__ == "__main__":
    # Initialize the inverted index
    index = Index()
    
    # Define a sample query and engagement data
    sample_query = "How to write a Python web scraper"
    engagement_data = {
        "doc1": 10,  # Example engagement scores for documents
        "doc2": 5,
        "doc3": 2,
    }
    
    # Evaluate the query using the retrieval models
    evaluate(sample_query, index, expansion=True, engagement_data=engagement_data)
