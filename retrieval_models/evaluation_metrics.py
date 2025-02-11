from collections import Counter
import math
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize


# Clarity Score which measures how focused the retrieved documents are compared to the overall collection (corpus) 
# A high Clarity Score means your search engine is pulling together results that are highly focused and distinct from the overall corpus.
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

# novel method: thinking -> cluster the retrieved documents into clusters to see if they are close together?
# potentially find outliers or documents that are not relevant to the query 
def compute_clustering_score_bm25_dbscan(retrieved_docs, eps=0.5, min_samples=2):
    """
    Computes clustering scores for retrieved documents using DBSCAN. 
    We want high silhouette score and low davies bouldin score?
    Can experiment with different methods.

    Args:
        retrieved_docs: List of retrieved document texts.
        eps (float): Maximum distance between two samples for them to be considered part of cluster. This may needs some finetuning
        min_samples (int): Minimum number of samples required to form a dense region. Also needs finetuning

    Returns:
        dict: Clustering scores and number of clusters.
    """

    if len(retrieved_docs) < 2:
        raise ValueError("At least two retrieved documents are required for clustering.")

    # Load CodeBERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    # Generate embeddings
    def get_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token representation

    doc_vectors = np.array([get_embedding(doc) for doc in retrieved_docs])

    # Normalize embeddings for better clustering
    doc_vectors = normalize(doc_vectors)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = dbscan.fit_predict(doc_vectors)

    # Count the number of clusters (excluding noise points labeled as -1)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Compute clustering evaluation metrics (only if there are actual clusters)
    if num_clusters > 1:
        silhouette = silhouette_score(doc_vectors, labels, metric='cosine')
        davies_bouldin = davies_bouldin_score(doc_vectors, labels)
    else:
        silhouette = None  # Not meaningful with one cluster
        davies_bouldin = None

    return {
        "num_clusters": num_clusters,
        "silhouette_score": silhouette,  # Higher means better clustering (range: -1 to 1)
        "davies_bouldin_score": davies_bouldin,  # Lower means better clustering
        "labels": labels.tolist()  # Cluster assignments for each document
    }

