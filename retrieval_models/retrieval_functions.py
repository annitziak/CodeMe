import math
from collections import defaultdict

# Given a query -> apply preprocessing needed   
"""The inverted index is structured as:
    - Each unique token in the corpus (across all documents) is associated with:
      - 'df': The document frequency (number of documents in which the token occurs)
      - 'doc_info': A list of tuples, where each tuple contains:
        - [docno]: The document number
        - [positions]: list with number of times the token appears in the document
        """

def preprocess_query(query: str) -> list:
    """
    Preprocess the input query.
    This is a placeholder for more advanced preprocessing.
    """
    # Convert to lowercase and tokenize
    return query.lower().split()

def query_expansion(preprocessed_query: list) -> list:
    """
    TO DO TOMORROW : EMBEDDINGS ETC
    Expand the query with synonyms or related terms.
    This is a placeholder for more advanced query expansion techniques.
    """
    return preprocessed_query

def compute_bm25(token, index, doc_lengths, avg_doc_length, total_docs, k1=1.5, b=0.75):
    """
    Compute BM25 score for each token over all documents in which it appears.

    Args:
        token (str): token
        index (dict): inverted index
        doc_lengths (dict): dictionary of document lengths
        avg_doc_length (float): average document length in the corpus.
        total_docs (int): number of documents in the corpus.
        k1 (float): term saturation.
        b (float): length normalization.

    Returns:
        dict: A dictionary with document numbers as keys and BM25 scores as values.
    """
    if token not in index:
        return {}  # Token not in the index

    df = index[token]['df']  # doc frequency
    idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)  # idf calculation

    scores = {}
    for docno, positions in index[token]['doc_info'].items():
        freq = len(positions)  # frequency is len of positions list
        doc_length = doc_lengths.get(docno, 0) #get the length of doc
        term_freq_component = freq * (k1 + 1) / (freq + k1 * (1 - b + b * (doc_length / avg_doc_length))) #the second component of formula
        scores[docno] = idf * term_freq_component

    return scores

def retrieval_function(query, index, doc_lengths, k1=1.5, b=0.75):
    """
    Retrieve documents based on a query using the BM25 scoring function.

    Args:
        query (str): The user query as a string.
        index (dict): The inverted index structure.
        doc_lengths (dict): A dictionary of document lengths ({docno: length}).
        k1 (float): BM25 hyperparameter for term saturation.
        b (float): BM25 hyperparameter for length normalization.

    Returns:
        list: A list of tuples (docno, score) sorted by descending score.
    """
    # Preprocess the query
    tokens = preprocess_query(query)  # returns a list of tokens

    # Calculate average document length
    total_docs = len(doc_lengths)
    avg_doc_length = sum(doc_lengths.values()) / total_docs if total_docs > 0 else 0

    # aggregate scores for all tokens in the query
    aggregated_scores = defaultdict(float)
    for token in tokens:
        token_scores = compute_bm25(token, index, doc_lengths, avg_doc_length, total_docs, k1, b)
        for docno, score in token_scores.items():
            aggregated_scores[docno] += score

    # Sort documents by score in descending order and return the top results
    #return only the first 100 - but we do have to calculate for all of them probably
    return sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)[:100]

if __name__ == "__main__":

    #This example is chatgpt generated
    # Mock data: inverted index structure
    index = {
    "example": {
        "df": 2,
        "doc_info": {
            "doc1": [5, 15, 30],  # Token "example" appears 3 times in doc1
            "doc2": [7, 25]       # Token "example" appears 2 times in doc2
        }
    },
    "query": {
        "df": 1,
        "doc_info": {
            "doc2": [10]          # Token "query" appears 1 time in doc2
        }
    }
}

    # Document lengths (total tokens per document)
    doc_lengths = {
        "doc1": 100,  # Document 1 has 100 tokens
        "doc2": 150   # Document 2 has 150 tokens
    }

    # Define the query string
    query = "example query"

    # Perform retrieval
    results = retrieval_function(query, index, doc_lengths)

    # Print results
    print("Top Results:", results)