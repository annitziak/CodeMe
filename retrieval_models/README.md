# Retrieval Models and Evaluation

## Project Overview

Several **information retrieval** models are used for document search, combining traditional methods like **Boolean Search** and **BM25** with techniques such as embedding-based **query expansion** and language model **reranking** to improve search relevance and efficiency.

---

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_folder>

2. **Install dependendencies**
    pip install -r requirements.txt

3. **Download pretrained models**
    The system utilizes CodeBERT for query expansion and reranking.

    *Precomputed embeddings are available in data/vocab_embeddings.pkl.*

4. **Prepare the index**
    Ensure the preprocessed inverted index and metadata (metadata_processed.csv) are located in the data/ directory.


## Retrieval models

### 1. Boolean Search

- Supports logical operators: `AND`, `OR`, `NOT`.
- Includes phrase searches e.g.: `"python tutorial"`.
- Includes proximity searches e.g.: `#5(python tutorial)`.

*Implementation Notes:*

- The search logic uses operator precedence (NOT > AND > OR) to parse queries.
- Phrase and proximity searches are extracted and processed before the main Boolean logic.
- A fallback mechanism is included to handle edge cases and unrecognized tokens.

### 2. BM25 Ranked Search

- Implements the **BM25** algorithm to rank documents based on term relevance.
- Applies term boosting for key programming terms like "python" and "java".

*Implementation Notes:*

- Utilizes TextBlob for spell correction when tokens are missing from the index.
- Parameters k1 and b fine-tune term frequency scaling and document length normalization.
- Handles edge cases like missing tokens and documents with unusual term distributions.


### 3. Query Expansion

- Utilizes an **Embedding Model** (CodeBERT) to expand queries with semantically related terms.
- It enhances recall by incorporating similar terms into the search.

*Implementation Notes:*

- Randomly selects between 1 to 3 similar terms per query word.
- Uses precomputed embeddings to speed up expansion.
- Boosts key programming terms (e.g., “python”, “list”, “array”, "java", "c", "javascript", "typescript", "rust", "golang", "swift", "php", "r", "matlab", "sql", "nosql", "html", "css", "ruby", "tree", "graph", "heap", "hashmap", "queue", "stack") to improve relevance.



## Reranking Methods

### 1. Metadata-Based Reranking

- Reranks retrieved documents based on metadata attributes such as:
  - `score`, `viewcount`, `commentcount`, `reputation_user`, and `days_since_creation`.
- Normalizes these features using MinMaxScaler and computes a weighted score favoring upvotes and user reputation.

*Implementation notes:*

- Weights are fine-tuned based on exploratory data analysis (e.g., heavier weight on upvotes and reputation).
- Includes a fallback mechanism if no metadata matches the retrieved documents.

### 2. Language Model-Based Reranking

- Employs **CodeBERT** to assess semantic similarity between queries and documents.
- Uses **cosine similarity** of embeddings to rank documents.

*Implementation Notes:*

- Query and document embeddings are generated using CodeBERT.
- Cosine similarity measures semantic closeness.
- Precomputed document embeddings speed up reranking.


## Evaluation

Evaluating the system's performance in this case was a bit challenging as it operates in an unsupervised setting, without ground truth labels for relevance. This lead to the use of indirect and clustering-based metrics to assess the quality of the retrieved results

1. **Ranking Quality Metrics:**

*Clarity Score:*
- Measured how focused the retrieved documents are compared to the corpus.
- Higher clarity score indicates more relevant results.

*Clustering Scores (DBSCAN):*

- Applied DBSCAN clustering on retrieved document embeddings.
- Evaluated with Silhouette Score (higher is better) and Davies-Bouldin Score (lower is better).

3. **Qualitative Analysis**

*Manual Review:*

- Analyzed selected queries and inspects the top results for relevance and coherence.
- Edge Case Testing: Tested how system handles more complex Boolean queries, proximity searches, etc.

**Evaluation Strategy:**

- The evaluation focused on balancing **efficiency** and **effectiveness**, ensuring the system delivers high-quality results without significant performance trade-offs.
