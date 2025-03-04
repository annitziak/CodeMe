# Retrieval Models and Evaluation

## Project Overview

This concerns the **BM25 model** for ranked retrieval with techniques such as embedding-based **query expansion**, **LM reranking** and **Metadata Reranking** to improve search relevance and efficiency.

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

    *Precomputed embeddings can be made available if needed, or can also be computed using CodeBERT.*

### BM25 Ranked Search

- Implements the **BM25** algorithm to rank documents based on term relevance.
- Applies term boosting for key programming terms like "python" and "java"
- Utilizes TextBlob for spell correction when tokens are missing from the index.

### Query Expansion

- Utilizes an **Embedding Model** (CodeBERT) to expand queries with semantically related terms.
- It enhances recall by incorporating similar terms into the search.

*Implementation Notes:*

- Randomly selects between 1 to 3 similar terms per query word with a limit of adding 10 extra words per query
- All embeddings are precomputed to speed up expansion.

## Reranking Methods

### 1. Metadata-Based Reranking

- Reranks retrieved documents based on metadata attributes such as:
  - `score`, `viewcount`, `commentcount`, `reputation_user`, and `days_since_creation`.
- Normalizes these features using MinMaxScaler and computes a weighted score favoring upvotes and user reputation. The paremeters were tuned qualitatively.

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

At the end it was decided that the two reranking models will be combined together, given weights of 0.6 for the **LM** and 0.4 for the **Metadata** reranking.

## Evaluation

Evaluating the system's performance in this case was a bit challenging as it operates in an unsupervised setting, without ground truth labels for relevance. This lead to the use of indirect and clustering-based metrics to assess the quality of the retrieved results

1. **Ranking Quality Metrics:**

*Clarity Score:*
- Measured how focused the retrieved documents are compared to the corpus.
- Higher clarity score indicates more relevant results.

* Jaccard Score*
- Measures how similar the retrieved documents are with each other.

Ideally we want a high clarity and jaccard score. You can view the this analysis in the `results` file.
The evaluation focused on balancing **efficiency** and **effectiveness**, ensuring the system delivers high-quality results without significant performance trade-offs.
