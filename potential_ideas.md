## Features to be integrated

### Essential features

1. Indexer 
2. Retrieval models (boolean, search, proximity search etc) → BM25 
3. Nice interface
→ focus on scalability

### Extra features

1. Query expansion :  pseudo-relevance feedback or LLM/Bert based on embeddings
2. L2R (consider  re-rankings of top N retrieved by some fixed method)
3. Classification of results into categories ( similar to LDA)
4. Query suggestion / Query correction → help user browse / re-formulate
5. Somehow log user actions and adjust retrieval model e.g. clickthrough data, preferences/choices presented
6. RAG?

## Corpora selection and project ideas

### **1. Coding Problems and Answers (e.g., Stack Overflow)**

### **Dataset Options**:

- **GitHub Discussions API**:
    - Link: [GitHub Discussions API Documentation](https://docs.github.com/en/graphql)
    - **Features included**
        - discussion titles and body for each question
        - tags (useful for organization + categories)
        - comment threads with reactions and engagement data e.g. upvotes (this might be helpful for retrieval evaluation e.g. ndcg metric etc)
- **Stack Exchange Data Dump ← can also combine with this**
    - You need to create an account to download them → follow this https://meta.stackexchange.com/help/data-dumps
    - **Features**:
        - Questions, answers, comments, and votes.
        - Tags for categorization.
        - User metadata

Look also : [**CodeSearchNet](https://github.com/github/CodeSearchNet/tree/master)** 

### **2. Research Paper Information Retrieval**

### **Dataset Option**:

- **arXiv Dataset**:
    - Link: [arXiv Bulk Data Access](https://info.arxiv.org/help/bulk_data/index.html)
    - **Features**:
        - Metadata (titles, abstracts, authors, categories).
        - Full-text content in PDF format > maybe not needed for us?
        - Categories in domain e.g. physics etc

To consider; **live indexing**
