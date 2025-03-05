## Parts of Backend Implementation 

**Languge Used**
The backend is built using Flask, acting as the interface between the database and the frontend,
with the API handling all search functionalities. Flask was chosen because it is lightweight, flexible,
and efficient for building scalable web applications and APIs. 

1. Search Modes: 
The search system operates in two modes:
Basic Search, which uses BM25 to retrieve relevant documents, and Advanced Search, which supports
Boolean search, proximity search, and phrase search for more refined results

2. Pre Serach:
Before retrieval, the query undergoes preprocessing and query expansion, incorporating similar words to improve recall. 

3. Search:
The search function first parses the query to extract terms (parsing and normalization the query), then checks the cache for existing results—returning them instantly if found or otherwise retrieving documents based on the search using BM25. 

4. Post Search:
Search results are formatted as per the required format(like time taken, has next page, has previos page for pagination). The retrieved results are then passed to post-search processing, where they are formatted, re-ranked based on metadata, and further refined using a Language Model (LM)-based reranking to ensure the most relevant documents appear at the top. 

5. Filtering
If filters are applied, the results undergo additional processing, where documents can be sorted by date in descending order or filtered by category, displaying only those matching the
five predefined tags by ordering the tags into these clusters usin LM clustering. The final set of results is then returned to the user via a POST request, ensuring efficiency and relevance in document retrieval.

Backend Port
8080

Command to run Backend 
`python back_end/backend.py --index-path .cache/index-doc-title-body-v2 --embedding-path .cache/embedding2.pkl --reranker-path .cache/embeddings_v2`
