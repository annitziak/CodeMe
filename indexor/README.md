# API Usage
Initialise an index by calling the following:
```python
from indexor.index import Index

index = Index(load_path="path_to_dir")
```

Ensure that the path references a directory to enable disk-based indexing.
If you wish to use memory-based indexing, you can initialise the index as follows:
```python
index = Index(load_path="path_to_file")
```

The following methods are available for use:
```python
index.get_document_frequency(term) # Returns the number of documents the term occurs with in the index
index.get_term_frequency(term, doc_id) # Returns the number of times the term occurs in the document
index.get_document_length(doc_id) # Returns the length of the document
index.get_postings_list(term) # Returns the postings list for the term
index.get_term(term) # Returns the term object
index.get_term(term, positions=True) # Returns the term object with positions
```
