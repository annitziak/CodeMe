File representations for the inverted index.

## Offset (Term Dictionary)
Read entirely into memory and provides direct mapping to the index file.

- Total number of terms
For each term
- term length (in bytes)
- term
- document frequency (number of postings for the term)
- offset in the posting list file
- offset in the positions files
- offset in the skiplist file
- offset in the term information file

Index File (Term Information)
- Total number of terms
For each term
- document frequency (number of postings for the term)
- offset in the posting list file
- offset in the positions file

Posting List File (Document IDs)
For each term and then for each document that contains the term
- document ID delta
- term frequency (number of occurrences of the term in the document)

Positions File (Term Positions)
For each term and then for each document that contains the term
- position delta
