# Preprocessing

Preprocessing, in its current form, is broken down into the following steps:
- Parsing
- Tokenization
- Normalization

## Parsing
The main goal is to convert the results from the SQL DB into a equal format.
```sql
SELECT body FROM posts;
```
The result of this query, is rendered HTML and therefore it is parsed by the `lxml` library.
We break up the result into a number of `blocks` based on the type of the HTML tag associated with the text.
The following `blocks` are created:
- `CodeBlock`
- `NormalTextBlock`
- `LinkBlock` (subclass of `NormalTextBlock`)
All of which are subclasses of `Block` (more may be required).

These blocks can then store properties relevant to the type of block they are.
For example `CodeBlock` stores the property `in_line` which determines whether the code block is inline or not.
`NormalTextBlock` stores a number of different properties such as its text-size (`h1`,...`h6`,`p`,`li`,`ul`,`ol`) and whether it is bold, italic or underlined.
The main idea is to use these properties to perhaps, augment our scoring function in the future.
Finally, the `LinkBlock` stores the URL of the link and any alt-text associated with it.

We parse the HTML based on the following [website](https://meta.stackexchange.com/questions/1777/what-html-tags-are-allowed-on-stack-exchange-sites).

For other queries, such as `SELECT title FROM posts;`, the parsing is done in a similar way, but the blocks are not as complex as they are only text.

## Tokenization
Tokenization involves splitting text into individual tokens.
Therefore, a post body can be made up of multiple blocks and each block can be made up of multiple tokens.
Internally the `Term` class is used to represent a token, and we make an effort to track:
- `term`: the actual token
- `position`: the position of the token in the block (maintained after normalization)
- `start_char_offset`: the start character offset of the token in the block (for highlighting)
- `end_char_offset`: the end character offset of the token in the block (for highlighting)

### Text Tokenization
For the moment, tokenization is done manually and is based off of the Elastic Search [Classic Tokenizer](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-classic-tokenizer.html)
The following steps are taken:
- Replace hyphens with whitespace (e.g. `non-linear` -> `non linear`)
- Recognizes emails and urls and tokenizes them as a single token
- Splits text based on regex defined word boundaries (i.e. `r'\b\w+\b'`)
- Contractions are combined into a single token (e.g. `can't` -> `cant`)
- Double whitespace is removed

### Code Tokenization (WIP)
Since code is not natural language, we tokenize it slightly differently.
Basically, we maintain a buffer while stepping though the code character by character.
When we encounter a non-whitespace character that is also not a symbol (e.g. !@#$%^&*()_+), we add it to the buffer.
The buffer is flushed and added as a token, once a whitespace character or a symbol is encountered.

e.g. `int main() {` -> [`int`, `main`, `(`, `)`, `{`]

(UNIMPLEMENTED: Perhaps just split based on underscores of camel case???)
Identifiers are then split into multiple tokens based on a BPE tokenizer that is trained on a large code corpus (CodeSearchNet).
This is done as code identifiers can be quite long and splitting them into subtokens can help with matching.

But, this is still a work in progress and we may require a more sophisticated approach.
Perhaps, a more semantics-based approach as code itself would not contain the same kind of tokens as a natural language query.

### Link Tokenization
Links are tokenized as tokens included within the `<a>` tag. The URL is tokenized as a single token and the alt-text is tokenized as a single token.
As is the case with `href` and `alt` attributes, they are not part of the text itself and therefore are not given a position or character offset (we should assume that it is the same as the block)

## Normalization
Normalization involves converting tokens into a standard form to make them easier to compare and match:
- Lowercasing
- Converting to NFD form (decomposing unicode)
- Stemming
- Stopword Removal
