#### Dataset

What kind of information you need for searching, and what kind of relevant information you might want to use to help with searching? For example, when searching for news, I might want to look for something in 2020, so the metadata "date and time" is useful for a news search engine (by adding a filter function etc.). Is your dataset large enough to fulfil its purpose?

What if users don't know the keywords? Let's say I want to find all articles/songs/books that is in English language (think about what users might want to do), is it possible to get the results with your data?

#### Indexing

How can you make the indexing more efficient and dynamic? Writing the index as a txt file takes up more storage, are there efficient ways to do this? Are there ways to do data compression to save storage? Do you have a function that can automatically/manually update documents?

#### Search

How realistic is your search engine at solving users needs? What functions do users want to see on your search engine? What type of searches they usually do on this kind of search engine? What kind of functions do they have in real life applications?

How robust is your search function? What happens when users write something random?

Do users have to follow a set of rules to use the search engine (A AND B in boolean search), or you can process their input in the background?

How can you maximise search results? Again, what if users can’t remember the keywords? Can it handle not just the exact words when searching, such as paraphrased words or wildcards?

#### Retrieval model

Is the model 'smart' and 'fast' enough to handle complex queries? Does the model use extra metadata to help with searching?

#### UI

Is your tool easy to interact with? Do users have to guess where a function is located and how it works?

#### Evaluation

What methods can you use to evaluate retrieval results. What are the most important features to evaluate for your search engine? Do the most relevant results always appear on the top? How can you evaluate the processing time for each query, especially longer ones?

It is important to think about the basic methods you have learnt in class and what are useful for your search engine. For example, how you can include efficient indexing, data compression, Boolean search, phrase search, proximity search, query expansion etc. to improve system performance.
