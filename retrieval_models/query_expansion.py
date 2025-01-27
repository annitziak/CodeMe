
#make embeddings for the vocabulary
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # For progress visualization

from data_loaders import *

class EmbeddingModel:
    def __init__(self, vocab:list, model_name="microsoft/codebert-base"):
        """
        Initialize the EmbeddingModel with the specified vocabulary and model.
        Args:
            vocab (list or set): A list or set of vocabulary words.
            model_name (str): The name of the pretrained model to load.
        """
        # Initialize and load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.vocab = vocab  # Store vocab as a list for processing

        #precompute embeddings for the vocabulary in batches
        self.embeddings = self.precompute_vocab_embeddings(batch_size = 100)


    def get_embeddings_batch(self, words):
        """
        Computes embeddings for a batch of words.
        Args:
            words (list): A batch of words.
        Returns:
            np.ndarray: A 2D array of embeddings for the input words.
        """
        # Tokenize the input words
        inputs = self.tokenizer(words, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Return the mean-pooled embeddings
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def precompute_vocab_embeddings(self, batch_size=50):
        """
        Precomputes embeddings for all words in the vocabulary using batching.
        Args:
            batch_size (int): The size of each batch for processing.
        Returns:
            np.ndarray: A 2D array of word embeddings.
            list: A list of words corresponding to the embeddings.
        """
        embeddings = []
        total_words = len(self.vocab)

        for i in tqdm(range(0, total_words, batch_size), desc="Processing batches"):
            batch_words = self.vocab[i:i + batch_size]
            try:
                batch_embeddings = self.get_embeddings_batch(batch_words)
                embeddings.append(batch_embeddings)
            except Exception as e:
                print(f"Failed to compute embeddings for batch: {batch_words}, error: {e}")

        return np.vstack(embeddings)
    
    def get_embedding(self, word):
        """
        Computes the embedding for a single word.
        Args:
            word (str): The input word.
        Returns:
            np.ndarray: The embedding of the input word.
        """
        embedding = self.get_embeddings_batch([word])
        return embedding

    def find_similar_words(self, word, top_k=5):
        """
        Finds the top_k most similar words to the given word using cosine similarity
        Args:
            word (str): The target word.
            vocab_embeddings (np.ndarray): The precomputed embeddings of the vocabulary.
            word_list (list): The vocabulary list corresponding to the embeddings.
            top_k (int): The number of similar words to return.
        Returns:
            list: A list of tuples containing similar words and their cosine similarities.
        """
        target_embedding = self.get_embedding(word)
        # Compute cosine similarity in a vectorized manner
        similarities = cosine_similarity(target_embedding, self.embeddings)[0]
        # Get the top_k most similar words
        top_indices = np.argsort(similarities)[::-1][:top_k]

        #return the top similar words
        return [self.vocab[i] for i in top_indices]

