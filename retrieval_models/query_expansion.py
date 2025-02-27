import torch
import numpy as np
import pickle
import os
import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingModel:
    def __init__(
        self,
        vocab: list,
        vocab_fn=None,
        model_name="microsoft/codebert-base",
        save_path="data/vocab_embeddings.pkl",
    ):
        """
        Initialize the EmbeddingModel with the specified vocabulary and model.
        Args:
            vocab (list or set): A list or set of vocabulary words.
            model_name (str): The name of the pretrained model to load.
            save_path (str): Path to save/load precomputed embeddings.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.vocab = list(vocab) if vocab is not None else []
        self.vocab_fn = vocab_fn
        self.save_path = save_path
        self.embeddings = None  # placeholder
        self.word_to_index = {}  # mapping

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # Load or compute embeddings
        if os.path.exists(self.save_path):
            print(f"Loading embeddings from {self.save_path}...")
            with open(self.save_path, "rb") as f:
                self.embeddings, self.word_to_index = pickle.load(f)
        else:
            print("Precomputing embeddings...")
            self.embeddings = self.precompute_vocab_embeddings(batch_size=100)
            self.save_embeddings()

    def save_embeddings(self):
        """Saves the precomputed embeddings and word-to-index mapping to a .pkl file."""
        self.word_to_index = {
            word: i for i, word in enumerate(self.vocab)
        }  # Create lookup dictionary

        try:
            with open(self.save_path, "wb") as f:
                pickle.dump((self.embeddings, self.word_to_index), f)
            print(f"✅ Embeddings successfully saved to {self.save_path}")
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")

        # Check if file exists after saving
        if os.path.exists(self.save_path):
            print(f"🔍 File confirmed at {self.save_path}")
        else:
            print("⚠ Warning: File was not created!")

    def get_embeddings_batch(self, words):
        """
        Computes embeddings for a batch of words.
        Args:
            words (list): A batch of words.
        Returns:
            np.ndarray: A 2D array of embeddings for the input words.
        """
        inputs = self.tokenizer(
            words, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def precompute_vocab_embeddings(self, batch_size=50):
        """
        Precomputes embeddings for all words in the vocabulary using batching.
        Args:
            batch_size (int): The size of each batch for processing.
        Returns:
            np.ndarray: A 2D array of word embeddings.
        """
        embeddings = []
        self.vocab = self.vocab_fn() if self.vocab_fn is not None else self.vocab
        total_words = len(self.vocab)

        for i in tqdm.tqdm(
            range(0, total_words, batch_size), desc="Processing batches"
        ):
            batch_words = self.vocab[i : i + batch_size]
            try:
                batch_embeddings = self.get_embeddings_batch(batch_words)
                embeddings.append(batch_embeddings)
            except Exception as e:
                print(
                    f"Failed to compute embeddings for batch: {batch_words}, error: {e}"
                )

        return np.vstack(embeddings)

    def get_embedding(self, word):
        """
        Retrieves the embedding for a single word from the precomputed embeddings.
        If the word is not in the vocabulary, it computes it on the fly.

        Args:
            word (str): The input word.
        Returns:
            np.ndarray: The embedding of the input word.
        """
        if word in self.word_to_index:
            return self.embeddings[
                self.word_to_index[word]
            ]  # Retrieve from precomputed embeddings
        else:
            print(f"Word '{word}' not found in precomputed embeddings")
            return []  # Compute dynamically

    def find_similar_words(self, word, top_k=5):
        """
        Finds the top_k most similar words to the given word using cosine similarity.

        Args:
            word (str): The target word.
            top_k (int): The number of similar words to return.
        Returns:
            list: A list of words most similar to the input word.
        """
        target_embedding = self.get_embedding(word)

        if len(target_embedding) == 0 or target_embedding is None:
            print(
                f"❌ Word '{word}' not found in vocabulary. Cannot find similar words."
            )
            return []

        target_embedding = target_embedding.reshape(
            1, -1
        )  # Correctly reshape for cosine similarity
        similarities = cosine_similarity(target_embedding, self.embeddings)[
            0
        ]  # Compute similarities

        top_indices = np.argsort(similarities)[::-1][
            1 : top_k + 1
        ]  # Skip the first as it's the same word
        print(top_indices)

        similar_words = [
            self.vocab[i] for i in top_indices
        ]  # Retrieve most similar words
        return similar_words
