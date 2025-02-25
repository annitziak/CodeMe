import torch
import os
import pickle
import logging

from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, do_rerank_lm=True):
        self.metadata = None
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.do_rerank_lm = do_rerank_lm
        self.lm_documents = self.load() if self.do_rerank_lm else {}

    def load(self):
        file_path_1 = os.path.join("retrieval_models", "data", "half_1.pkl")
        if not os.path.exists(file_path_1):
            raise FileNotFoundError(f"File '{file_path_1}' not found")

        with open(file_path_1, "rb") as f:
            data_1 = pickle.load(f)  # Load first dictionary

        file_path_2 = os.path.join("retrieval_models", "data", "half_2.pkl")
        if not os.path.exists(file_path_2):
            raise FileNotFoundError(f"File '{file_path_2}' not found")

        with open(file_path_2, "rb") as f:
            data_2 = pickle.load(f)  # Load second dictionary

        # Ensure both are dictionaries before merging
        if isinstance(data_1, dict) and isinstance(data_2, dict):
            lm_documents = {
                **data_1,
                **data_2,
            }  # Merge both dictionaries (overwrites duplicates)
        else:
            lm_documents = {}  # Empty dictionary if loading failed

        return lm_documents

    def rerank_metadata(self, retrieved_documents):
        "rerank the top retrieved documents based on metadata"

        return sorted(
            retrieved_documents, key=lambda x: x.get("metadata_score", 0), reverse=True
        )

    def rerank_lm(self, retrieved_documents, query):
        "rerank the top retrieved documents based on language model"
        # lm_queries is doc_id : encoded query

        # Encode the query
        encoded_query_tokenized = self.tokenizer(
            query, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded_query_tokenized)
        encoded_query = (
            outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        )  # Get the mean embedding?

        # Rerank documents based on cosine similarity
        reranked_documents = []
        for doc in retrieved_documents:
            doc_id = str(doc["doc_id"])
            if doc_id not in self.lm_documents:
                logger.info(
                    f"Document ID {doc['doc_id']} not found in language model documents {len(self.lm_documents)}"
                )
                continue

            doc_embedding = self.lm_documents[doc_id]

            # Compute cosine similarity
            similarity = 1 - cosine(
                encoded_query.flatten(), doc_embedding.flatten()
            )  # Cosine similarity

            doc["lm_score"] = similarity.item()
            reranked_documents.append(doc)

        # Sort documents by similarity (higher is better)
        reranked_documents.sort(key=lambda x: x["lm_score"], reverse=True)

        # Return only document IDs in sorted order
        return reranked_documents
