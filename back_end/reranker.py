import torch
import os
import pickle
import logging

from semantic_search.embedded_search import EmbeddingSearchIndex
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(
        self,
        do_rerank_lm=True,
        load_dir=".cache/doc_embeddings",
        pretrained_model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.metadata = None
        if "sentence-transformers" in pretrained_model_name_or_path:
            self.model = SentenceTransformer(pretrained_model_name_or_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            self.model = AutoModel.from_pretrained("microsoft/codebert-base")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.do_rerank_lm = do_rerank_lm
        self.load_dir = load_dir

        h5_path = os.path.join(self.load_dir, "embeddings.h5")
        metadata_path = os.path.join(self.load_dir, "metadata.pkl")
        faiss_path = os.path.join(self.load_dir, "faiss_cosine.index")
        self.lm_documents = EmbeddingSearchIndex(h5_path, metadata_path).build_index(
            faiss_path
        )

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

    def fuse_scores(self, retrieved_documents, weights=(0, 0.6, 0.4)):
        if not retrieved_documents:
            return []

        min_bm25 = min([doc.get("bm25_score", 0) for doc in retrieved_documents])
        mid_bm25 = sum([doc.get("bm25_score", 0) for doc in retrieved_documents]) / len(
            retrieved_documents
        )
        max_bm25 = max([doc.get("bm25_score", 0) for doc in retrieved_documents])
        diff_bm25 = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1

        min_lm = min([doc.get("lm_score", 0) for doc in retrieved_documents])
        mid_lm = sum([doc.get("lm_score", 0) for doc in retrieved_documents]) / len(
            retrieved_documents
        )
        max_lm = max([doc.get("lm_score", 0) for doc in retrieved_documents])
        diff_lm = max_lm - min_lm if max_lm != min_lm else 1

        min_md = min([doc.get("metadata_score", 0) for doc in retrieved_documents])
        mid_md = sum(
            [doc.get("metadata_score", 0) for doc in retrieved_documents]
        ) / len(retrieved_documents)
        max_md = max([doc.get("metadata_score", 0) for doc in retrieved_documents])
        diff_md = max_md - min_md if max_md != min_md else 1

        for doc in retrieved_documents:
            bm25 = (doc.get("bm25_score", mid_bm25) - min_bm25) / diff_bm25
            lm = (doc.get("lm_score", mid_lm) - min_lm) / diff_lm
            md = (doc.get("metadata_score", mid_md) - min_md) / diff_md
            doc["final_score_normalized"] = (
                weights[0] * bm25 + weights[1] * lm + weights[2] * md
            )

        retrieved_documents = sorted(
            retrieved_documents,
            key=lambda x: x.get("final_score_normalized", 0),
            reverse=True,
        )

        return retrieved_documents

    def rerank_metadata(self, retrieved_documents):
        "rerank the top retrieved documents based on metadata"

        for doc in retrieved_documents:
            # TEMP SOLUTION WITHOUT INDEX RUN
            if len(doc.get("title", "")) == 0:
                doc["metadata_score"] -= 1
                continue

        return sorted(
            retrieved_documents, key=lambda x: x.get("metadata_score", 0), reverse=True
        )

    def rerank_lm(self, retrieved_documents, query):
        "rerank the top retrieved documents based on language model"
        # lm_queries is doc_id : encoded query
        if isinstance(self.model, SentenceTransformer):
            return self.rerank_lm_sentence_transformer(retrieved_documents, query)

        # Encode the query
        encoded_query_tokenized = self.tokenizer(
            query, return_tensors="pt", truncation=True
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

            similarity = 1 - cosine(encoded_query.flatten(), doc_embedding.flatten())

            doc["lm_score"] = similarity.item()
            reranked_documents.append(doc)

        reranked_documents.sort(key=lambda x: x["lm_score"], reverse=False)

        # Return only document IDs in sorted order
        return reranked_documents

    def rerank_lm_sentence_transformer(self, retrieved_documents, query):
        encoded_query = self.model.encode(query, convert_to_tensor=True).cpu()
        doc_map = {doc["doc_id"]: doc for doc in retrieved_documents}
        doc_ids = [x["doc_id"] for x in retrieved_documents]
        reranked_documents = self.lm_documents.search_filtered(
            encoded_query, doc_ids, k=len(retrieved_documents)
        )

        reranked_documents.sort(key=lambda x: x["lm_score"], reverse=True)
        for idx, doc in enumerate(reranked_documents):
            lm_score = (
                doc["lm_score"].item()
                if hasattr(doc["lm_score"], "item")
                else doc["lm_score"]
            )

            doc.update(doc_map[doc["doc_id"]])
            doc["lm_score"] = lm_score

        return reranked_documents

    def semantic_search(self, query, top_k=10):
        "rerank the top retrieved documents based on language model"
        query = self.model.encode(query, convert_to_tensor=True).cpu()

        retrieved_documents = self.lm_documents.search(query, top_k)
        retrieved_documents = sorted(
            retrieved_documents, key=lambda x: x.get("lm_score", 0), reverse=True
        )

        return retrieved_documents
