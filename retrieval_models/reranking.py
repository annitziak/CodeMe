import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine


class Reranker:
    def __init__(self):
        self.metadata = None
        self.load()
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def load(self):
        #temporary
        # this just has only the data we 
        self.metadata = pd.read_csv("data/metadata_processed.csv")

    def rerank_metadata(self, retrieved_documents):
        "rerank the top retrieved documents based on metadata"
        
        # only get useful docs
        metadata_df = self.metadata[self.metadata["id"].isin(retrieved_documents)].copy()

        if metadata_df.empty:
            print("⚠ Warning: No retrieved documents found in metadata.")
            return retrieved_documents

        # Features for ranking - these showed the most discimination in the EDA
        features = ["score", "viewcount",  "commentcount", "reputation_user", "days_since_creation"]

        # Normalize features
        scaler = MinMaxScaler()
        metadata_df[features] = scaler.fit_transform(metadata_df[features])


        # Compute ranking score - this can be stored and precomputed for each document
        metadata_df["ranking_score"] = (
            (metadata_df["score"] * 1.5) +  # Strong weight on upvotes
            (metadata_df["viewcount"] * 1.2) +  # Normalize view counts
            #(metadata_df["favoritecount"] * 1) +  # Prioritize favorited posts
            #(metadata_df["commentcount"] * 1.5) +  # Active discussions indicate usefulness
            (metadata_df["reputation_user"] * 1.5) +  # Normalize reputation for fair scaling
            (metadata_df["days_since_creation"] * 1.5)  # Boost newer documents - this actually might not matter this much
        )

        # Sort by ranking score (descending order)
        ranked_documents = metadata_df.sort_values(by="ranking_score", ascending=False)["id"].tolist()

        return ranked_documents
    
    def rerank_lm(self, retrieved_documents, query, lm_queries):
        "rerank the top retrieved documents based on language model"
        #lm_queries is doc_id : encoded query

        # Encode the query
        encoded_query_tokenized = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded_query_tokenized)
        encoded_query = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Get the mean embedding?

        # Rerank documents based on cosine similarity
        reranked_documents = []
        for doc_id in retrieved_documents:
            doc_embedding = lm_queries[doc_id]

            # Compute cosine similarity
            similarity = 1 - cosine(encoded_query.flatten(), doc_embedding.flatten())  # Cosine similarity
            
            reranked_documents.append((doc_id, similarity))

        # Sort documents by similarity (higher is better)
        reranked_documents.sort(key=lambda x: x[1], reverse=True)

        # Return only document IDs in sorted order
        return [doc_id for doc_id, _ in reranked_documents]

