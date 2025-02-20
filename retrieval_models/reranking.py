import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

class Reranker:
    def __init__(self):
        self.metadata = None
        self.load()

    def load(self):
        #temporary
        self.metadata = pd.read_csv("data/metadata_processed.csv")

    def rerank(self, retrieved_documents):
        # rerank the retrieved documents based on metadata
        # this will change but we can do some testing for now
        
        # only get useful docs
        metadata_df = self.metadata[self.metadata["id"].isin(retrieved_documents)].copy()

        if metadata_df.empty:
            print("⚠ Warning: No retrieved documents found in metadata.")
            return retrieved_documents

        # Features for ranking
        features = ["score", "viewcount",  "commentcount", "reputation_user", "days_since_creation"]

        # Normalize features
        scaler = MinMaxScaler()
        metadata_df[features] = scaler.fit_transform(metadata_df[features])


        # Compute ranking score
        metadata_df["ranking_score"] = (
            (metadata_df["score"] * 1.5) +  # Strong weight on upvotes
            (metadata_df["viewcount"] * 1) +  # Normalize view counts
            #(metadata_df["favoritecount"] * 1) +  # Prioritize favorited posts
            (metadata_df["commentcount"] * 1.5) +  # Active discussions indicate usefulness
            (metadata_df["reputation_user"] * 1.5) +  # Normalize reputation for fair scaling
            (metadata_df["days_since_creation"] * 2)  # Boost newer documents - this actually might not matter this much
        )

        # Sort by ranking score (descending order)
        ranked_documents = metadata_df.sort_values(by="ranking_score", ascending=False)["id"].tolist()

        return ranked_documents
#will give list and will rerank the top based on metadata
