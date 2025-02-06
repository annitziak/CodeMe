import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

class Reranker:
    def __init__(self, retrieved_documents):
        self.retrieved_documents = retrieved_documents
        self.metadata = None
        self.load()

    def load(self):
        #temporary
        metadata = pd.read_csv("data/metadata_processed.csv")
        self.metadata = metadata[metadata["id"].isin(self.retrieved_documents)]
    
    def rerank(self):
        # rerank the retrieved documents based on metadata
        # this will change but we can do some testing for now

        features = ["score", "viewcount", "favoritecount", "commentcount", "reputation_user", "days_since_creation"]

        # between 0 and 1
        scaler = MinMaxScaler()
        self.metadata[features] = scaler.fit_transform(self.metadata[features])

        # Compute ranking score with weighted factors
        self.metadata["ranking_score"] = (
            (self.metadata["score"] * 1.5) +  # Strong weight on upvotes
            (self.metadata["viewcount"]) +  # Normalize view counts
            (self.metadata["favoritecount"] * 1 ) +  # Prioritize favorited posts
            (self.metadata["commentcount"] * 1.5) +  # Active discussions indicate usefulness
            (self.metadata["reputation_user"] * 1.5) +  # Normalize reputation for fair scaling
            (self.metadata["days_since_creation"] * 2)  # Boost newer documents
        )

        # Sort documents by ranking score in descending order
        self.metadata = self.metadata.sort_values(by="ranking_score", ascending=False)

        return self.metadata["id"].tolist()

#will give list and will rerank the top based on metadata
