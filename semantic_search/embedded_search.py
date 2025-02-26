import faiss
import h5py
import numpy as np
import os
import pickle
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EmbeddingSearchIndex:
    def __init__(self, h5_path, metadata_path):
        """Initialize the search index with paths to stored embeddings"""
        self.h5_path = h5_path
        self.metadata_path = metadata_path
        self.index = None

        # Load metadata
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
            print(
                f"Loaded metadata from {metadata_path} with |documents|={len(self.metadata.get('document_ids', []))}"
            )

        # Get dimension from metadata or H5 file
        if "embedding_dim" in self.metadata:
            self.dim = self.metadata["embedding_dim"]
        else:
            with h5py.File(h5_path, "r") as f:
                self.dim = f["embeddings"].shape[1]

        print(f"Embedding dimension: {self.dim}")

    def build_index(self, index_path=None, use_gpu=False):
        """Build a FAISS index optimized for cosine similarity search"""
        if index_path and os.path.exists(index_path):
            print(f"Loading existing index from {index_path}")
            # Keeps the index in memory-mapped mode for faster loading
            self.index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
            return self

        print("Building new FAISS index...")
        start_time = time.time()

        # For cosine similarity, we use IndexFlatIP (inner product) with normalized vectors
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatIP(self.dim)
            index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            index = faiss.IndexFlatIP(self.dim)

        # For larger datasets, you might want to use IVF for better scalability
        # Uncomment the following for a more scalable index:

        # Create quantizer
        quantizer = faiss.IndexFlatIP(self.dim)
        # Create IVF index (1024 clusters for 50M vectors is reasonable)
        nlist = 1024
        index = faiss.IndexIVFFlat(
            quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT
        )

        # Need to train this index
        print("Training IVF index...")
        with h5py.File(self.h5_path, "r") as f:
            # Use a sample of vectors for training
            train_size = min(100000, f["embeddings"].shape[0])
            train_vectors = f["embeddings"][:train_size].astype(np.float32)
            faiss.normalize_L2(train_vectors)
            index.train(train_vectors)

        # Add vectors in batches
        with h5py.File(self.h5_path, "r") as f:
            embeddings = f["embeddings"]
            total_vectors = embeddings.shape[0]
            batch_size = 100000  # Adjust based on your RAM

            for i in range(0, total_vectors, batch_size):
                end = min(i + batch_size, total_vectors)
                batch = embeddings[i:end].astype(np.float32)

                # Normalize each vector to unit length for cosine similarity
                faiss.normalize_L2(batch)

                # Add to index
                index.add(batch)
                print(f"Added {end}/{total_vectors} vectors to index")

        self.index = index

        # Save index if path provided
        if index_path:
            print(f"Saving index to {index_path}")
            faiss.write_index(index, index_path)

        print(f"Index built in {time.time() - start_time:.2f} seconds")
        return self

    def search(self, query_vector, k=10):
        """Search for similar vectors"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Ensure query is a numpy array of the right shape and type
        query = np.asarray(query_vector).reshape(1, self.dim).astype(np.float32)

        # Normalize query for cosine similarity
        faiss.normalize_L2(query)

        # Search
        distances, indices = self.index.search(query, k)

        # Map indices to document IDs if available
        results = []
        doc_ids = self.metadata.get("document_ids", [])

        for i, idx in enumerate(indices[0]):
            doc_id = doc_ids[idx] if idx < len(doc_ids) else str(idx)
            similarity = distances[0][
                i
            ]  # For normalized vectors, this is cosine similarity
            results.append({"id": doc_id, "similarity": similarity})

        return results

    def search_filtered(self, query_vector, filter_doc_ids, k=100):
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        query = np.asarray(query_vector).reshape(1, self.dim).astype(np.float32)
        faiss.normalize_L2(query)

        valid_indicies = []
        doc_id_map = {}
        doc_ids = self.metadata.get("document_ids", [])

        for doc_id in filter_doc_ids:
            if doc_id not in doc_ids:
                continue

            idx = doc_ids.index(doc_id)
            valid_indicies.append(idx)
            doc_id_map[len(valid_indicies) - 1] = doc_id

        if len(valid_indicies) == 0:
            return []

        id_selector = faiss.IDSelectorArray(valid_indicies)
        distances, indices = self.index.search(
            query, k, params=faiss.SearchParametersIVF(sel=id_selector)
        )

        if len(indices) == 0:
            return [{"doc_id": doc_id, "lm_score": 0.0} for doc_id in filter_doc_ids]

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            doc_id = doc_id_map[idx]
            similarity = distances[0][i]
            results.append({"doc_id": doc_id, "lm_score": similarity})

        if len(results) == 0:
            return [{"doc_id": doc_id, "lm_score": 0.0} for doc_id in filter_doc_ids]

        return results


# Example usage
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    index = EmbeddingSearchIndex(
        h5_path="./embeddings/embeddings.h5", metadata_path="./embeddings/metadata.pkl"
    )

    # Build or load index
    index.build_index(index_path="./embeddings/faiss_cosine.index")

    # Example search with a random vector (replace with your query embedding)
    query = np.random.random(index.dim).astype(np.float32)
    results = index.search(query, k=5)

    print("Search results for random query vector:")
    for result in results:
        print(f"Document ID: {result['id']}, Similarity: {result['similarity']:.4f}")

    start = time.time()
    query = "What is a linked list in Python?"
    query_embedding = model.encode(query)
    results = index.search(query_embedding.reshape((1, len(query_embedding))), k=5)

    print("Search results for query: 'What is a linked list in Python?'")
    for result in results:
        print(f"Document ID: {result['id']}, Similarity: {result['similarity']:.4f}")
    end = time.time()

    print(f"Time taken: {end - start:.2f} seconds")
