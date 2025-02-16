import torch
import os
import faiss
import json

from utils.db_connection import DBConnection
from constants import DB_PARAMS

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class Embedder:
    def __init__(self, pretrained_model_name_or_path: str, device: str = "cpu"):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path).to(device)

    def embed(self, text: str):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            embeddings = self.model(input_ids).last_hidden_state[:, 0, :]
        return embeddings

    def embed_batch(self, texts: list, bodies: list):
        texts = [f"{title} {body}" for title, body in zip(texts, bodies)]
        input_ids = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        with torch.no_grad():
            # use [CLS] token
            embeddings = self.model(
                **input_ids, output_hidden_states=True
            ).last_hidden_state[:, 0, :]
        return embeddings

    def __call__(self, text: str):
        return self.embed(text)


class EmbedManager:
    def __init__(
        self,
        db_params: dict,
        pretrained_model_name_or_path: str,
        device: str = "cpu",
        batch_size: int = 128,
        save_dir: str = ".cache",
    ):
        self.embedder = Embedder(pretrained_model_name_or_path, device)
        self.db_connection = DBConnection(db_params)

        self.index = None
        self.index_map = {}

        self.batch_size = batch_size
        self.device = device

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def embed_and_save(self, limit=-1):
        limit_stmt = f"LIMIT {limit}" if limit > 0 else ""
        sql_query = f"""
            SELECT id, title, body
            FROM posts
            {limit_stmt}
        """

        pbar = tqdm(total=limit)

        self.index = faiss.IndexFlatL2(self.embedder.model.config.hidden_size)

        with self.db_connection as conn:
            conn.execute(sql_query)

            while True:
                batch = conn.fetchmany(self.batch_size)
                if not batch:
                    break

                ids, titles, bodies = zip(*batch)
                embeddings = self.embedder.embed_batch(titles, bodies)

                start_idx = self.index.ntotal
                self.index.add(embeddings.cpu().numpy().astype("float32"))

                for idx, doc_id in enumerate(ids):
                    self.index_map[doc_id] = start_idx + idx

                pbar.update(len(batch))

        save_index = self.index
        faiss.write_index(save_index, os.path.join(self.save_dir, "index.faiss"))

        with open(os.path.join(self.save_dir, "index_map.json"), "w") as f:
            json.dump(self.index_map, f)

        pbar.close()

    def load(self):
        index = faiss.read_index(os.path.join(self.save_dir, "index.faiss"))
        with open(os.path.join(self.save_dir, "index_map.json"), "r") as f:
            index_map = json.load(f)
        return index, index_map

    def search(self, query, k=10):
        query_embedding = self.embedder.embed(query)
        query_embedding = query_embedding.cpu().numpy().astype("float32")

        distances, indices = self.index.search(query_embedding, k)
        doc_ids = [
            doc_id for doc_id in self.index_map if self.index_map[doc_id] in indices
        ]
        return distances, doc_ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--db_name", type=str, default="stack_overflow")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default=".cache")
    parser.add_argument(
        "--pretrained-model-name-or-path", type=str, default="bert-base-uncased"
    )
    args = parser.parse_args()

    DB_PARAMS["database"] = args.db_name

    args.device = (
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    embed_manager = EmbedManager(
        DB_PARAMS,
        args.pretrained_model_name_or_path,
        args.device,
        save_dir=args.save_dir,
    )
    embed_manager.embed_and_save(args.limit)
    res = embed_manager.search("How to write a Python function?", k=10)
    print(res)

    embed_manager.load()
    res = embed_manager.search("How to write a Python function?", k=10)
    print(res)
