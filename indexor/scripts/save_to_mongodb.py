import glob
import os
import logging

from indexor.index import Index
from indexor.structures import Term

from pymongo import MongoClient
from concurrent.futures import ProcessPoolExecutor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_files(index_path: str, prefix: str):
    shard_files = glob.glob(f"{index_path}/{prefix}_*.index")
    offset_files = glob.glob(f"{index_path}/{prefix}_*.offset")

    return (
        list(zip(sorted(shard_files), sorted(offset_files)))
        if len(shard_files) > 0
        else []
    )


class MongoInvertedIndexBuilder:
    def __init__(self, index_path: str):
        self.index_path = index_path

        self.index = Index(self.index_path)
        self.client = MongoClient("localhost", 27017)

        self.db = self.client["index"]
        self.terms = self.db["terms"]
        self.postings = self.db["postings"]
        self.positions = self.db["positions"]

    def save_term_to_mongo(
        self,
        term: str,
    ):
        logger.info(f"Processing term: {term}")

        term_obj: Term = self.index.get_term(term)

        for doc_id, positions in term_obj.posting_lists.items():
            position_id = f"{term}_{doc_id}"
            self.positions.update_on(
                {"_id": position_id},
                {
                    "$set": {
                        "term": term,
                        "doc_id": doc_id,
                        "positions": positions,
                    }
                },
                upsert=True,
            )

            posting_doc = {
                "doc_id": doc_id,
                "doc_term_frequency": len(positions),
                "position_id": position_id,
            }

            self.postings.update_on(
                {"_id": f"postings_{term}"},
                {"$push": {"posting_lists": posting_doc}},
                upsert=True,
            )

        self.terms.update_on(
            {"_id": term},
            {
                "$set": {
                    "doc_freq": term_obj.document_frequency,
                    "posting_id": f"postings_{term}",
                }
            },
            upsert=True,
        )

    def __call__(self, *args, **kwargs):
        self.build(*args, **kwargs)

    def build(self):
        with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            for term in self.index.terms():
                executor.submit(
                    self.save_term_to_mongo,
                    term,
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save the index to MongoDB")
    parser.add_argument(
        "--index-path", help="The index file to save to MongoDB", required=True
    )
    args = parser.parse_args()

    builder = MongoInvertedIndexBuilder(index_path=args.index_path)
    builder()
