import glob
import os
import logging

from indexor.index_builder.shard_reader import ShardReader

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


def save_shard_to_mongo(
    shard_file: str,
    offset_file: str,
    positions: bool = False,
    collection_name="index",
    batch_size=25000,
):
    client = MongoClient("localhost", 27017)
    db = client["test"]
    collection = db[collection_name]

    count = 0

    logger.info(
        f"Processing {shard_file} and {offset_file}. Saving to {collection_name}"
    )
    shard_reader = ShardReader(shard_file, offset_file)
    collection = db[collection_name]

    next_term, next_offset = shard_reader.next_term()

    while next_term is not None and next_offset is not None:
        count += 1
        postings = shard_reader.read_postings(next_offset, read_positions=positions)

        for i in range(0, len(postings), batch_size):
            end = min(i + batch_size, len(postings))

            item = {
                "term": next_term,
                "shard": shard_file,
                "batch": i // batch_size,
                "doc_freq": len(postings[i:end]),
                "postings": [
                    {
                        "doc_id": x.doc_id,
                        "doc_term_frequency": x.doc_term_frequency,
                        "positions": x.positions,
                    }
                    for x in postings[i:end]
                ],
            }
            try:
                collection.insert_one(item)
            except Exception as e:
                logger.error(
                    f"Error saving {next_term} of size {len(postings)} to MongoDB: {e}"
                )

        if count % 1000 == 0:
            logger.info(f"Processed {count} terms in {shard_file}")

        next_term, next_offset = shard_reader.next_term()

    shard_reader.close()
    logger.info(f"Finished processing {shard_file}. Processed {count} terms")


def main(index_path):
    files = get_files(index_path, "shard")
    pos_files = get_files(index_path, "pos_shard")

    logger.info(f"Processing {len(files)} shard files")
    logger.info(f"Processing {len(pos_files)} position shard files")

    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = []
        for shard_file, offset_file in files:
            futures.append(
                executor.submit(
                    save_shard_to_mongo, shard_file, offset_file, False, "index"
                )
            )

        for pos_shard_file, pos_offset_file in pos_files:
            futures.append(
                executor.submit(
                    save_shard_to_mongo,
                    pos_shard_file,
                    pos_offset_file,
                    True,
                    "pos_index",
                )
            )

        for future in futures:
            future.result()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save the index to MongoDB")
    parser.add_argument(
        "--index-path", help="The index file to save to MongoDB", required=True
    )
    args = parser.parse_args()

    main(args.index_path)
