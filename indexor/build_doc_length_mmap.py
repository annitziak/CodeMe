import os
import struct
import glob
import numpy as np

from indexor.index_builder.constants import SIZE_KEY, READ_SIZE_KEY


def find_offset_files(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index path {index_path} does not exist")

    return glob.glob(os.path.join(index_path, "doc_shard_*.offset"))


def build_doc_length_mmap(doc_offset_files, index_path):
    for offset_file in doc_offset_files:
        doc_ids = []
        doc_lengths = []
        doc_offsets = []
        shard_num = int(offset_file.split("_")[-1].split(".")[0])
        with open(offset_file, "rb") as sub_offset_f:
            local_doc_count = struct.unpack(
                SIZE_KEY["doc_count"],
                sub_offset_f.read(READ_SIZE_KEY[SIZE_KEY["doc_count"]]),
            )[0]
            _ = struct.unpack(
                SIZE_KEY["offset"],
                sub_offset_f.read(READ_SIZE_KEY[SIZE_KEY["offset"]]),
            )[0]  # sum_doc_length

            for _ in range(local_doc_count):
                doc_id = struct.unpack(
                    SIZE_KEY["doc_id"],
                    sub_offset_f.read(READ_SIZE_KEY[SIZE_KEY["doc_id"]]),
                )[0]
                # OFFSET IS ONE HERE
                # MISSING ONE DOC 9999
                offset = struct.unpack(
                    SIZE_KEY["offset"],
                    sub_offset_f.read(READ_SIZE_KEY[SIZE_KEY["offset"]]),
                )[0]
                doc_length = struct.unpack(
                    SIZE_KEY["doc_length"],
                    sub_offset_f.read(READ_SIZE_KEY[SIZE_KEY["doc_length"]]),
                )[0]
                doc_ids.append(doc_id)
                doc_offsets.append(offset)
                doc_lengths.append(doc_length)

        doc_ids = np.array(doc_ids, dtype=np.uint32)
        doc_lengths = np.array(doc_lengths, dtype=np.uint16)
        doc_offsets = np.array(doc_offsets, dtype=np.uint64)

        sorted_indicies = np.argsort(doc_ids)
        doc_ids = doc_ids[sorted_indicies]
        doc_offsets = doc_offsets[sorted_indicies]
        doc_lengths = doc_lengths[sorted_indicies]

        np.save(os.path.join(index_path, f"doc_ids_{shard_num}.npy"), doc_ids)
        np.save(os.path.join(index_path, f"doc_offsets_{shard_num}.npy"), doc_offsets)
        np.save(os.path.join(index_path, f"doc_lengths_{shard_num}.npy"), doc_lengths)

        print("Completed building doc length mmap for shard", shard_num)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build document length mmap")
    parser.add_argument(
        "--index-path", type=str, required=True, help="Path to the index directory"
    )
    args = parser.parse_args()

    doc_offset_files = find_offset_files(args.index_path)
    build_doc_length_mmap(doc_offset_files, args.index_path)
