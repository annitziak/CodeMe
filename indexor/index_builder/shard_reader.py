import struct
import filelock
import logging

from indexor.index_builder.constants import SIZE_KEY, READ_SIZE_KEY
from indexor.structures import PostingList

logger = logging.getLogger(__name__)


class ShardReader:
    def __init__(self, shard_path: str, position_path: str, offset_path: str):
        self.shard_path = shard_path
        self.position_path = position_path
        self.offset_path = offset_path

        self.lock_shard_file = filelock.FileLock(shard_path + ".lock")
        self.lock_position_file = filelock.FileLock(position_path + ".lock")
        self.lock_offset_file = filelock.FileLock(offset_path + ".lock")

        logger.debug(f"Locking {shard_path} and {offset_path}")
        self.lock_shard_file.acquire()
        self.lock_position_file.acquire()
        self.lock_offset_file.acquire()

        self.f_shard = open(shard_path, "rb")
        self.f_position = open(position_path, "rb")
        self.f_offset = open(offset_path, "rb")

    def __del__(self):
        self.close()

    def close(self):
        if not hasattr(self, "f_shard") or not hasattr(self, "f_offset"):
            return

        self.f_shard.close()
        self.f_position.close()
        self.f_offset.close()

        self.lock_shard_file.release()
        self.lock_position_file.release()
        self.lock_offset_file.release()
        logger.debug(f"Unlocked {self.shard_path} and {self.offset_path}")

    def next_term(self):
        try:
            term_length = struct.unpack(
                SIZE_KEY["term_bytes"],
                self.f_offset.read(READ_SIZE_KEY[SIZE_KEY["term_bytes"]]),
            )[0]
            term = self.f_offset.read(term_length).decode("utf-8")
            offset = struct.unpack(
                SIZE_KEY["offset"],
                self.f_offset.read(READ_SIZE_KEY[SIZE_KEY["offset"]]),
            )[0]
            pos_offset = struct.unpack(
                SIZE_KEY["offset"],
                self.f_offset.read(READ_SIZE_KEY[SIZE_KEY["offset"]]),
            )[0]

            return term, offset, pos_offset
        except struct.error:
            return None, None, None

    def read_postings(self, offset: int, pos_offset=-1) -> list[PostingList]:
        self.f_shard.seek(offset)

        count = struct.unpack(
            SIZE_KEY["postings_count"],
            self.f_shard.read(READ_SIZE_KEY[SIZE_KEY["postings_count"]]),
        )[0]
        postings = []
        curr_doc_id = 0

        for _ in range(count):
            doc_id_delta, doc_term_frequency = struct.unpack(
                SIZE_KEY["deltaTF"],
                self.f_shard.read(READ_SIZE_KEY[SIZE_KEY["deltaTF"]]),
            )
            position_count = struct.unpack(
                SIZE_KEY["position_count"],
                self.f_shard.read(READ_SIZE_KEY[SIZE_KEY["position_count"]]),
            )[0]
            curr_doc_id += doc_id_delta

            positions = []
            if pos_offset >= 0:
                curr_position = 0
                for _ in range(position_count):
                    position_delta = struct.unpack(
                        SIZE_KEY["position_delta"],
                        self.f_position.read(READ_SIZE_KEY[SIZE_KEY["position_delta"]]),
                    )[0]
                    curr_position += position_delta
                    positions.append(curr_position)

            postings.append(
                PostingList(curr_doc_id, doc_term_frequency, sorted(positions))
            )

        return postings
