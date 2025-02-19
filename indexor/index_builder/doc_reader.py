import struct
import logging

from indexor.index_builder.constants import READ_SIZE_KEY, SIZE_KEY

logger = logging.getLogger(__name__)


def read_doc(shard_f, offset_f, sub_shard_f, sub_offset_f, docs_offset={}):
    local_doc_count = 0
    try:
        local_doc_count = struct.unpack(
            SIZE_KEY["doc_count"],
            sub_offset_f.read(READ_SIZE_KEY[SIZE_KEY["doc_count"]]),
        )[0]

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

            sub_shard_f.seek(offset)
            doc_length = struct.unpack(
                SIZE_KEY["doc_length"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_length"]]),
            )[0]
            score = struct.unpack(
                SIZE_KEY["doc_score"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_score"]]),
            )[0]
            view_count = struct.unpack(
                SIZE_KEY["doc_viewcount"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_viewcount"]]),
            )[0]
            owneruserid = struct.unpack(
                SIZE_KEY["doc_owneruserid"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_owneruserid"]]),
            )[0]
            answer_count = struct.unpack(
                SIZE_KEY["doc_answercount"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_answercount"]]),
            )[0]
            comment_count = struct.unpack(
                SIZE_KEY["doc_commentcount"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_commentcount"]]),
            )[0]
            favorite_count = struct.unpack(
                SIZE_KEY["doc_favoritecount"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_favoritecount"]]),
            )[0]
            size_display_name = struct.unpack(
                SIZE_KEY["doc_ownerdisplayname"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_ownerdisplayname"]]),
            )[0]
            ownerdisplayname = sub_shard_f.read(size_display_name).decode("utf-8")

            raw_tag_size = struct.unpack(
                SIZE_KEY["doc_tags"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_tags"]]),
            )[0]
            tags = sub_shard_f.read(raw_tag_size).decode("utf-8")

            raw_creation_date_size = struct.unpack(
                SIZE_KEY["doc_creationdate"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_creationdate"]]),
            )[0]
            creation_date = sub_shard_f.read(raw_creation_date_size).decode("utf-8")

            docs_offset[int(doc_id)] = shard_f.tell()

            shard_f.write(struct.pack(SIZE_KEY["doc_length"], doc_length))
            shard_f.write(struct.pack(SIZE_KEY["doc_score"], score))
            shard_f.write(struct.pack(SIZE_KEY["doc_viewcount"], view_count))
            shard_f.write(struct.pack(SIZE_KEY["doc_owneruserid"], owneruserid))
            shard_f.write(struct.pack(SIZE_KEY["doc_answercount"], answer_count))
            shard_f.write(struct.pack(SIZE_KEY["doc_commentcount"], comment_count))
            shard_f.write(struct.pack(SIZE_KEY["doc_favoritecount"], favorite_count))
            shard_f.write(
                struct.pack(SIZE_KEY["doc_ownerdisplayname"], size_display_name)
            )
            shard_f.write(ownerdisplayname.encode("utf-8"))
            shard_f.write(struct.pack(SIZE_KEY["doc_tags"], raw_tag_size))
            shard_f.write(tags.encode("utf-8"))

            shard_f.write(
                struct.pack(SIZE_KEY["doc_creationdate"], raw_creation_date_size)
            )
            shard_f.write(creation_date.encode("utf-8"))

    except struct.error as e:
        import traceback

        logger.error(
            f"Error reading {shard_f} after length {len(docs_offset)} {e} {local_doc_count}"
        )
        logger.error(traceback.format_exc())

        raise ValueError()

    return docs_offset
