import struct
import logging

from indexor.index_builder.constants import READ_SIZE_KEY, SIZE_KEY
from indexor.metadata_score import metadata_score
from indexor.structures import DocMetadata

logger = logging.getLogger(__name__)


def read_doc(
    shard_f,
    offset_f,
    sub_shard_f,
    sub_offset_f,
    docs_offset={},
    docs_metadata=DocMetadata.default(),
    minmax_stat=None,
):
    """
    If `minmax_stat` is specified then we can update with the specific stats
    """
    local_doc_count = 0
    local_doc_count = struct.unpack(
        SIZE_KEY["doc_count"],
        sub_offset_f.read(READ_SIZE_KEY[SIZE_KEY["doc_count"]]),
    )[0]
    _ = struct.unpack(
        SIZE_KEY["offset"],
        sub_offset_f.read(READ_SIZE_KEY[SIZE_KEY["offset"]]),
    )[0]  # sum_doc_length

    for _ in range(local_doc_count):
        doc_length, score, view_count, owneruserid = 0, 0, 0, 0
        answer_count, comment_count, favorite_count = 0, 0, 0
        size_display_name, ownerdisplayname = 0, ""
        raw_tag_size, tags = 0, ""
        creation_date = 0
        has_accepted_answer = 0
        user_reputation = 0
        raw_body_size, body = 0, ""
        raw_title_size, title = 0, ""

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

        try:
            sub_shard_f.seek(offset)
            doc_length = struct.unpack(
                SIZE_KEY["doc_length"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_length"]]),
            )[0]
            # CUSTOM SCORE
            _ = struct.unpack(
                SIZE_KEY["doc_metadatascore"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_metadatascore"]]),
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

            creation_date = struct.unpack(
                SIZE_KEY["doc_creationdate"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_creationdate"]]),
            )[0]

            has_accepted_answer = struct.unpack(
                SIZE_KEY["doc_hasacceptedanswer"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_hasacceptedanswer"]]),
            )[0]

            user_reputation = struct.unpack(
                SIZE_KEY["doc_userreputation"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_userreputation"]]),
            )[0]

            raw_title_size = struct.unpack(
                SIZE_KEY["doc_title"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_title"]]),
            )[0]
            title = sub_shard_f.read(raw_title_size).decode("utf-8")
            raw_body_size = struct.unpack(
                SIZE_KEY["doc_body"],
                sub_shard_f.read(READ_SIZE_KEY[SIZE_KEY["doc_body"]]),
            )[0]
            body = sub_shard_f.read(raw_body_size).decode("utf-8")

        except (struct.error, UnicodeDecodeError) as e:
            import traceback

            logger.error(
                f"Error reading {shard_f} after length {len(docs_offset)} {e} {local_doc_count}"
            )
            logger.error(f"Doc length: {doc_length}")
            logger.error(f"Doc ID: {doc_id}")
            logger.error(f"Doc Offset: {offset}")
            logger.error(f"Doc Score: {score}")
            logger.error(f"Doc View Count: {view_count}")
            logger.error(f"Doc Owner User ID: {owneruserid}")
            logger.error(f"Doc Answer Count: {answer_count}")
            logger.error(f"Doc Comment Count: {comment_count}")
            logger.error(f"Doc Favorite Count: {favorite_count}")
            logger.error(f"Doc Owner Display Name: {size_display_name}")
            logger.error(f"Doc Owner Display Name: {ownerdisplayname}")
            logger.error(f"Doc Tags: {raw_tag_size}")
            logger.error(f"Doc Tags: {tags}")
            logger.error(f"Doc Creation Date: {creation_date}")
            logger.error(f"Doc Has Accepted Answer: {has_accepted_answer}")
            logger.error(f"Doc User Reputation: {user_reputation}")
            logger.error(f"Doc Title: {raw_title_size}")
            logger.error(f"Doc Title: {title}")
            logger.error(f"Doc Body: {raw_body_size}")
            logger.error(f"Doc Body: {body}")
            logger.error(traceback.format_exc())
            raise ValueError("Error reading doc")

        docs_offset[int(doc_id)] = (shard_f.tell(), doc_length)

        shard_f.write(struct.pack(SIZE_KEY["doc_length"], doc_length))
        doc_custom_score = 0
        if minmax_stat is not None:
            doc_custom_score = metadata_score(
                score=score,
                viewcount=view_count,
                creationdate=creation_date,
                answercount=answer_count,
                commentcount=comment_count,
                favoritecount=favorite_count,
                userreputation=user_reputation,
                minmax_dict=minmax_stat,
            )
        shard_f.write(struct.pack(SIZE_KEY["doc_metadatascore"], doc_custom_score))
        shard_f.write(struct.pack(SIZE_KEY["doc_score"], score))
        shard_f.write(struct.pack(SIZE_KEY["doc_viewcount"], view_count))
        shard_f.write(struct.pack(SIZE_KEY["doc_owneruserid"], owneruserid))
        shard_f.write(struct.pack(SIZE_KEY["doc_answercount"], answer_count))
        shard_f.write(struct.pack(SIZE_KEY["doc_commentcount"], comment_count))
        shard_f.write(struct.pack(SIZE_KEY["doc_favoritecount"], favorite_count))
        shard_f.write(struct.pack(SIZE_KEY["doc_ownerdisplayname"], size_display_name))
        shard_f.write(ownerdisplayname.encode("utf-8"))
        shard_f.write(struct.pack(SIZE_KEY["doc_tags"], raw_tag_size))
        shard_f.write(tags.encode("utf-8"))

        shard_f.write(struct.pack(SIZE_KEY["doc_creationdate"], creation_date))
        shard_f.write(
            struct.pack(SIZE_KEY["doc_hasacceptedanswer"], has_accepted_answer)
        )
        shard_f.write(struct.pack(SIZE_KEY["doc_userreputation"], user_reputation))
        shard_f.write(struct.pack(SIZE_KEY["doc_title"], raw_title_size))
        shard_f.write(title.encode("utf-8"))
        shard_f.write(struct.pack(SIZE_KEY["doc_body"], raw_body_size))
        shard_f.write(body.encode("utf-8"))

        docs_metadata.update_with_raw(
            doc_length=doc_length,
            creationdate=creation_date,
            score=score,
            viewcount=view_count,
            owneruserid=owneruserid,
            answercount=answer_count,
            commentcount=comment_count,
            favoritecount=favorite_count,
            userreputation=user_reputation,
        )
    return docs_offset, docs_metadata
