SIZE_KEY = {
    "offset": "<Q",
    "offset_doc_length": "<QI",
    "pos_offset": "<QQ",
    "offset_shard": "<HQ",
    "offset_shard_doc_length": "<HQI",
    "pos_offset_shard": "<HQQ",
    "postings_count": "<I",
    "deltaTF": "<IH",
    "position_count": "<H",
    "position_delta": "<H",
    "df": "<I",
    "tf": "<H",
    "term_bytes": "<I",
    "doc_id": "<Q",
    "doc_length": "<I",
    "doc_count": "<I",
    "doc_score": "<i",
    "doc_viewcount": "<I",
    "doc_owneruserid": "<Q",
    "doc_ownerdisplayname": "<I",
    "doc_tags": "<I",
    "doc_answercount": "<I",
    "doc_commentcount": "<I",
    "doc_favoritecount": "<I",
    "doc_creationdate": "<I",
    "doc_metadatascore": "<f",
    "doc_hasacceptedanswer": "<H",
    "doc_title": "<I",
    "doc_body": "<I",
}

READ_SIZE_KEY = {
    "<Q": 8,
    "<I": 4,
    "<H": 2,
    "<IH": 6,
    "<HQ": 10,
    "<i": 4,
    "<f": 4,
    "<QI": 12,
    "<HQI": 14,
}
