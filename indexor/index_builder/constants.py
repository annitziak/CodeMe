SIZE_KEY = {
    "offset": "<Q",
    "pos_offset": "<QQ",
    "offset_shard": "<HQ",
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
}

READ_SIZE_KEY = {
    "<Q": 8,
    "<I": 4,
    "<H": 2,
    "<IH": 6,
    "<HQ": 10,
}
