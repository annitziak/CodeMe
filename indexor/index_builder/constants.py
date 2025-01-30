SIZE_KEY = {
    "offset": "<Q",
    "postings_count": "<I",
    "deltaTF": "<IH",
    "position_count": "<H",
    "position_delta": "<H",
    "df": "<I",
    "tf": "<H",
    "term_bytes": "<I",
}

READ_SIZE_KEY = {
    "<Q": 8,
    "<I": 4,
    "<H": 2,
    "<IH": 6,
}
