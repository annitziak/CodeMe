DEFAULT_WEIGHTS = {
    "score": 1.5,
    "viewcount": 1.2,
    "creationdate": 1.5,
    "reputationuser": 1.5,
}


def normalize(data, minmax_dict, key):
    min_key = f"{key}_min"
    max_key = f"{key}_max"

    if min_key not in minmax_dict:
        return 0
    if max_key not in minmax_dict:
        return 0
    
    if minmax_dict[max_key] == minmax_dict[min_key]:
        return 0
    
    return (data - minmax_dict[min_key]) / (minmax_dict[max_key] - minmax_dict[min_key])


def metadata_score(
    score=0,
    viewcount=0,
    creationdate=0,
    reputationuser=0,
    answercount=0,
    commentcount=0,
    favoritecount=0,
    minmax_dict={},
    weights=DEFAULT_WEIGHTS,
):
    score = 0
    if "score" in weights:
        score += normalize(score, minmax_dict, "score") * weights["score"]
    if "viewcount" in weights:
        score += normalize(viewcount, minmax_dict, "viewcount") * weights["viewcount"]
    if "creationdate" in weights:
        score += (
            normalize(creationdate, minmax_dict, "creationdate")
            * weights["creationdate"]
        )
    if "reputationuser" in weights:
        score += (
            normalize(reputationuser, minmax_dict, "reputationuser")
            * weights["reputationuser"]
        )

    return score
