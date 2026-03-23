from lab.text import tokenize


def _rank(ranked, doc_id):
    for i, item in enumerate(ranked, start=1):
        if item == doc_id:
            return i
    return None


def _head(text, n=220):
    return " ".join(text.split())[:n]


def _soft_hits(query, passage):
    q_tokens = tokenize(query)
    p_tokens = set(tokenize(passage))
    exact = 0
    soft = 0
    for token in q_tokens:
        if token in p_tokens:
            exact += 1
            continue
        if len(token) < 4:
            continue
        for other in p_tokens:
            if len(other) >= 4 and (other.startswith(token) or token.startswith(other)):
                soft += 1
                break
    return exact, soft, len(q_tokens)


def collect(questions, passages, ranks, hyde_texts):
    by_id = {p["id"]: p["text"] for p in passages}
    names = list(ranks)
    counts = {
        "morph": 0,
        "hyde_drift": 0,
        "rrf_agree": 0,
        "all_miss": 0,
    }
    rows = []
    all_miss_kept = 0
    for question in questions:
        qid = question["id"]
        gold = question["passage_id"]
        per = {name: _rank(ranks[name][qid], gold) for name in names}
        exact, soft, n_q = _soft_hits(question["text"], by_id[gold])
        tags = []
        dense_rank = per.get("dense")
        bm25_rank = per.get("bm25")
        hyde_rank = per.get("hyde")
        rrf_rank = per.get("rrf")
        if (
            n_q
            and exact / n_q < 0.34
            and soft > 0
            and (bm25_rank is None or bm25_rank > 10)
            and dense_rank is not None
            and dense_rank <= 10
        ):
            tags.append("morph")
        if dense_rank is not None and dense_rank <= 3 and (hyde_rank is None or hyde_rank > 10):
            tags.append("hyde_drift")
        rrf_top = ranks["rrf"][qid][0] if ranks.get("rrf", {}).get(qid) else None
        top_detail = {}
        if rrf_top:
            top_detail = {
                "id": rrf_top,
                "bm25_rank": _rank(ranks["bm25"][qid], rrf_top),
                "dense_rank": _rank(ranks["dense"][qid], rrf_top),
                "head": _head(by_id.get(rrf_top, "")),
            }
            if (
                bm25_rank == 1
                and (rrf_rank is None or rrf_rank > 1)
                and top_detail["bm25_rank"] is not None
                and top_detail["dense_rank"] is not None
                and top_detail["bm25_rank"] <= 10
                and top_detail["dense_rank"] <= 10
            ):
                tags.append("rrf_agree")
        in_top10 = any(rank is not None and rank <= 10 for rank in per.values())
        if not in_top10:
            counts["all_miss"] += 1
            if all_miss_kept < 15:
                tags.append("all_miss")
                all_miss_kept += 1
        for tag in tags:
            if tag != "all_miss":
                counts[tag] += 1
        if not tags:
            continue
        tops = {}
        for name in names:
            top_id = ranks[name][qid][0]
            tops[name] = {"id": top_id, "head": _head(by_id.get(top_id, ""))}
        rows.append(
            {
                "qid": qid,
                "question": question["text"],
                "answer": question["answer"],
                "gold_id": gold,
                "gold_head": _head(by_id[gold]),
                "ranks": per,
                "exact_overlap": exact,
                "soft_overlap": soft,
                "query_tokens": n_q,
                "tops": tops,
                "rrf_top": top_detail,
                "hyde": hyde_texts.get(qid, ""),
                "tags": tags,
            }
        )
    return rows, counts
