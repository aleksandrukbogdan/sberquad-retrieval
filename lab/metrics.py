import math


def recall_at(ranked, relevant, k):
    if not relevant:
        return 0.0
    hit = sum(1 for doc_id in ranked[:k] if doc_id in relevant)
    return hit / len(relevant)


def mrr_at(ranked, relevant, k):
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked, relevant, k):
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal = min(len(relevant), k)
    if ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg


def summarize(rankings, qrels):
    acc = {"recall@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0}
    n = len(qrels)
    if n == 0:
        return acc
    for qid, rel in qrels.items():
        ranked = rankings[qid]
        relset = set(rel)
        acc["recall@5"] += recall_at(ranked, relset, 5)
        acc["recall@10"] += recall_at(ranked, relset, 10)
        acc["mrr@10"] += mrr_at(ranked, relset, 10)
        acc["ndcg@10"] += ndcg_at(ranked, relset, 10)
    return {key: value / n for key, value in acc.items()}
