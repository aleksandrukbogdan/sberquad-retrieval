def rrf(lists, k=60, depth=100):
    # Ранг с единицы. Документ из одного списка всё равно получает 1/(k+rank).
    scores = {}
    for ranked in lists:
        for rank, doc_id in enumerate(ranked[:depth], start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    order = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return order, scores
