from lab.metrics import summarize


def test_known_ranking():
    # Релевантный на позиции 2. nDCG@10 = 1/log2(3).
    rankings = {"q": ["a", "b", "c"]}
    qrels = {"q": ["b"]}
    stats = summarize(rankings, qrels)
    assert stats["recall@5"] == 1.0
    assert abs(stats["mrr@10"] - 0.5) < 1e-12
    assert abs(stats["ndcg@10"] - 0.6309297535714574) < 1e-12


def test_hit_at_rank_one():
    stats = summarize({"q": ["b", "a"]}, {"q": ["b"]})
    assert stats["recall@5"] == 1.0
    assert stats["recall@10"] == 1.0
    assert stats["mrr@10"] == 1.0
    assert stats["ndcg@10"] == 1.0


def test_miss():
    stats = summarize({"q": ["a", "c"]}, {"q": ["b"]})
    assert stats["recall@10"] == 0.0
    assert stats["mrr@10"] == 0.0
    assert stats["ndcg@10"] == 0.0
