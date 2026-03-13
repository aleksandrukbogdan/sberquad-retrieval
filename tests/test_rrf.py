from lab.rrf import rrf


def test_rrf_k60():
    order, scores = rrf([["d1", "d2", "d3"], ["d2", "d4"]], k=60)
    assert order == ["d2", "d1", "d4", "d3"]
    assert abs(scores["d1"] - (1 / 61)) < 1e-12
    assert abs(scores["d2"] - (1 / 62 + 1 / 61)) < 1e-12
    assert abs(scores["d4"] - (1 / 62)) < 1e-12
    assert abs(scores["d3"] - (1 / 63)) < 1e-12
