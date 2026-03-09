from lab.bm25 import BM25
from lab.text import tokenize


def test_lucene_score_on_three_docs():
    # N=3, запрос «кот», df=2, avgdl=8/3, k1=1.2, b=0.75
    # idf = ln(1.6) = 0.47000362924573563
    # d1 tf=1 dl=2 -> 0.523548346501579
    # d2 tf=2 dl=2 -> 0.6951314180441133
    docs = [
        tokenize("кот сидел"),
        tokenize("кот кот"),
        tokenize("пёс сидел на ковре"),
    ]
    index = BM25(docs)
    scores = index.scores(["кот"], k1=1.2, b=0.75)
    assert abs(scores[0] - 0.523548346501579) < 1e-12
    assert abs(scores[1] - 0.6951314180441133) < 1e-12
    assert 2 not in scores
    assert scores[1] > scores[0]


def test_yo_folds_to_e():
    assert tokenize("Ёж, Ёж!") == ["еж", "еж"]
