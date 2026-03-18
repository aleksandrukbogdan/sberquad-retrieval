from lab.chunks import covers, windows


def test_cut_span_is_not_relevant():
    text = "abcdefghij" * 2
    spans = windows(text, 10, 0)
    assert spans == [(0, "abcdefghij"), (10, "abcdefghij")]
    assert covers(0, 10, 8, 12) is False
    assert covers(10, 20, 8, 12) is False
    assert covers(0, 10, 2, 5) is True


def test_short_text_is_one_window():
    assert windows("абв", 10, 2) == [(0, "абв")]
