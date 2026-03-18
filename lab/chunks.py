def windows(text, size, overlap):
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    if len(text) <= size:
        return [(0, text)]
    step = size - overlap
    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        out.append((start, text[start:end]))
        if end == len(text):
            break
        start += step
    return out


def covers(chunk_start, chunk_end, answer_start, answer_end):
    # Спан должен лежать внутри целиком. Разрез по границе — не релевантный чанк.
    return chunk_start <= answer_start and answer_end <= chunk_end
