import hashlib
import math


def _xorshift(token, dim, seed):
    # Свой RNG, чтобы mini-прогон не зависел от версии numpy.
    digest = hashlib.sha256(f"{seed}:{token}".encode("utf-8")).digest()
    x = int.from_bytes(digest[:8], "little") or 1
    out = []
    for _ in range(dim):
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 7
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        x &= 0xFFFFFFFFFFFFFFFF
        out.append(((x >> 11) & 0xFFFFFFFF) / 2147483648.0 - 1.0)
    return out


def embed(text, tokenize, dim=64, seed=42):
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * dim
    acc = [0.0] * dim
    for token in tokens:
        vec = _xorshift(token, dim, seed)
        for i, value in enumerate(vec):
            acc[i] += value
    inv = 1.0 / len(tokens)
    for i in range(dim):
        acc[i] *= inv
    norm = math.sqrt(sum(value * value for value in acc))
    if norm == 0.0:
        return acc
    return [value / norm for value in acc]


def search(doc_vecs, query_vec, doc_ids, topn):
    scored = []
    for doc_id, vec in zip(doc_ids, doc_vecs):
        score = 0.0
        for a, b in zip(vec, query_vec):
            score += a * b
        scored.append((score, doc_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc_id for _, doc_id in scored[:topn]]
