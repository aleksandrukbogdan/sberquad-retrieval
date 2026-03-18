import json
from itertools import product
from pathlib import Path

from lab.text import tokenize

FEATURES = ("bm25", "dense", "unigram", "bigram")
WEIGHTS_PATH = Path(__file__).with_name("rerank_weights.json")


def _minmax(values):
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.0] * len(values)
    span = hi - lo
    return [(value - lo) / span for value in values]


def _grams(tokens, n):
    if n == 1:
        return set(tokens)
    return set(zip(tokens, tokens[1:]))


def coverage(query_tokens, doc_tokens, n):
    query_grams = _grams(query_tokens, n)
    if not query_grams:
        return 0.0
    return len(query_grams & _grams(doc_tokens, n)) / len(query_grams)


def score_pool(cands, weights):
    # cands: list of {id, bm25, dense, unigram, bigram}
    if not cands:
        return []
    columns = {name: _minmax([cand[name] for cand in cands]) for name in FEATURES}
    scored = []
    for i, cand in enumerate(cands):
        total = 0.0
        for name in FEATURES:
            total += weights[name] * columns[name][i]
        scored.append((total, cand["id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc_id for _, doc_id in scored]


def load_weights():
    return json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))


def save_weights(weights):
    WEIGHTS_PATH.write_text(
        json.dumps({name: int(weights[name]) for name in FEATURES}, indent=2) + "\n",
        encoding="utf-8",
    )


def grid_weights(pools, qrels, depths):
    # pools[qid] = candidates in fused order, raw features, already long enough
    best = None
    rows = []
    for depth in depths:
        depth_best = None
        for combo in product((0, 1, 2), repeat=len(FEATURES)):
            if sum(combo) == 0:
                continue
            weights = dict(zip(FEATURES, combo))
            rankings = {}
            for qid, cands in pools.items():
                head = cands[:depth]
                reranked = score_pool(head, weights)
                seen = set(reranked)
                tail = [cand["id"] for cand in cands if cand["id"] not in seen]
                rankings[qid] = reranked + tail
            from lab.metrics import summarize

            stats = summarize(rankings, qrels)
            row = {"depth": depth, "weights": weights, **stats}
            if depth_best is None or _better(row, depth_best):
                depth_best = row
        rows.append(
            {
                "depth": depth,
                "mrr@10": depth_best["mrr@10"],
                "recall@10": depth_best["recall@10"],
                "weights": depth_best["weights"],
            }
        )
        if best is None or _better(depth_best, best):
            best = depth_best
    return best, rows


def _better(row, other):
    return (row["mrr@10"], row["recall@10"], -sum(row["weights"].values())) > (
        other["mrr@10"],
        other["recall@10"],
        -sum(other["weights"].values()),
    )


def build_candidates(question, fused_ids, bm25_scores, dense_scores, doc_tokens, q_tokens):
    cands = []
    for doc_id in fused_ids:
        cands.append(
            {
                "id": doc_id,
                "bm25": bm25_scores.get(doc_id, 0.0),
                "dense": dense_scores.get(doc_id, 0.0),
                "unigram": coverage(q_tokens, doc_tokens[doc_id], 1),
                "bigram": coverage(q_tokens, doc_tokens[doc_id], 2),
            }
        )
    return cands


def apply_linear(questions, fused, bm25_by_q, dense_by_q, doc_tokens, weights, depth):
    rankings = {}
    for question in questions:
        qid = question["id"]
        cands = build_candidates(
            question,
            fused[qid],
            bm25_by_q[qid],
            dense_by_q[qid],
            doc_tokens,
            tokenize(question["text"]),
        )
        head = cands[:depth]
        reranked = score_pool(head, weights)
        seen = set(reranked)
        tail = [doc_id for doc_id in fused[qid] if doc_id not in seen]
        rankings[qid] = reranked + tail
    return rankings


def _load_ce(model_name, revision):
    import os

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
    if torch.cuda.is_available():
        model.to("cuda")
        print("ce device cuda", flush=True)
    else:
        print("ce device cpu", flush=True)
    model.eval()
    return tokenizer, model


def _ce_forward(pairs, tokenizer, model, max_length, batch_size, logit_sign):
    import torch

    scores = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        left = [item[0] for item in batch]
        right = [item[1] for item in batch]
        encoded = tokenizer(
            left,
            right,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
        if logits.size(-1) == 1:
            batch_scores = logits.squeeze(-1)
        else:
            batch_scores = logits[:, -1]
        scores.extend((logit_sign * batch_scores).tolist())
        if start % (batch_size * 20) == 0:
            print(f"ce {start}/{len(pairs)}", flush=True)
    return scores


def ce_scores(pairs, model_name, revision, max_length, batch_size, logit_sign):
    tokenizer, model = _load_ce(model_name, revision)
    try:
        return _ce_forward(pairs, tokenizer, model, max_length, batch_size, logit_sign)
    finally:
        del model


def rerank_ce(questions, fused, passages, depth, cfg):
    by_id = {passage["id"]: passage["text"] for passage in passages}
    block = cfg["reranker_ce"]
    pairs = []
    spans = []
    for question in questions:
        head = fused[question["id"]][:depth]
        spans.append(head)
        for doc_id in head:
            pairs.append((question["text"], by_id[doc_id]))
    raw = ce_scores(
        pairs,
        block["model"],
        block["revision"],
        block["max_length"],
        block["batch_size"],
        block["logit_sign"],
    )
    rankings = {}
    cursor = 0
    for question, head in zip(questions, spans):
        chunk = raw[cursor : cursor + len(head)]
        cursor += len(head)
        order = sorted(range(len(head)), key=lambda i: (-chunk[i], head[i]))
        reranked = [head[i] for i in order]
        seen = set(reranked)
        tail = [doc_id for doc_id in fused[question["id"]] if doc_id not in seen]
        rankings[question["id"]] = reranked + tail
    return rankings


def logit_sign(questions, passages, cfg, n=30):
    import random

    by_id = {passage["id"]: passage["text"] for passage in passages}
    ids = [passage["id"] for passage in passages]
    rng = random.Random(cfg["seed"])
    gold_pairs = []
    bad_pairs = []
    for question in questions[:n]:
        gold_pairs.append((question["text"], by_id[question["passage_id"]]))
        bad = rng.choice(ids)
        while bad == question["passage_id"]:
            bad = rng.choice(ids)
        bad_pairs.append((question["text"], by_id[bad]))
    block = cfg["reranker_ce"]
    tokenizer, model = _load_ce(block["model"], block["revision"])
    try:
        gold_scores = _ce_forward(
            gold_pairs, tokenizer, model, block["max_length"], block["batch_size"], 1
        )
        bad_scores = _ce_forward(
            bad_pairs, tokenizer, model, block["max_length"], block["batch_size"], 1
        )
    finally:
        del model
    gold_mean = sum(gold_scores) / len(gold_scores)
    bad_mean = sum(bad_scores) / len(bad_scores)
    sign = 1 if gold_mean >= bad_mean else -1
    print(f"ce sign {sign} gold {gold_mean:.3f} bad {bad_mean:.3f}", flush=True)
    return sign
