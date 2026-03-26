import json

from lab.bm25 import BM25, top_docs
from lab.data import ROOT, load_config
from lab.hashvec import embed, search
from lab.metrics import summarize
from lab.rerank import apply_linear, load_weights
from lab.rrf import rrf
from lab.text import tokenize


def _read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _qrels(questions):
    return {q["id"]: [q["passage_id"]] for q in questions}


def run_mini():
    cfg = load_config()
    mini = ROOT / "data" / "mini"
    passages = _read_jsonl(mini / "passages.jsonl")
    questions = _read_jsonl(mini / "questions.jsonl")
    hyde_rows = _read_jsonl(mini / "hyde.jsonl")
    hyde = {row["id"]: row["text"] for row in hyde_rows}
    missing = [q["id"] for q in questions if q["id"] not in hyde]
    if missing:
        raise SystemExit(f"mini hyde cache missing {missing[:3]}")
    doc_ids = [p["id"] for p in passages]
    doc_tokens = [tokenize(p["text"]) for p in passages]
    index = BM25(doc_tokens)
    k1 = cfg["bm25"]["k1"]
    b = cfg["bm25"]["b"]
    topn = cfg["retrieve_topk"]
    bm25_ranks = {}
    bm25_scores = {}
    for question in questions:
        scores = index.scores(tokenize(question["text"]), k1, b)
        bm25_ranks[question["id"]] = top_docs(scores, doc_ids, topn)
        bm25_scores[question["id"]] = {doc_ids[i]: scores[i] for i in scores}
    doc_vecs = [embed(p["text"], tokenize, seed=cfg["seed"]) for p in passages]
    q_vecs = [embed(q["text"], tokenize, seed=cfg["seed"]) for q in questions]
    h_vecs = [embed(hyde[q["id"]] or q["text"], tokenize, seed=cfg["seed"]) for q in questions]
    dense_ranks = {}
    dense_scores = {}
    hyde_ranks = {}
    mix_ranks = {}
    for question, q_vec, h_vec in zip(questions, q_vecs, h_vecs):
        qid = question["id"]
        dense_ranks[qid] = search(doc_vecs, q_vec, doc_ids, topn)
        hyde_ranks[qid] = search(doc_vecs, h_vec, doc_ids, topn)
        mixed = [a + b for a, b in zip(q_vec, h_vec)]
        norm = sum(value * value for value in mixed) ** 0.5 or 1.0
        mixed = [value / norm for value in mixed]
        mix_ranks[qid] = search(doc_vecs, mixed, doc_ids, topn)
        dense_scores[qid] = {
            doc_id: sum(a * b for a, b in zip(vec, q_vec))
            for doc_id, vec in zip(doc_ids, doc_vecs)
        }
    fused = {}
    for question in questions:
        qid = question["id"]
        order, _ = rrf(
            [bm25_ranks[qid], dense_ranks[qid]],
            k=cfg["rrf"]["k"],
            depth=cfg["rrf"]["depth"],
        )
        fused[qid] = order
    doc_token_map = {p["id"]: tokenize(p["text"]) for p in passages}
    linear_ranks = apply_linear(
        questions,
        fused,
        bm25_scores,
        dense_scores,
        doc_token_map,
        load_weights(),
        cfg["rerank"]["depth"],
    )
    qrels = _qrels(questions)
    systems = {
        "bm25": bm25_ranks,
        "dense": dense_ranks,
        "rrf": fused,
        "rrf_linear": linear_ranks,
        "hyde": hyde_ranks,
        "hyde_mix": mix_ranks,
    }
    packed = {}
    for name, ranks in systems.items():
        cut = {qid: ranked[: cfg["store_topk"]] for qid, ranked in ranks.items()}
        packed[name] = {"rankings": cut, **summarize(cut, qrels)}
    return {
        "encoder": "hash",
        "seed": cfg["seed"],
        "qrels": qrels,
        "systems": packed,
    }


def write_baseline(report):
    path = ROOT / "reports" / "mini_baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def check_baseline(report):
    path = ROOT / "reports" / "mini_baseline.json"
    if not path.exists():
        raise SystemExit(f"нет {path}")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for name, block in baseline["systems"].items():
        got = report["systems"][name]
        if got["rankings"] != block["rankings"]:
            problems.append(f"{name} rankings")
        for key in ("recall@5", "recall@10", "mrr@10", "ndcg@10"):
            if abs(got[key] - block[key]) > 1e-9:
                problems.append(f"{name} {key}")
    if problems:
        raise SystemExit("mini regression: " + ", ".join(problems))
    print("mini ok", flush=True)
