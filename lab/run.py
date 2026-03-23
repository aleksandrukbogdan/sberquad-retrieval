import json

from lab.bm25 import BM25, top_docs
from lab.chunks import covers, windows
from lab.data import (
    ROOT,
    ensure_splits,
    load_config,
    load_manifest,
    load_subset,
    qrels_of,
    save_config,
)
from lab.failures import collect
from lab.metrics import summarize
from lab.models import dense_search, encode_many, generate_hyde, mix_vectors, pin_revisions
from lab.report import SYSTEMS, pack, write_json
from lab.rerank import apply_linear, grid_weights, logit_sign, rerank_ce, save_weights
from lab.rrf import rrf
from lab.text import tokenize


def _show(title, stats):
    print(
        f"{title} R@5={stats['recall@5']:.3f} R@10={stats['recall@10']:.3f} "
        f"MRR={stats['mrr@10']:.3f} nDCG={stats['ndcg@10']:.3f}",
        flush=True,
    )


def _pick(rows, *keys):
    def sort_key(row):
        head = [row["mrr@10"], row["recall@10"]]
        tail = [-row[key] for key in keys]
        return tuple(head + tail)

    return max(rows, key=sort_key)


def _bm25_index(passages):
    ids = [p["id"] for p in passages]
    index = BM25([tokenize(p["text"]) for p in passages])
    return index, ids


def _bm25_ranks(index, ids, questions, k1, b, topn):
    ranks = {}
    scores = {}
    for question in questions:
        raw = index.scores(tokenize(question["text"]), k1, b)
        ranks[question["id"]] = top_docs(raw, ids, topn)
        scores[question["id"]] = {ids[i]: float(raw[i]) for i in raw}
    return ranks, scores


def _fuse(questions, left, right, k, depth):
    fused = {}
    for question in questions:
        qid = question["id"]
        order, _ = rrf([left[qid], right[qid]], k=k, depth=depth)
        fused[qid] = order
    return fused


def _dense_for(sims, questions, dids, wanted):
    id_to_i = {doc_id: i for i, doc_id in enumerate(dids)}
    out = {}
    for qi, question in enumerate(questions):
        qid = question["id"]
        row = sims[qi]
        out[qid] = {doc_id: float(row[id_to_i[doc_id]]) for doc_id in wanted[qid]}
    return out


def _hyde_texts(questions, mapping):
    texts = []
    empty = 0
    for question in questions:
        text = (mapping.get(question["id"]) or "").strip()
        if not text:
            empty += 1
            text = question["text"]
        texts.append(text)
    if empty:
        print(f"hyde empty {empty}", flush=True)
    return texts


def _chunk_break(passages, questions, para_ranks, k1, b, overlap, sizes):
    by_id = {p["id"]: p["text"] for p in passages}
    found = None
    rows = []
    for size in sizes:
        chunks = []
        rel = {q["id"]: [] for q in questions}
        for passage in passages:
            for start, text in windows(passage["text"], size, overlap):
                chunks.append({"id": f"{passage['id']}:{start}", "text": text})
        chunk_ids = [c["id"] for c in chunks]
        index = BM25([tokenize(c["text"]) for c in chunks])
        for question in questions:
            for start, text in windows(by_id[question["passage_id"]], size, overlap):
                end = start + len(text)
                if covers(start, end, question["answer_start"], question["answer_end"]):
                    rel[question["id"]].append(f"{question['passage_id']}:{start}")
        ranks, _ = _bm25_ranks(index, chunk_ids, questions, k1, b, 10)
        stats = summarize(ranks, rel)
        rows.append(
            {
                "mode": "window",
                "size": size,
                "overlap": overlap,
                "n_chunks": len(chunks),
                **stats,
            }
        )
        _show(f"chunk {size}", stats)
        if found is not None:
            continue
        for question in questions:
            gold = question["passage_id"]
            paragraph_rank = None
            for i, doc_id in enumerate(para_ranks[question["id"]], start=1):
                if doc_id == gold:
                    paragraph_rank = i
                    break
            if paragraph_rank is None or paragraph_rank > 5:
                continue
            if question["answer_end"] - question["answer_start"] >= size:
                continue
            covered = False
            for start, text in windows(by_id[gold], size, overlap):
                if covers(start, start + len(text), question["answer_start"], question["answer_end"]):
                    covered = True
                    break
            if covered:
                continue
            text = by_id[gold]
            a0 = question["answer_start"]
            a1 = question["answer_end"]
            found = {
                "qid": question["id"],
                "question": question["text"],
                "answer": question["answer"],
                "size": size,
                "paragraph_rank": paragraph_rank,
                "answer_start": a0,
                "passage_len": len(text),
                "snippet": text[max(0, a0 - 80) : a1 + 80],
            }
    return rows, found


def _write_mini_hyde(mapping):
    path = ROOT / "data" / "mini" / "questions.jsonl"
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(json.loads(line))
    lines = []
    for question in questions:
        if question["id"] not in mapping:
            raise SystemExit(f"нет hyde для mini {question['id']}")
        lines.append(json.dumps({"id": question["id"], "text": mapping[question["id"]]}, ensure_ascii=False))
    (ROOT / "data" / "mini" / "hyde.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _systems_from(ranks, qrels, store):
    packed = {}
    for name in SYSTEMS:
        packed[name] = pack(ranks[name], qrels, store)
        _show(name, packed[name])
    return packed


def run_tune(rebuild_hyde=False):
    cfg = pin_revisions(load_config())
    splits = ensure_splits(cfg)
    passages, questions = load_subset("train", splits["tune_passage_ids"], splits["tune_question_ids"])
    qrels = qrels_of(questions)
    topn = cfg["retrieve_topk"]
    print(f"tune questions {len(questions)} passages {len(passages)}", flush=True)

    index, ids = _bm25_index(passages)
    bm25_grid = []
    for k1 in (0.6, 1.2, 1.8, 2.0):
        for b in (0.3, 0.5, 0.75, 1.0):
            ranks, _ = _bm25_ranks(index, ids, questions, k1, b, 10)
            stats = summarize(ranks, qrels)
            bm25_grid.append({"k1": k1, "b": b, **stats})
            _show(f"bm25 k1={k1} b={b}", stats)
    chosen_bm25 = _pick(bm25_grid, "k1", "b")
    cfg["bm25"]["k1"] = chosen_bm25["k1"]
    cfg["bm25"]["b"] = chosen_bm25["b"]
    print(f"chosen bm25 {cfg['bm25']}", flush=True)

    q_prefix = cfg["dense"]["query_prefix"]
    p_prefix = cfg["dense"]["passage_prefix"]
    doc_texts = [p["text"] for p in passages]
    query_texts = [q["text"] for q in questions]
    doc_pref, query_pref, doc_raw, query_raw = encode_many(
        [
            {"texts": doc_texts, "prefix": p_prefix, "role": "passage-prefix"},
            {"texts": query_texts, "prefix": q_prefix, "role": "query-prefix"},
            {"texts": doc_texts, "prefix": "", "role": "passage-raw"},
            {"texts": query_texts, "prefix": "", "role": "query-raw"},
        ],
        cfg,
    )
    dids = [p["id"] for p in passages]
    qids = [q["id"] for q in questions]
    dense_ranks, dense_sims = dense_search(query_pref, doc_pref, qids, dids, topn)
    raw_ranks, _ = dense_search(query_raw, doc_raw, qids, dids, topn)
    prefix_stats = {
        "with": summarize(dense_ranks, qrels),
        "without": summarize(raw_ranks, qrels),
    }
    _show("dense prefix", prefix_stats["with"])
    _show("dense raw", prefix_stats["without"])

    bm25_ranks, bm25_scores = _bm25_ranks(
        index, ids, questions, cfg["bm25"]["k1"], cfg["bm25"]["b"], topn
    )
    rrf_grid = []
    for k in (10, 30, 60, 100):
        for depth in (20, 50, 100):
            fused = _fuse(questions, bm25_ranks, dense_ranks, k, depth)
            stats = summarize(fused, qrels)
            rrf_grid.append({"k": k, "depth": depth, **stats})
            _show(f"rrf k={k} depth={depth}", stats)
    chosen_rrf = _pick(rrf_grid, "k", "depth")
    cfg["rrf"]["k"] = chosen_rrf["k"]
    cfg["rrf"]["depth"] = chosen_rrf["depth"]
    fused = _fuse(questions, bm25_ranks, dense_ranks, cfg["rrf"]["k"], cfg["rrf"]["depth"])
    print(f"chosen rrf {cfg['rrf']}", flush=True)

    doc_tokens = {p["id"]: tokenize(p["text"]) for p in passages}
    dense_scores = _dense_for(dense_sims, questions, dids, fused)
    pools = {}
    for question in questions:
        from lab.rerank import build_candidates

        pools[question["id"]] = build_candidates(
            question,
            fused[question["id"]],
            bm25_scores[question["id"]],
            dense_scores[question["id"]],
            doc_tokens,
            tokenize(question["text"]),
        )
    best_linear, depth_rows = grid_weights(pools, qrels, (5, 20, 50))
    cfg["rerank"]["depth"] = best_linear["depth"]
    save_weights(best_linear["weights"])
    print(f"chosen linear depth={best_linear['depth']} {best_linear['weights']}", flush=True)
    linear_ranks = apply_linear(
        questions,
        fused,
        bm25_scores,
        dense_scores,
        doc_tokens,
        best_linear["weights"],
        best_linear["depth"],
    )

    mapping = generate_hyde(questions, cfg, rebuild=rebuild_hyde)
    hyde_input = _hyde_texts(questions, mapping)
    hyde_vecs = encode_many(
        [{"texts": hyde_input, "prefix": q_prefix, "role": "hyde-query"}],
        cfg,
    )[0]
    hyde_ranks, _ = dense_search(hyde_vecs, doc_pref, qids, dids, topn)
    mix = mix_vectors(query_pref, hyde_vecs)
    mix_ranks, _ = dense_search(mix, doc_pref, qids, dids, topn)
    hyde_stats = {
        "hyde": summarize(hyde_ranks, qrels),
        "hyde_mix": summarize(mix_ranks, qrels),
    }
    _show("hyde", hyde_stats["hyde"])
    _show("hyde_mix", hyde_stats["hyde_mix"])

    chunk_rows, chunk_example = _chunk_break(
        passages,
        questions,
        bm25_ranks,
        cfg["bm25"]["k1"],
        cfg["bm25"]["b"],
        cfg["chunk"]["overlap"],
        cfg["chunk"]["sizes"],
    )
    paragraph_stats = summarize(bm25_ranks, qrels)
    chunk_rows.insert(0, {"mode": "paragraph", "size": None, "n_chunks": len(passages), **paragraph_stats})

    sign = logit_sign(questions, passages, cfg)
    cfg["reranker_ce"]["logit_sign"] = sign
    ce_ranks = rerank_ce(questions, fused, passages, cfg["rerank"]["depth"], cfg)

    ranks = {
        "bm25": bm25_ranks,
        "dense": dense_ranks,
        "dense_no_prefix": raw_ranks,
        "rrf": fused,
        "rrf_linear": linear_ranks,
        "rrf_ce": ce_ranks,
        "hyde": hyde_ranks,
        "hyde_mix": mix_ranks,
    }
    systems = {name: summarize(ranks[name], qrels) for name in SYSTEMS}
    for name, stats in systems.items():
        _show(f"tune {name}", stats)

    cfg["frozen"] = True
    save_config(cfg)
    report = {
        "split": "tune",
        "seed": cfg["seed"],
        "n_questions": len(questions),
        "n_passages": len(passages),
        "bm25_grid": bm25_grid,
        "chosen_bm25": {"k1": chosen_bm25["k1"], "b": chosen_bm25["b"], "mrr@10": chosen_bm25["mrr@10"]},
        "prefix": prefix_stats,
        "rrf_grid": rrf_grid,
        "chosen_rrf": {
            "k": chosen_rrf["k"],
            "depth": chosen_rrf["depth"],
            "mrr@10": chosen_rrf["mrr@10"],
        },
        "rerank_depth": depth_rows,
        "chosen_linear": {
            "depth": best_linear["depth"],
            "weights": best_linear["weights"],
            "mrr@10": best_linear["mrr@10"],
        },
        "chunks": chunk_rows,
        "chunk_break": chunk_example,
        "hyde": hyde_stats,
        "ce_logit_sign": sign,
        "systems": systems,
    }
    write_json(ROOT / "reports" / "tune.json", report)
    print("wrote reports/tune.json", flush=True)


def run_dev(rebuild_hyde=False):
    cfg = load_config()
    if not cfg.get("frozen"):
        raise SystemExit("сначала python -m lab.eval --split tune")
    if not cfg["dense"]["revision"] or not cfg["reranker_ce"]["revision"] or not cfg["hyde"]["revision"]:
        raise SystemExit("в config пустые revision")
    splits = ensure_splits(cfg)
    passages, questions = load_subset("dev", splits["dev_passage_ids"], splits["dev_question_ids"])
    qrels = qrels_of(questions)
    topn = cfg["retrieve_topk"]
    print(f"dev questions {len(questions)} passages {len(passages)}", flush=True)

    index, ids = _bm25_index(passages)
    bm25_ranks, bm25_scores = _bm25_ranks(
        index, ids, questions, cfg["bm25"]["k1"], cfg["bm25"]["b"], topn
    )
    q_prefix = cfg["dense"]["query_prefix"]
    p_prefix = cfg["dense"]["passage_prefix"]
    doc_texts = [p["text"] for p in passages]
    query_texts = [q["text"] for q in questions]
    mapping = generate_hyde(questions, cfg, rebuild=rebuild_hyde)
    _write_mini_hyde(mapping)
    hyde_input = _hyde_texts(questions, mapping)
    doc_pref, query_pref, doc_raw, query_raw, hyde_vecs = encode_many(
        [
            {"texts": doc_texts, "prefix": p_prefix, "role": "passage-prefix"},
            {"texts": query_texts, "prefix": q_prefix, "role": "query-prefix"},
            {"texts": doc_texts, "prefix": "", "role": "passage-raw"},
            {"texts": query_texts, "prefix": "", "role": "query-raw"},
            {"texts": hyde_input, "prefix": q_prefix, "role": "hyde-query"},
        ],
        cfg,
    )
    dids = [p["id"] for p in passages]
    qids = [q["id"] for q in questions]
    dense_ranks, dense_sims = dense_search(query_pref, doc_pref, qids, dids, topn)
    raw_ranks, _ = dense_search(query_raw, doc_raw, qids, dids, topn)
    hyde_ranks, _ = dense_search(hyde_vecs, doc_pref, qids, dids, topn)
    mix_ranks, _ = dense_search(mix_vectors(query_pref, hyde_vecs), doc_pref, qids, dids, topn)
    fused = _fuse(questions, bm25_ranks, dense_ranks, cfg["rrf"]["k"], cfg["rrf"]["depth"])
    doc_tokens = {p["id"]: tokenize(p["text"]) for p in passages}
    dense_scores = _dense_for(dense_sims, questions, dids, fused)
    from lab.rerank import load_weights

    linear_ranks = apply_linear(
        questions,
        fused,
        bm25_scores,
        dense_scores,
        doc_tokens,
        load_weights(),
        cfg["rerank"]["depth"],
    )
    ce_ranks = rerank_ce(questions, fused, passages, cfg["rerank"]["depth"], cfg)
    ranks = {
        "bm25": bm25_ranks,
        "dense": dense_ranks,
        "dense_no_prefix": raw_ranks,
        "rrf": fused,
        "rrf_linear": linear_ranks,
        "rrf_ce": ce_ranks,
        "hyde": hyde_ranks,
        "hyde_mix": mix_ranks,
    }
    failures, counts = collect(questions, passages, ranks, mapping)
    fail_path = ROOT / "reports" / "failures.jsonl"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    with fail_path.open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    packed = _systems_from(ranks, qrels, cfg["store_topk"])
    manifest = load_manifest()
    device_name = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    report = {
        "split": "dev",
        "seed": cfg["seed"],
        "archive_sha256": manifest["archive_sha256"],
        "n_questions": len(questions),
        "n_passages": len(passages),
        "config": {
            "bm25": cfg["bm25"],
            "rrf": cfg["rrf"],
            "rerank": cfg["rerank"],
            "dense_model": cfg["dense"]["model"],
            "dense_revision": cfg["dense"]["revision"],
            "ce_model": cfg["reranker_ce"]["model"],
            "ce_revision": cfg["reranker_ce"]["revision"],
            "ce_logit_sign": cfg["reranker_ce"]["logit_sign"],
            "hyde_model": cfg["hyde"]["model"],
            "hyde_revision": cfg["hyde"]["revision"],
            "linear_weights": load_weights(),
            "device": device_name,
        },
        "failure_counts": counts,
        "qrels": qrels,
        "systems": packed,
    }
    write_json(ROOT / "reports" / "dev.json", report)
    print(f"failures {counts}", flush=True)
    print("wrote reports/dev.json", flush=True)
