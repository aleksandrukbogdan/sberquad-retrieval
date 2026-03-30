"""Дополнительные системы поверх уже замороженного dev-прогона.

Старые строки отчёта не пересчитываются и не переподбираются.
Новые используют те же вопросы, корпус, k1, b, k и глубину RRF.
"""

import gc
import json

from lab.data import ensure_splits, load_config, load_subset, qrels_of
from lab.metrics import summarize
from lab.models import HYDE_PROMPT, _cache_file, _device, _load_cached, _store, dense_search, encode
from lab.report import pack, write_json
from lab.rerank import ce_scores
from lab.run import _bm25_index, _bm25_ranks, _fuse
from lab.text import tokenize

def _root_report():
    from lab.data import ROOT

    return ROOT / "reports" / "dev.json"


def _ranks_from_vector(scores, ids, topn):
    positive = [i for i in range(len(ids)) if float(scores[i]) > 0.0]
    order = sorted(positive, key=lambda i: (-float(scores[i]), ids[i]))
    return [ids[i] for i in order[:topn]]


def _bm25_library(passages, questions, k1, b, topn):
    from rank_bm25 import BM25Okapi

    ids = [p["id"] for p in passages]
    docs = [tokenize(p["text"]) for p in passages]
    index = BM25Okapi(docs, k1=k1, b=b)
    ranks = {}
    for question in questions:
        scores = index.get_scores(tokenize(question["text"]))
        ranks[question["id"]] = _ranks_from_vector(scores, ids, topn)
    return ranks


def _lemma_tokens(tokens, morph):
    lemmas = []
    for token in tokens:
        parsed = morph.parse(token)
        lemmas.append(parsed[0].normal_form if parsed else token)
    return lemmas


def _bm25_lemma(passages, questions, k1, b, topn):
    import pymorphy3

    morph = pymorphy3.MorphAnalyzer()
    ids = [p["id"] for p in passages]
    docs = [_lemma_tokens(tokenize(p["text"]), morph) for p in passages]
    from lab.bm25 import BM25, top_docs

    index = BM25(docs)
    ranks = {}
    changed = 0
    for question in questions:
        raw_tokens = tokenize(question["text"])
        lemmas = _lemma_tokens(raw_tokens, morph)
        if lemmas != raw_tokens:
            changed += 1
        raw = index.scores(lemmas, k1, b)
        ranks[question["id"]] = top_docs(raw, ids, topn)
    return ranks, changed


def _encode_cls(texts, model_name, revision, batch_size, max_length, role):
    path = _cache_file(revision, role, texts)
    cached = _load_cached(path, len(texts))
    if cached is not None:
        print(f"cache {role} {len(texts)}", flush=True)
        return cached
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision, torch_dtype=torch.float16)
    device = _device()
    model.to(device)
    model.eval()
    print(f"{role} device {device}", flush=True)
    pieces = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state[:, 0]
                hidden = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
            pieces.append(hidden.cpu().numpy().astype(np.float32))
            if start % (batch_size * 10) == 0:
                print(f"{role} {start}/{len(texts)}", flush=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    array = np.vstack(pieces)
    _store(path, array)
    return array


def _revision(name):
    from huggingface_hub import HfApi

    return HfApi().model_info(name).sha


def _load_jsonl_cache(path):
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["id"])] = row["text"]
    return out


def _save_jsonl_cache(path, mapping):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"id": qid, "text": mapping[qid]}, ensure_ascii=False)
        for qid in sorted(mapping)
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _clean(text):
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    if text.lower().startswith("абзац:"):
        text = text.split(":", 1)[1].strip()
    return text


def _generate(questions, model_name, revision, cache_path, max_new_tokens):
    cache = _load_jsonl_cache(cache_path)
    missing = [q for q in questions if q["id"] not in cache]
    if not missing:
        return {q["id"]: cache[q["id"]] for q in questions}
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.float16,
    )
    device = _device()
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"hyde generator {model_name} {device}", flush=True)
    try:
        for i, question in enumerate(missing):
            messages = [{"role": "user", "content": HYDE_PROMPT.format(q=question["text"])}]
            kwargs = {
                "add_generation_prompt": True,
                "return_tensors": "pt",
                "enable_thinking": False,
            }
            try:
                encoded = tokenizer.apply_chat_template(messages, **kwargs)
            except (TypeError, ValueError):
                kwargs.pop("enable_thinking")
                encoded = tokenizer.apply_chat_template(messages, **kwargs)
            input_ids = encoded["input_ids"] if hasattr(encoded, "input_ids") else encoded
            input_ids = input_ids.to(device)
            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(output[0, input_ids.shape[-1] :], skip_special_tokens=True)
            cache[question["id"]] = _clean(text)
            if i % 10 == 0:
                _save_jsonl_cache(cache_path, cache)
                print(f"hyde3 {i}/{len(missing)}", flush=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _save_jsonl_cache(cache_path, cache)
    return {q["id"]: cache[q["id"]] for q in questions}


def _show(title, stats):
    print(
        f"{title} R@5={stats['recall@5']:.3f} R@10={stats['recall@10']:.3f} "
        f"MRR={stats['mrr@10']:.3f} nDCG={stats['ndcg@10']:.3f}",
        flush=True,
    )


def _agree(left, right, k=10):
    same = 0
    for qid, ranked in left.items():
        if ranked[:k] == right[qid][:k]:
            same += 1
    return same / len(left) if left else 0.0


def run_extra():
    from lab.data import ROOT

    cfg = load_config()
    if not cfg.get("frozen"):
        raise SystemExit("сначала нужен замороженный прогон dev")
    report_path = _root_report()
    if not report_path.exists():
        raise SystemExit("нет reports/dev.json")
    splits = ensure_splits(cfg)
    passages, questions = load_subset("dev", splits["dev_passage_ids"], splits["dev_question_ids"])
    qrels = qrels_of(questions)
    k1 = cfg["bm25"]["k1"]
    b = cfg["bm25"]["b"]
    topn = cfg["retrieve_topk"]
    store = cfg["store_topk"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.setdefault("supplement", {})
    have = set(report["systems"])
    print(f"extra questions {len(questions)} passages {len(passages)}", flush=True)

    def save(name, ranks):
        report["systems"][name] = pack(ranks, qrels, store)
        write_json(report_path, report)
        _show(name, report["systems"][name])

    need_bm25 = not {"bm25_lib", "bm25_lemma", "bge_m3", "rrf_bge", "rrf_bge_rerank"} <= have
    own_ranks = None
    if need_bm25 or "hyde_qwen3" not in have:
        own_ranks, _ = _bm25_ranks(*_bm25_index(passages), questions, k1, b, topn)

    if "bm25_lib" not in have or "bm25_lemma" not in have:
        lib_ranks = _bm25_library(passages, questions, k1, b, topn)
        lemma_ranks, lemma_changed = _bm25_lemma(passages, questions, k1, b, topn)
        agree = _agree(own_ranks, lib_ranks)
        print(f"bm25 top10 identical {agree:.3f} lemma queries changed {lemma_changed}", flush=True)
        report["supplement"]["bm25_top10_identical"] = agree
        report["supplement"]["lemma_queries_changed"] = lemma_changed
        report["supplement"]["bm25_lib"] = (
            "rank_bm25.BM25Okapi, same tokens, k1 and b; zero scores dropped"
        )
        report["supplement"]["bm25_lemma"] = "pymorphy3 normal_form, same k1 and b"
        if "bm25_lib" not in have:
            save("bm25_lib", lib_ranks)
        if "bm25_lemma" not in have:
            save("bm25_lemma", lemma_ranks)

    doc_texts = [p["text"] for p in passages]
    query_texts = [q["text"] for q in questions]
    dids = [p["id"] for p in passages]
    qids = [q["id"] for q in questions]
    q_prefix = cfg["dense"]["query_prefix"]
    p_prefix = cfg["dense"]["passage_prefix"]

    if "bge_m3" not in have or "rrf_bge" not in have:
        bge_name = "BAAI/bge-m3"
        bge_rev = _revision(bge_name)
        bge_docs = _encode_cls(doc_texts, bge_name, bge_rev, 16, 512, "bge-m3-passage")
        bge_queries = _encode_cls(query_texts, bge_name, bge_rev, 16, 512, "bge-m3-query")
        bge_ranks, _ = dense_search(bge_queries, bge_docs, qids, dids, topn)
        report["supplement"]["bge_m3"] = {
            "model": bge_name,
            "revision": bge_rev,
            "pooling": "cls",
            "max_length": 512,
        }
        if "bge_m3" not in have:
            save("bge_m3", bge_ranks)
        if "rrf_bge" not in have:
            save("rrf_bge", _fuse(questions, own_ranks, bge_ranks, cfg["rrf"]["k"], cfg["rrf"]["depth"]))

    if "rrf_bge_rerank" not in have:
        doc_pref = encode(doc_texts, cfg, p_prefix, "passage-prefix")
        query_pref = encode(query_texts, cfg, q_prefix, "query-prefix")
        dense_ranks, _ = dense_search(query_pref, doc_pref, qids, dids, topn)
        fused = _fuse(questions, own_ranks, dense_ranks, cfg["rrf"]["k"], cfg["rrf"]["depth"])
        rerank_name = "BAAI/bge-reranker-v2-m3"
        rerank_rev = _revision(rerank_name)
        by_id = {p["id"]: p["text"] for p in passages}
        depth = cfg["rerank"]["depth"]
        pairs = []
        spans = []
        for question in questions:
            head = fused[question["id"]][:depth]
            spans.append(head)
            for doc_id in head:
                pairs.append((question["text"], by_id[doc_id]))
        raw_scores = ce_scores(pairs, rerank_name, rerank_rev, 512, 16, 1)
        bge_rerank = {}
        cursor = 0
        for question, head in zip(questions, spans):
            chunk = raw_scores[cursor : cursor + len(head)]
            cursor += len(head)
            order = sorted(range(len(head)), key=lambda i: (-chunk[i], head[i]))
            reranked = [head[i] for i in order]
            seen = set(reranked)
            tail = [doc_id for doc_id in fused[question["id"]] if doc_id not in seen]
            bge_rerank[question["id"]] = reranked + tail
        report["supplement"]["bge_reranker"] = {
            "model": rerank_name,
            "revision": rerank_rev,
            "pool": "rrf(bm25, e5)",
            "depth": depth,
        }
        save("rrf_bge_rerank", bge_rerank)
        del doc_pref, query_pref

    if "rrf_lemma" not in have:
        if own_ranks is None:
            own_ranks, _ = _bm25_ranks(*_bm25_index(passages), questions, k1, b, topn)
        lemma_for_fuse, _ = _bm25_lemma(passages, questions, k1, b, topn)
        doc_pref = encode(doc_texts, cfg, p_prefix, "passage-prefix")
        query_pref = encode(query_texts, cfg, q_prefix, "query-prefix")
        dense_ranks, _ = dense_search(query_pref, doc_pref, qids, dids, topn)
        report["supplement"]["rrf_lemma"] = "RRF(bm25_lemma, e5 with prefix), frozen k and depth"
        save("rrf_lemma", _fuse(questions, lemma_for_fuse, dense_ranks, cfg["rrf"]["k"], cfg["rrf"]["depth"]))
        del doc_pref, query_pref, dense_ranks

    if "hyde_qwen3" not in have or "hyde_qwen3_mix" not in have:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free = torch.cuda.mem_get_info()[0]
        else:
            free = 0
        hyde_name = "Qwen/Qwen3-8B" if free >= 22 * 1024**3 else "Qwen/Qwen3-4B"
        print(f"hyde model {hyde_name} free_bytes {free}", flush=True)
        hyde_rev = _revision(hyde_name)
        cache_path = ROOT / "data" / "cache" / "hyde_qwen3.jsonl"
        mapping = _generate(
            questions, hyde_name, hyde_rev, cache_path, cfg["hyde"]["max_new_tokens"]
        )
        hyde_input = []
        empty = 0
        for question in questions:
            text = (mapping.get(question["id"]) or "").strip()
            if not text:
                empty += 1
                text = question["text"]
            hyde_input.append(text)
        if empty:
            print(f"hyde3 empty {empty}", flush=True)
        doc_pref = encode(doc_texts, cfg, p_prefix, "passage-prefix")
        query_pref = encode(query_texts, cfg, q_prefix, "query-prefix")
        hyde_vecs = encode(hyde_input, cfg, q_prefix, "hyde-qwen3-query")
        hyde_ranks, _ = dense_search(hyde_vecs, doc_pref, qids, dids, topn)
        from lab.models import mix_vectors

        mix_ranks, _ = dense_search(mix_vectors(query_pref, hyde_vecs), doc_pref, qids, dids, topn)
        report["supplement"]["hyde_qwen3"] = {
            "model": hyde_name,
            "revision": hyde_rev,
            "prompt": "same as hyde",
            "retriever": "e5 with prefix",
            "thinking": False,
            "empty": empty,
        }
        if "hyde_qwen3" not in have:
            save("hyde_qwen3", hyde_ranks)
        if "hyde_qwen3_mix" not in have:
            save("hyde_qwen3_mix", mix_ranks)

    report["supplement"]["added_after_freeze"] = True
    write_json(report_path, report)
    print("updated reports/dev.json", flush=True)
