import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from lab.data import ROOT, load_config, save_config

CACHE = ROOT / "data" / "cache"
HYDE_CACHE = CACHE / "hyde.jsonl"

HYDE_PROMPT = (
    "Напиши короткий абзац на русском, который отвечает на вопрос. "
    "Только абзац, без пояснений.\n\nВопрос: {q}\n\nАбзац:"
)


def pin_revisions(cfg):
    from huggingface_hub import HfApi

    api = HfApi()
    changed = False
    for key in ("dense", "reranker_ce", "hyde"):
        if not cfg[key].get("revision"):
            cfg[key]["revision"] = api.model_info(cfg[key]["model"]).sha
            print(f"pin {cfg[key]['model']} {cfg[key]['revision']}", flush=True)
            changed = True
    if changed:
        save_config(cfg)
    return cfg


def _threads():
    import torch

    torch.set_num_threads(min(8, os.cpu_count() or 1))


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _texts_key(texts):
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _cache_file(revision, role, texts):
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{revision[:8]}_{role}_{_texts_key(texts)}.npy"


def _load_cached(path, n_rows):
    meta = Path(str(path) + ".meta.json")
    if not path.exists() or not meta.exists():
        return None
    info = json.loads(meta.read_text(encoding="utf-8"))
    if info.get("rows") != n_rows:
        return None
    return np.load(path)


def _store(path, array):
    np.save(path, array)
    Path(str(path) + ".meta.json").write_text(
        json.dumps({"rows": int(array.shape[0]), "dim": int(array.shape[1])}) + "\n",
        encoding="utf-8",
    )


def _average_pool(hidden, mask):
    import torch

    weights = mask.unsqueeze(-1).type_as(hidden)
    summed = (hidden * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_with(tokenizer, model, texts, prefix, block, role):
    import torch

    device = next(model.parameters()).device
    pieces = []
    for start in range(0, len(texts), block["batch_size"]):
        batch = [prefix + text for text in texts[start : start + block["batch_size"]]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=block["max_length"],
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            pooled = _average_pool(hidden, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        pieces.append(pooled.cpu().numpy().astype(np.float32))
        if start % (block["batch_size"] * 10) == 0:
            print(f"e5 {role} {start}/{len(texts)}", flush=True)
    return np.vstack(pieces)


def encode_many(jobs, cfg):
    # jobs: list of {texts, prefix, role}
    block = cfg["dense"]
    results = [None] * len(jobs)
    pending = []
    for i, job in enumerate(jobs):
        path = _cache_file(block["revision"], job["role"], [job["prefix"] + text for text in job["texts"]])
        cached = _load_cached(path, len(job["texts"]))
        if cached is not None:
            print(f"cache {job['role']} {len(job['texts'])}", flush=True)
            results[i] = cached
        else:
            pending.append((i, job, path))
    if not pending:
        return results
    import torch
    from transformers import AutoModel, AutoTokenizer

    _threads()
    tokenizer = AutoTokenizer.from_pretrained(block["model"], revision=block["revision"])
    model = AutoModel.from_pretrained(block["model"], revision=block["revision"])
    device = _device()
    model.to(device)
    model.eval()
    print(f"e5 device {device}", flush=True)
    try:
        for i, job, path in pending:
            array = _encode_with(tokenizer, model, job["texts"], job["prefix"], block, job["role"])
            _store(path, array)
            results[i] = array
    finally:
        del model
        gc.collect()
    return results


def encode(texts, cfg, prefix, role):
    return encode_many([{"texts": texts, "prefix": prefix, "role": role}], cfg)[0]


def dense_search(q_mat, d_mat, qids, dids, topn):
    sims = q_mat @ d_mat.T
    ranks = {}
    for qi, qid in enumerate(qids):
        row = sims[qi]
        n = int(row.shape[0])
        k = min(topn, n)
        if k < n:
            idx = np.argpartition(-row, k - 1)[:k]
        else:
            idx = np.arange(n)
        chosen = sorted(idx.tolist(), key=lambda i: (-float(row[i]), dids[i]))
        ranks[qid] = [dids[i] for i in chosen]
    return ranks, sims


def mix_vectors(left, right):
    mixed = left.astype(np.float32) + right.astype(np.float32)
    norms = np.linalg.norm(mixed, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    return (mixed / norms).astype(np.float32)


def load_hyde_cache():
    if not HYDE_CACHE.exists():
        return {}
    out = {}
    for line in HYDE_CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["id"])] = row["text"]
    return out


def save_hyde_cache(mapping):
    CACHE.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"id": qid, "text": mapping[qid]}, ensure_ascii=False)
        for qid in sorted(mapping)
    ]
    HYDE_CACHE.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _clean_hyde(text):
    text = text.strip()
    if text.lower().startswith("абзац:"):
        text = text.split(":", 1)[1].strip()
    return text


def generate_hyde(questions, cfg, rebuild=False):
    cache = {} if rebuild else load_hyde_cache()
    missing = [q for q in questions if q["id"] not in cache]
    if not missing:
        return {q["id"]: cache[q["id"]] for q in questions}
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _threads()
    block = cfg["hyde"]
    tokenizer = AutoTokenizer.from_pretrained(block["model"], revision=block["revision"])
    model = AutoModelForCausalLM.from_pretrained(block["model"], revision=block["revision"])
    device = _device()
    model.to(device)
    model.eval()
    print(f"hyde device {device}", flush=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        for i, question in enumerate(missing):
            prompt = HYDE_PROMPT.format(q=question["text"])
            messages = [{"role": "user", "content": prompt}]
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(encoded, "input_ids"):
                input_ids = encoded["input_ids"]
            else:
                input_ids = encoded
            input_ids = input_ids.to(device)
            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    max_new_tokens=block["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = output[0, input_ids.shape[-1] :]
            cache[question["id"]] = _clean_hyde(tokenizer.decode(new_tokens, skip_special_tokens=True))
            if i % 10 == 0:
                save_hyde_cache(cache)
                print(f"hyde {i}/{len(missing)}", flush=True)
    finally:
        del model
        gc.collect()
    save_hyde_cache(cache)
    return {q["id"]: cache[q["id"]] for q in questions}
