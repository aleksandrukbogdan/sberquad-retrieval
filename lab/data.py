import hashlib
import json
import random
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST_PATH = ROOT / "data" / "MANIFEST.json"
SPLITS_PATH = ROOT / "data" / "splits.json"
ARCHIVE_NAME = "sber_squad-v1.1.tar.gz"
ARCHIVE_URL = "https://files.deeppavlov.ai/datasets/sber_squad-v1.1.tar.gz"

# В tar одна статья-обёртка SberChallenge: абзац = контекст, вопрос один.
# Заголовков Википедии в файле нет, поэтому корпус — сами абзацы.


def load_config():
    return json.loads((ROOT / "lab" / "config.json").read_text(encoding="utf-8"))


def save_config(cfg):
    path = ROOT / "lab" / "config.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def ensure_raw():
    manifest = load_manifest()
    RAW.mkdir(parents=True, exist_ok=True)
    archive = RAW / ARCHIVE_NAME
    if not archive.exists() or _sha256(archive) != manifest["archive_sha256"]:
        print(f"download {ARCHIVE_URL}", flush=True)
        req = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "raglab"})
        with urllib.request.urlopen(req, timeout=180) as response, archive.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
        got = _sha256(archive)
        if got != manifest["archive_sha256"]:
            raise SystemExit(f"archive sha256 {got} != {manifest['archive_sha256']}")
    needed = [f"{name}-v1.1.json" for name in ("train", "dev")]
    if not all((RAW / name).exists() for name in needed):
        with tarfile.open(archive, "r:gz") as tar:
            try:
                tar.extractall(RAW, filter="data")
            except TypeError:
                tar.extractall(RAW)
    for name, meta in manifest["files"].items():
        path = RAW / name
        got = _sha256(path)
        if got != meta["sha256"]:
            raise SystemExit(f"{name} sha256 {got} != {meta['sha256']}")
    return manifest


def _align_span(ctx, answer, start):
    n = len(answer)
    if n == 0:
        start = min(max(int(start), 0), len(ctx))
        return start, start
    if 0 <= start < len(ctx):
        sl = ctx[start : start + n]
        if sl.casefold() == answer.casefold():
            return start, start + n
    pos = ctx.casefold().find(answer.casefold())
    if pos >= 0:
        return pos, pos + n
    start = min(max(int(start), 0), len(ctx))
    return start, min(len(ctx), start + n)


def load_squad(split):
    path = RAW / f"{split}-v1.1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    passages = []
    questions = []
    seen = set()
    for article in data["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            pid = "p" + hashlib.sha256(ctx.encode("utf-8")).hexdigest()[:16]
            if pid not in seen:
                seen.add(pid)
                passages.append({"id": pid, "text": ctx})
            for qa in para["qas"]:
                answers = qa.get("answers") or []
                if not answers:
                    continue
                answer = answers[0]
                text = answer.get("text") or ""
                start, end = _align_span(ctx, text, int(answer.get("answer_start") or 0))
                questions.append(
                    {
                        "id": str(qa["id"]),
                        "text": qa["question"],
                        "passage_id": pid,
                        "answer": text,
                        "answer_start": start,
                        "answer_end": end,
                    }
                )
    return passages, questions


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_splits(cfg):
    ensure_raw()
    if SPLITS_PATH.exists():
        return json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    train_passages, train_questions = load_squad("train")
    dev_passages, dev_questions = load_squad("dev")
    overlap = {q["id"] for q in train_questions} & {q["id"] for q in dev_questions}
    if overlap:
        raise SystemExit(f"train/dev question ids overlap: {len(overlap)}")
    rng = random.Random(cfg["seed"])
    tune_ids = sorted(rng.sample(sorted(q["id"] for q in train_questions), cfg["tune_questions"]))
    dev_ids = sorted(rng.sample(sorted(q["id"] for q in dev_questions), cfg["eval_questions"]))
    tune_set = set(tune_ids)
    by_q = {q["id"]: q for q in train_questions}
    gold = {by_q[qid]["passage_id"] for qid in tune_ids}
    rest = sorted(p["id"] for p in train_passages if p["id"] not in gold)
    extra = rng.sample(rest, cfg["train_distractors"])
    splits = {
        "seed": cfg["seed"],
        "tune_question_ids": tune_ids,
        "dev_question_ids": dev_ids,
        "tune_passage_ids": sorted(gold | set(extra)),
        "dev_passage_ids": sorted(p["id"] for p in dev_passages),
        "counts": {
            "train_passages": len(train_passages),
            "train_questions": len(train_questions),
            "dev_passages": len(dev_passages),
            "dev_questions": len(dev_questions),
        },
    }
    _write_json(SPLITS_PATH, splits)
    _write_mini(dev_passages, dev_questions, dev_ids)
    print(
        f"splits tune_q={len(tune_ids)} tune_p={len(splits['tune_passage_ids'])} "
        f"dev_q={len(dev_ids)} dev_p={len(splits['dev_passage_ids'])}",
        flush=True,
    )
    return splits


def _write_mini(passages, questions, dev_ids):
    mini = ROOT / "data" / "mini"
    mini.mkdir(parents=True, exist_ok=True)
    by_q = {q["id"]: q for q in questions}
    by_p = {p["id"]: p for p in passages}
    qids = dev_ids[:20]
    chosen_q = [by_q[qid] for qid in qids]
    pids = []
    seen = set()
    for question in chosen_q:
        if question["passage_id"] not in seen:
            seen.add(question["passage_id"])
            pids.append(question["passage_id"])
    for pid in sorted(by_p):
        if len(pids) >= 40:
            break
        if pid not in seen:
            seen.add(pid)
            pids.append(pid)
    with (mini / "passages.jsonl").open("w", encoding="utf-8") as handle:
        for pid in pids:
            handle.write(json.dumps(by_p[pid], ensure_ascii=False) + "\n")
    with (mini / "questions.jsonl").open("w", encoding="utf-8") as handle:
        for question in chosen_q:
            handle.write(json.dumps(question, ensure_ascii=False) + "\n")


def load_subset(split, passage_ids, question_ids):
    passages, questions = load_squad(split)
    want_p = set(passage_ids)
    want_q = set(question_ids)
    passages = [p for p in passages if p["id"] in want_p]
    questions = [q for q in questions if q["id"] in want_q]
    passages.sort(key=lambda p: p["id"])
    questions.sort(key=lambda q: q["id"])
    missing = [q["id"] for q in questions if q["passage_id"] not in want_p]
    if missing:
        raise SystemExit(f"gold passage missing for {missing[:3]}")
    if len(questions) != len(want_q):
        raise SystemExit(f"questions {len(questions)} != {len(want_q)}")
    return passages, questions


def qrels_of(questions):
    return {q["id"]: [q["passage_id"]] for q in questions}


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
