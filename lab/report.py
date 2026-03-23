import json
from pathlib import Path

from lab.data import ROOT
from lab.metrics import summarize

SYSTEMS = (
    "bm25",
    "dense",
    "dense_no_prefix",
    "rrf",
    "rrf_linear",
    "rrf_ce",
    "hyde",
    "hyde_mix",
    "bm25_lib",
    "bm25_lemma",
    "bge_m3",
    "rrf_bge",
    "rrf_lemma",
    "rrf_bge_rerank",
    "hyde_qwen3",
    "hyde_qwen3_mix",
)

METRIC_KEYS = ("recall@5", "recall@10", "mrr@10", "ndcg@10")


def pack(ranks, qrels, store):
    cut = {qid: ranked[:store] for qid, ranked in ranks.items()}
    stats = summarize(cut, qrels)
    return {"rankings": cut, **stats}


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def recompute(report):
    qrels = report["qrels"]
    out = {}
    for name, block in report["systems"].items():
        out[name] = summarize(block["rankings"], qrels)
    return out


def parse_main_table(text):
    lines = text.splitlines()
    header_i = None
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and "recall@5" in line and "mrr@10" in line:
            header_i = i
            break
    if header_i is None:
        raise AssertionError("в README нет таблицы метрик")
    rows = {}
    for line in lines[header_i + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[0] == "прогон":
            continue
        rows[cells[0]] = {
            "recall@5": float(cells[1]),
            "recall@10": float(cells[2]),
            "mrr@10": float(cells[3]),
            "ndcg@10": float(cells[4]),
        }
    return rows


def table_matches(readme, report):
    parsed = parse_main_table(readme)
    problems = []
    for name in SYSTEMS:
        if name not in parsed:
            problems.append(f"нет строки {name}")
            continue
        block = report["systems"][name]
        for key in METRIC_KEYS:
            shown = f"{parsed[name][key]:.4f}"
            expected = f"{block[key]:.4f}"
            if shown != expected:
                problems.append(f"{name} {key}: readme {shown} report {expected}")
    return problems


def dev_report_path():
    return ROOT / "reports" / "dev.json"
