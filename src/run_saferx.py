"""
Run SafeRx-Agent (multi-agent generation: Route -> Summarize/Generate -> Critique)
on a patient JSONL and write one ``eval_overlay_XXXX.json`` per case plus a
``summary.json`` with P / R / F1 / Jaccard.

Safety verification (FindFlags + Verify) runs afterwards on the same log dir:
    python -m scripts.llm_safety_filter --config <cfg> --patient-jsonl $MIMIC4_JSONL --eval-dir <log-dir> --out-dir <log-dir>_verified

Examples:
  # full SafeRx-Agent (skip the 100 prompt-development cases)
  python -m src.run_saferx --config configs/example.yaml --jsonl $MIMIC4_JSONL \
      --skip 100 --limit 100000 --log-dir logs/saferx

  # ablation: w/o Specialty Experts (single generalist expert)
  python -m src.run_saferx ... --general-agent
  # ablations
  python -m src.run_saferx ... --no-summarizer | --no-medi | --no-critique
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.saferx import PatientCase, build_saferx_agent
from src.utils import MockLLM, OpenAIChat, load_llm_section
from src.utils.medi2 import load_medi2
from src.utils.metrics import batch_codes_to_multihot, compute_metrics
from src.utils.vocab import load_med_vocab

PROMPT_SUFFIX = "Then, the patient should be prescribed:"


def read_jsonl(path: Path, limit: int, skip: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(l for l in handle if l.strip()):
            if i < skip:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def strip_suffix(text: str) -> str:
    idx = text.rfind(PROMPT_SUFFIX)
    return text[:idx].rstrip() if idx != -1 else text


def prior_med_codes(row: dict[str, Any]) -> list[str]:
    """ATC-L4 codes from all visits before the target visit (empty if no prior visit)."""
    meds = (row.get("records") or {}).get("medication") or []
    seen: set[str] = set()
    out: list[str] = []
    for visit in meds[:-1]:
        for c in visit:
            code = str(c).strip()[:5]
            if code and code not in seen:
                seen.add(code)
                out.append(code)
    return out


def gold_codes(row: dict[str, Any]) -> list[str]:
    target = row.get("target")
    if not target:
        meds = (row.get("records") or {}).get("medication") or []
        target = meds[-1] if meds else []
    return list(dict.fromkeys(str(c).strip()[:5] for c in target if str(c).strip()))


def build_llm(cfg: dict[str, Any], use_mock: bool):
    if use_mock:
        return MockLLM()
    llm_cfg = load_llm_section(cfg)
    return OpenAIChat(model=llm_cfg["model"], api_key=llm_cfg["api_key"], base_url=llm_cfg.get("base_url"))


def bootstrap_ci(gold, pred, vocab, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(gold)
    f1s, jacs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(
            batch_codes_to_multihot([gold[i] for i in idx], vocab),
            batch_codes_to_multihot([pred[i] for i in idx], vocab),
        )
        f1s.append(m["micro_f1"])
        jacs.append(m["jaccard"])
    pct = lambda xs: (float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5)))
    return pct(f1s), pct(jacs)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=10, help="Evaluate at most N rows.")
    ap.add_argument("--skip", type=int, default=0, help="Skip the first N rows (e.g. the 100 prompt-dev cases).")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--general-agent", action="store_true", help="Ablation: w/o Specialty Experts (single generalist expert).")
    ap.add_argument("--no-summarizer", action="store_true", help="Ablation: w/o Summarizer.")
    ap.add_argument("--no-medi", action="store_true", help="Ablation: w/o Medical Resource (MEDI).")
    ap.add_argument("--no-critique", action="store_true", help="Ablation: w/o Critique.")
    ap.add_argument("--use-mock-llm", action="store_true", help="Offline dry run with a stub LLM.")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    run_cfg = cfg.get("run", {})
    vocab = load_med_vocab(
        Path(run_cfg.get("med_vocab_json", "data/med_vocab.json")),
        atc_csv_path=Path(run_cfg.get("vocab_csv", "data/atc_hierarchy.csv")),
    )
    medi2 = None if args.no_medi else load_medi2(Path(run_cfg.get("medi2_csv", "data/MEDI2_with_ATC_ICD10.csv")))
    args.log_dir.mkdir(parents=True, exist_ok=True)

    agent = build_saferx_agent(
        llm=build_llm(cfg, args.use_mock_llm),
        vocab=vocab,
        log_dir=args.log_dir / "traces",
        medi2=medi2,
        general_agent=args.general_agent,
        use_summarizer=not args.no_summarizer,
        use_critique=not args.no_critique,
    )

    rows = read_jsonl(args.jsonl, args.limit, skip=args.skip)
    print(f"Loaded {len(rows)} cases; vocab={len(vocab.allowed_label_space)} codes")
    sem = asyncio.Semaphore(args.concurrency)
    results: list[tuple[list[str], list[str]] | None] = [None] * len(rows)

    async def run_one(idx: int, row: dict[str, Any]) -> None:
        overlay_path = args.log_dir / f"eval_overlay_{idx:04d}.json"
        if overlay_path.exists():  # resume
            done = json.loads(overlay_path.read_text())
            results[idx] = (done["gold_codes"], done["evaluated_pred_codes"])
            return
        async with sem:
            case = PatientCase(
                case_id=f"case_{row.get('subject_id', idx)}",
                patient_text=strip_suffix(str(row.get("input", ""))),
                patient_record=row,
            )
            result = await agent.run(case)
        pred = [d["code"] for d in result["final_prediction"]["final_predicted_drugs"] if d.get("code")]
        gold = gold_codes(row)
        overlay = {
            "idx": idx,
            "subject_id": row.get("subject_id", idx),
            "evaluated_pred_codes": pred,
            "gold_codes": gold,
            "prior_meds": prior_med_codes(row),
            "report": result.get("report"),
        }
        overlay_path.write_text(json.dumps(overlay, indent=2, ensure_ascii=False))
        results[idx] = (gold, pred)
        print(f"[{idx}] gold={len(gold)} pred={len(pred)}")

    await asyncio.gather(*(run_one(i, r) for i, r in enumerate(rows)))

    gold = [r[0] for r in results if r]
    pred = [r[1] for r in results if r]
    m = compute_metrics(batch_codes_to_multihot(gold, vocab), batch_codes_to_multihot(pred, vocab))
    (f1_lo, f1_hi), (j_lo, j_hi) = bootstrap_ci(gold, pred, vocab)
    summary = {
        "n_cases": len(gold),
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "micro_f1": round(m["micro_f1"], 4),
        "jaccard": round(m["jaccard"], 4),
        "avg_pred": round(float(np.mean([len(p) for p in pred])) if pred else 0.0, 2),
        "f1_ci": [round(f1_lo, 4), round(f1_hi, 4)],
        "jaccard_ci": [round(j_lo, 4), round(j_hi, 4)],
        "general_agent": args.general_agent,
        "summarizer": not args.no_summarizer,
        "medi": not args.no_medi,
        "critique": not args.no_critique,
        "config": str(args.config),
        "jsonl": str(args.jsonl),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (args.log_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
