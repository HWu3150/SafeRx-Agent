"""
Efficiency benchmark (Table 5): SafeRx-Agent (+ FindFlags/Verify) or, with
--general-agent (ablation: w/o Specialty Experts), on N random cases.

Collects per-case:
  - LLM calls (total and per-phase)
  - Input / output tokens (total and per-phase)
  - MEDI2 lookups
  - Wall-clock latency

Usage:
  python -m scripts.efficiency_benchmark \
    --config configs/example.yaml \
    --jsonl $MIMIC4_JSONL \
    --n 20 --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import yaml
from openai import OpenAI

from src.saferx import PatientCase, build_saferx_agent
from src.utils.llm import load_llm_section
from src.utils.vocab import load_med_vocab
from src.utils.medi2 import MEDI2Lookup, load_medi2
from scripts import llm_safety_filter as sf

DEFAULT_SUFFIX = "Then, the patient should be prescribed:"

# ── Instrumented LLM wrapper ────────────────────────────────────────────

@dataclass
class LLMCallRecord:
    phase: str
    input_tokens: int
    output_tokens: int
    latency_s: float


class InstrumentedLLM:
    """Wraps OpenAI client to record token usage, latency, and phase labels."""

    def __init__(self, model: str, api_key: str, base_url: str):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.records: list[LLMCallRecord] = []
        self._phase = "unknown"

    def set_phase(self, phase: str):
        self._phase = phase

    async def chat(self, messages: Sequence[Dict[str, str]], **kwargs) -> str:
        from src.utils.llm import _sanitize_messages
        cleaned = _sanitize_messages(messages)
        phase = self._phase

        def _call():
            t0 = time.perf_counter()
            resp = self._client.chat.completions.create(
                model=self.model, messages=cleaned, **kwargs,
            )
            elapsed = time.perf_counter() - t0
            content = resp.choices[0].message.content or ""
            usage = resp.usage
            rec = LLMCallRecord(
                phase=phase,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_s=elapsed,
            )
            return content, rec

        content, rec = await asyncio.to_thread(_call)
        self.records.append(rec)
        return content

    def pop_records(self) -> list[LLMCallRecord]:
        out = self.records[:]
        self.records.clear()
        return out


# ── Instrumented MEDI2 wrapper ──────────────────────────────────────────

class InstrumentedMEDI2(MEDI2Lookup):
    def __init__(self, inner: MEDI2Lookup):
        self._map = inner._map
        self.lookup_count = 0

    def lookup(self, icd_codes, **kw):
        self.lookup_count += 1
        return super().lookup(icd_codes, **kw)

    def selective_lookup(self, icd_codes, **kw):
        self.lookup_count += 1
        return super().selective_lookup(icd_codes, **kw)

    def reset_count(self):
        c = self.lookup_count
        self.lookup_count = 0
        return c


# ── Monkey-patch pipeline to label phases ────────────────────────────────

def patch_agent_phases(agent, llm: InstrumentedLLM):
    """Wrap pipeline methods to set phase labels on the instrumented LLM."""
    orig_summarize = agent._summarize
    orig_generate = agent._generate
    orig_critique = agent._critique

    async def _summarize(case, skill):
        llm.set_phase(f"summarize:{skill.skill_id}")
        return await orig_summarize(case, skill)

    async def _generate(state, skill, **kw):
        llm.set_phase(f"generate:{skill.skill_id}")
        return await orig_generate(state, skill, **kw)

    async def _critique(merged, case, skill_runs, prior_meds):
        llm.set_phase("critique")
        return await orig_critique(merged, case, skill_runs, prior_meds)

    agent._summarize = _summarize
    agent._generate = _generate
    agent._critique = _critique


# ── Safety filter: FindFlags + Verify (same logic as llm_safety_filter.py) ─

_SAFETY_RESOURCES = None


def _safety_resources(patient_jsonl):
    global _SAFETY_RESOURCES
    if _SAFETY_RESOURCES is None:
        _SAFETY_RESOURCES = (
            sf.load_resources(str(patient_jsonl)),
            sf.load_patient_texts(str(patient_jsonl)),
            sf.load_indication_resource(),
        )
    return _SAFETY_RESOURCES


async def run_safety_filter(llm: InstrumentedLLM, pred_codes: list[str], row: dict, patient_jsonl) -> list[str]:
    """FindFlags (deterministic) then Verify (one LLM call, only if any flag is raised)."""
    (ddi_b, _ddi_w, d2i, _i2d, deg, contra_b, _contra_w, code_name, by_sid), patient_texts, medi2 = \
        _safety_resources(patient_jsonl)
    records = row.get("records", {}) if isinstance(row.get("records"), dict) else {}
    med_visits = records.get("medication", []) or []
    prior = med_visits[0] if len(med_visits) >= 2 else []  # only a real prior visit
    prior_set = {str(c).strip()[:5] for c in prior if str(c).strip()}
    dx = by_sid.get(str(row.get("subject_id", "")), [])

    ddi_pairs = sf.get_ddi_pairs(pred_codes, ddi_b, d2i, deg)
    contra_pairs = sf.get_contra_pairs(pred_codes, dx, contra_b)
    if not ddi_pairs and not contra_pairs:
        return list(pred_codes)  # no flags -> verifier not invoked

    llm.set_phase("verify")
    sid = str(row.get("subject_id", ""))
    flagged = sf.flagged_codes(ddi_pairs, contra_pairs)
    user_prompt = sf.build_user_prompt(
        pred_codes, prior_set, ddi_pairs, contra_pairs, code_name,
        patient_text=patient_texts.get(sid, ""),
        indication_evidence=sf.get_indication_evidence(medi2, dx, flagged),
    )
    raw = await llm.chat([
        {"role": "system", "content": sf.SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ], temperature=0.0)
    return sf.parse_llm_response(raw, pred_codes, flagged)


# ── Main ─────────────────────────────────────────────────────────────────

def strip_suffix(text: str, suffix: str = DEFAULT_SUFFIX) -> str:
    idx = text.rfind(suffix)
    return text[:idx].rstrip() if idx != -1 else text


def extract_codes(items):
    codes = []
    if isinstance(items, dict):
        for key in ("final_predicted_drugs", "should_prescribe", "predicted_drugs"):
            if key in items:
                return extract_codes(items[key])
        for key in ("code", "atc"):
            if key in items and isinstance(items[key], str):
                return [items[key].strip()]
        return codes
    if isinstance(items, list):
        for item in items:
            codes.extend(extract_codes(item))
        seen = set()
        return [c for c in codes if c and not (c in seen or seen.add(c))]
    if isinstance(items, str):
        return [items.strip()] if items.strip() else []
    return codes


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--general-agent", action="store_true",
                        help="Benchmark the w/o Specialty Experts variant (single generalist, no safety filter).")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    vllm_cfg = load_llm_section(cfg)
    run_cfg = cfg.get("run", {})

    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    rng = random.Random(args.seed)
    sampled = rng.sample(rows, min(args.n, len(rows)))
    print(f"Sampled {len(sampled)} cases (seed={args.seed}) from {len(rows)} total")

    llm = InstrumentedLLM(
        model=vllm_cfg.get("model", "Gemma3-27B"),
        api_key=vllm_cfg.get("api_key", "EMPTY"),
        base_url=vllm_cfg.get("base_url", "http://localhost:18000/v1"),
    )

    vocab_csv = Path(run_cfg.get("vocab_csv", "data/atc_hierarchy.csv"))
    med_vocab_json = Path(run_cfg.get("med_vocab_json", "data/med_vocab.json"))
    vocab = load_med_vocab(med_vocab_json, atc_csv_path=vocab_csv)

    medi2_raw = load_medi2(Path(run_cfg.get("medi2_csv", "data/MEDI2_with_ATC_ICD10.csv")))
    medi2 = InstrumentedMEDI2(medi2_raw)

    log_dir = "logs/efficiency_bench"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    agent = build_saferx_agent(
        llm=llm,
        vocab=vocab,
        log_dir=log_dir,
        medi2=medi2,
        general_agent=args.general_agent,
    )
    patch_agent_phases(agent, llm)

    case_stats: list[dict] = []

    for i, row in enumerate(sampled):
        subject_id = row.get("subject_id", i)
        raw_text = row.get("input", "")
        patient_text = strip_suffix(str(raw_text))
        case_id = f"eff_{subject_id}"

        print(f"\n[{i+1}/{len(sampled)}] case={case_id} ...", flush=True)

        medi2.reset_count()
        llm.pop_records()

        t_start = time.perf_counter()
        case = PatientCase(case_id=case_id, patient_text=patient_text, patient_record=row)
        result = await agent.run(case)
        t_pipeline = time.perf_counter() - t_start

        final_pred = result.get("final_prediction", {}) or {}
        pred_codes = extract_codes(final_pred.get("final_predicted_drugs", []))

        t_sf_start = time.perf_counter()
        filtered_codes = pred_codes if args.general_agent else await run_safety_filter(llm, pred_codes, row, args.jsonl)
        t_sf = time.perf_counter() - t_sf_start

        t_total = time.perf_counter() - t_start
        records = llm.pop_records()
        medi2_count = medi2.reset_count()

        phase_stats: dict[str, dict] = {}
        for rec in records:
            p = rec.phase.split(":")[0]
            if p not in phase_stats:
                phase_stats[p] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0}
            phase_stats[p]["calls"] += 1
            phase_stats[p]["input_tokens"] += rec.input_tokens
            phase_stats[p]["output_tokens"] += rec.output_tokens
            phase_stats[p]["latency_s"] += rec.latency_s

        total_calls = len(records)
        total_in = sum(r.input_tokens for r in records)
        total_out = sum(r.output_tokens for r in records)

        stat = {
            "case_id": case_id,
            "subject_id": subject_id,
            "total_llm_calls": total_calls,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "medi2_lookups": medi2_count,
            "pipeline_latency_s": round(t_pipeline, 2),
            "safety_filter_latency_s": round(t_sf, 2),
            "total_latency_s": round(t_total, 2),
            "n_pred_codes": len(pred_codes),
            "n_filtered_codes": len(filtered_codes),
            "phase_breakdown": phase_stats,
            "n_skills": len(result.get("active_skills", [])),
        }
        case_stats.append(stat)
        print(f"  calls={total_calls}  in_tok={total_in}  out_tok={total_out}  "
              f"medi2={medi2_count}  latency={t_total:.1f}s  skills={stat['n_skills']}")

    n = len(case_stats)
    print(f"\n{'='*70}")
    print(f"EFFICIENCY BENCHMARK — {n} cases, model={vllm_cfg.get('model')}")
    print(f"{'='*70}")

    def avg(key):
        return sum(s[key] for s in case_stats) / n

    def fmt(val, decimals=1):
        return f"{val:.{decimals}f}"

    print(f"\n## Per-case averages")
    print(f"  Avg LLM calls:       {fmt(avg('total_llm_calls'))}")
    print(f"  Avg input tokens:    {fmt(avg('total_input_tokens'), 0)}")
    print(f"  Avg output tokens:   {fmt(avg('total_output_tokens'), 0)}")
    print(f"  Avg MEDI2 lookups:   {fmt(avg('medi2_lookups'))}")
    print(f"  Avg total latency:   {fmt(avg('total_latency_s'))} s")
    print(f"  Avg pipeline latency:{fmt(avg('pipeline_latency_s'))} s")
    print(f"  Avg safety latency:  {fmt(avg('safety_filter_latency_s'))} s")
    print(f"  Avg active skills:   {fmt(avg('n_skills'))}")

    all_phases = set()
    for s in case_stats:
        all_phases.update(s["phase_breakdown"].keys())

    print(f"\n## Phase-level averages")
    print(f"  {'Phase':<20} {'Calls':>6} {'InTok':>8} {'OutTok':>8} {'Latency':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for phase in sorted(all_phases):
        p_calls = sum(s["phase_breakdown"].get(phase, {}).get("calls", 0) for s in case_stats) / n
        p_in = sum(s["phase_breakdown"].get(phase, {}).get("input_tokens", 0) for s in case_stats) / n
        p_out = sum(s["phase_breakdown"].get(phase, {}).get("output_tokens", 0) for s in case_stats) / n
        p_lat = sum(s["phase_breakdown"].get(phase, {}).get("latency_s", 0) for s in case_stats) / n
        print(f"  {phase:<20} {p_calls:>6.1f} {p_in:>8.0f} {p_out:>8.0f} {p_lat:>7.1f}s")

    out_path = Path(log_dir) / "efficiency_summary.json"
    out_path.write_text(json.dumps({
        "config": str(args.config),
        "n_cases": n,
        "seed": args.seed,
        "model": vllm_cfg.get("model"),
        "avg_llm_calls": round(avg("total_llm_calls"), 2),
        "avg_input_tokens": round(avg("total_input_tokens"), 1),
        "avg_output_tokens": round(avg("total_output_tokens"), 1),
        "avg_medi2_lookups": round(avg("medi2_lookups"), 2),
        "avg_total_latency_s": round(avg("total_latency_s"), 2),
        "avg_pipeline_latency_s": round(avg("pipeline_latency_s"), 2),
        "avg_safety_filter_latency_s": round(avg("safety_filter_latency_s"), 2),
        "per_case": case_stats,
    }, indent=2))
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
