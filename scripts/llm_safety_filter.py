"""Safety verification: FindFlags (deterministic) + Verify (LLM, clinical pharmacist).

FindFlags raises DDI pairs among the predicted drugs and drug-diagnosis
contraindications. Verify sees the patient record, the predicted drugs,
prior medications, the flags and MEDI indication evidence for flagged drugs,
and may only RETAIN or REMOVE flagged drugs: it never adds or replaces codes,
and unflagged drugs are always retained. If no flags are raised, the
verifier is not invoked.

Input: eval_overlay_*.json written by src.run_saferx. Output: verified overlays
+ safety_summary.md with P/R/F1/Jaccard and DDI-B/W, Contra-B/W (GT-normalized).
Works with any OpenAI-compatible endpoint (vLLM / API).
"""

import argparse
import asyncio
import json
import glob
import pickle
import re
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent  # repo root
DDI_BIN   = ROOT / 'data/ddi_all_binary_A_final.pkl'
DDI_WT    = ROOT / 'data/ddi_all_weighted_A_final.pkl'
CONTRA_B  = ROOT / 'data/contraind_all_binary_atc_icd.csv'
CONTRA_W  = ROOT / 'data/contraind_all_weighted_atc_icd.csv'
VOCAB     = ROOT / 'data/med_vocab_300.json'
ATC_CSV   = ROOT / 'data/atc_hierarchy.csv'
MEDI2_CSV = ROOT / 'data/MEDI2_with_ATC_ICD10.csv'
PROMPT_SUFFIX = 'Then, the patient should be prescribed:'


# ── Loaders ──────────────────────────────────────────────────────────────

def load_resources(patient_jsonl):
    ddi_b = pickle.load(open(DDI_BIN, 'rb'))
    ddi_w = pickle.load(open(DDI_WT, 'rb'))
    v = json.load(open(VOCAB))
    d2i = v['drug_to_index']
    i2d = v['index_to_drug']
    deg = {d: int(ddi_b[i].sum()) for d, i in d2i.items()}
    contra_b = pd.read_csv(CONTRA_B, index_col=0)
    contra_w = pd.read_csv(CONTRA_W, index_col=0)
    # ATC L4 code → name
    atc = pd.read_csv(ATC_CSV)
    code_name = dict(zip(atc['ATC_L4_Code'], atc['ATC_L4_Name']))
    # Patient diagnoses
    by_sid = {}
    with open(patient_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            dx = rec['records']['diagnosis']
            by_sid[str(rec['subject_id'])] = list(dx[-1]) if isinstance(dx[-1], list) else list(dx)
    return ddi_b, ddi_w, d2i, i2d, deg, contra_b, contra_w, code_name, by_sid


def load_patient_texts(patient_jsonl) -> dict:
    """subject_id -> serialized patient record X_T (prediction suffix stripped)."""
    texts = {}
    with open(patient_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            text = str(rec.get('input', ''))
            idx = text.rfind(PROMPT_SUFFIX)
            texts[str(rec['subject_id'])] = text[:idx].rstrip() if idx != -1 else text
    return texts


def load_indication_resource(medi2_csv=None):
    """Indication resource R (MEDI) used by Verify."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.utils.medi2 import load_medi2
    return load_medi2(str(medi2_csv or MEDI2_CSV))


def get_indication_evidence(medi2, patient_dx, flagged) -> dict:
    """For each flagged drug, MEDI indications linking it to the patient's current diagnoses."""
    if medi2 is None or not flagged:
        return {}
    evidence = {}
    for hit in medi2.lookup(list(patient_dx), high_precision_only=True):
        if hit['atc'] in flagged:
            evidence.setdefault(hit['atc'], []).append((hit['icd_code'], hit['indication']))
    return evidence


# ── Safety info extraction ──────────────────────────────────────────────

def get_ddi_pairs(preds, ddi_b, d2i, deg):
    """Return list of (drugA, drugB, degA, degB) for interacting pairs."""
    preds_in = [p for p in preds if p in d2i]
    pairs = []
    for i in range(len(preds_in)):
        for j in range(i + 1, len(preds_in)):
            a, b = preds_in[i], preds_in[j]
            if ddi_b[d2i[a], d2i[b]] > 0:
                pairs.append((a, b, deg.get(a, 0), deg.get(b, 0)))
    return pairs


def get_contra_pairs(preds, patient_dx, contra_b):
    """Return list of (drug, diagnosis_code) contraindicated pairs."""
    rows = set(contra_b.index)
    cols = set(contra_b.columns)
    dxs = [c for c in patient_dx if c in cols]
    if not dxs:
        return []
    pairs = []
    for p in preds:
        c4 = p[:5] if isinstance(p, str) and len(p) >= 5 else p
        if c4 not in rows:
            continue
        for dx in dxs:
            if contra_b.loc[c4, dx] > 0:
                pairs.append((c4, dx))
    return pairs


# ── Prompt builder ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a clinical pharmacist safety reviewer. Given the patient record, a \
list of predicted prescription drugs, safety signals (drug-drug interactions \
and contraindications) and indication evidence linking flagged drugs to the \
patient's diagnoses, decide, for each flagged drug, whether to RETAIN or \
REMOVE it.

Rules:
1. If a drug is in prior_meds (patient was on it before this admission), \
prefer to KEEP it unless the interaction is severe.
2. When a DDI pair is flagged, prefer removing the drug with the higher \
global DDI degree (more interactions overall) and lower clinical necessity.
3. When a drug is contraindicated with a patient diagnosis, remove it \
unless it was in prior_meds and the benefit clearly outweighs the risk.
4. RETAIN a flagged drug when the patient context supports its use (e.g. a \
previously tolerated pair, or a documented indication); REMOVE it when the \
risk lacks patient-specific justification.
5. Only flagged drugs (those appearing in a DDI or contraindication pair) \
may be removed. Never add new drugs and never suggest replacements.
6. Be conservative: only remove drugs with clear safety signals. Do NOT \
remove drugs just because they have a low DDI degree.
7. Output ONLY valid JSON — no markdown fences, no commentary.

Output format (JSON):
{
  "kept_drugs": ["CODE1", "CODE2", ...],
  "removed_drugs": [
    {"code": "CODE3", "reason": "DDI with CODE1 (degree 45)"},
    {"code": "CODE4", "reason": "contraindicated with E119"}
  ]
}
"""


def build_user_prompt(preds, prior_meds, ddi_pairs, contra_pairs, code_name,
                      patient_text="", indication_evidence=None):
    lines = []

    lines.append("## Patient record")
    lines.append(patient_text.strip() if patient_text else "(not available)")
    lines.append("")

    lines.append("## Predicted drugs")
    for p in preds:
        name = code_name.get(p, "unknown")
        prior_tag = " [PRIOR-MED]" if p in prior_meds else ""
        lines.append(f"  - {p}: {name}{prior_tag}")

    lines.append("")
    lines.append("## Prior medications (active in previous visit)")
    if prior_meds:
        for pm in sorted(prior_meds):
            lines.append(f"  - {pm}: {code_name.get(pm, 'unknown')}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("## Drug-Drug Interaction (DDI) pairs detected")
    if ddi_pairs:
        for a, b, da, db in ddi_pairs:
            na = code_name.get(a, "?")
            nb = code_name.get(b, "?")
            a_prior = " [PRIOR]" if a in prior_meds else ""
            b_prior = " [PRIOR]" if b in prior_meds else ""
            lines.append(
                f"  - {a} ({na}, degree={da}){a_prior}  ↔  {b} ({nb}, degree={db}){b_prior}"
            )
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("## Contraindication pairs detected (drug ↔ patient diagnosis)")
    if contra_pairs:
        for drug, dx in contra_pairs:
            nd = code_name.get(drug, "?")
            drug_prior = " [PRIOR]" if drug in prior_meds else ""
            lines.append(f"  - {drug} ({nd}){drug_prior}  ↔  diagnosis {dx}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("## Indication evidence for flagged drugs (MEDI: patient diagnosis → drug)")
    if indication_evidence:
        for drug in sorted(indication_evidence):
            items = "; ".join(f"{icd}: {ind}" for icd, ind in indication_evidence[drug][:3])
            lines.append(f"  - {drug} ({code_name.get(drug, '?')}): {items}")
    else:
        lines.append("  (no indication evidence found for the flagged drugs)")

    lines.append("")
    lines.append(
        "Based on the patient record, the safety signals and the indication "
        "evidence, decide for each flagged drug whether to retain or remove it. "
        "Output JSON only."
    )
    return "\n".join(lines)


# ── LLM call ────────────────────────────────────────────────────────────

def flagged_codes(ddi_pairs, contra_pairs) -> set[str]:
    """Drugs that FindFlags raised at least one flag for."""
    flagged = set()
    for a, b, _da, _db in ddi_pairs:
        flagged.add(a)
        flagged.add(b)
    for drug, _dx in contra_pairs:
        flagged.add(drug)
    return flagged


def parse_llm_response(text: str, original_preds: list[str], flagged: set[str] | None = None) -> list[str]:
    """Apply Verify decisions (Ret/Rem) to the candidate set.

    Only drugs in ``flagged`` can be removed; nothing is ever added.
    Unparseable output -> keep the original set unchanged.
    """
    flagged = set(original_preds) if flagged is None else set(flagged)
    # Strip thinking tags (Qwen3 thinking mode)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Strip markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start < 0 or end <= start:
            return list(original_preds)
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return list(original_preds)
    if not isinstance(data, dict):
        return list(original_preds)

    def _norm(c):
        return str(c).strip().upper()[:5]

    removed_items = data.get('removed_drugs')
    if isinstance(removed_items, list):
        removed = {
            _norm(item.get('code', '') if isinstance(item, dict) else item)
            for item in removed_items
        }
    else:
        # No removed list: infer Rem as flagged drugs missing from kept_drugs.
        kept = data.get('kept_drugs')
        if not isinstance(kept, list):
            return list(original_preds)
        kept_set = {_norm(c) for c in kept}
        removed = {p for p in flagged if _norm(p) not in kept_set}

    return [p for p in original_preds if not (p in flagged and _norm(p) in removed)]


def verify_decisions(text: str, preds: list[str], filtered: list[str], flagged: set[str]) -> list[dict]:
    """Per-flag Retain/Remove decisions with the verifier's reasons (for the report)."""
    reasons = {}
    try:
        cleaned = re.sub(r'<think>.*?</think>', '', text or '', flags=re.DOTALL)
        data = json.loads(cleaned[cleaned.find('{'):cleaned.rfind('}') + 1])
        for item in data.get('removed_drugs', []) or []:
            if isinstance(item, dict):
                reasons[str(item.get('code', '')).strip().upper()[:5]] = str(item.get('reason', ''))
    except Exception:
        pass
    kept = set(filtered)
    return [
        {'code': p, 'action': 'Ret' if p in kept else 'Rem', 'reason': reasons.get(p, '')}
        for p in preds if p in flagged
    ]


def update_report(case: dict, filtered: list[str], decisions: list[dict]) -> dict | None:
    """Apply Verify decisions to the pipeline's traceable report, if present."""
    report = case.get('report')
    if not isinstance(report, dict):
        return report
    kept = set(filtered)
    report = dict(report)
    report['medications'] = [m for m in report.get('medications', []) if m.get('code') in kept]
    report['removed'] = list(report.get('removed', [])) + [
        {'code': d['code'], 'stage': 'verify', 'reason': d['reason']} for d in decisions if d['action'] == 'Rem'
    ]
    report['safety_verification'] = decisions
    return report


# ── Metrics ──────────────────────────────────────────────────────

def codes_to_indices(codes, d2i):
    seen, out = set(), []
    for c in codes:
        c4 = c[:5] if isinstance(c, str) and len(c) >= 5 else ''
        if c4 in d2i and c4 not in seen:
            out.append(d2i[c4]); seen.add(c4)
    return out


def c_ddi(pred_idx, adj, weighted):
    s = 0.0
    n = len(pred_idx)
    for i in range(n):
        for j in range(i + 1, n):
            v = adj[pred_idx[i], pred_idx[j]]
            s += float(v) if weighted else (1.0 if v > 0 else 0.0)
    return s


def gtnorm_visit(num, gold_size, denom_kind, dx_size=None):
    if denom_kind == 'pair':
        denom = max(gold_size * (gold_size - 1) / 2, 1)
    else:
        denom = max(gold_size * dx_size, 1)
    return min(num / denom, 1.0)


def keep_atc(codes, mat_rows):
    seen, out = set(), []
    for c in codes:
        c4 = c[:5] if isinstance(c, str) and len(c) >= 5 else ''
        if c4 in mat_rows and c4 not in seen:
            out.append(c4); seen.add(c4)
    return out


def keep_icd(codes, mat_cols):
    seen, out = set(), []
    for c in codes:
        c = c.strip()
        if c in mat_cols and c not in seen:
            out.append(c); seen.add(c)
    return out


def aggregate(pred_lists, gold_lists, dx_lists,
              ddi_b, ddi_w, contra_b, contra_w, d2i):
    n = len(pred_lists)
    tp = fp = fn = 0
    j_sum = 0.0
    pred_size_sum = 0
    ddi_b_scores, ddi_w_scores = [], []
    cb_scores, cw_scores = [], []
    contra_b_rows = set(contra_b.index)
    contra_b_cols = set(contra_b.columns)
    contra_w_rows = set(contra_w.index)
    contra_w_cols = set(contra_w.columns)
    for preds, gold, dx in zip(pred_lists, gold_lists, dx_lists):
        pset = {p[:5] for p in preds if len(p) >= 5}
        gset = {g[:5] for g in gold if len(g) >= 5}
        tp += len(pset & gset)
        fp += len(pset - gset)
        fn += len(gset - pset)
        u = pset | gset
        j_sum += len(pset & gset) / len(u) if u else 0.0
        pred_size_sum += len(pset)
        gold_size = len(set(gold))
        dx_size = len(set(dx))
        pi = codes_to_indices(preds, d2i)
        ddi_b_scores.append(gtnorm_visit(c_ddi(pi, ddi_b, False), gold_size, 'pair'))
        ddi_w_scores.append(gtnorm_visit(c_ddi(pi, ddi_w, True), gold_size, 'pair'))
        pred_drugs_b = keep_atc(preds, contra_b_rows)
        dxs_b = keep_icd(dx, contra_b_cols)
        c_b = float(contra_b.loc[pred_drugs_b, dxs_b].values.sum()) if pred_drugs_b and dxs_b else 0.0
        cb_scores.append(gtnorm_visit(c_b, gold_size, 'bipartite', dx_size))
        pred_drugs_w = keep_atc(preds, contra_w_rows)
        dxs_w = keep_icd(dx, contra_w_cols)
        c_w = float(contra_w.loc[pred_drugs_w, dxs_w].values.sum()) if pred_drugs_w and dxs_w else 0.0
        cw_scores.append(gtnorm_visit(c_w, gold_size, 'bipartite', dx_size))
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return dict(
        P=P, R=R, F1=F1, J=j_sum / n,
        TP=tp, FP=fp, FN=fn,
        avg_pred=pred_size_sum / n,
        ddi_b=float(np.mean(ddi_b_scores)),
        ddi_w=float(np.mean(ddi_w_scores)),
        contra_b=float(np.mean(cb_scores)),
        contra_w=float(np.mean(cw_scores)),
    )


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--eval-dir', required=True, help='Log dir written by src.run_saferx')
    parser.add_argument('--out-dir', required=True, help='Where verified overlays are written')
    parser.add_argument('--out-md', default=None, help='Summary markdown (default: <out-dir>/safety_summary.md)')
    parser.add_argument('--patient-jsonl', required=True,
                        help='Patient JSONL for diagnosis lookup')
    parser.add_argument('--medi2-csv', default=str(MEDI2_CSV),
                        help='MEDI indication resource used by Verify')
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_md = Path(args.out_md) if args.out_md else out_dir / 'safety_summary.md'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config for LLM endpoint
    cfg = yaml.safe_load(open(args.config))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.utils.llm import load_llm_section
    llm_cfg = load_llm_section(cfg)

    from openai import OpenAI
    client = OpenAI(
        api_key=llm_cfg['api_key'],
        base_url=llm_cfg['base_url'],
    )
    client._model_name = llm_cfg['model']

    # Load resources
    print("Loading resources...")
    ddi_b, ddi_w, d2i, i2d, deg, contra_b, contra_w, code_name, by_sid = load_resources(args.patient_jsonl)
    patient_texts = load_patient_texts(args.patient_jsonl)
    medi2 = load_indication_resource(args.medi2_csv)

    files = sorted(glob.glob(str(eval_dir / 'eval_overlay_*.json')))
    print(f'N eval files: {len(files)}')

    # Skip already-processed files (resume support)
    todo = []
    for fp in files:
        out_fp = out_dir / Path(fp).name
        if out_fp.exists():
            continue
        todo.append(fp)
    print(f'Already done: {len(files) - len(todo)}, remaining: {len(todo)}')

    # Process cases
    sem = asyncio.Semaphore(args.concurrency)

    async def process_one(fp: str):
        async with sem:
            d = json.load(open(fp))
            sid = str(d.get('subject_id') or '')
            preds = list(d.get('evaluated_pred_codes') or [])
            gold = list(d.get('gold_codes') or [])
            prior = set(d.get('prior_meds') or [])
            dx = by_sid.get(sid, [])

            ddi_pairs = get_ddi_pairs(preds, ddi_b, d2i, deg)
            contra_pairs = get_contra_pairs(preds, dx, contra_b)

            flagged = flagged_codes(ddi_pairs, contra_pairs)
            # If no safety signals, keep as-is (verifier not invoked)
            if not ddi_pairs and not contra_pairs:
                filtered = list(preds)
                llm_raw = ""
            else:
                user_prompt = build_user_prompt(
                    preds, prior, ddi_pairs, contra_pairs, code_name,
                    patient_text=patient_texts.get(sid, ""),
                    indication_evidence=get_indication_evidence(medi2, dx, flagged),
                )
                try:
                    def _sync_call():
                        kwargs = dict(
                            model=client._model_name,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.0,
                        )
                        if 'gpt-5' in client._model_name or 'o3' in client._model_name or 'o4' in client._model_name:
                            kwargs['max_completion_tokens'] = 2048
                        else:
                            kwargs['max_tokens'] = 2048
                        response = client.chat.completions.create(**kwargs)
                        return response.choices[0].message.content or ""

                    llm_raw = await asyncio.to_thread(_sync_call)
                    filtered = parse_llm_response(llm_raw, preds, flagged)
                except Exception as e:
                    print(f"  LLM error for {Path(fp).name}: {e}")
                    traceback.print_exc()
                    filtered = list(preds)
                    llm_raw = f"ERROR: {e}"

            decisions = verify_decisions(llm_raw, preds, filtered, flagged)
            out_d = dict(d,
                         evaluated_pred_codes=filtered,
                         _verify_decisions=decisions,
                         _filter='llm_safety',
                         _llm_safety_raw=llm_raw,
                         _ddi_pairs_found=len(ddi_pairs),
                         _contra_pairs_found=len(contra_pairs))
            if isinstance(d.get('report'), dict):
                out_d['report'] = update_report(d, filtered, decisions)
            out_fp = out_dir / Path(fp).name
            out_fp.write_text(json.dumps(out_d, indent=2, ensure_ascii=False))
            n_removed = len(set(preds) - set(filtered))
            tag = f"  -{n_removed}" if n_removed else ""
            if not ddi_pairs and not contra_pairs:
                tag = "  (no signals)"
            print(f"  {Path(fp).name}: {len(preds)}→{len(filtered)}{tag}")

    async def run_all():
        tasks = [process_one(fp) for fp in todo]
        await asyncio.gather(*tasks)

    asyncio.run(run_all())

    # ── Compute metrics ──────────────────────────────────────────────────
    print("\nComputing metrics...")
    all_files = sorted(glob.glob(str(out_dir / 'eval_overlay_*.json')))
    raw_preds, filtered_preds, golds, dxs = [], [], [], []

    removal_counter = Counter()
    cases_with_signals = 0
    cases_with_removal = 0
    total_removed = 0
    tp_loss = 0
    fp_clean = 0

    for fp in all_files:
        d = json.load(open(fp))
        sid = str(d.get('subject_id') or '')
        filt = list(d.get('evaluated_pred_codes') or [])
        # Load original preds from source
        orig_fp = eval_dir / Path(fp).name
        orig_d = json.load(open(orig_fp))
        orig = list(orig_d.get('evaluated_pred_codes') or [])
        gold = list(orig_d.get('gold_codes') or [])
        dx = by_sid.get(sid, [])
        gold_set = set(gold)

        removed = set(orig) - set(filt)
        if d.get('_ddi_pairs_found', 0) > 0 or d.get('_contra_pairs_found', 0) > 0:
            cases_with_signals += 1
        if removed:
            cases_with_removal += 1
            total_removed += len(removed)
            for r in removed:
                removal_counter[r] += 1
                if r in gold_set:
                    tp_loss += 1
                else:
                    fp_clean += 1

        raw_preds.append(orig)
        filtered_preds.append(filt)
        golds.append(gold)
        dxs.append(dx)

    raw_m = aggregate(raw_preds, golds, dxs, ddi_b, ddi_w, contra_b, contra_w, d2i)
    filt_m = aggregate(filtered_preds, golds, dxs, ddi_b, ddi_w, contra_b, contra_w, d2i)
    gold_m = aggregate(golds, golds, dxs, ddi_b, ddi_w, contra_b, contra_w, d2i)
    n = len(all_files)

    # ── Write markdown report ────────────────────────────────────────────
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, 'w') as f:
        f.write('# Safety verification (FindFlags + Verify)\n\n')
        f.write(f'Cases: n={n}. Model: `{client._model_name}`\n\n')

        f.write('## Method\n\n')
        f.write('For each flagged case, Verify is given:\n')
        f.write('- The serialized patient record\n')
        f.write('- Predicted drug list with ATC names\n')
        f.write('- All DDI pairs among predicted drugs (with global DDI degree)\n')
        f.write('- All contraindication pairs (drug ↔ patient diagnosis)\n')
        f.write('- Prior-medication status for each drug\n')
        f.write('- MEDI indication evidence for flagged drugs\n\n')
        f.write('The LLM decides, for each flagged drug, whether to retain or remove it (no additions).\n\n')

        f.write('## Metrics (lower-is-better for DDI / Contra)\n\n')
        f.write('| Metric | raw | LLM-filtered | Gold (self) |\n')
        f.write('|---|---:|---:|---:|\n')
        for k, lab in [('F1', 'F1'), ('P', 'Precision'), ('R', 'Recall'), ('J', 'Jaccard'),
                       ('avg_pred', 'Avg pred size'),
                       ('ddi_b', 'DDI_GTnorm (binary)'), ('ddi_w', 'DDI_GTnorm (weighted)'),
                       ('contra_b', 'Contra_GTnorm (binary)'), ('contra_w', 'Contra_GTnorm (weighted)')]:
            r = raw_m[k]; a = filt_m[k]; g = gold_m[k]
            if k == 'avg_pred':
                f.write(f'| {lab} | {r:.2f} | {a:.2f} | {g:.2f} |\n')
            else:
                f.write(f'| {lab} | {r:.4f} | {a:.4f} | {g:.4f} |\n')
        f.write('\n')

        f.write('### Δ vs raw\n\n')
        f.write('| Metric | LLM Δ |\n|---|---:|\n')
        for k, lab in [('F1', 'F1'), ('P', 'Precision'), ('R', 'Recall'), ('J', 'Jaccard'),
                       ('avg_pred', 'Avg pred size'),
                       ('ddi_b', 'DDI_GTnorm (binary)'), ('ddi_w', 'DDI_GTnorm (weighted)'),
                       ('contra_b', 'Contra_GTnorm (binary)'), ('contra_w', 'Contra_GTnorm (weighted)')]:
            d1 = filt_m[k] - raw_m[k]
            if k == 'avg_pred':
                f.write(f'| {lab} | {d1:+.2f} |\n')
            else:
                f.write(f'| {lab} | {d1:+.4f} |\n')
        f.write('\n')

        f.write('## Removal accounting\n\n')
        f.write(f'- Cases with safety signals: **{cases_with_signals}/{n}**\n')
        f.write(f'- Cases with ≥1 removal: **{cases_with_removal}/{n}**\n')
        f.write(f'- Total drugs removed: **{total_removed}**\n')
        f.write(f'- FP cleanups (removed drug NOT in gold): **{fp_clean}**\n')
        f.write(f'- TP losses (removed drug IN gold): **{tp_loss}**\n\n')

        f.write('### Top removed drugs\n\n')
        f.write('| Drug | Name | Times removed |\n|---|---|---:|\n')
        for drug, c in removal_counter.most_common(15):
            f.write(f'| {drug} | {code_name.get(drug, "?")} | {c} |\n')


    print(f'\nWrote {out_md}')

    # Summary table to stdout
    print('\n' + '=' * 80)
    print(f'{"Metric":<28} {"raw":>10} {"LLM-filter":>12} {"Gold":>10}')
    print('-' * 80)
    for k, lab in [('F1', 'F1'), ('P', 'Precision'), ('R', 'Recall'), ('J', 'Jaccard'),
                   ('avg_pred', 'Avg pred size'),
                   ('ddi_b', 'DDI_GTnorm bin'), ('ddi_w', 'DDI_GTnorm wt'),
                   ('contra_b', 'Contra_GTnorm bin'), ('contra_w', 'Contra_GTnorm wt')]:
        r = raw_m[k]; a = filt_m[k]; g = gold_m[k]
        if k == 'avg_pred':
            print(f'{lab:<28} {r:>10.2f} {a:>12.2f} {g:>10.2f}')
        else:
            print(f'{lab:<28} {r:>10.4f} {a:>12.4f} {g:>10.4f}')


if __name__ == '__main__':
    main()
