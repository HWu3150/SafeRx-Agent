#!/usr/bin/env python3
"""Compute bootstrap 95% CIs for all Table 2 & Table 3 metrics.

Reads the eval_overlay_*.json files written by src.run_saferx /
scripts.llm_safety_filter, pairs them with ground truth and diagnoses from
--patient-jsonl, computes per-patient metrics, then bootstraps (resampling
patients with replacement).

Table 2 metrics: Precision, Recall, F1, Jaccard (ATC-L4)
Table 3 metrics: DDI-B, DDI-W, Contra-B, Contra-W

Usage:
    python scripts/compute_bootstrap_ci.py \
        --patient-jsonl /path/to/mimic4_patients.jsonl \
        --run "SafeRx w/o filter=logs/saferx" \
        --run "SafeRx w/ filter=logs/saferx_verified"
    ... --n-boot 5000 --latex --json-out results/ci.json
"""

import argparse
import json
import os
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path):
    data = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            sid = str(rec.get("subject_id", rec.get("hadm_id", "")))
            data[sid] = rec
    return data


def get_gold(rec):
    records = rec.get("records", {})
    meds = records.get("medication", [])
    v = meds[-1] if meds else []
    return [c for c in v if isinstance(c, str)]


def get_dx(rec):
    records = rec.get("records", {})
    dx_raw = records.get("diagnosis", records.get("diagnoses", []))
    codes = []
    for v in (dx_raw if isinstance(dx_raw, list) else []):
        if isinstance(v, list):
            codes.extend(v)
        else:
            codes.append(v)
    return set(codes)


def pred_auto(c):
    """Predicted codes from a SafeRx (or verified SafeRx) overlay."""
    if "evaluated_pred_codes" in c:
        return c["evaluated_pred_codes"]
    pd_ = c.get("final_prediction", {}).get("final_predicted_drugs", [])
    if pd_ and isinstance(pd_[0], dict):
        return [d["code"] for d in pd_ if "code" in d]
    return pd_ or []


PREFIXES = ("eval_overlay_",)


def collect_per_patient(logdir, jdata, d2i, ddi_bin, ddi_wt,
                        contra_bin_df, contra_wt_df):
    """Collect per-patient metrics: returns list of dicts, one per patient."""
    files = [f for f in os.listdir(logdir)
             if f.startswith(PREFIXES) and f.endswith(".json")]
    if not files:
        return []

    patients = []
    for fname in sorted(files):
        with open(os.path.join(logdir, fname)) as f:
            case = json.load(f)
        prefix = next(p for p in PREFIXES if fname.startswith(p))
        sid = str(case.get("subject_id") or fname[len(prefix):-5])
        rec = jdata.get(sid)
        if not rec:
            continue

        gold = get_gold(rec)
        pred = pred_auto(case)

        g4 = set(g[:5] for g in gold)
        p4 = set(p[:5] for p in pred)
        tp = len(g4 & p4)
        fp = len(p4) - tp
        fn = len(g4) - tp
        jaccard = len(g4 & p4) / len(g4 | p4) if (g4 | p4) else 0.0

        # DDI metrics
        ngp = len(list(combinations(list(g4), 2)))
        db = dw = 0
        for a, b in combinations(p4, 2):
            ia, ib = d2i.get(a), d2i.get(b)
            if ia is not None and ib is not None:
                ia, ib = int(ia), int(ib)
                if ddi_bin[ia][ib] > 0:
                    db += 1
                dw += ddi_wt[ia][ib]
        ddi_b = db / ngp if ngp > 0 else None
        ddi_w = dw / ngp if ngp > 0 else None

        # Contra metrics
        dx = get_dx(rec)
        ngd = len(g4) * len(dx)
        cb_ = cw_ = 0
        for drug in p4:
            for d in dx:
                if drug in contra_bin_df.index and d in contra_bin_df.columns:
                    if contra_bin_df.loc[drug, d] > 0:
                        cb_ += 1
                    cw_ += contra_wt_df.loc[drug, d]
        contra_b = cb_ / ngd if ngd > 0 else None
        contra_w = cw_ / ngd if ngd > 0 else None

        patients.append({
            "sid": sid,
            "tp": tp, "fp": fp, "fn": fn,
            "jaccard": jaccard,
            "n_pred": len(p4), "n_gold": len(g4),
            "ddi_b": ddi_b, "ddi_w": ddi_w,
            "contra_b": contra_b, "contra_w": contra_w,
        })

    return patients


def micro_metrics_from_patients(patients):
    """Compute micro precision/recall/F1 from a list of patient dicts."""
    tp = sum(p["tp"] for p in patients)
    fp = sum(p["fp"] for p in patients)
    fn = sum(p["fn"] for p in patients)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    jacc = np.mean([p["jaccard"] for p in patients])
    return prec, rec, f1, jacc


def safety_metrics_from_patients(patients):
    """Compute mean safety metrics, skipping patients with None."""
    db = [p["ddi_b"] for p in patients if p["ddi_b"] is not None]
    dw = [p["ddi_w"] for p in patients if p["ddi_w"] is not None]
    cb = [p["contra_b"] for p in patients if p["contra_b"] is not None]
    cw = [p["contra_w"] for p in patients if p["contra_w"] is not None]
    return (
        np.mean(db) if db else 0.0,
        np.mean(dw) if dw else 0.0,
        np.mean(cb) if cb else 0.0,
        np.mean(cw) if cw else 0.0,
    )


def bootstrap_ci(patients, metric_fn, n_boot=2000, ci=0.95, seed=42):
    """Bootstrap CI by resampling patients with replacement."""
    rng = np.random.RandomState(seed)
    n = len(patients)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        sampled = [patients[i] for i in idx]
        boot_vals.append(metric_fn(sampled))
    boot_vals = np.array(boot_vals)
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_vals, 100 * alpha))
    hi = float(np.percentile(boot_vals, 100 * (1 - alpha)))
    mean = float(np.mean(boot_vals))
    return mean, lo, hi


def compute_all_bootstrap(patients, n_boot=2000, seed=42):
    """Compute point estimates and 95% CIs for all metrics."""
    prec, rec, f1, jacc = micro_metrics_from_patients(patients)
    ddi_b, ddi_w, contra_b, contra_w = safety_metrics_from_patients(patients)

    results = {"n": len(patients)}

    metric_fns = {
        "precision": lambda ps: micro_metrics_from_patients(ps)[0],
        "recall":    lambda ps: micro_metrics_from_patients(ps)[1],
        "f1":        lambda ps: micro_metrics_from_patients(ps)[2],
        "jaccard":   lambda ps: micro_metrics_from_patients(ps)[3],
        "ddi_b":     lambda ps: safety_metrics_from_patients(ps)[0],
        "ddi_w":     lambda ps: safety_metrics_from_patients(ps)[1],
        "contra_b":  lambda ps: safety_metrics_from_patients(ps)[2],
        "contra_w":  lambda ps: safety_metrics_from_patients(ps)[3],
    }

    point_estimates = {
        "precision": prec, "recall": rec, "f1": f1, "jaccard": jacc,
        "ddi_b": ddi_b, "ddi_w": ddi_w, "contra_b": contra_b, "contra_w": contra_w,
    }

    for name, fn in metric_fns.items():
        mean, lo, hi = bootstrap_ci(patients, fn, n_boot=n_boot, seed=seed)
        results[name] = {
            "point": point_estimates[name],
            "boot_mean": mean,
            "ci_lo": lo,
            "ci_hi": hi,
        }

    return results




def fmt_ci(val, lo, hi, latex=False):
    if latex:
        return f"${val:.4f}_{{[{lo:.4f},{hi:.4f}]}}$"
    return f"{val:.4f} [{lo:.4f}, {hi:.4f}]"


def main():
    parser = argparse.ArgumentParser(
        description="Compute bootstrap 95% CIs for Table 2 & Table 3 metrics")
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=LOG_DIR",
                        help="Run to evaluate. Repeatable, e.g. "
                             "--run 'SafeRx=logs/saferx_verified'")
    parser.add_argument("--patient-jsonl", type=Path, required=True,
                        help="Patient JSONL (gold = last-visit medications, diagnoses for Contra)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latex", action="store_true",
                        help="Output in LaTeX format")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Save full results to JSON file")
    args = parser.parse_args()

    root = PROJECT_ROOT

    # Load safety resources
    ddi_bin = pickle.load(open(root / "data/ddi_all_binary_A_final.pkl", "rb"))
    ddi_wt = pickle.load(open(root / "data/ddi_all_weighted_A_final.pkl", "rb"))
    contra_bin_df = pd.read_csv(root / "data/contraind_all_binary_atc_icd.csv", index_col=0)
    contra_wt_df = pd.read_csv(root / "data/contraind_all_weighted_atc_icd.csv", index_col=0)
    vocab = json.load(open(root / "data/med_vocab_300.json"))
    d2i = vocab["drug_to_index"]

    # Load ground truth
    jdata = load_jsonl(args.patient_jsonl)

    experiments = []
    for spec in args.run:
        label, rest = spec.split("=", 1)
        experiments.append((label, rest))

    all_json = {}
    table2_rows = []
    table3_rows = []

    for name, logdir in experiments:
        full_logdir = Path(logdir) if Path(logdir).is_absolute() else root / logdir
        if not full_logdir.is_dir():
            print(f"  {name}: {full_logdir} not found, skipping")
            continue

        patients = collect_per_patient(
            str(full_logdir), jdata,
            d2i, ddi_bin, ddi_wt, contra_bin_df, contra_wt_df
        )
        if not patients:
            print(f"  {name}: no cases found, skipping")
            continue

        res = compute_all_bootstrap(patients, n_boot=args.n_boot, seed=args.seed)
        all_json[name] = res

        # Table 2 row
        t2 = (
            name, res["n"],
            res["precision"]["point"], res["precision"]["ci_lo"], res["precision"]["ci_hi"],
            res["recall"]["point"],    res["recall"]["ci_lo"],    res["recall"]["ci_hi"],
            res["f1"]["point"],        res["f1"]["ci_lo"],        res["f1"]["ci_hi"],
            res["jaccard"]["point"],   res["jaccard"]["ci_lo"],   res["jaccard"]["ci_hi"],
        )
        table2_rows.append(t2)

        # Table 3 row
        t3 = (
            name, res["n"],
            res["ddi_b"]["point"],   res["ddi_b"]["ci_lo"],   res["ddi_b"]["ci_hi"],
            res["ddi_w"]["point"],   res["ddi_w"]["ci_lo"],   res["ddi_w"]["ci_hi"],
            res["contra_b"]["point"],res["contra_b"]["ci_lo"],res["contra_b"]["ci_hi"],
            res["contra_w"]["point"],res["contra_w"]["ci_lo"],res["contra_w"]["ci_hi"],
        )
        table3_rows.append(t3)

    # Print Table 2
    print("\n" + "=" * 140)
    print("TABLE 2: Prediction Performance (ATC-L4) with 95% Bootstrap CIs")
    print("=" * 140)
    hdr = f"{'Method':<28s} {'N':>4s}  {'Precision':>22s}  {'Recall':>22s}  {'F1':>22s}  {'Jaccard':>22s}"
    print(hdr)
    print("-" * 140)
    for row in table2_rows:
        name, n = row[0], row[1]
        prec_s = fmt_ci(row[2], row[3], row[4], args.latex)
        rec_s  = fmt_ci(row[5], row[6], row[7], args.latex)
        f1_s   = fmt_ci(row[8], row[9], row[10], args.latex)
        j_s    = fmt_ci(row[11], row[12], row[13], args.latex)
        print(f"{name:<28s} {n:>4d}  {prec_s:>22s}  {rec_s:>22s}  {f1_s:>22s}  {j_s:>22s}")

    # Print Table 3
    print("\n" + "=" * 140)
    print("TABLE 3: Safety Metrics with 95% Bootstrap CIs (lower is better)")
    print("=" * 140)
    hdr = f"{'Method':<28s} {'N':>4s}  {'DDI-B':>22s}  {'DDI-W':>22s}  {'Contra-B':>22s}  {'Contra-W':>22s}"
    print(hdr)
    print("-" * 140)
    for row in table3_rows:
        name, n = row[0], row[1]
        db_s = fmt_ci(row[2], row[3], row[4], args.latex)
        dw_s = fmt_ci(row[5], row[6], row[7], args.latex)
        cb_s = fmt_ci(row[8], row[9], row[10], args.latex)
        cw_s = fmt_ci(row[11], row[12], row[13], args.latex)
        print(f"{name:<28s} {n:>4d}  {db_s:>22s}  {dw_s:>22s}  {cb_s:>22s}  {cw_s:>22s}")

    if args.json_out:
        out_path = Path(args.json_out)
        with open(out_path, "w") as f:
            json.dump(all_json, f, indent=2, default=float)
            f.write("\n")
        print(f"\nFull results saved to {out_path}")

    if not table2_rows:
        print("\nNo experiment log directories found. Make sure logs/ contains per-case eval JSONs.")

    return all_json


if __name__ == "__main__":
    main()
