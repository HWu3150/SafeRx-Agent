# SafeRx-Agent

SafeRx-Agent is a knowledge-grounded multi-agent framework for safe and explainable ATC-L4 medication recommendation: routed specialty experts propose medications grounded in MEDI, a global critique reconciles them, and a safety verifier adjudicates DDI and contraindication flags.

## Repository structure

```
src/
  run_saferx.py              entry point: multi-agent generation + P/R/F1/Jaccard
  saferx/
    router/                  ROUTE: ICD-chapter + keyword scoring, ≤3 specialties + universal expert
    orchestrator/            SUMMARIZE / GENERATE / CRITIQUE and the traceable report
    skills/<expert>/         expert panel: summarize_template, drug_pred_checklist, critic_focus, report_template
    general_agent/           single generalist expert (ablation: w/o Specialty Experts)
  utils/                     LLM client, ATC vocabulary, MEDI lookup, metrics
scripts/
  llm_safety_filter.py       FINDFLAGS + VERIFY
  compute_bootstrap_ci.py    prediction and safety metrics with bootstrap CIs
  efficiency_benchmark.py    LLM calls, tokens and latency per case
configs/example.yaml         backbone config (OpenAI / Anthropic / vLLM)
data/                        vocabulary and knowledge resources (see Data)
```

## Data

Patient data is **not** included. SafeRx-Agent is evaluated on MIMIC-IV, which requires credentialed access through [PhysioNet](https://physionet.org/content/mimiciv/); after obtaining access, serialize each patient into one JSON line with `subject_id`, the EHR text `input`, `records` (per-visit `diagnosis` ICD-10, `procedure`, `medication` ATC codes) and the target-visit `target` ATC-L4 codes.

`data/` contains only the non-patient resources used for routing, grounding and safety checking:

| File | Content | Source |
|---|---|---|
| `med_vocab.json` | ATC-L4 prediction vocabulary (274 classes) | ATC codes retained in the MIMIC-IV cohort |
| `atc_hierarchy.csv` | ATC names and hierarchy | [WHO ATC classification](https://atcddd.fhi.no/) |
| `MEDI2_with_ATC_ICD10.csv` | ICD-10 → ATC-L4 indication relations | MEDI (Wei et al., 2013), mapped to ATC-L4 |
| `ddi_all_{binary,weighted}_A_final.pkl` | ATC-L4 drug–drug interaction matrices | TWOSIDES (Tatonetti et al., 2012) |
| `contraind_all_{binary,weighted}_atc_icd.csv` | ATC-L4 × ICD-10 contraindication matrices | openFDA drug labels (snapshot 2026-03-04) |
| `med_vocab_300.json` | row/column index of the DDI and contraindication matrices | — |

## Running

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...                            # or ANTHROPIC_API_KEY; vLLM needs none
export MIMIC4_JSONL=/path/to/mimic4_patients.jsonl   # one serialized patient per line (PhysioNet access required)

# 1. multi-agent generation (the first 100 cases are the prompt-development split)
python -m src.run_saferx --config configs/example.yaml --jsonl $MIMIC4_JSONL \
    --skip 100 --limit 100000 --log-dir logs/saferx

# 2. safety verification
python -m scripts.llm_safety_filter --config configs/example.yaml --patient-jsonl $MIMIC4_JSONL \
    --eval-dir logs/saferx --out-dir logs/saferx_verified

# 3. metrics
python scripts/compute_bootstrap_ci.py --patient-jsonl $MIMIC4_JSONL \
    --run "SafeRx w/o filter=logs/saferx" --run "SafeRx=logs/saferx_verified"

# ablations: --general-agent | --no-summarizer | --no-medi | --no-critique
python -m src.run_saferx ... --no-critique

# efficiency
python -m scripts.efficiency_benchmark --config configs/example.yaml --jsonl $MIMIC4_JSONL
```
