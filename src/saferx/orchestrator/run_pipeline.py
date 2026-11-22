from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..router import RoutingConfig, load_routing_config, route_patient_case
from ..skills import ExpertiseSkill, discover_skills
from ...utils import LLMClient, Vocabulary
from ...utils.medi2 import MEDI2Lookup
from .merge_outputs import merge_skill_outputs
from .select_skills import select_skills


@dataclass
class PatientCase:
    case_id: str
    patient_text: str
    patient_record: Dict[str, Any] | None = None


@dataclass
class SkillState:
    skill_id: str
    skill_name: str
    case_id: str
    patient_text: str
    routing_context: Dict[str, Any]
    summary: Dict[str, Any] = field(default_factory=dict)  # (s_i, rho_i) from Summarize
    proposal: Dict[str, Any] | None = None
    final_output: Dict[str, Any] | None = None
    trace: List[Dict[str, Any]] = field(default_factory=list)


def _safe_json_parse(text: str, fallback: Dict[str, Any] | None = None) -> Dict[str, Any]:
    fallback = fallback or {}
    if not text:
        return fallback
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.index("{")
        end = text.rindex("}")
        return json.loads(text[start : end + 1])
    except Exception:
        return fallback


class _SafeDict(dict):
    def __missing__(self, key):
        return ""


def _render(template: str, **kwargs) -> str:
    return template.format_map(_SafeDict(**kwargs))


def _dedupe_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = str(item).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


GENERAL_AGENT_DIR = Path(__file__).resolve().parent.parent / "general_agent"
SUPPORTIVE_ID = "universal_supportive"
GENERAL_AGENT_ID = "general_agent"


class SafeRxAgent:
    """SafeRx-Agent multi-agent generation (Algorithm 1, lines 1-5).

    Route (deterministic) -> per activated expert: Summarize -> Generate
    -> one global attending Critique over the union of proposals.
    FindFlags + Verify run afterwards (scripts/llm_safety_filter.py).

    Switches used for the paper's ablations:
      general_agent=True   one generalist expert instead of the routed panel
                           ("w/o Specialty Experts")
      use_summarizer=False Generate works from the raw record only ("w/o Summarizer")
      medi2=None           no MEDI indication candidates ("w/o Medical Resource")
      use_critique=False   skip the global Critique ("w/o Critique")
    """

    def __init__(
        self,
        llm: LLMClient,
        vocab: Vocabulary,
        log_dir: str | Path | None = "logs/saferx",
        skills_root: str | Path | None = None,
        routing_config_path: str | Path | None = None,
        medi2: Optional[MEDI2Lookup] = None,
        general_agent: bool = False,
        use_summarizer: bool = True,
        use_critique: bool = True,
    ):
        self.llm = llm
        self.vocab = vocab
        self.log_dir = Path(log_dir) if log_dir else None
        self.general_agent = general_agent
        self.skills = discover_skills(root_dir=GENERAL_AGENT_DIR if general_agent else skills_root)
        self.routing_config: RoutingConfig = load_routing_config(routing_config_path)
        self.routing_config_path = routing_config_path
        self.medi2 = medi2
        self.use_summarizer = use_summarizer
        self.use_critique = use_critique

    async def _summarize(self, case: PatientCase, skill: ExpertiseSkill) -> Dict[str, Any]:
        """Per-expert Summarize operator: returns (s_i, rho_i) as one JSON dict.

        s_i  = typed summary (Sec. 4.3): active problems, acute organ dysfunction,
               procedures, care context, medication-relevant risks, prior prescriptions
        rho_i = visit_rationales (per prior visit: diagnoses/procedures -> medications,
               continuing vs. transient)
        """
        prior_meds = self._get_prior_med_codes(case)
        prior_meds_block = ""
        if prior_meds:
            lines = ["Prior-visit medication codes (from the previous admission):"]
            for code in prior_meds:
                path = self.vocab.hierarchy_path(code) if self.vocab else None
                name = self.vocab.code_name(code) if self.vocab else "unknown"
                label = path if path else f"{code}: {name or 'unknown'}"
                lines.append(f"  - {label}")
            prior_meds_block = "\n".join(lines)

        diag_codes = self._get_diagnosis_codes(case)
        diag_block = "Current-visit ICD-10 diagnosis codes: " + ", ".join(diag_codes[:30]) if diag_codes else ""

        prior_diag_block = ""
        record = case.patient_record or {}
        records = record.get("records", {})
        if isinstance(records, dict):
            diag_visits = records.get("diagnosis", [])
            if diag_visits and len(diag_visits) >= 2:
                prior_diag = diag_visits[0] if isinstance(diag_visits[0], list) else []
                if prior_diag:
                    prior_diag_block = "Prior-visit ICD-10 diagnosis codes: " + ", ".join(str(c) for c in prior_diag[:30])

        prompt = _render(
            """You are the activated `{skill_name}` specialty module in a routed inpatient medication prediction system.

Your job at this stage is to SUMMARIZE the patient record from the perspective of your specialty, so that your own medication generator can later predict ATC L4 medication classes for the CURRENT admission.

## Specialty playbook
{summarize_template}

## Rules
- Ground every statement in the patient record. Do not invent clinical details.
- Do NOT predict medications at this stage. Only analyze what is documented.
- Highlight risks and care-context factors that matter for your specialty.
- For EACH prior visit, explain which diagnoses/procedures most likely led to which prior medications, and whether that indication is still active in the current admission (continuing) or was transient.
- If there is no prior visit or no prior medication, say so explicitly in visit_rationales.

## Patient Record
{patient_text}

{diag_block}

{prior_diag_block}

{prior_meds_block}

Return strict JSON only:
{{
  "expertise": "{skill_name}",
  "active_problems": ["active diagnoses/problems relevant to this specialty, acute vs. chronic"],
  "acute_organ_dysfunction": ["e.g. AKI stage 1, respiratory failure; empty list if none"],
  "procedures": ["procedures/devices in this admission relevant to medication decisions"],
  "care_context": "care setting and acuity (e.g. MICU, post-operative ward)",
  "medication_relevant_risks": ["bleeding, renal/hepatic impairment, interaction or allergy risks, etc."],
  "prior_prescriptions": ["ATC_L4_CODE: name — still indicated? (yes/no/unclear)"],
  "visit_rationales": [
    {{
      "visit": "prior visit index or label",
      "diagnoses_procedures": ["key prior diagnoses/procedures"],
      "medications": ["ATC_L4_CODE"],
      "rationale": "why these medications were likely given",
      "status": "continuing" | "transient" | "unclear"
    }}
  ]
}}
""",
            skill_name=skill.name,
            summarize_template=skill.summarize_template,
            patient_text=case.patient_text,
            diag_block=diag_block,
            prior_diag_block=prior_diag_block,
            prior_meds_block=prior_meds_block,
        )
        raw = await self.llm.chat([{"role": "system", "content": prompt}], temperature=0.0)
        parsed = _safe_json_parse(raw, fallback={})
        if not parsed:
            # Keep the raw text so downstream operators still see the summary.
            parsed = {"expertise": skill.name, "unparsed_summary": raw.strip()}
        parsed.setdefault("visit_rationales", [])
        return {"summary": parsed, "prompt": prompt, "raw": raw}

    @staticmethod
    def _split_summary(summary: Dict[str, Any]) -> tuple[str, str]:
        """Return (s_i JSON, rho_i JSON) strings for prompt injection."""
        s_i = {k: v for k, v in (summary or {}).items() if k != "visit_rationales"}
        rho_i = (summary or {}).get("visit_rationales", [])
        return (
            json.dumps(s_i, ensure_ascii=False, indent=2),
            json.dumps(rho_i, ensure_ascii=False, indent=2),
        )


    def _build_prior_meds_block(self, prior_codes: List[str]) -> str:
        """Format prior-visit medications as continuation candidates for the generator."""
        if not prior_codes:
            return ""
        lines = [
            "Prior-visit medications (medication reconciliation — these were active in the previous visit):",
            "Continue a code if the current admission context supports ongoing use; discontinue if the clinical situation has changed.",
            "",
        ]
        for code in prior_codes:
            path = self.vocab.hierarchy_path(code) if self.vocab else None
            name = self.vocab.code_name(code) if self.vocab else None
            label = path if path else f"{code}: {name or 'unknown'}"
            lines.append(f"  • {label}")
        return "\n".join(lines)

    def _build_medi2_block(self, candidates: List[dict]) -> str:
        """Format MEDI2 lookup results as a prompt block for the generator.

        Candidates carry ``score`` and ``supporting_icd_codes``; classes
        supported by several diagnoses are marked as converging.
        """
        if not candidates:
            return ""
        lines = [
            "Knowledge-base drug candidates (MEDI2 ICD-10 → ATC L4, ranked by evidence strength):",
            "Include a code if the clinical summary supports it; reject if contraindicated.",
            "",
        ]
        for c in candidates[:15]:
            path = self.vocab.hierarchy_path(c["atc"]) if self.vocab else None
            code_label = path if path else f"{c['atc']}: {c['atc_name']}"
            # Show ICD convergence when several diagnoses support the same class
            icd_support = c.get("supporting_icd_codes", [])
            score = c.get("score")
            if icd_support and len(icd_support) > 1:
                icd_str = ", ".join(icd_support[:4])
                lines.append(f"  • {code_label}  [converging: {icd_str}]")
            else:
                lines.append(f"  • {code_label}")
            lines.append(f"    ↳ Evidence: {c['indication'][:80]}")
        return "\n".join(lines)

    async def _generate(
        self,
        state: SkillState,
        skill: ExpertiseSkill,
        medi2_candidates: List[dict] | None = None,
        prior_meds: List[str] | None = None,
    ) -> Dict[str, Any]:
        routing_json = json.dumps(state.routing_context, ensure_ascii=True, indent=2)

        medi2_block = self._build_medi2_block(medi2_candidates or [])
        medi2_section = f"\n{medi2_block}\n" if medi2_block else ""

        prior_meds_block = self._build_prior_meds_block(prior_meds or [])
        prior_meds_section = f"\n{prior_meds_block}\n" if prior_meds_block else ""

        s_i, rho_i = self._split_summary(state.summary)

        prompt = _render(
            """
You are the medication generator for the `{skill_name}` expertise skill.

This is a medication-list prediction task for the current admission. Work from your typed specialty summary, visit rationales, raw patient record, and routing context.
Do not output guideline-only wish lists. Predict what is plausibly on the medication list.

Task:
- Predict which ATC L4 medication classes are likely PRESENT during the current admission.
- Output ONLY ATC L4 class codes in 5-character format, for example `C10AA`.
- Never output ingredient-level or product-level ATC codes longer than 5 characters.
- If you think of a more specific code such as `J01CA04`, convert it to its parent ATC L4 class `J01CA`.
- Map from clinical scene -> plausible medication class, then choose the fitting ATC L4 code.
- Do not leave the list empty when the summary clearly supports likely inpatient medication classes.

Evidence priority (strongest to weakest):
1) Explicit medication evidence in the summary or record.
2) Prior-visit medications (if provided below) — continue unless the clinical situation warrants discontinuation.
3) Procedure/device/care-context evidence.
4) Current diagnoses/problems and acute organ dysfunction.
5) Knowledge-base drug candidates (if provided below) — include when supported by the summary.
6) Supportive inpatient medication priors only when they fit the documented scene.

Routing context:
{routing_json}

Skill checklist:
{drug_pred_checklist}

Rationale template (use it to write reason_from_input for each code):
{report_template}
{prior_meds_section}
{medi2_section}

Typed specialty summary (from your Summarize step):
{expert_summary}

Visit rationales (prior visit -> medication links; continuing vs. transient):
{visit_rationales}

Raw Patient Record:
{patient_text}

Return strict JSON only:
{{
  "should_prescribe": [
    {{
      "code": "ATC_L4_CODE_EXACTLY_5_CHARACTERS",
      "confidence": 0.0,
      "reason_from_input": "specialty-grounded reason citing the summary fields"
    }}
  ],
  "assumptions": [
    "missing information that limits certainty"
  ],
  "alternatives": [
    {{
      "code": "ATC_L4_CODE_EXACTLY_5_CHARACTERS",
      "reason": "plausible alternative class if the evidence resolves differently"
    }}
  ]
}}
""",
            skill_name=skill.name,
            routing_json=routing_json,
            drug_pred_checklist=skill.drug_pred_checklist,
            report_template=skill.report_template,
            prior_meds_section=prior_meds_section,
            medi2_section=medi2_section,
            expert_summary=s_i,
            visit_rationales=rho_i,
            patient_text=state.patient_text,
        )
        base_messages = [{"role": "system", "content": prompt}]

        parsed: Dict[str, Any] = {}
        raw = ""
        for attempt in range(2):
            raw = await self.llm.chat(base_messages, temperature=0.2)
            parsed = _safe_json_parse(raw, fallback={})
            codes = [(item.get("code") or "").strip() for item in parsed.get("should_prescribe", []) or []]
            invalid = [code for code in codes if code and self._normalize_atc_l4_code(code) is None]
            bad_format = [code for code in codes if code and (len(code.strip()) != 5 or not self.vocab.has_code(code.strip()))]
            if invalid and attempt == 0:
                correction = (
                    "Some proposed medication codes were not valid ATC L4 class codes: "
                    + ", ".join(invalid)
                    + ". Keep the other proposals unchanged. "
                    + "Replace ONLY the invalid codes with plausible ATC L4 codes in 5-character format."
                )
                raw = await self.llm.chat(
                    base_messages + [{"role": "user", "content": correction}],
                    temperature=0.2,
                )
                parsed = _safe_json_parse(raw, fallback={})
                break
            if bad_format and attempt == 0:
                correction = (
                    "Some proposed codes were too specific or not in ATC L4 format: "
                    + ", ".join(bad_format)
                    + ". Rewrite them as parent ATC L4 class codes using exactly 5 characters. "
                    + "Example: `J01CA04` -> `J01CA`. Keep the JSON schema unchanged."
                )
                raw = await self.llm.chat(
                    base_messages + [{"role": "user", "content": correction}],
                    temperature=0.2,
                )
                parsed = _safe_json_parse(raw, fallback={})
                break
            if not invalid:
                break

        parsed = self._normalize_proposal(parsed)
        state.trace.append(
            {
                "node": f"DrugGenerator:{skill.skill_id}",
                "prompt": prompt,
                "raw": raw,
                "parsed": parsed,
            }
        )
        return parsed

    def _normalize_atc_l4_code(self, code: str) -> str | None:
        cleaned = str(code or "").strip().upper()
        if not cleaned:
            return None
        if len(cleaned) == 5 and self.vocab.has_code(cleaned):
            return cleaned
        if len(cleaned) > 5:
            parent = cleaned[:5]
            if self.vocab.has_code(parent):
                return parent
        return None

    def _normalize_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(proposal, list):
            proposal = {"should_prescribe": proposal, "assumptions": []}
        drugs = proposal.get("should_prescribe") or proposal.get("predicted_drugs") or []
        cleaned: list[dict[str, Any]] = []
        for item in drugs:
            if isinstance(item, str):
                item = {"code": item, "name": "", "confidence": 0.5, "reason_from_input": ""}
            code = self._normalize_atc_l4_code(item.get("code") or "")
            if not code:
                continue
            cleaned.append(
                {
                    "code": code,
                    "name": self.vocab.code_name(code) or item.get("name") or "",
                    "l4_name": self.vocab.code_name(code) or item.get("name") or "",
                    "confidence": max(0.0, min(float(item.get("confidence", 0.0) or 0.0), 1.0)),
                    "reason_from_input": str(item.get("reason_from_input", "")).strip(),
                }
            )
        proposal["should_prescribe"] = cleaned
        proposal["assumptions"] = _dedupe_list([str(item) for item in proposal.get("assumptions", []) or []])
        return proposal

    def _get_diagnosis_codes(self, case: PatientCase) -> List[str]:
        """Extract current-visit ICD-10 diagnosis codes from patient record."""
        record = case.patient_record or {}
        records = record.get("records", {})
        if not isinstance(records, dict):
            return []
        diag_visits = records.get("diagnosis", [])
        if not diag_visits:
            return []
        # Use the last (current) visit diagnosis list
        current = diag_visits[-1] if isinstance(diag_visits[-1], list) else []
        return [str(c).strip() for c in current if str(c).strip()]

    def _get_prior_med_codes(self, case: PatientCase) -> List[str]:
        """Extract prior-visit medication ATC L4 codes from patient record."""
        record = case.patient_record or {}
        records = record.get("records", {})
        if not isinstance(records, dict):
            return []
        med_visits = records.get("medication", [])
        if not med_visits or len(med_visits) < 2:
            return []
        prior = med_visits[0] if isinstance(med_visits[0], list) else []
        codes: List[str] = []
        seen: set[str] = set()
        for c in prior:
            code = str(c).strip()[:5].upper()
            if code and self.vocab.has_code(code) and code not in seen:
                seen.add(code)
                codes.append(code)
        return codes

    async def _run_skill(
        self,
        case: PatientCase,
        skill: ExpertiseSkill,
        routing_context: Dict[str, Any],
        selected_skill: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Per-expert branch of Algorithm 1: Summarize -> Generate (one pass each)."""
        state = SkillState(
            skill_id=skill.skill_id,
            skill_name=skill.name,
            case_id=case.case_id,
            patient_text=case.patient_text,
            routing_context=routing_context,
        )

        if self.use_summarizer:
            summ = await self._summarize(case, skill)
            state.summary = summ["summary"]
            state.trace.append(
                {
                    "node": f"Summarizer:{skill.skill_id}",
                    "prompt": summ["prompt"],
                    "raw": summ["raw"],
                    "parsed": state.summary,
                }
            )
        else:
            state.summary = {"note": "Summarizer disabled; work from the raw patient record."}

        medi2_candidates: List[dict] = []
        if self.medi2 is not None:
            icd_codes = self._get_diagnosis_codes(case)
            medi2_candidates = self.medi2.selective_lookup(
                icd_codes, high_precision_only=True, max_total=15, convergence_bonus=True,
            )

        prior_meds = self._get_prior_med_codes(case)

        state.proposal = await self._generate(state, skill, medi2_candidates=medi2_candidates, prior_meds=prior_meds)
        final_codes = [d["code"] for d in state.proposal.get("should_prescribe", []) or [] if d.get("code")]
        state.final_output = {
            "final_predicted_drugs": [
                {"code": code, "name": self.vocab.code_name(code) or ""} for code in _dedupe_list(final_codes)
            ],
        }
        return {
            "case_id": case.case_id,
            "skill_id": skill.skill_id,
            "skill_name": skill.name,
            "routing_score": selected_skill.get("routing_score"),
            "summary": state.summary,
            "final_proposal": state.proposal,
            "final_prediction": state.final_output,
            "medi2_candidates": medi2_candidates,
            "proposal_history": [
                item for item in state.trace if str(item.get("node", "")).startswith("DrugGenerator:")
            ],
            "trace": state.trace,
        }

    async def run(self, case: PatientCase) -> Dict[str, Any]:
        """Algorithm 1 (pre-filter part): Route -> {Summarize, Generate} per expert -> global Critique.

        FindFlags + Verify (the safety filter) run afterwards in
        ``scripts/llm_safety_filter.py``.
        """
        if self.general_agent:
            routing_result = {"status": "general_agent", "primary_domain_name": "General ICU"}
            selected_skills = [
                {
                    "skill_id": GENERAL_AGENT_ID,
                    "skill_name": self.skills[GENERAL_AGENT_ID].name,
                    "routing_score": 1.0,
                    "rank": 1,
                    "activation_reason": "general_agent_mode",
                }
            ]
        else:
            routing_result = route_patient_case(
                patient_text=case.patient_text,
                patient_record=case.patient_record,
                routing_config_path=self.routing_config_path,
            )
            selected_skills = select_skills(
                routing_result=routing_result,
                routing_config=self.routing_config,
                skill_registry=self.skills,
            )
            # E_univ (universal supportive-care expert) is always activated.
            if SUPPORTIVE_ID in self.skills and not any(s["skill_id"] == SUPPORTIVE_ID for s in selected_skills):
                selected_skills.append(
                    {
                        "skill_id": SUPPORTIVE_ID,
                        "skill_name": self.skills[SUPPORTIVE_ID].name,
                        "routing_score": 1.0,
                        "rank": len(selected_skills) + 1,
                        "activation_reason": "universal_supportive_always_active",
                    }
                )

        skill_runs = await asyncio.gather(
            *[
                self._run_skill(
                    case=case,
                    skill=self.skills[selected["skill_id"]],
                    routing_context=routing_result,
                    selected_skill=selected,
                )
                for selected in selected_skills
            ]
        )

        prior_meds = self._get_prior_med_codes(case)

        result = merge_skill_outputs(
            case_id=case.case_id,
            routing_result=routing_result,
            selected_skills=selected_skills,
            skill_runs=skill_runs,
        )

        if self.use_critique:
            result = await self._critique(result, case, skill_runs, prior_meds)
        else:
            result["report"] = self._build_report(result["final_prediction"]["final_predicted_drugs"], [])
        self._save_log(result)
        return result

    async def _critique(
        self,
        merged_result: Dict[str, Any],
        case: PatientCase,
        skill_runs: List[Dict[str, Any]],
        prior_meds: List[str],
    ) -> Dict[str, Any]:
        """Global attending Critique: may only RETAIN or REMOVE codes from the union of expert proposals."""
        expert_blocks: List[str] = []
        medi2_candidates: List[dict] = []
        for sr in skill_runs:
            sid = sr.get("skill_id", "unknown")
            sname = sr.get("skill_name", sid)
            if not medi2_candidates and sr.get("medi2_candidates"):
                medi2_candidates = sr["medi2_candidates"]
            s_i, rho_i = self._split_summary(sr.get("summary") or {})
            proposal = sr.get("final_proposal") or {}
            drugs = proposal.get("should_prescribe", []) or []
            skill = self.skills.get(sid)
            critic_focus = skill.critic_focus if skill else ""
            lines = [
                f"### {sname} ({sid})",
                "Summary:", s_i,
                "Visit rationales:", rho_i,
                "Critique focus for this specialty (apply when judging this expert's codes):",
                critic_focus or "(none)",
                "Proposed codes:",
            ]
            if not drugs:
                lines.append("  (none)")
            for d in drugs:
                code = d.get("code", "")
                name = d.get("name") or d.get("l4_name") or ""
                conf = d.get("confidence", 0)
                reason = d.get("reason_from_input", "")
                lines.append(f"  - {code} ({name}, conf={conf:.2f}): {reason[:200]}")
            expert_blocks.append("\n".join(lines))

        prior_block = ""
        if prior_meds:
            prior_lines = ["Prior-visit medications (continued from previous admission):"]
            for code in prior_meds:
                name = self.vocab.code_name(code) if self.vocab else ""
                prior_lines.append(f"  - {code}: {name or 'unknown'}")
            prior_block = "\n".join(prior_lines)

        medi2_block = self._build_medi2_block(medi2_candidates)

        merged_drugs = merged_result.get("final_prediction", {}).get("final_predicted_drugs", [])
        union_codes = [d["code"] for d in merged_drugs if d.get("code")]
        if not union_codes:
            merged_result["report"] = self._build_report([], [])
            return merged_result

        prompt = _render(
            """You are the attending physician critique for a routed team of specialist medication experts. Each activated expert has summarized the patient and proposed ATC L4 medication classes for this inpatient admission. Your job is to produce the FINAL medication list by merging the proposals and removing codes that are not well-supported.

## Expert summaries and proposals
{expert_proposals}

{prior_block}

{medi2_block}

## Raw Patient Record
{patient_text}

## Union of all proposed codes
{union_codes}

## Instructions
1. KEEP a code if:
   - At least one expert gives a concrete, patient-specific reason for it, OR
   - It is an ongoing prior-visit medication and the underlying condition persists, OR
   - It is a plausible supportive or prophylactic inpatient medication (anticoagulants, analgesics, electrolytes, antiemetics, laxatives, PPIs, IV fluids) given the documented context.
2. REMOVE a code only if it is unsupported by the record AND has no plausible inpatient reason.
3. Do NOT remove a code just because it is outside one specialist's scope. Apply each specialty's critique focus to the codes that specialty proposed.
4. Remove duplicates only when two codes clearly share the same therapeutic intent without added benefit.
5. When in doubt, keep the code.
6. Do not add codes outside the union.

Return strict JSON only:
{{
  "retained_codes": ["ATC_L4_CODE", ...],
  "removed_codes": [
    {{"code": "ATC_L4_CODE", "reason": "why removed"}}
  ],
  "rationale": "brief overall rationale",
  "missing_information": ["information absent from the record that limits certainty"]
}}
""",
            expert_proposals="\n\n".join(expert_blocks),
            prior_block=prior_block,
            medi2_block=medi2_block,
            patient_text=case.patient_text,
            union_codes=json.dumps(union_codes),
        )
        raw = await self.llm.chat([{"role": "system", "content": prompt}], temperature=0.0)
        parsed = _safe_json_parse(raw, fallback={})

        kept_codes = parsed.get("retained_codes") or []
        union_set = set(union_codes)
        kept_set: set[str] = set()
        for c in kept_codes or []:
            code = self._normalize_atc_l4_code(str(c).strip())
            if code and code in union_set:  # critique can never add codes
                kept_set.add(code)

        if not kept_set:
            # Unparseable / empty critique output: fall back to the full union.
            merged_result["report"] = self._build_report(merged_drugs, [])
            return merged_result

        filtered_drugs = [d for d in merged_drugs if d.get("code") in kept_set]
        removed_info = parsed.get("removed_codes", [])

        merged_result["pre_critique_prediction"] = merged_result["final_prediction"].copy()
        merged_result["final_prediction"] = {
            **merged_result["final_prediction"],
            "final_predicted_drugs": filtered_drugs,
        }
        merged_result["report"] = self._build_report(filtered_drugs, removed_info)
        merged_result["critique"] = {
            "kept": len(filtered_drugs),
            "removed": len(merged_drugs) - len(filtered_drugs),
            "removed_details": removed_info,
            "reasoning": parsed.get("rationale", parsed.get("reasoning", "")),
            "missing_information": parsed.get("missing_information", []),
            "prompt": prompt,
            "raw": raw,
        }
        return merged_result

    def _build_report(self, final_drugs: List[Dict[str, Any]], critique_removed: List[Any]) -> Dict[str, Any]:
        """Traceable report: per-medication rationales (no extra LLM call).

        Rationales come from the experts' template-guided reasons; the safety
        filter later appends its Retain/Remove decisions to this report.
        """
        medications = []
        for d in final_drugs:
            code = d.get("code", "")
            medications.append(
                {
                    "code": code,
                    "name": d.get("name") or self.vocab.code_name(code) or "",
                    "supporting_experts": d.get("supporting_skills", []),
                    "rationale": d.get("reasons", []),
                }
            )
        removed = []
        for item in critique_removed or []:
            if isinstance(item, dict):
                removed.append({"code": item.get("code", ""), "stage": "critique", "reason": item.get("reason", "")})
            else:
                removed.append({"code": str(item), "stage": "critique", "reason": ""})
        return {"medications": medications, "removed": removed}

    def _save_log(self, result: Dict[str, Any]) -> None:
        if not self.log_dir:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.log_dir / f"{result.get('case_id', 'case')}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


def build_saferx_agent(llm: LLMClient, vocab: Vocabulary, **kwargs) -> SafeRxAgent:
    return SafeRxAgent(llm=llm, vocab=vocab, **kwargs)


def run_sync(agent: SafeRxAgent, case: PatientCase) -> Dict[str, Any]:
    return asyncio.run(agent.run(case))
