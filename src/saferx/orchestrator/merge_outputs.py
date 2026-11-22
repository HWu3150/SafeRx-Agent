from __future__ import annotations

from typing import Any


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = str(item).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def merge_skill_outputs(
    case_id: str,
    routing_result: dict[str, Any],
    selected_skills: list[dict[str, Any]],
    skill_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    merged_by_code: dict[str, dict[str, Any]] = {}
    skill_justifications: list[dict[str, Any]] = []
    uncertainty_items: list[str] = []

    for skill_run in skill_runs:
        skill_id = skill_run.get("skill_id", "")
        skill_name = skill_run.get("skill_name", skill_id)
        proposal = skill_run.get("final_proposal") or {}
        final_prediction = skill_run.get("final_prediction") or {}

        skill_justifications.append(
            {
                "skill_id": skill_id,
                "skill_name": skill_name,
                "routing_score": skill_run.get("routing_score"),
                "justification": final_prediction.get("justification", ""),
            }
        )

        uncertainty = final_prediction.get("uncertainty")
        if isinstance(uncertainty, str):
            uncertainty_items.append(uncertainty)

        assumptions = proposal.get("assumptions", []) or []
        uncertainty_items.extend(str(item) for item in assumptions if str(item).strip())

        for drug in proposal.get("should_prescribe", []) or []:
            code = str(drug.get("code", "")).strip()
            if not code:
                continue

            current = merged_by_code.setdefault(
                code,
                {
                    "code": code,
                    "name": drug.get("name") or drug.get("l4_name") or "",
                    "confidence": 0.0,
                    "supporting_skills": [],
                    "reasons": [],
                },
            )

            current["name"] = current["name"] or drug.get("name") or drug.get("l4_name") or ""
            current["confidence"] = max(current["confidence"], float(drug.get("confidence", 0.0) or 0.0))
            current["supporting_skills"] = _dedupe_strings(current["supporting_skills"] + [skill_id])
            reason = str(drug.get("reason_from_input", "")).strip()
            if reason:
                current["reasons"] = _dedupe_strings(current["reasons"] + [f"{skill_id}: {reason}"])

        for item in final_prediction.get("final_predicted_drugs", []) or []:
            code = str(item.get("code", "")).strip()
            if not code:
                continue
            current = merged_by_code.setdefault(
                code,
                {
                    "code": code,
                    "name": item.get("name", ""),
                    "confidence": 0.0,
                    "supporting_skills": [skill_id],
                    "reasons": [],
                },
            )
            current["name"] = current["name"] or item.get("name", "")
            current["supporting_skills"] = _dedupe_strings(current["supporting_skills"] + [skill_id])

    merged_drugs = sorted(
        (
            {
                **value,
                "confidence": min(1.0, value["confidence"] + 0.05 * max(len(value["supporting_skills"]) - 1, 0)),
            }
            for value in merged_by_code.values()
        ),
        key=lambda item: (-float(item.get("confidence", 0.0)), item["code"]),
    )

    selected_skill_ids = [item["skill_id"] for item in selected_skills]
    summary_line = (
        f"Routed to {routing_result.get('primary_domain_name')} with {len(selected_skill_ids)} active skill(s): "
        + ", ".join(selected_skill_ids)
    )

    return {
        "case_id": case_id,
        "routing": routing_result,
        "active_skills": selected_skills,
        "skill_justifications": skill_justifications,
        "final_prediction": {
            "final_predicted_drugs": merged_drugs,
            "justification": summary_line,
            "uncertainty": " | ".join(_dedupe_strings(uncertainty_items)),
            "requires_clinician_review": True,
        },
        "skill_outputs": skill_runs,
    }
