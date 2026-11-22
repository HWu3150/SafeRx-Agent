from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROUTER_DIR = Path(__file__).resolve().parent
DEFAULT_TAXONOMY_PATH = ROUTER_DIR / "taxonomy.json"
DEFAULT_ROUTING_CONFIG_PATH = ROUTER_DIR / "routing_config.json"

DIAGNOSIS_BLOCK_RE = re.compile(
    r"diagnosis:\s*(.*?)(?:; procedures:|; the patient was prescribed drugs:|$)",
    re.IGNORECASE | re.DOTALL,
)
DIAGNOSIS_ITEM_WITH_CODE_RE = re.compile(r"\s*(.+?)\s+\(([A-Z][A-Z0-9]{2,8})\)(?=,\s|$)")
CODE_IN_TEXT_RE = re.compile(r"\(([A-Z][A-Z0-9]{2,8})\)")


@dataclass(frozen=True)
class DomainGroup:
    id: str
    name: str
    chapter_ids: tuple[str, ...]
    description: str
    keypoints: tuple[str, ...]


@dataclass(frozen=True)
class RoutingConfig:
    code_weight: float = 0.70
    keyword_weight: float = 0.10
    mixed_margin: float = 0.08
    selection_margin: float = 0.12
    min_skill_score: float = 0.12
    secondary_score_floor: float = 0.22
    activate_top_n: int = 3
    max_active_skills: int = 3
    fallback_mode: str = "uniform"


def load_taxonomy(path: str | Path | None = None) -> list[DomainGroup]:
    taxonomy_path = Path(path) if path else DEFAULT_TAXONOMY_PATH
    payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    return [
        DomainGroup(
            id=item["id"],
            name=item["name"],
            chapter_ids=tuple(item["chapter_ids"]),
            description=item["description"],
            keypoints=tuple(item["keypoints"]),
        )
        for item in payload
    ]


def load_routing_config(path: str | Path | None = None) -> RoutingConfig:
    config_path = Path(path) if path else DEFAULT_ROUTING_CONFIG_PATH
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return RoutingConfig(**payload)


def normalize_codes(items: list[str] | None) -> list[str]:
    if not items:
        return []
    return [str(item).strip().upper().replace(".", "") for item in items if str(item).strip()]


def parse_code_prefix_number(code: str) -> int | None:
    digits = "".join(ch for ch in code[1:] if ch.isdigit())
    if len(digits) < 2:
        return None
    return int(digits[:2])


def map_icd10_chapter(code: str) -> str | None:
    code = code.strip().upper().replace(".", "")
    if len(code) < 3:
        return None

    first = code[0]
    if first in {"A", "B"}:
        return "I"
    if first == "C":
        return "II"
    if first == "D":
        number = parse_code_prefix_number(code)
        if number is None:
            return "III"
        return "II" if number < 50 else "III"
    if first == "E":
        return "IV"
    if first == "F":
        return "V"
    if first == "G":
        return "VI"
    if first == "H":
        number = parse_code_prefix_number(code)
        if number is None:
            return "VII"
        return "VII" if number < 60 else "VIII"
    if first == "I":
        return "IX"
    if first == "J":
        return "X"
    if first == "K":
        return "XI"
    if first == "L":
        return "XII"
    if first == "M":
        return "XIII"
    if first == "N":
        return "XIV"
    if first == "O":
        return "XV"
    if first == "P":
        return "XVI"
    if first == "Q":
        return "XVII"
    if first == "R":
        return "XVIII"
    if first in {"S", "T"}:
        return "XIX"
    if first in {"V", "W", "X", "Y"}:
        return "XX"
    if first == "Z":
        return "XXI"
    if first == "U":
        return "SPECIAL"
    return None


def extract_diagnosis_codes(patient_record: dict[str, Any] | None, patient_text: str) -> list[str]:
    codes: list[str] = []

    if isinstance(patient_record, dict):
        records = patient_record.get("records", {})
        if isinstance(records, dict):
            for visit_codes in records.get("diagnosis", []) or []:
                codes.extend(normalize_codes(visit_codes))

        direct_codes = patient_record.get("diagnosis_codes")
        if isinstance(direct_codes, list):
            codes.extend(normalize_codes(direct_codes))

    if not codes:
        codes.extend(normalize_codes(CODE_IN_TEXT_RE.findall(patient_text or "")))

    return codes


def _split_diagnosis_fallback(block: str) -> list[str]:
    parts = [part.strip(" ,.;") for part in re.split(r",\s+", block) if part.strip(" ,.;")]
    seen: set[str] = set()
    ordered: list[str] = []
    for part in parts:
        cleaned = CODE_IN_TEXT_RE.sub("", part).strip(" ,.;")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def extract_diagnosis_descriptions(patient_text: str) -> list[str]:
    descriptions: list[str] = []
    seen: set[str] = set()

    for block in DIAGNOSIS_BLOCK_RE.findall(patient_text or ""):
        coded_matches = list(DIAGNOSIS_ITEM_WITH_CODE_RE.finditer(block))
        if coded_matches:
            items = [match.group(1).strip() for match in coded_matches if match.group(1).strip()]
        else:
            items = _split_diagnosis_fallback(block)
        for item in items:
            if item not in seen:
                seen.add(item)
                descriptions.append(item)

    return descriptions


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-12, None)
    return matrix / row_sums


def _score_case_chapters(codes: list[str]) -> dict[str, float]:
    counter: dict[str, float] = {}
    for code in codes:
        chapter_id = map_icd10_chapter(code)
        if chapter_id is None:
            continue
        counter[chapter_id] = counter.get(chapter_id, 0.0) + 1.0
    return counter


def _build_code_score_vector(case_chapter_scores: dict[str, float], domains: list[DomainGroup]) -> np.ndarray:
    scores = np.zeros(len(domains), dtype=np.float32)
    for idx, domain in enumerate(domains):
        scores[idx] = float(sum(case_chapter_scores.get(chapter_id, 0.0) for chapter_id in domain.chapter_ids))
    return scores


def _build_keyword_score_vector(text: str, domains: list[DomainGroup]) -> np.ndarray:
    lowered = (text or "").lower()
    scores = np.zeros(len(domains), dtype=np.float32)
    for idx, domain in enumerate(domains):
        hits = 0.0
        for keyword in domain.keypoints:
            if keyword in lowered:
                hits += 1.0
        scores[idx] = hits
    return scores


def _combine_scores(
    code_scores: np.ndarray,
    keyword_scores: np.ndarray,
    config: RoutingConfig,
) -> tuple[np.ndarray, str, list[str]]:
    components: list[np.ndarray] = []
    weights: list[float] = []
    signals: list[str] = []

    if np.any(code_scores > 0):
        components.append(_normalize_rows(code_scores[None, :])[0])
        weights.append(config.code_weight)
        signals.append("icd_codes")

    if np.any(keyword_scores > 0):
        components.append(_normalize_rows(keyword_scores[None, :])[0])
        weights.append(config.keyword_weight)
        signals.append("keywords")

    if not components:
        if config.fallback_mode == "uniform":
            uniform = np.full(len(code_scores), 1.0 / max(len(code_scores), 1), dtype=np.float32)
            return uniform, "fallback_uniform", ["uniform_prior"]
        raise RuntimeError("Router produced no usable signals and fallback_mode is disabled.")

    total_weight = float(sum(weights))
    combined = np.zeros_like(components[0])
    for component, weight in zip(components, weights):
        combined += component * (weight / total_weight)
    combined = _normalize_rows(combined[None, :])[0]
    return combined, "ok", signals


def route_patient_case(
    patient_text: str,
    patient_record: dict[str, Any] | None = None,
    taxonomy_path: str | Path | None = None,
    routing_config_path: str | Path | None = None,
) -> dict[str, Any]:
    domains = load_taxonomy(taxonomy_path)
    config = load_routing_config(routing_config_path)

    diagnosis_codes = extract_diagnosis_codes(patient_record, patient_text)
    diagnosis_descriptions = extract_diagnosis_descriptions(patient_text)
    description_text = " ".join(diagnosis_descriptions).strip()
    keyword_text = description_text or patient_text or ""

    code_scores = _build_code_score_vector(_score_case_chapters(diagnosis_codes), domains)
    keyword_scores = _build_keyword_score_vector(keyword_text, domains)
    domain_scores, status, signals = _combine_scores(code_scores, keyword_scores, config)

    ranked_indices = np.argsort(domain_scores)[::-1]
    primary_idx = int(ranked_indices[0])
    secondary_idx = int(ranked_indices[1]) if len(ranked_indices) > 1 else primary_idx

    primary_score = float(domain_scores[primary_idx])
    secondary_score = float(domain_scores[secondary_idx]) if len(ranked_indices) > 1 else 0.0
    score_margin = primary_score - secondary_score

    ranked_domains = [
        {
            "rank": rank + 1,
            "id": domains[idx].id,
            "name": domains[idx].name,
            "score": float(domain_scores[idx]),
            "chapter_ids": list(domains[idx].chapter_ids),
            "description": domains[idx].description,
        }
        for rank, idx in enumerate(ranked_indices)
    ]

    return {
        "status": status,
        "signals_used": signals,
        "diagnosis_codes": diagnosis_codes,
        "diagnosis_descriptions": diagnosis_descriptions,
        "primary_domain_id": domains[primary_idx].id,
        "primary_domain_name": domains[primary_idx].name,
        "primary_score": primary_score,
        "secondary_domain_id": domains[secondary_idx].id,
        "secondary_domain_name": domains[secondary_idx].name,
        "secondary_score": secondary_score,
        "score_margin": score_margin,
        "is_mixed": score_margin < config.mixed_margin,
        "ranked_domains": ranked_domains,
    }
