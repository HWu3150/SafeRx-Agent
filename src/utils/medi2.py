"""
MEDI2 ICD-10 → ATC L4 drug-indication lookup utility.

Loads MEDI2_with_ATC_ICD10.csv and provides candidate ATC L4 drug classes
for a given set of ICD-10-CM diagnosis codes.

Usage:
    from src.utils.medi2 import load_medi2
    medi2 = load_medi2("data/MEDI2_with_ATC_ICD10.csv")
    candidates = medi2.lookup(["I214", "N186", "I5033"])
    # Returns list of dicts: {icd_code, matched_key, atc, atc_name, indication}
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


class MEDI2Lookup:
    """ICD-10 diagnosis code → ATC L4 drug class candidates."""

    def __init__(self, icd_to_entries: Dict[str, List[dict]]) -> None:
        self._map = icd_to_entries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _candidate_keys(code: str) -> List[str]:
        """
        Generate lookup keys for an ICD-10-CM code (which may or may not have a dot).

        Patient records store codes WITHOUT dots (e.g., "I214", "N186").
        MEDI2 stores codes WITH dots (e.g., "I21.4", "N18.6").

        Strategy (most-specific first):
          1. Insert dot at position 3 then keep full code   (e.g., N186 → N18.6)
          2. Insert dot at position 3, keep 5 chars          (e.g., N3001 → N30.01)
          3. Raw code as-is (in case MEDI2 happens to store it without dot)
          4. First 3 characters (category-level fallback)    (e.g., N186 → N18)
        """
        clean = code.strip().upper().replace(".", "")
        keys: List[str] = []
        if len(clean) > 3:
            # Insert dot after 3rd char and keep entire remainder
            keys.append(clean[:3] + "." + clean[3:])   # e.g. N186  → N18.6
        if len(clean) > 4:
            # Insert dot after 3rd char, keep only 2 more chars
            keys.append(clean[:3] + "." + clean[3:5])  # e.g. N3001 → N30.01
        keys.append(clean)             # raw (no dot)
        keys.append(clean[:3])         # 3-char category prefix
        return keys

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self,
        icd_codes: List[str],
        high_precision_only: bool = True,
    ) -> List[dict]:
        """
        Return deduplicated ATC L4 candidates for a list of ICD-10-CM codes.

        Parameters
        ----------
        icd_codes : list[str]
            ICD-10-CM codes from patient records (without dots).
        high_precision_only : bool
            If True (default), only return entries where HIGH_PRECISION_SUBSET == 'TRUE'.

        Returns
        -------
        list of dicts with keys:
            icd_code    – original input code
            matched_key – the MEDI2 key that matched (may differ due to dot insertion)
            atc         – ATC L4 code (5 chars, e.g., "B01AB")
            atc_name    – ATC L4 human-readable name
            drug        – example drug name from MEDI2
            indication  – ICD description from MEDI2
        """
        seen_atc: set[str] = set()
        results: List[dict] = []

        for icd in icd_codes:
            for key in self._candidate_keys(icd):
                entries = self._map.get(key, [])
                for entry in entries:
                    if high_precision_only and not entry["high_precision"]:
                        continue
                    atc = entry["atc"]
                    if atc in seen_atc:
                        continue
                    seen_atc.add(atc)
                    results.append(
                        {
                            "icd_code": icd,
                            "matched_key": key,
                            "atc": atc,
                            "atc_name": entry["atc_name"],
                            "drug": entry["drug"],
                            "indication": entry["indication"],
                        }
                    )
                if entries:
                    # Use first (most specific) key that has entries
                    break

        return results

    def selective_lookup(
        self,
        icd_codes: List[str],
        high_precision_only: bool = True,
        max_total: int = 15,
        convergence_bonus: bool = True,
    ) -> List[dict]:
        """
        Return *scored and ranked* ATC L4 candidates, prioritising specificity
        and convergence across multiple ICD codes.

        Scoring per candidate:
          - full-code match (dotted form): 3
          - raw undotted match: 2
          - 3-char category fallback: 1
          - each additional ICD code pointing to the same ATC: +2  (convergence)

        Parameters
        ----------
        max_total : int
            Hard cap on returned candidates (after global ranking).
        convergence_bonus : bool
            If True, boost ATC codes supported by ≥2 ICD codes.

        Returns
        -------
        list of dicts (same schema as ``lookup``, plus ``score`` and
        ``supporting_icd_codes`` fields).  Sorted descending by score.
        """
        # Phase 1: collect all (atc, entry) pairs with match-quality scores
        atc_entries: Dict[str, dict] = {}  # atc -> best entry + metadata
        atc_icd_support: Dict[str, List[str]] = defaultdict(list)

        for icd in icd_codes:
            keys = self._candidate_keys(icd)
            matched = False
            for key_idx, key in enumerate(keys):
                entries = self._map.get(key, [])
                for entry in entries:
                    if high_precision_only and not entry["high_precision"]:
                        continue
                    atc = entry["atc"]
                    # Specificity score: exact (key_idx 0,1) vs category fallback (key_idx 3)
                    specificity = 3 if key_idx <= 1 else (2 if key_idx == 2 else 1)

                    if atc not in atc_entries or specificity > atc_entries[atc].get("_specificity", 0):
                        atc_entries[atc] = {
                            "icd_code": icd,
                            "matched_key": key,
                            "atc": atc,
                            "atc_name": entry["atc_name"],
                            "drug": entry["drug"],
                            "indication": entry["indication"],
                            "_specificity": specificity,
                        }
                    atc_icd_support[atc].append(icd)
                    matched = True
                if matched:
                    break  # most-specific key that matched

        # Phase 2: compute final score with convergence bonus
        scored: List[dict] = []
        for atc, entry in atc_entries.items():
            icd_support = list(set(atc_icd_support[atc]))
            score = entry["_specificity"]
            if convergence_bonus and len(icd_support) >= 2:
                score += 2 * (len(icd_support) - 1)
            result = {k: v for k, v in entry.items() if not k.startswith("_")}
            result["score"] = score
            result["supporting_icd_codes"] = icd_support
            scored.append(result)

        # Phase 3: rank and cap
        scored.sort(key=lambda x: (-x["score"], x["atc"]))
        return scored[:max_total]


def load_medi2(csv_path: str | Path) -> MEDI2Lookup:
    """
    Load MEDI2_with_ATC_ICD10.csv into a MEDI2Lookup instance.

    Parameters
    ----------
    csv_path : str | Path
        Path to MEDI2_with_ATC_ICD10.csv.

    Returns
    -------
    MEDI2Lookup
    """
    icd_to_entries: Dict[str, List[dict]] = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            icd = row.get("CODE", "").strip()
            atc = row.get("ATC", "").strip().upper()
            if not icd or not atc or len(atc) != 5:
                continue
            high_prec = row.get("HIGH_PRECISION_SUBSET", "").strip().upper() == "TRUE"
            icd_to_entries[icd].append(
                {
                    "atc": atc,
                    "atc_name": row.get("ATC_NAME", "").strip(),
                    "drug": row.get("DRUG_DESCRIPTION", "").strip(),
                    "indication": row.get("INDICATION_DESCRIPTION", "").strip(),
                    "high_precision": high_prec,
                }
            )

    return MEDI2Lookup(dict(icd_to_entries))
