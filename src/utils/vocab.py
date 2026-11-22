"""
Closed ATC-L4 prediction vocabulary V_med (``data/med_vocab.json``), with names
and hierarchy paths from ``data/atc_hierarchy.csv``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


@dataclass
class Vocabulary:
    code_to_name: Dict[str, str]
    name_to_code: Dict[str, str]
    allowed_label_space: List[str]
    code_set: Set[str]
    # Optional: hierarchy path per L4 code (populated by load_med_vocab from atc_hierarchy.csv)
    hierarchy_paths: Dict[str, str] = None  # type: ignore[assignment]

    def has_code(self, code: str) -> bool:
        return code in self.code_set

    def code_name(self, code: str) -> str | None:
        return self.code_to_name.get(code)

    def hierarchy_path(self, code: str) -> str | None:
        """Return a human-readable path like 'B: Blood → B01: Antithrombotic → B01A: ... → B01AB: Heparin'."""
        if self.hierarchy_paths:
            return self.hierarchy_paths.get(code)
        return None


def load_med_vocab(
    json_path: str | Path,
    atc_csv_path: str | Path | None = None,
) -> Vocabulary:
    """
    Build a Vocabulary restricted to the drugs in med_vocab.json.

    Parameters
    ----------
    json_path : str | Path
        Path to ``data/med_vocab.json``.
        Must contain ``drug_to_index`` and ``drug_name`` fields.
    atc_csv_path : str | Path | None
        Optional path to ``data/atc_hierarchy.csv`` for hierarchy path strings.
    """
    import json as _json

    with open(json_path, encoding="utf-8") as f:
        med = _json.load(f)

    drug_to_index: Dict[str, int] = med["drug_to_index"]
    drug_name_map: Dict[str, str] = med.get("drug_name", {})

    code_to_name: Dict[str, str] = {}
    name_to_code: Dict[str, str] = {}
    for code, name in drug_name_map.items():
        if code in drug_to_index:
            code_to_name[code] = name
            name_to_code.setdefault(name.lower(), code)
    for code in drug_to_index:
        code_to_name.setdefault(code, code)

    allowed_label_space = sorted(code_to_name.keys())
    code_set = set(allowed_label_space)

    hierarchy_paths: Dict[str, str] = {}
    if atc_csv_path is not None:
        with open(atc_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("ATC_L4_Code", "").strip()
                if code not in code_set or code in hierarchy_paths:
                    continue
                l1c = row.get("ATC_L1_Code", "").strip()
                l1n = row.get("ATC_L1_Name", "").strip()
                l2c = row.get("ATC_L2_Code", "").strip()
                l2n = row.get("ATC_L2_Name", "").strip()
                l3c = row.get("ATC_L3_Code", "").strip()
                l3n = row.get("ATC_L3_Name", "").strip()
                name = row.get("ATC_L4_Name", "").strip()
                if l1c and l2c and l3c:
                    hierarchy_paths[code] = (
                        f"{l1c}: {l1n.title()} → {l2c}: {l2n.title()} → "
                        f"{l3c}: {l3n.title()} → {code}: {name}"
                    )

    return Vocabulary(
        code_to_name=code_to_name,
        name_to_code=name_to_code,
        allowed_label_space=allowed_label_space,
        code_set=code_set,
        hierarchy_paths=hierarchy_paths if hierarchy_paths else None,
    )
