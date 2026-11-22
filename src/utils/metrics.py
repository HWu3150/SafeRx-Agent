"""Prediction-quality metrics. Safety metrics (DDI/Contra) live in scripts/llm_safety_filter.py."""

import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score


def atc_to_l4(code: str) -> str | None:
    if not isinstance(code, str):
        return None
    code = code.strip()
    return code[:5] if len(code) >= 5 else None


def codes_to_multihot(codes, vocab) -> np.ndarray:
    """Convert iterable of ATC codes (any level) to multi-hot over vocab.allowed_label_space (L4)."""
    vec = np.zeros(len(vocab.allowed_label_space), dtype=int)
    index_map = {c: i for i, c in enumerate(vocab.allowed_label_space)}
    seen = set()
    for c in codes:
        l4 = atc_to_l4(c)
        if l4 and l4 in index_map and l4 not in seen:
            vec[index_map[l4]] = 1
            seen.add(l4)
    return vec


def batch_codes_to_multihot(batch_codes, vocab) -> np.ndarray:
    return np.stack([codes_to_multihot(codes, vocab) for codes in batch_codes], axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Sample-averaged Jaccard; micro-averaged precision / recall / F1 (paper App. E)."""
    jaccard = float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    recall = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    return {
        "jaccard": jaccard,
        "micro_f1": micro_f1,
        "precision": precision,
        "recall": recall,
    }
