from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def classification_metrics(logits: Tensor, labels: Tensor) -> Dict[str, float]:
    """Compute F1 (primary), precision, recall, accuracy, and AUC on CPU."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    labels_int = labels.long()

    tp = ((preds == 1) & (labels_int == 1)).sum().item()
    fp = ((preds == 1) & (labels_int == 0)).sum().item()
    fn = ((preds == 0) & (labels_int == 1)).sum().item()
    tn = ((preds == 0) & (labels_int == 0)).sum().item()

    def _safe_div(num, den):
        return num / den if den != 0 else 0.0

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    try:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(labels_int.cpu(), probs.cpu())
    except Exception:
        auc = float("nan")

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "auc": float(auc),
    }
