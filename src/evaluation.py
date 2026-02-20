"""Metrics computation for link prediction.

Provides both *classification* metrics (F1, precision, recall, AUC)
and *ranking* metrics (MRR, Hits@K) for future use.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def classification_metrics(logits: Tensor, labels: Tensor) -> Dict[str, float]:
    """Compute classification metrics from raw logits and binary labels.

    Metrics
    -------
    * **f1** — harmonic mean of precision & recall (primary).
    * **precision** — TP / (TP + FP).
    * **recall** — TP / (TP + FN).
    * **accuracy** — (TP + TN) / total.
    * **auc** — area under the ROC curve (via sklearn if available).

    Parameters
    ----------
    logits:
        ``[N]`` raw model output (before sigmoid).
    labels:
        ``[N]`` binary ground-truth (1 = positive, 0 = negative).

    Returns
    -------
    Dict with string keys and float values.
    """
    logits = logits.detach().view(-1).cpu()
    labels = labels.detach().view(-1).long().cpu()

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    tp = int(((preds == 1) & (labels == 1)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())

    def _safe_div(num: float, den: float) -> float:
        return float(num / den) if den != 0 else 0.0

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(labels.numpy(), probs.numpy()))
    except Exception:
        auc = float("nan")

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "auc": float(auc),
    }


def ranking_metrics(
    scores: Tensor,
    positive_mask: Tensor,
    k_values: list[int] | None = None,
) -> Dict[str, float]:
    """Compute ranking-based metrics.

    Intended for the "rank all proteins for a given disease" scenario.

    Metrics
    -------
    * **mrr** — Mean Reciprocal Rank.
    * **hits@k** — fraction of positives ranked in top-*k*.

    Parameters
    ----------
    scores:
        ``[N]`` predicted scores for one query (e.g. one disease).
    positive_mask:
        ``[N]`` bool mask — True for the ground-truth positive targets.
    k_values:
        Which *k* to evaluate for Hits@K.  Default: ``[1, 3, 10]``.

    Returns
    -------
    Dict, e.g. ``{"mrr": 0.42, "hits@1": 0.30, "hits@10": 0.78}``.
    """
    if k_values is None:
        k_values = [1, 3, 10]

    scores = scores.detach().view(-1).cpu()
    positive_mask = positive_mask.detach().view(-1).bool().cpu()

    if scores.numel() == 0:
        out = {"mrr": 0.0}
        out.update({f"hits@{k}": 0.0 for k in k_values})
        return out

    pos_idx = torch.where(positive_mask)[0]
    if pos_idx.numel() == 0:
        out = {"mrr": 0.0}
        out.update({f"hits@{k}": 0.0 for k in k_values})
        return out

    sorted_idx = torch.argsort(scores, descending=True)
    rank_positions = torch.empty_like(sorted_idx)
    rank_positions[sorted_idx] = torch.arange(1, scores.numel() + 1)

    pos_ranks = rank_positions[pos_idx].float()
    mrr = float((1.0 / pos_ranks).mean().item())

    out = {"mrr": mrr}
    for k in k_values:
        out[f"hits@{k}"] = float((pos_ranks <= float(k)).float().mean().item())

    return out
