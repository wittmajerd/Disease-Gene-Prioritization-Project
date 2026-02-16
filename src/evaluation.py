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
    ...


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
    ...
