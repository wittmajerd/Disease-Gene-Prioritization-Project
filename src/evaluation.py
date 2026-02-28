"""Metrics computation for link prediction.

Provides both *classification* metrics (F1, precision, recall, AUC)
and *ranking* metrics (MRR, Hits@K).

The ranking module supports the **filtered** evaluation protocol
standard in the KGE literature: when ranking a test triple, other
known-true triples are excluded so the model is not penalised for
ranking valid answers highly.
"""

from __future__ import annotations

from collections import defaultdict
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


# ═══════════════════════════════════════════════════════════════════
# Filtered ranking evaluation (KG completion protocol)
# ═══════════════════════════════════════════════════════════════════


def build_filter_dicts(
    all_positives: dict[tuple[str, str, str], Tensor],
) -> tuple[
    dict[tuple[tuple[str, str, str], int], Tensor],
    dict[tuple[tuple[str, str, str], int], Tensor],
]:
    """Pre-compute per-query true targets for the filtered protocol.

    For each ``(edge_type, head_id)`` → all known true tail ids.
    For each ``(edge_type, tail_id)`` → all known true head ids.

    These sets include train + val + test triples so that **no** valid
    triple is unfairly counted as a wrong answer during evaluation.

    Parameters
    ----------
    all_positives:
        ``{edge_type: [2, E]}`` — all positive edges **before** splitting.
        Typically ``KGSplits.all_positives``.

    Returns
    -------
    tail_filter:
        ``{(edge_type, head_id): LongTensor of true tail ids}``
    head_filter:
        ``{(edge_type, tail_id): LongTensor of true head ids}``
    """
    tail_groups: dict[tuple[tuple[str, str, str], int], list[int]] = defaultdict(list)
    head_groups: dict[tuple[tuple[str, str, str], int], list[int]] = defaultdict(list)

    for edge_type, ei in all_positives.items():
        heads = ei[0].cpu().tolist()
        tails = ei[1].cpu().tolist()
        for h, t in zip(heads, tails):
            tail_groups[(edge_type, h)].append(t)
            head_groups[(edge_type, t)].append(h)

    tail_filter = {k: torch.tensor(v, dtype=torch.long) for k, v in tail_groups.items()}
    head_filter = {k: torch.tensor(v, dtype=torch.long) for k, v in head_groups.items()}
    return tail_filter, head_filter


def compute_filtered_rank(
    all_scores: Tensor,
    target_idx: int,
    true_targets: Tensor,
) -> int:
    """Compute the filtered rank of a single target entity.

    The **filtered** setting masks out all other known-true answers so
    the model is not penalised for ranking them highly.

    Parameters
    ----------
    all_scores:
        ``[num_candidates]`` predicted scores for every candidate entity.
    target_idx:
        Index of the entity we want to rank (the true answer).
    true_targets:
        Indices of **all** known-true entities for this query
        (including ``target_idx`` itself).  All except ``target_idx``
        will have their scores replaced with ``-inf``.

    Returns
    -------
    1-based rank of ``target_idx`` (lower is better).
    """
    scores = all_scores.clone()

    # Build filter mask: all true targets EXCEPT the current one
    filter_mask = torch.zeros(scores.size(0), dtype=torch.bool, device=scores.device)
    filter_mask[true_targets.to(scores.device)] = True
    filter_mask[target_idx] = False  # keep the actual target unmasked

    scores[filter_mask] = float("-inf")

    target_score = scores[target_idx]
    rank = int((scores >= target_score).sum().item())
    return max(rank, 1)  # ensure at least rank 1


def aggregate_ranks(
    ranks: list[int],
    k_values: list[int] | None = None,
) -> Dict[str, float]:
    """Compute MRR and Hits@K from a list of 1-based ranks.

    Parameters
    ----------
    ranks:
        List of integer ranks (1 = best).
    k_values:
        Which *k* to evaluate for Hits@K.  Default: ``[1, 3, 10]``.

    Returns
    -------
    Dict, e.g. ``{"mrr": 0.42, "hits@1": 0.30, "hits@3": 0.55, "hits@10": 0.78}``.
    """
    if k_values is None:
        k_values = [1, 3, 10]

    if not ranks:
        out: Dict[str, float] = {"mrr": 0.0}
        out.update({f"hits@{k}": 0.0 for k in k_values})
        return out

    ranks_t = torch.tensor(ranks, dtype=torch.float)
    mrr = float((1.0 / ranks_t).mean().item())

    out = {"mrr": mrr}
    for k in k_values:
        out[f"hits@{k}"] = float((ranks_t <= k).float().mean().item())
    return out


def filtered_metrics_from_scores(
    tail_scores: list[tuple[Tensor, int, Tensor]],
    head_scores: list[tuple[Tensor, int, Tensor]],
    k_values: list[int] | None = None,
) -> Dict[str, float]:
    """Compute filtered MRR / Hits@K from pre-scored candidates.

    This is a convenience wrapper: the caller (Trainer) iterates over
    test triples, scores all candidates, and passes them here.

    Parameters
    ----------
    tail_scores:
        List of ``(all_tail_scores, true_tail_idx, true_tails_tensor)``
        for tail prediction queries.
    head_scores:
        List of ``(all_head_scores, true_head_idx, true_heads_tensor)``
        for head prediction queries.
    k_values:
        Which *k* for Hits@K.

    Returns
    -------
    Dict with ``mrr``, ``hits@1``, ``hits@3``, ``hits@10`` (averaged
    over both head and tail predictions).
    """
    if k_values is None:
        k_values = [1, 3, 10]

    all_ranks: list[int] = []

    for scores, target, true_targets in tail_scores:
        all_ranks.append(compute_filtered_rank(scores, target, true_targets))

    for scores, target, true_targets in head_scores:
        all_ranks.append(compute_filtered_rank(scores, target, true_targets))

    return aggregate_ranks(all_ranks, k_values)
