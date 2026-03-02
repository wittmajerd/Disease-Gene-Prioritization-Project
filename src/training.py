"""Training loop: Trainer class, early stopping, checkpointing.

Supports two modes controlled by ``split.mode`` in the config:

* **target** (legacy) — single edge type, classification metrics,
  BCE or self-adversarial loss.
* **all** (KG completion) — every relation split, ranking metrics
  (filtered MRR / Hits@K), ranking losses (self-adversarial, BPR,
  margin, InfoNCE).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BCEWithLogitsLoss

from src.config import PipelineConfig
from src.data import EdgeSplits, KGSplits, corrupt_triples, negative_sampling
from src.evaluation import (
    aggregate_ranks,
    build_filter_dicts,
    classification_metrics,
    compute_filtered_rank,
    ranking_metrics,
)
from src.models import HeteroLinkPredictor, MultiRelationDistMultDecoder
from src.utils import load_checkpoint, log_metrics, save_checkpoint


# ── Early stopping ──────────────────────────────────────────────────


class EarlyStopper:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience:
        How many evaluation rounds without improvement before stopping.
    mode:
        ``"max"`` (higher is better, e.g. F1) or ``"min"`` (lower is
        better, e.g. loss).
    """

    def __init__(self, patience: int = 10, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.counter: int = 0

    def step(self, metric: float) -> bool:
        """Return ``True`` when training should stop."""
        if self.best is None:
            self.best = metric
            self.counter = 0
            return False

        improved = metric > self.best if self.mode == "max" else metric < self.best
        if improved:
            self.best = metric
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


# ── Ranking loss functions ──────────────────────────────────────────


def _bpr_loss(pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
    """Bayesian Personalised Ranking loss.

    .. math::
        \\mathcal{L} = -\\frac{1}{N} \\sum \\log \\sigma(s_{pos} - s_{neg})
    """
    n_pos = pos_scores.size(0)
    k = neg_scores.size(0) // n_pos
    if k > 1:
        neg_scores = neg_scores.view(n_pos, k)
        pos_scores = pos_scores.unsqueeze(1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def _margin_loss(pos_scores: Tensor, neg_scores: Tensor, margin: float = 6.0) -> Tensor:
    """Margin-based ranking loss (TransE-style).

    .. math::
        \\mathcal{L} = \\frac{1}{N} \\sum \\max(0, \\gamma + s_{neg} - s_{pos})
    """
    n_pos = pos_scores.size(0)
    k = neg_scores.size(0) // n_pos
    if k > 1:
        neg_scores = neg_scores.view(n_pos, k)
        pos_scores = pos_scores.unsqueeze(1)
    return F.relu(margin + neg_scores - pos_scores).mean()


def _infonce_loss(pos_scores: Tensor, neg_scores: Tensor, temperature: float = 0.07) -> Tensor:
    """InfoNCE / cross-entropy ranking loss.

    Treats each positive + its K negatives as a softmax classification.
    Most aligned with ranking metrics.
    """
    n_pos = pos_scores.size(0)
    k = neg_scores.size(0) // n_pos
    neg_scores = neg_scores.view(n_pos, k)
    pos_scores = pos_scores.unsqueeze(1)

    logits = torch.cat([pos_scores, neg_scores], dim=1) / temperature
    labels = torch.zeros(n_pos, device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


def _self_adversarial_loss(
    pos_scores: Tensor,
    neg_scores: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Self-adversarial negative sampling loss (RotatE)."""
    weights = torch.softmax(temperature * neg_scores.detach(), dim=0)
    pos_loss = -F.logsigmoid(pos_scores).mean()
    neg_loss = -(weights * F.logsigmoid(-neg_scores)).sum()
    return pos_loss + neg_loss


# ── Trainer ─────────────────────────────────────────────────────────


class Trainer:
    """Manages the full training lifecycle for a ``HeteroLinkPredictor``.

    Supports both the legacy single-edge mode (``EdgeSplits``) and the
    new KG-completion mode (``KGSplits``).
    """

    def __init__(
        self,
        model: HeteroLinkPredictor,
        cfg: PipelineConfig,
        device: torch.device,
        logger: logging.Logger,
        run_dir: Path,
    ) -> None:
        self.model: HeteroLinkPredictor = model
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.run_dir = run_dir

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.criterion = BCEWithLogitsLoss()

        # Determine which mode we're operating in
        self.split_mode = getattr(cfg.split, "mode", "target")

        # Early stopping mode
        if self.split_mode == "all":
            es_mode = "max"  # MRR is "higher is better"
        else:
            primary = str(cfg.training.primary_metric)
            es_mode = "min" if primary in ("loss",) else "max"

        self.stopper = EarlyStopper(
            patience=cfg.training.patience,
            mode=es_mode,
        )

        # Legacy single-edge convenience
        self.src_type: str = cfg.graph.target_edge[0]
        self.dst_type: str = cfg.graph.target_edge[2]

        # Tracking
        self._best_state: dict | None = None
        self._best_metric: float = float("-inf")
        self._best_epoch: int = 0
        self._best_metrics: Dict[str, float] | None = None
        self._ckpt_path: Path = self.run_dir / "best_model.pth"

    # ================================================================
    # Legacy single-edge mode (backward compatible)
    # ================================================================

    def _train_one_epoch_legacy(
        self,
        msg_edge_index_dict: dict,
        pos_edges: Tensor,
    ) -> float:
        """Run one training step (single target edge, BCE or self-adversarial)."""
        self.model.train()
        self.optimizer.zero_grad()

        pos_logits = self.model(msg_edge_index_dict, pos_edges, self.src_type, self.dst_type)

        neg_count = int(max(1, round(pos_edges.size(1) * float(self.cfg.training.neg_ratio))))
        num_src = int(self.model.emb[self.src_type].size(0))
        num_dst = int(self.model.emb[self.dst_type].size(0))

        neg_sampling_kind = str(getattr(self.cfg.training, "neg_sampling", "uniform")).lower()
        if neg_sampling_kind == "self_adversarial":
            cand_mult = int(max(1, getattr(self.cfg.training, "adversarial_candidates_multiplier", 4)))
            cand_count = int(max(neg_count, neg_count * cand_mult))

            neg_edges = negative_sampling(
                num_src=num_src,
                num_dst=num_dst,
                num_samples=cand_count,
                positive_edges=pos_edges,
                device=self.device,
            )
            neg_logits = self.model(msg_edge_index_dict, neg_edges, self.src_type, self.dst_type)

            temp = float(getattr(self.cfg.training, "adversarial_temperature", 1.0))
            weights = torch.softmax(temp * neg_logits.detach(), dim=0)

            pos_loss = -F.logsigmoid(pos_logits).mean()
            neg_loss = -(weights * F.logsigmoid(-neg_logits)).sum()
            loss = pos_loss + neg_loss
        else:
            pos_labels = torch.ones(pos_logits.size(0), device=self.device)
            neg_edges = negative_sampling(
                num_src=num_src,
                num_dst=num_dst,
                num_samples=neg_count,
                positive_edges=pos_edges,
                device=self.device,
            )
            neg_logits = self.model(msg_edge_index_dict, neg_edges, self.src_type, self.dst_type)
            neg_labels = torch.zeros(neg_logits.size(0), device=self.device)

            logits = torch.cat([pos_logits, neg_logits], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)
            loss = self.criterion(logits, labels)

        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    @torch.no_grad()
    def _evaluate_legacy(
        self,
        msg_edge_index_dict: dict,
        pos_edges: Tensor,
    ) -> Dict[str, float]:
        """Classification-based evaluation (legacy)."""
        self.model.eval()

        pos_logits = self.model(msg_edge_index_dict, pos_edges, self.src_type, self.dst_type)
        num_src = int(self.model.emb[self.src_type].size(0))
        num_dst = int(self.model.emb[self.dst_type].size(0))
        neg_edges = negative_sampling(
            num_src=num_src,
            num_dst=num_dst,
            num_samples=pos_edges.size(1),
            positive_edges=pos_edges,
            device=self.device,
        )
        neg_logits = self.model(msg_edge_index_dict, neg_edges, self.src_type, self.dst_type)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat(
            [
                torch.ones(pos_logits.size(0), device=self.device),
                torch.zeros(neg_logits.size(0), device=self.device),
            ],
            dim=0,
        )
        return classification_metrics(logits, labels)

    @torch.no_grad()
    def _evaluate_ranking_legacy(
        self,
        msg_edge_index_dict: dict,
        pos_edges: Tensor,
    ) -> Dict[str, float]:
        """Query-wise ranking metrics on test positives (legacy)."""
        self.model.eval()

        k_values = list(getattr(self.cfg.training, "ranking_k_values", [1, 3, 10]))
        max_queries = getattr(self.cfg.training, "max_ranking_queries", None)

        unique_sources = torch.unique(pos_edges[0]).detach()
        if max_queries is not None:
            unique_sources = unique_sources[: int(max_queries)]

        num_dst = int(self.model.emb[self.dst_type].size(0))
        dst_idx = torch.arange(num_dst, device=self.device, dtype=torch.long)

        per_query: list[Dict[str, float]] = []
        for src_idx in unique_sources:
            src_id = int(src_idx.item())

            edge_mask = pos_edges[0] == src_id
            pos_dsts = pos_edges[1][edge_mask]
            if pos_dsts.numel() == 0:
                continue

            src_col = torch.full((num_dst,), src_id, device=self.device, dtype=torch.long)
            edge_label_index = torch.stack([src_col, dst_idx], dim=0)
            logits = self.model(msg_edge_index_dict, edge_label_index, self.src_type, self.dst_type)

            positive_mask = torch.zeros(num_dst, device=self.device, dtype=torch.bool)
            positive_mask[pos_dsts] = True
            per_query.append(ranking_metrics(logits, positive_mask, k_values=k_values))

        if not per_query:
            out: Dict[str, float] = {"mrr": 0.0}
            out.update({f"hits@{k}": 0.0 for k in k_values})
            return out

        agg: Dict[str, float] = {}
        for key in per_query[0].keys():
            agg[key] = float(sum(m[key] for m in per_query) / len(per_query))
        return agg

    # ================================================================
    # KG-completion mode (all edge types, ranking metrics)
    # ================================================================

    def _train_one_epoch_kg(
        self,
        splits: KGSplits,
    ) -> float:
        """One training epoch over all supervision edge types.

        Each edge type is processed in mini-batches of size
        ``cfg.training.kg_batch_size`` to avoid OOM on large relations.
        The encoder is run once; gradients accumulate across batches
        then a single optimiser step is taken.
        """
        self.model.train()
        self.optimizer.zero_grad()

        z_dict = self.model.encode(splits.msg_edge_index_dict)

        total_loss = torch.tensor(0.0, device=self.device)
        total_triples = 0

        loss_fn = getattr(self.cfg.training, "loss_fn", "self_adversarial")
        corruption_mode = getattr(self.cfg.training, "corruption_mode", "both")
        neg_per_pos = max(1, int(self.cfg.training.neg_ratio))
        batch_size = int(getattr(self.cfg.training, "kg_batch_size", 16384))

        for edge_type in splits.supervision_edge_types:
            src_type, rel, dst_type = edge_type
            pos_edges = splits.train[edge_type]
            rel_key = MultiRelationDistMultDecoder._key(edge_type)

            n_edges = pos_edges.size(1)
            if n_edges == 0:
                continue

            num_src = splits.num_nodes_dict[src_type]
            num_dst = splits.num_nodes_dict[dst_type]

            # Shuffle edges each epoch
            perm = torch.randperm(n_edges, device=self.device)

            for start in range(0, n_edges, batch_size):
                batch_idx = perm[start : start + batch_size]
                batch_pos = pos_edges[:, batch_idx]
                b = batch_pos.size(1)

                pos_scores = self.model.decode(z_dict, batch_pos, src_type, dst_type, rel_key)

                neg_edges = corrupt_triples(
                    batch_pos, num_src, num_dst,
                    num_neg_per_pos=neg_per_pos,
                    mode=corruption_mode,
                )
                neg_scores = self.model.decode(z_dict, neg_edges, src_type, dst_type, rel_key)

                if loss_fn == "bce":
                    all_scores = torch.cat([pos_scores, neg_scores])
                    labels = torch.cat([
                        torch.ones_like(pos_scores),
                        torch.zeros_like(neg_scores),
                    ])
                    edge_loss = self.criterion(all_scores, labels)
                elif loss_fn == "bpr":
                    edge_loss = _bpr_loss(pos_scores, neg_scores)
                elif loss_fn == "margin":
                    edge_loss = _margin_loss(
                        pos_scores, neg_scores,
                        margin=getattr(self.cfg.training, "margin", 6.0),
                    )
                elif loss_fn == "infonce":
                    edge_loss = _infonce_loss(
                        pos_scores, neg_scores,
                        temperature=getattr(self.cfg.training, "infonce_temperature", 0.07),
                    )
                elif loss_fn == "self_adversarial":
                    edge_loss = _self_adversarial_loss(
                        pos_scores, neg_scores,
                        temperature=getattr(self.cfg.training, "adversarial_temperature", 1.0),
                    )
                else:
                    raise ValueError(f"Unknown loss function: {loss_fn}")

                # Scale loss by batch proportion so gradient magnitude
                # is independent of batch_size.
                weight = b / max(1, n_edges)
                (edge_loss * weight).backward(retain_graph=True)

                total_loss = total_loss + edge_loss.detach() * b
                total_triples += b

        self.optimizer.step()

        if total_triples > 0:
            return float((total_loss / total_triples).item())
        return 0.0

    @torch.no_grad()
    def _evaluate_ranking_kg(
        self,
        splits: KGSplits,
        eval_edges: dict[tuple[str, str, str], Tensor],
        tail_filter: dict,
        head_filter: dict,
        max_queries: int | None = None,
    ) -> Dict[str, float]:
        """Filtered ranking evaluation over all edge types.

        For each triple ``(h, r, t)``:
        - Tail prediction: score ``(h, r, e)`` for all *e*, filter, rank.
        - Head prediction: score ``(e, r, t)`` for all *e*, filter, rank.
        """
        self.model.eval()
        z_dict = self.model.encode(splits.msg_edge_index_dict)

        k_values = list(getattr(self.cfg.training, "ranking_k_values", [1, 3, 10]))
        all_ranks: list[int] = []
        queries_used = 0

        for edge_type in splits.supervision_edge_types:
            src_type, rel, dst_type = edge_type
            pos_edges = eval_edges.get(edge_type)
            if pos_edges is None or pos_edges.size(1) == 0:
                continue

            rel_key = MultiRelationDistMultDecoder._key(edge_type)
            num_src = splits.num_nodes_dict[src_type]
            num_dst = splits.num_nodes_dict[dst_type]

            num_triples = pos_edges.size(1)
            indices = list(range(num_triples))

            if max_queries is not None:
                per_rel_budget = max(1, max_queries // max(1, len(splits.supervision_edge_types)))
                if num_triples > per_rel_budget:
                    perm = torch.randperm(num_triples)[:per_rel_budget]
                    indices = perm.tolist()

            for idx in indices:
                h = int(pos_edges[0, idx].item())
                t = int(pos_edges[1, idx].item())

                # Tail prediction
                src_col = torch.full((num_dst,), h, device=self.device, dtype=torch.long)
                dst_col = torch.arange(num_dst, device=self.device, dtype=torch.long)
                edge_idx = torch.stack([src_col, dst_col], dim=0)
                tail_scores = self.model.decode(z_dict, edge_idx, src_type, dst_type, rel_key)

                true_tails = tail_filter.get((edge_type, h), torch.tensor([], dtype=torch.long))
                all_ranks.append(compute_filtered_rank(tail_scores, t, true_tails.to(self.device)))

                # Head prediction
                src_col = torch.arange(num_src, device=self.device, dtype=torch.long)
                dst_col = torch.full((num_src,), t, device=self.device, dtype=torch.long)
                edge_idx = torch.stack([src_col, dst_col], dim=0)
                head_scores = self.model.decode(z_dict, edge_idx, src_type, dst_type, rel_key)

                true_heads = head_filter.get((edge_type, t), torch.tensor([], dtype=torch.long))
                all_ranks.append(compute_filtered_rank(head_scores, h, true_heads.to(self.device)))

                queries_used += 1
                if max_queries is not None and queries_used >= max_queries:
                    break

            if max_queries is not None and queries_used >= max_queries:
                break

        return aggregate_ranks(all_ranks, k_values)

    # ================================================================
    # Unified fit / test interface
    # ================================================================

    def fit(self, splits: Union[EdgeSplits, KGSplits]) -> Dict[str, Any]:
        """Execute the full training loop.

        Dispatches to the legacy or KG-completion path based on
        ``cfg.split.mode``.
        """
        if self.split_mode == "all" and isinstance(splits, KGSplits):
            return self._fit_kg(splits)
        elif isinstance(splits, EdgeSplits):
            return self._fit_legacy(splits)
        else:
            raise TypeError(
                f"Expected EdgeSplits for mode='target' or KGSplits for mode='all', "
                f"got {type(splits).__name__}"
            )

    def test(self, splits: Union[EdgeSplits, KGSplits]) -> Dict[str, float]:
        """Load best checkpoint and evaluate on the test set."""
        if self.split_mode == "all" and isinstance(splits, KGSplits):
            return self._test_kg(splits)
        elif isinstance(splits, EdgeSplits):
            return self._test_legacy(splits)
        else:
            raise TypeError(f"Unexpected splits type: {type(splits).__name__}")

    # ── Legacy fit/test ─────────────────────────────────────────

    def _fit_legacy(self, splits: EdgeSplits) -> Dict[str, Any]:
        eval_every = max(1, int(self.cfg.training.eval_every))
        primary = str(self.cfg.training.primary_metric)

        for epoch in range(1, int(self.cfg.training.epochs) + 1):
            loss = self._train_one_epoch_legacy(splits.msg_edge_index_dict, splits.train_edges)

            if epoch % eval_every != 0:
                continue

            val_metrics = self._evaluate_legacy(splits.msg_edge_index_dict, splits.val_edges)
            if primary not in val_metrics:
                raise KeyError(
                    f"Primary metric '{primary}' not found in metrics: {list(val_metrics.keys())}"
                )

            metric_value = float(val_metrics[primary])
            log_metrics(self.logger, epoch=epoch, loss=loss, metrics=val_metrics, phase="val")

            if metric_value > self._best_metric:
                self._best_metric = metric_value
                self._best_epoch = epoch
                self._best_metrics = dict(val_metrics)
                save_checkpoint(self.model, self._ckpt_path)

            if self.stopper.step(metric_value):
                self.logger.info("Early stopping at epoch %d", epoch)
                break

        return {
            "best_epoch": self._best_epoch,
            "best_metric": self._best_metric,
            "best_val_metrics": self._best_metrics,
            "checkpoint": str(self._ckpt_path),
        }

    def _test_legacy(self, splits: EdgeSplits) -> Dict[str, float]:
        if self._ckpt_path.exists():
            self.model = load_checkpoint(self.model, self._ckpt_path, self.device)  # type: ignore[assignment]

        metrics = self._evaluate_legacy(splits.msg_edge_index_dict, splits.test_edges)
        if bool(getattr(self.cfg.training, "compute_ranking_on_test", True)):
            metrics.update(
                self._evaluate_ranking_legacy(splits.msg_edge_index_dict, splits.test_edges)
            )

        log_metrics(
            self.logger,
            epoch=self._best_epoch if self._best_epoch > 0 else 0,
            loss=0.0,
            metrics=metrics,
            phase="test",
        )
        return metrics

    # ── KG-completion fit/test ──────────────────────────────────

    def _fit_kg(self, splits: KGSplits) -> Dict[str, Any]:
        eval_every = max(1, int(self.cfg.training.eval_every))
        max_val_queries = int(getattr(self.cfg.training, "max_val_ranking_queries", 200))

        self.logger.info("Building filter dicts for filtered evaluation...")
        tail_filter, head_filter = build_filter_dicts(splits.all_positives)

        primary = "mrr"

        for epoch in range(1, int(self.cfg.training.epochs) + 1):
            loss = self._train_one_epoch_kg(splits)

            if epoch % eval_every != 0:
                continue

            val_metrics = self._evaluate_ranking_kg(
                splits,
                eval_edges=splits.val,
                tail_filter=tail_filter,
                head_filter=head_filter,
                max_queries=max_val_queries,
            )

            metric_value = float(val_metrics[primary])
            log_metrics(self.logger, epoch=epoch, loss=loss, metrics=val_metrics, phase="val")

            if metric_value > self._best_metric:
                self._best_metric = metric_value
                self._best_epoch = epoch
                self._best_metrics = dict(val_metrics)
                save_checkpoint(self.model, self._ckpt_path)

            if self.stopper.step(metric_value):
                self.logger.info("Early stopping at epoch %d", epoch)
                break

        return {
            "best_epoch": self._best_epoch,
            "best_metric": self._best_metric,
            "best_val_metrics": self._best_metrics,
            "checkpoint": str(self._ckpt_path),
        }

    def _test_kg(self, splits: KGSplits) -> Dict[str, float]:
        if self._ckpt_path.exists():
            self.model = load_checkpoint(self.model, self._ckpt_path, self.device)  # type: ignore[assignment]

        max_queries = getattr(self.cfg.training, "max_ranking_queries", None)

        self.logger.info("Building filter dicts for test evaluation...")
        tail_filter, head_filter = build_filter_dicts(splits.all_positives)

        metrics = self._evaluate_ranking_kg(
            splits,
            eval_edges=splits.test,
            tail_filter=tail_filter,
            head_filter=head_filter,
            max_queries=max_queries,
        )

        log_metrics(
            self.logger,
            epoch=self._best_epoch if self._best_epoch > 0 else 0,
            loss=0.0,
            metrics=metrics,
            phase="test",
        )
        return metrics