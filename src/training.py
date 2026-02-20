"""Training loop: Trainer class, early stopping, checkpointing.

The ``Trainer`` encapsulates the full train → evaluate → early-stop →
checkpoint cycle so that ``run.py`` stays minimal.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss

from src.config import PipelineConfig
from src.data import EdgeSplits, negative_sampling
from src.evaluation import classification_metrics
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


# ── Trainer ─────────────────────────────────────────────────────────


class Trainer:
    """Manages the full training lifecycle for a ``HeteroLinkPredictor``.

    Responsibilities
    ────────────────
    * One-epoch training step (positive + negative loss).
    * Periodic evaluation on validation edges.
    * Early stopping based on ``training.primary_metric``.
    * Best-model checkpointing.
    * Logging to console/file and optionally to W&B.

    Parameters
    ----------
    model:
        A ``HeteroLinkPredictor`` instance (already on *device*).
    cfg:
        The full ``PipelineConfig``.
    device:
        CUDA or CPU.
    logger:
        Python logger (file + console handler setup by ``utils``).
    run_dir:
        Directory for checkpoints and artefacts.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: PipelineConfig,
        device: torch.device,
        logger: logging.Logger,
        run_dir: Path,
    ) -> None:
        self.model = model
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
        self.stopper = EarlyStopper(
            patience=cfg.training.patience,
            mode="max",
        )

        # Extracted from target_edge config for convenience.
        self.src_type: str = cfg.graph.target_edge[0]
        self.dst_type: str = cfg.graph.target_edge[2]

        # Filled during training
        self._best_state: dict | None = None
        self._best_metric: float = float("-inf")
        self._best_epoch: int = 0
        self._best_metrics: Dict[str, float] | None = None
        self._ckpt_path: Path = self.run_dir / "best_model.pth"

    # ── Single epoch ────────────────────────────────────────────

    def _train_one_epoch(
        self,
        msg_edge_index_dict: dict,
        pos_edges: Tensor,
    ) -> float:
        """Run one training step.

        1. Forward positive edges through encode → decode.
        2. Sample negatives (filtered) of equal size × ``neg_ratio``.
        3. Compute BCE loss on [pos_logits; neg_logits].
        4. Backward + optimiser step.

        Returns
        -------
        Scalar loss value.
        """
        self.model.train()
        self.optimizer.zero_grad()

        pos_logits = self.model(msg_edge_index_dict, pos_edges, self.src_type, self.dst_type)
        pos_labels = torch.ones(pos_logits.size(0), device=self.device)

        neg_count = int(max(1, round(pos_edges.size(1) * float(self.cfg.training.neg_ratio))))
        num_src = int(self.model.emb[self.src_type].size(0))
        num_dst = int(self.model.emb[self.dst_type].size(0))
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

    # ── Evaluation ──────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(
        self,
        msg_edge_index_dict: dict,
        pos_edges: Tensor,
    ) -> Dict[str, float]:
        """Evaluate on a set of positive edges (val or test).

        Generates negatives of the same size, computes logits,
        and returns the full metrics dict.
        """
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

    # ── Full loop ───────────────────────────────────────────────

    def fit(self, splits: EdgeSplits) -> Dict[str, Any]:
        """Execute the full training loop.

        Parameters
        ----------
        splits:
            Contains train/val/test edges **and**
            ``msg_edge_index_dict`` for message passing.

        Returns
        -------
        A summary dict with ``best_val_metrics``, ``best_epoch``, etc.
        """
        eval_every = max(1, int(self.cfg.training.eval_every))
        primary = str(self.cfg.training.primary_metric)

        for epoch in range(1, int(self.cfg.training.epochs) + 1):
            loss = self._train_one_epoch(splits.msg_edge_index_dict, splits.train_edges)

            if epoch % eval_every != 0:
                continue

            val_metrics = self._evaluate(splits.msg_edge_index_dict, splits.val_edges)
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

    # ── Test ────────────────────────────────────────────────────

    def test(self, splits: EdgeSplits) -> Dict[str, float]:
        """Load the best checkpoint and evaluate on the test set.

        Uses the **full** ``edge_index_dict`` (including all target edges
        used for training) for message passing at test time, following
        the transductive link-prediction convention.
        """
        if self._ckpt_path.exists():
            self.model = load_checkpoint(self.model, self._ckpt_path, self.device)

        metrics = self._evaluate(splits.msg_edge_index_dict, splits.test_edges)
        log_metrics(
            self.logger,
            epoch=self._best_epoch if self._best_epoch > 0 else 0,
            loss=0.0,
            metrics=metrics,
            phase="test",
        )
        return metrics


