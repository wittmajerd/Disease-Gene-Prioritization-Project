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
        ...


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

        self.optimizer: torch.optim.Optimizer = ...
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
        ...

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
        ...

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
        ...

    # ── Test ────────────────────────────────────────────────────

    def test(self, splits: EdgeSplits) -> Dict[str, float]:
        """Load the best checkpoint and evaluate on the test set.

        Uses the **full** ``edge_index_dict`` (including all target edges
        used for training) for message passing at test time, following
        the transductive link-prediction convention.
        """
        ...


