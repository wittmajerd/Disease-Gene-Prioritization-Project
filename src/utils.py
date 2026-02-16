"""Shared utilities: seeding, I/O, logging, checkpointing.

Every function here is stateless (no class needed).
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml


# ── Paths ───────────────────────────────────────────────────────────


def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[1]


# ── Seeding ─────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (CPU + all CUDA devices)."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Run directory ───────────────────────────────────────────────────


def setup_run_dir(run_name: str | None) -> Path:
    """Create (or reuse) a run directory under ``saves/``.

    If *run_name* is ``None`` or empty, generates a timestamp-based name.

    Returns
    -------
    Absolute path to the run directory.
    """
    ...


# ── Logging ─────────────────────────────────────────────────────────


def setup_logger(run_dir: Path) -> logging.Logger:
    """Configure a logger that writes to both **console** and
    ``run_dir/run.log``.

    Returns
    -------
    A ``logging.Logger`` instance named ``"biokg.pipeline"``.
    """
    ...


def setup_wandb(cfg: Any) -> None:
    """Initialise a Weights & Biases run (if ``cfg.logging.use_wandb``).

    Called once at the start of the pipeline.  Silently skips if wandb
    is not installed or ``use_wandb`` is ``False``.

    Logs the full ``PipelineConfig`` as the W&B config so that every
    hyper-parameter is searchable / filterable in the dashboard.
    """
    ...


def log_metrics(
    logger: logging.Logger,
    epoch: int,
    loss: float,
    metrics: dict[str, float],
    phase: str = "val",
) -> None:
    """Log one evaluation round to the Python logger **and** to W&B
    (if active).

    Parameters
    ----------
    phase:
        ``"val"`` or ``"test"`` — used as prefix in the log line and
        in the W&B metric names.
    """
    ...


# ── Checkpointing ──────────────────────────────────────────────────


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Save model ``state_dict`` to *path* (creates parent dirs)."""
    ...


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a ``state_dict`` into *model* and return it."""
    ...


# ── Result persistence ──────────────────────────────────────────────


def save_yaml(path: Path, payload: Any) -> None:
    """Dump *payload* as YAML to *path*."""
    ...


def append_jsonl(path: Path, payload: Any) -> None:
    """Append one JSON line to a ``.jsonl`` file (creates if missing)."""
    ...


def save_results(run_dir: Path, summary: dict) -> None:
    """Persist a run summary as ``summary.yaml`` inside *run_dir* and
    append it to the global ``saves/results.jsonl``."""
    ...


def pretty(obj: Any) -> str:
    """JSON-pretty-print for logging."""
    return json.dumps(obj, indent=4, default=str)