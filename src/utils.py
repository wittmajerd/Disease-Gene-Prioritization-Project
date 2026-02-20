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

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import wandb  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    wandb = None


# ── Paths ───────────────────────────────────────────────────────────


def project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[1]


# ── Seeding ─────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (CPU + all CUDA devices)."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
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
    runs_root = project_root() / "saves"
    runs_root.mkdir(parents=True, exist_ok=True)

    if run_name is None or not run_name.strip():
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ── Logging ─────────────────────────────────────────────────────────


def setup_logger(run_dir: Path) -> logging.Logger:
    """Configure a logger that writes to both **console** and
    ``run_dir/run.log``.

    Returns
    -------
    A ``logging.Logger`` instance named ``"biokg.pipeline"``.
    """
    logger = logging.getLogger("biokg.pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _to_plain_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_plain_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_dict(v) for v in obj]
    return obj


def setup_wandb(cfg: Any) -> None:
    """Initialise a Weights & Biases run (if ``cfg.logging.use_wandb``).

    Called once at the start of the pipeline.  Silently skips if wandb
    is not installed or ``use_wandb`` is ``False``.

    Logs the full ``PipelineConfig`` as the W&B config so that every
    hyper-parameter is searchable / filterable in the dashboard.
    """
    if not getattr(cfg.logging, "use_wandb", False):
        return
    if wandb is None:
        return

    config_payload = _to_plain_dict(cfg)
    wandb.init(
        project=getattr(cfg.logging, "wandb_project", "disease-gene-prioritization"),
        name=getattr(cfg.logging, "run_name", None),
        config=config_payload,
    )


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
    metric_parts = " | ".join([f"{phase} {k} {v:.4f}" for k, v in metrics.items()])
    logger.info("Epoch %03d | loss %.4f | %s", epoch, loss, metric_parts)

    if wandb is not None and wandb.run is not None:
        payload = {f"{phase}/{k}": v for k, v in metrics.items()}
        payload[f"{phase}/loss"] = loss
        payload["epoch"] = epoch
        wandb.log(payload)


# ── Checkpointing ──────────────────────────────────────────────────


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Save model ``state_dict`` to *path* (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a ``state_dict`` into *model* and return it."""
    state_dict = torch.load(str(path), map_location=device)
    model.load_state_dict(state_dict)
    return model


# ── Result persistence ──────────────────────────────────────────────


def save_yaml(path: Path, payload: Any) -> None:
    """Dump *payload* as YAML to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_to_plain_dict(payload), sort_keys=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    """Append one JSON line to a ``.jsonl`` file (creates if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_plain_dict(payload), default=str) + "\n")


def save_results(run_dir: Path, summary: dict) -> None:
    """Persist a run summary as ``summary.yaml`` inside *run_dir* and
    append it to the global ``saves/results.jsonl``."""
    save_yaml(run_dir / "summary.yaml", summary)
    append_jsonl(project_root() / "saves" / "results.jsonl", summary)


def pretty(obj: Any) -> str:
    """JSON-pretty-print for logging."""
    return json.dumps(obj, indent=4, default=str)