"""Pipeline configuration with nested dataclasses.

Single source of truth for every tuneable parameter.
YAML → nested dataclasses via ``load_config``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ── Sub-configs ─────────────────────────────────────────────────────


@dataclass
class GraphConfig:
    """What goes into the heterogeneous graph."""

    data_path: str = "ogbl_biokg_raw.pkl"
    node_types: list[str] = field(default_factory=lambda: ["disease", "protein"])
    # Each element is [src_type, relation, dst_type].
    edge_types: list[list[str]] = field(
        default_factory=lambda: [
            ["disease", "disease-protein", "protein"],
        ]
    )
    add_reverse_edges: bool = True
    # The edge type on which we do link prediction.
    target_edge: list[str] = field(
        default_factory=lambda: ["disease", "disease-protein", "protein"]
    )


@dataclass
class SplitConfig:
    """Train / validation / test split ratios (test = 1 - train - val)."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    mode: str = "target"  # "target" = single target edge | "all" = KG completion


@dataclass
class ModelConfig:
    """GNN encoder + decoder hyper-parameters."""

    encoder_type: str = "sage"  # sage | gat | gcn | transformer | hgt
    hidden_dim: int = 256
    num_layers: int = 3
    output_dim: int = 128
    heads: int = 4
    dropout: float = 0.0
    decoder_type: str = "dot_product"  # dot_product | mlp | distmult


@dataclass
class Node2VecConfig:
    """Optional Node2Vec pre-training for initial node features."""

    enabled: bool = False
    use_full_graph: bool = True  # train on FULL BioKG, then slice
    embedding_dim: int = 128
    walk_length: int = 20
    context_size: int = 10
    walks_per_node: int = 10
    p: float = 1.0
    q: float = 1.0
    epochs: int = 20
    batch_size: int = 256
    lr: float = 0.01


@dataclass
class TrainingConfig:
    """Optimiser & training loop settings."""

    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 200
    patience: int = 10
    eval_every: int = 1
    neg_ratio: float = 1.0  # #negatives / #positives
    neg_sampling: str = "uniform"  # uniform | self_adversarial
    adversarial_temperature: float = 1.0
    adversarial_candidates_multiplier: int = 4
    primary_metric: str = "f1"  # metric used for early stopping & best model
    compute_ranking_on_test: bool = True
    ranking_k_values: list[int] = field(default_factory=lambda: [1, 3, 10])
    max_ranking_queries: int | None = None

    # ── KG completion settings (used when split.mode == "all") ───
    loss_fn: str = "self_adversarial"  # bce | self_adversarial | bpr | margin | infonce
    corruption_mode: str = "both"      # head | tail | both
    margin: float = 6.0               # margin for margin-based ranking loss
    infonce_temperature: float = 0.07  # temperature for InfoNCE loss
    max_val_ranking_queries: int = 200 # subsample validation ranking queries
    kg_batch_size: int = 16384         # mini-batch size per edge type in KG mode


@dataclass
class LoggingConfig:
    """Logging & run output settings."""

    use_wandb: bool = False
    wandb_project: str = "disease-gene-prioritization"
    run_name: str | None = None


# ── Top-level config ────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Root configuration object assembled from all sub-configs."""

    seed: int = 42
    graph: GraphConfig = field(default_factory=GraphConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    node2vec: Node2VecConfig = field(default_factory=Node2VecConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ── Loaders ─────────────────────────────────────────────────────────

# Mapping from YAML section name → dataclass
_SECTION_MAP: dict[str, type] = {
    "graph": GraphConfig,
    "split": SplitConfig,
    "model": ModelConfig,
    "node2vec": Node2VecConfig,
    "training": TrainingConfig,
    "logging": LoggingConfig,
}


def _build_section(cls: type, raw: dict[str, Any]) -> Any:
    """Instantiate a dataclass from a raw dict, ignoring unknown keys."""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in raw.items() if k in known})


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config file and return a fully typed ``PipelineConfig``.

    Unknown top-level keys are silently ignored so the YAML can contain
    comments or experimental fields without breaking the loader.
    """
    path = Path(path)
    raw: dict = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    kwargs: dict[str, Any] = {}
    if "seed" in raw:
        kwargs["seed"] = int(raw["seed"])

    for section_name, section_cls in _SECTION_MAP.items():
        if section_name in raw and isinstance(raw[section_name], dict):
            kwargs[section_name] = _build_section(section_cls, raw[section_name])

    return PipelineConfig(**kwargs)


def parse_args() -> PipelineConfig:
    """CLI entry-point: ``python -m src.run --config path/to/config.yaml``."""
    parser = argparse.ArgumentParser(description="Disease-Gene Prioritization Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: <project_root>/run_config.yaml)",
    )
    args = parser.parse_args()

    from src.utils import project_root  # avoid circular import

    config_path = Path(args.config) if args.config else project_root() / "run_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    return load_config(config_path)
