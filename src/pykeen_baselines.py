"""PyKEEN KGE baselines on the same BioKG splits.

Trains standard KGE models (TransE, RotatE, ComplEx, DistMult) through
PyKEEN and evaluates them with the standard filtered ranking protocol,
producing MRR / Hits@K numbers directly comparable to the GNN pipeline.

Usage
─────
    # From project root:
    python -m src.pykeen_baselines                          # default config
    python -m src.pykeen_baselines --config my_config.yaml  # custom config

    # Or call from a notebook / script:
    from src.pykeen_baselines import run_pykeen_baselines
    results = run_pykeen_baselines(cfg)

Design
──────
1. Load data & build HeteroData exactly as ``run.py`` does.
2. Convert ``KGSplits`` into PyKEEN ``TriplesFactory`` objects.
3. For each requested model, train via ``pykeen.pipeline``.
4. Collect & log filtered MRR / Hits@K.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import Tensor

# ── PyKEEN imports (guarded) ────────────────────────────────────────

try:
    from pykeen.pipeline import pipeline as pykeen_pipeline  # type: ignore[import-unresolved]
    from pykeen.triples import CoreTriplesFactory, TriplesFactory  # type: ignore[import-unresolved]

    _PYKEEN_AVAILABLE = True
except ImportError:
    _PYKEEN_AVAILABLE = False

from src.config import PipelineConfig, parse_args
from src.data import (
    KGSplits,
    build_hetero_graph,
    load_raw_data,
    split_all_edges,
)
from src.utils import (
    pretty,
    save_yaml,
    seed_everything,
    setup_logger,
    setup_run_dir,
)


# ═══════════════════════════════════════════════════════════════════
# Config for baselines
# ═══════════════════════════════════════════════════════════════════


@dataclass
class BaselineConfig:
    """Hyper-parameters for PyKEEN baseline runs."""

    models: list[str] = field(
        default_factory=lambda: ["TransE", "RotatE", "ComplEx", "DistMult"]
    )
    embedding_dim: int = 128
    epochs: int = 100
    batch_size: int = 1024
    lr: float = 0.001
    num_negs_per_pos: int = 64
    # Evaluator
    filtered: bool = True
    ks: list[int] = field(default_factory=lambda: [1, 3, 10])


def _default_baseline_cfg() -> BaselineConfig:
    return BaselineConfig()


# ═══════════════════════════════════════════════════════════════════
# Conversion helpers
# ═══════════════════════════════════════════════════════════════════


def _kg_splits_to_mapped_triples(
    edge_dict: dict[tuple[str, str, str], Tensor],
    num_nodes_dict: dict[str, int],
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Convert our edge dicts to a flat ``(h, r, t)`` int array.

    Because BioKG node IDs are local per type, we create a *global*
    entity ID space: ``entity_id = offset[type] + local_id``.

    Returns
    -------
    triples : np.ndarray of shape ``[N, 3]`` (int64)
        Columns: ``[head, relation, tail]``.
    entity_to_id : dict[str, int]
        Maps ``"type:local_id"`` → global int.
    relation_to_id : dict[str, int]
        Maps ``"src__rel__dst"`` → relation int.
    """
    # Build global entity offsets
    sorted_types = sorted(num_nodes_dict.keys())
    type_offset: dict[str, int] = {}
    offset = 0
    for t in sorted_types:
        type_offset[t] = offset
        offset += num_nodes_dict[t]

    # Build relation mapping
    relation_to_id: dict[str, int] = {}
    for edge_type in edge_dict:
        key = "__".join(edge_type)
        if key not in relation_to_id:
            relation_to_id[key] = len(relation_to_id)

    rows: list[list[int]] = []
    for edge_type, ei in edge_dict.items():
        src_type, rel, dst_type = edge_type
        rel_id = relation_to_id["__".join(edge_type)]
        src_offset = type_offset[src_type]
        dst_offset = type_offset[dst_type]

        heads = ei[0].numpy() + src_offset
        tails = ei[1].numpy() + dst_offset

        for h, t in zip(heads, tails):
            rows.append([int(h), rel_id, int(t)])

    triples = np.array(rows, dtype=np.int64) if rows else np.empty((0, 3), dtype=np.int64)

    entity_to_id: dict[str, int] = {}
    for t in sorted_types:
        for i in range(num_nodes_dict[t]):
            entity_to_id[f"{t}:{i}"] = type_offset[t] + i

    return triples, entity_to_id, relation_to_id


def kg_splits_to_pykeen(
    splits: KGSplits,
) -> tuple["CoreTriplesFactory", "CoreTriplesFactory", "CoreTriplesFactory"]:
    """Convert ``KGSplits`` → PyKEEN ``CoreTriplesFactory`` for train / val / test.

    Uses a unified entity and relation mapping across all three splits
    so that IDs are consistent.
    """
    if not _PYKEEN_AVAILABLE:
        raise ImportError(
            "PyKEEN is not installed. Run: pip install pykeen"
        )

    # Merge all edges to build a complete entity + relation vocabulary
    all_edges: dict[tuple[str, str, str], Tensor] = {}
    for d in [splits.train, splits.val, splits.test]:
        for k, v in d.items():
            if k in all_edges:
                all_edges[k] = torch.cat([all_edges[k], v], dim=1)
            else:
                all_edges[k] = v

    _, entity_to_id, relation_to_id = _kg_splits_to_mapped_triples(
        all_edges, splits.num_nodes_dict
    )

    num_entities = sum(splits.num_nodes_dict.values())
    num_relations = len(relation_to_id)

    def _make_factory(
        edge_dict: dict[tuple[str, str, str], Tensor],
    ) -> "CoreTriplesFactory":
        triples, _, _ = _kg_splits_to_mapped_triples(
            edge_dict, splits.num_nodes_dict
        )
        mapped = torch.from_numpy(triples).long()
        return CoreTriplesFactory.create(  # type: ignore[possibly-undefined]
            mapped_triples=mapped,
            num_entities=num_entities,
            num_relations=num_relations,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
        )

    train_tf = _make_factory(splits.train)
    val_tf = _make_factory(splits.val)
    test_tf = _make_factory(splits.test)

    return train_tf, val_tf, test_tf


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════


def run_single_baseline(
    model_name: str,
    train_tf: "CoreTriplesFactory",
    val_tf: "CoreTriplesFactory",
    test_tf: "CoreTriplesFactory",
    baseline_cfg: BaselineConfig,
    device: str = "cuda",
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """Train & evaluate a single PyKEEN model.

    Returns
    -------
    A dict with ``model``, ``metrics`` (flat MRR / Hits@K), and the
    full PyKEEN pipeline result object under ``raw``.
    """
    if not _PYKEEN_AVAILABLE:
        raise ImportError("PyKEEN is not installed.")

    if logger:
        logger.info("Training PyKEEN model: %s", model_name)

    result = pykeen_pipeline(  # type: ignore[possibly-undefined]
        training=train_tf,
        validation=val_tf,
        testing=test_tf,
        model=model_name,
        model_kwargs={"embedding_dim": baseline_cfg.embedding_dim},
        optimizer="Adam",
        optimizer_kwargs={"lr": baseline_cfg.lr},
        training_kwargs={
            "num_epochs": baseline_cfg.epochs,
            "batch_size": baseline_cfg.batch_size,
        },
        negative_sampler_kwargs={"num_negs_per_pos": baseline_cfg.num_negs_per_pos},
        evaluator_kwargs={"filtered": baseline_cfg.filtered},
        device=device,
        random_seed=42,
    )

    # Extract key metrics
    metrics: Dict[str, float] = {}
    mr = result.metric_results
    try:
        metrics["mrr"] = float(mr.get_metric("both.realistic.inverse_harmonic_mean_rank"))
    except Exception:
        metrics["mrr"] = 0.0

    for k in baseline_cfg.ks:
        try:
            metrics[f"hits@{k}"] = float(
                mr.get_metric(f"both.realistic.hits_at_{k}")
            )
        except Exception:
            metrics[f"hits@{k}"] = 0.0

    if logger:
        logger.info("  %s results: %s", model_name, pretty(metrics))

    return {
        "model": model_name,
        "metrics": metrics,
        "raw": result,
    }


def run_pykeen_baselines(
    pipeline_cfg: PipelineConfig,
    baseline_cfg: BaselineConfig | None = None,
    logger: logging.Logger | None = None,
    run_dir: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    """Run all configured PyKEEN baselines end-to-end.

    Parameters
    ----------
    pipeline_cfg:
        The same config used by the GNN pipeline (we reuse ``graph``
        and ``split`` sections).
    baseline_cfg:
        PyKEEN-specific hyper-parameters.  ``None`` → defaults.
    logger:
        Optional logger.
    run_dir:
        Where to save results.

    Returns
    -------
    ``{model_name: {metric_name: value}}``
    """
    if not _PYKEEN_AVAILABLE:
        raise ImportError(
            "PyKEEN is not installed. Run:  pip install pykeen"
        )

    if baseline_cfg is None:
        baseline_cfg = _default_baseline_cfg()

    seed_everything(pipeline_cfg.seed)

    if logger is None:
        logger = logging.getLogger("pykeen_baselines")
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── 1. Data ─────────────────────────────────────────────────
    raw = load_raw_data(pipeline_cfg.graph.data_path)
    data = build_hetero_graph(raw, pipeline_cfg.graph)

    # Force mode to "all" for baseline comparison
    pipeline_cfg.split.mode = "all"
    splits: KGSplits = split_all_edges(
        data,
        split_cfg=pipeline_cfg.split,
        seed=pipeline_cfg.seed,
    )

    logger.info(
        "KGSplits: %d supervision types, %d total train triples",
        len(splits.supervision_edge_types),
        sum(v.size(1) for v in splits.train.values()),
    )

    # ── 2. Convert to PyKEEN ────────────────────────────────────
    train_tf, val_tf, test_tf = kg_splits_to_pykeen(splits)
    logger.info(
        "PyKEEN triples — train: %d, val: %d, test: %d",
        train_tf.num_triples,
        val_tf.num_triples,
        test_tf.num_triples,
    )

    # ── 3. Train each model ─────────────────────────────────────
    all_results: Dict[str, Dict[str, float]] = {}

    for model_name in baseline_cfg.models:
        res = run_single_baseline(
            model_name=model_name,
            train_tf=train_tf,
            val_tf=val_tf,
            test_tf=test_tf,
            baseline_cfg=baseline_cfg,
            device=device,
            logger=logger,
        )
        all_results[model_name] = res["metrics"]

    # ── 4. Summary ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PyKEEN Baseline Summary")
    logger.info("=" * 60)
    header = f"{'Model':<15} {'MRR':>8}"
    for k in baseline_cfg.ks:
        header += f" {'H@'+str(k):>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for name, m in all_results.items():
        line = f"{name:<15} {m.get('mrr', 0):.4f}"
        for k in baseline_cfg.ks:
            line += f" {m.get(f'hits@{k}', 0):.4f}"
        logger.info(line)

    # Save if run_dir provided
    if run_dir is not None:
        save_yaml(run_dir / "pykeen_baselines.yaml", all_results)
        logger.info("Results saved to %s", run_dir / "pykeen_baselines.yaml")

    return all_results


# ═══════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point: ``python -m src.pykeen_baselines``."""
    pipeline_cfg = parse_args()

    run_dir = setup_run_dir("pykeen_baselines")
    logger = setup_logger(run_dir)
    save_yaml(run_dir / "config.yaml", pipeline_cfg.__dict__)

    all_results = run_pykeen_baselines(
        pipeline_cfg=pipeline_cfg,
        logger=logger,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
