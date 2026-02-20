"""Entry point for the Disease-Gene Prioritization pipeline.

Usage
─────
    python -m src.run                         # uses default run_config.yaml
    python -m src.run --config path/to.yaml   # custom config
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.config import PipelineConfig, parse_args
from src.data import (
    EdgeSplits,
    build_full_homogeneous_graph,
    build_homogeneous_from_hetero,
    build_hetero_graph,
    get_graph_summary,
    get_node_offsets,
    load_raw_data,
    split_target_edges,
)
from src.models import HeteroLinkPredictor, Node2VecFeaturizer
from src.training import Trainer
from src.utils import (
    pretty,
    save_results,
    save_yaml,
    seed_everything,
    setup_logger,
    setup_run_dir,
    setup_wandb,
)


def _get_node_counts(data) -> dict[str, int]:
    """Extract ``{node_type: num_nodes}`` from a ``HeteroData``."""
    return {ntype: data[ntype].num_nodes for ntype in data.node_types}


def _maybe_node2vec(
    cfg: PipelineConfig,
    raw,
    data,
    device: torch.device,
    logger,
) -> dict[str, torch.Tensor] | None:
    """Run Node2Vec pre-training if enabled in config.

    Returns ``{node_type: Tensor}`` or ``None``.
    """
    if not cfg.node2vec.enabled:
        return None

    featurizer = Node2VecFeaturizer(cfg.node2vec)

    if cfg.node2vec.use_full_graph:
        # Train on the FULL BioKG — then slice out needed types.
        edge_index, total_nodes = build_full_homogeneous_graph(raw)
        node_offsets = get_node_offsets(raw)
        logger.info(
            "Node2Vec: training on full BioKG (%d nodes, %d edges).",
            total_nodes,
            edge_index.size(1),
        )
    else:
        edge_index, total_nodes, node_offsets = build_homogeneous_from_hetero(
            data,
            node_type_order=cfg.graph.node_types,
        )
        logger.info(
            "Node2Vec: training on configured subgraph (%d nodes, %d edges).",
            total_nodes,
            edge_index.size(1),
        )

    return featurizer.fit_and_get(
        edge_index=edge_index,
        num_nodes=total_nodes,
        node_offsets=node_offsets,
        node_types=cfg.graph.node_types,
        device=device,
    )


def main() -> None:
    # ── 1. Config & setup ───────────────────────────────────────
    cfg: PipelineConfig = parse_args()
    seed_everything(cfg.seed)

    run_dir = setup_run_dir(cfg.logging.run_name)
    logger = setup_logger(run_dir)
    save_yaml(run_dir / "config.yaml", cfg.__dict__)
    setup_wandb(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── 2. Data ─────────────────────────────────────────────────
    raw = load_raw_data(cfg.graph.data_path)
    data = build_hetero_graph(raw, cfg.graph)
    logger.info("Graph summary:\n%s", pretty(get_graph_summary(data)))

    # ── 3. Split ────────────────────────────────────────────────
    splits: EdgeSplits = split_target_edges(
        data,
        target_edge=cfg.graph.target_edge,
        split_cfg=cfg.split,
        seed=cfg.seed,
    )

    # ── 4. Optional Node2Vec features ───────────────────────────
    init_embs = _maybe_node2vec(cfg, raw, data, device, logger)

    # ── 5. Model ────────────────────────────────────────────────
    node_counts = _get_node_counts(data)
    model = HeteroLinkPredictor(
        metadata=data.metadata(),
        node_counts=node_counts,
        model_cfg=cfg.model,
        init_embeddings=init_embs,
    ).to(device)
    logger.info("Model:\n%s", model)

    # Move data to device
    data = data.to(device)
    splits.train_edges = splits.train_edges.to(device)
    splits.val_edges = splits.val_edges.to(device)
    splits.test_edges = splits.test_edges.to(device)
    splits.msg_edge_index_dict = {
        k: v.to(device) for k, v in splits.msg_edge_index_dict.items()
    }

    # ── 6. Train ────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        cfg=cfg,
        device=device,
        logger=logger,
        run_dir=run_dir,
    )
    train_summary = trainer.fit(splits)

    # ── 7. Test ─────────────────────────────────────────────────
    test_metrics = trainer.test(splits)
    logger.info("Test metrics:\n%s", pretty(test_metrics))

    # ── 8. Save results ─────────────────────────────────────────
    summary = {
        "run_dir": str(run_dir),
        "config": cfg.__dict__,
        "graph_summary": get_graph_summary(data),
        "train_summary": train_summary,
        "test_metrics": test_metrics,
    }
    save_results(run_dir, summary)
    logger.info("Run complete. Results saved to %s", run_dir)


if __name__ == "__main__":
    main()

