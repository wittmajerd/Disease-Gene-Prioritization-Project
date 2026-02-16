from __future__ import annotations


from pathlib import Path

import torch

from src.config import parse_args
from src.data_loading import load_biokg_pickle
from src.training import (
    seed_everything,
    train_with_validation,
)
from src.evaluation import evaluate_hetero, evaluate_homo
from src.model_builder import build_model, build_node2vec_features, resolve_layers
from src.preprocessing import prepare_graph_artifacts, random_split_edges
from src.utils import _append_jsonl, _pretty, _project_root, _save_yaml, _setup_logger, _setup_run_dir


def _normalized_config(cfg, layer_sizes: list[int], node2vec_dim: int, device: torch.device) -> dict:
    normalized = dict(cfg.__dict__)
    normalized["layer_sizes"] = layer_sizes
    normalized["node2vec_dim"] = node2vec_dim
    normalized["device"] = str(device)
    return normalized


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)

    run_dir = _setup_run_dir(cfg.run_name)
    logger = _setup_logger(run_dir)
    _save_yaml(run_dir / "config.yaml", cfg.__dict__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    raw = load_biokg_pickle()

    artifacts = prepare_graph_artifacts(raw, use_hetero=cfg.use_hetero, add_ppi=cfg.add_ppi)
    logger.info("Graph summary:\n%s", _pretty(artifacts.summary))

    splits = random_split_edges(artifacts.edge_index, cfg.train_ratio, cfg.val_ratio, cfg.seed)

    layer_kind, layer_sizes = resolve_layers(cfg)
    node2vec_features, node2vec_dim = build_node2vec_features(cfg, artifacts, layer_sizes, device, logger=logger)

    model = build_model(cfg, artifacts, layer_kind, layer_sizes, node2vec_features, device)

    artifacts.graph_data = artifacts.graph_data.to(device)
    splits.train_edges = splits.train_edges.to(device)
    splits.val_edges = splits.val_edges.to(device)
    splits.test_edges = splits.test_edges.to(device)

    ckpt_path = Path(cfg.ckpt_path) if cfg.ckpt_path else (run_dir / "best_model.pth")

    train_result = train_with_validation(
        cfg=cfg,
        model=model,
        artifacts=artifacts,
        splits=splits,
        device=device,
        logger=logger,
        ckpt_path=ckpt_path,
    )

    if cfg.use_hetero:
        test_metrics = evaluate_hetero(
            model,
            artifacts.graph_data,
            splits.test_edges,
            device,
            artifacts.num_diseases,
            artifacts.num_proteins,
        )
    else:
        test_metrics = evaluate_homo(model, artifacts.graph_data.edge_index, splits.test_edges, device)

    logger.info("Test metrics:\n%s", _pretty(test_metrics))

    normalized_config = _normalized_config(cfg, layer_sizes, node2vec_dim, device)
    _save_yaml(run_dir / "config.yaml", normalized_config)

    summary = {
        "run_dir": str(run_dir),
        "model_kind": cfg.model_kind,
        "use_hetero": cfg.use_hetero,
        "add_ppi": cfg.add_ppi,
        "use_node2vec": cfg.use_node2vec,
        "node2vec_dim": node2vec_dim,
        "layer_kind": layer_kind,
        "layer_sizes": layer_sizes,
        "best_val_f1": train_result["best_val_f1"],
        "best_epoch": train_result["best_epoch"],
        "best_val_metrics": train_result["best_val_metrics"],
        "test_metrics": test_metrics,
        "config": normalized_config,
    }
    _save_yaml(run_dir / "summary.yaml", summary)
    _append_jsonl(_project_root() / "saves" / "results.jsonl", summary)


if __name__ == "__main__":
    main()

    # python -m src.run_pipeline
