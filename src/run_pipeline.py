from __future__ import annotations

import json
import yaml
import logging
from datetime import datetime
from pathlib import Path

import torch

from src.config import parse_args
from src.data_loading import load_biokg_pickle
from src.graph_builders import (
    add_ppi_edges,
    build_bipartite_graph,
    build_hetero_graph,
    summarize_bipartite,
    summarize_hetero,
)
from src.models import GNNLinkPredictor, HeteroLinkPredictor, Node2VecFeaturizer
from src.splits import random_split_edges
from src.training import (
    EarlyStopper,
    evaluate_hetero,
    evaluate_homo,
    save_ckpt,
    seed_everything,
    train_one_epoch_hetero,
    train_one_epoch_homo,
)


def _pretty(obj) -> str:
    return json.dumps(obj, indent=2)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _setup_run_dir(run_name: str | None) -> Path:
    runs_root = _project_root() / "saves"
    runs_root.mkdir(parents=True, exist_ok=True)
    if run_name is None or not run_name.strip():
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("biokg.pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _save_yaml(path: Path, payload) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

def _append_jsonl(path: Path, payload) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _resolve_layers(cfg):
    if cfg.layer_sizes:
        layer_sizes = list(cfg.layer_sizes)
    else:
        if cfg.layer_count < 1:
            raise ValueError("layer_count must be >= 1")
        layer_sizes = [cfg.layer_size] * max(cfg.layer_count - 1, 0) + [cfg.output_size]

    return cfg.model_kind, layer_sizes


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
    # _save_json(run_dir / "config.json", cfg.__dict__)
    _save_yaml(run_dir / "config.yaml", cfg.__dict__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    raw = load_biokg_pickle()

    if cfg.use_hetero:
        data, num_diseases, num_proteins = build_hetero_graph(raw, include_ppi=cfg.add_ppi)
        logger.info("Graph summary:\n%s", _pretty(summarize_hetero(data)))
        dp_key = ("disease", "disease_protein", "protein")
        edge_index = data[dp_key].edge_index
    else:
        graph, num_diseases, num_proteins = build_bipartite_graph(raw)
        if cfg.add_ppi:
            graph = add_ppi_edges(graph, int(num_diseases), raw)
        logger.info("Graph summary:\n%s", _pretty(summarize_bipartite(graph, int(num_diseases), int(num_proteins))))
        edge_index = graph.edge_index

    train_e, val_e, test_e = random_split_edges(edge_index, cfg.train_ratio, cfg.val_ratio, cfg.seed)

    layer_kind, layer_sizes = _resolve_layers(cfg)

    node2vec_features = None
    node2vec_dim = cfg.node2vec_dim
    if cfg.use_node2vec:
        if node2vec_dim != layer_sizes[0]:
            logger.info(
                "node2vec_dim (%s) does not match layer_sizes[0] (%s). Using %s.",
                node2vec_dim,
                layer_sizes[0],
                layer_sizes[0],
            )
            node2vec_dim = layer_sizes[0]
        featurizer = Node2VecFeaturizer(embedding_dim=node2vec_dim)
        if cfg.use_hetero:
            n2v_graph, n2v_d, n2v_p = build_bipartite_graph(raw)
            if cfg.add_ppi:
                n2v_graph = add_ppi_edges(n2v_graph, int(n2v_d), raw)
            node2vec_features = featurizer.fit_transform(
                n2v_graph.edge_index,
                num_nodes=int(n2v_d + n2v_p),
                device=device,
                epochs=cfg.node2vec_epochs,
                lr=cfg.node2vec_lr,
            )
        else:
            node2vec_features = featurizer.fit_transform(
                graph.edge_index,
                num_nodes=int(num_diseases + num_proteins),
                device=device,
                epochs=cfg.node2vec_epochs,
                lr=cfg.node2vec_lr,
            )

    if cfg.use_hetero:
        data = data.to(device)
        train_e, val_e, test_e = train_e.to(device), val_e.to(device), test_e.to(device)
        disease_features = None
        protein_features = None
        if node2vec_features is not None:
            disease_features = node2vec_features[: int(num_diseases)]
            protein_features = node2vec_features[int(num_diseases) :]
        model = HeteroLinkPredictor(
            metadata=data.metadata(),
            num_diseases=int(num_diseases),
            num_proteins=int(num_proteins),
            layer_kind=layer_kind,
            layer_sizes=layer_sizes,
            heads=cfg.heads,
            disease_features=disease_features,
            protein_features=protein_features,
        ).to(device)
    else:
        graph = graph.to(device)
        train_e, val_e, test_e = train_e.to(device), val_e.to(device), test_e.to(device)
        model = GNNLinkPredictor(
            num_nodes=int(num_diseases + num_proteins),
            layer_kind=layer_kind,
            layer_sizes=layer_sizes,
            heads=cfg.heads,
            init_features=node2vec_features,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(patience=cfg.patience, mode="max")
    best_val = float("-inf")
    best_epoch = 0
    best_val_metrics = None
    ckpt_path = Path(cfg.ckpt_path) if cfg.ckpt_path else (run_dir / "best_model.pth")

    for epoch in range(1, cfg.epochs + 1):
        if cfg.use_hetero:
            loss = train_one_epoch_hetero(model, optimizer, data, train_e, device, int(num_diseases), int(num_proteins))
        else:
            loss = train_one_epoch_homo(model, optimizer, graph.edge_index, train_e, device)

        if epoch % cfg.eval_every == 0:
            if cfg.use_hetero:
                val_metrics = evaluate_hetero(model, data, val_e, device, int(num_diseases), int(num_proteins))
            else:
                val_metrics = evaluate_homo(model, graph.edge_index, val_e, device)

            improved = val_metrics["f1"] > best_val
            if improved:
                best_val = val_metrics["f1"]
                best_epoch = epoch
                best_val_metrics = val_metrics
                save_ckpt(model, ckpt_path)

            logger.info(
                "Epoch %03d | loss %.4f | val f1 %.4f | precision %.4f | recall %.4f",
                epoch,
                loss,
                val_metrics["f1"],
                val_metrics["precision"],
                val_metrics["recall"],
            )

            if stopper.step(val_metrics["f1"]):
                logger.info("Early stopping.")
                break

    if cfg.use_hetero:
        test_metrics = evaluate_hetero(model, data, test_e, device, int(num_diseases), int(num_proteins))
    else:
        test_metrics = evaluate_homo(model, graph.edge_index, test_e, device)

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
        "best_val_f1": best_val,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "config": normalized_config,
    }
    # _save_json(run_dir / "summary.json", summary)
    _save_yaml(run_dir / "summary.yaml", summary)
    _append_jsonl(_project_root() / "saves" / "results.jsonl", summary)


if __name__ == "__main__":
    main()

    # python -m src.run_pipeline
