from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import parse_args
from .data_loading import load_biokg_pickle
from .graph_builders import add_ppi_edges, build_bipartite_graph, build_hetero_graph, summarize_bipartite, summarize_hetero
from .models import HeteroLinkPredictor, LinkPredictorSAGE
from .splits import random_split_edges
from .training import EarlyStopper, evaluate_hetero, evaluate_homo, save_ckpt, seed_everything, train_one_epoch_hetero, train_one_epoch_homo


def _pretty(obj) -> str:
    return json.dumps(obj, indent=2)


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)

    device = (
        torch.device("cuda")
        if cfg.device == "auto" and torch.cuda.is_available()
        else torch.device(cfg.device)
    )
    print(f"Using device: {device}")

    raw = load_biokg_pickle()

    if cfg.use_hetero:
        data, num_diseases, num_proteins = build_hetero_graph(raw, include_ppi=cfg.add_ppi)
        print(_pretty(summarize_hetero(data)))
        dp_key = ("disease", "disease_protein", "protein")
        edge_index = data[dp_key].edge_index
    else:
        graph, num_diseases, num_proteins = build_bipartite_graph(raw)
        if cfg.add_ppi:
            graph = add_ppi_edges(graph, int(num_diseases), raw)
        print(_pretty(summarize_bipartite(graph, int(num_diseases), int(num_proteins))))
        edge_index = graph.edge_index

    train_e, val_e, test_e = random_split_edges(edge_index, cfg.train_ratio, cfg.val_ratio, cfg.seed)

    if cfg.use_hetero:
        data = data.to(device)
        train_e, val_e, test_e = train_e.to(device), val_e.to(device), test_e.to(device)
        model = HeteroLinkPredictor(
            metadata=data.metadata(),
            num_diseases=int(num_diseases),
            num_proteins=int(num_proteins),
            hidden=cfg.hidden,
            out=cfg.out,
            model_kind=cfg.model_kind,
            heads=cfg.heads,
            num_bases=cfg.num_bases,
        ).to(device)
    else:
        graph = graph.to(device)
        train_e, val_e, test_e = train_e.to(device), val_e.to(device), test_e.to(device)
        model = LinkPredictorSAGE(num_nodes=int(num_diseases + num_proteins), hidden=cfg.hidden, out=cfg.out).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(patience=cfg.patience, mode="max")
    best_val = float("-inf")
    ckpt_path = Path(cfg.ckpt_path) if cfg.ckpt_path else Path(f"hetero_{cfg.model_kind}_best_model.pth")

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
                save_ckpt(model, ckpt_path)

            print(
                f"Epoch {epoch:03d} | loss {loss:.4f} | val f1 {val_metrics['f1']:.4f} | "
                f"precision {val_metrics['precision']:.4f} | recall {val_metrics['recall']:.4f}"
            )

            if stopper.step(val_metrics["f1"]):
                print("Early stopping.")
                break

    if cfg.use_hetero:
        test_metrics = evaluate_hetero(model, data, test_e, device, int(num_diseases), int(num_proteins))
    else:
        test_metrics = evaluate_homo(model, graph.edge_index, test_e, device)

    print("Test metrics:")
    print(_pretty(test_metrics))


if __name__ == "__main__":
    main()
