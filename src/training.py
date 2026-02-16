from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss

from src.evaluation import evaluate_hetero, evaluate_homo
from src.preprocessing import EdgeSplits, GraphArtifacts, negative_sampling_hetero, negative_sampling_homo


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopper:
    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False
        improve = metric > self.best if self.mode == "max" else metric < self.best
        if improve:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def save_ckpt(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def train_one_epoch_homo(model, optimizer, edge_index: Tensor, pos_edges: Tensor, device) -> float:
    model.train()
    optimizer.zero_grad()
    pos_out = model(edge_index, pos_edges)
    criterion = BCEWithLogitsLoss()
    pos_loss = criterion(pos_out, torch.ones(pos_out.size(0), device=device))
    neg_edges = negative_sampling_homo(model.embedding.num_embeddings, pos_edges.size(1), device)
    neg_out = model(edge_index, neg_edges)
    neg_loss = criterion(neg_out, torch.zeros(neg_out.size(0), device=device))
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_one_epoch_hetero(
    model, optimizer, data, pos_edges: Tensor, device, num_diseases: int, num_proteins: int
) -> float:
    model.train()
    optimizer.zero_grad()
    pos_out = model(data, pos_edges)
    criterion = BCEWithLogitsLoss()
    pos_loss = criterion(pos_out, torch.ones(pos_out.size(0), device=device))
    neg_edges = negative_sampling_hetero(num_diseases, num_proteins, pos_edges.size(1), device)
    neg_out = model(data, neg_edges)
    neg_loss = criterion(neg_out, torch.zeros(neg_out.size(0), device=device))
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_with_validation(
    cfg,
    model,
    artifacts: GraphArtifacts,
    splits: EdgeSplits,
    device,
    logger,
    ckpt_path: Path,
) -> Dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(patience=cfg.patience, mode="max")
    best_val = float("-inf")
    best_epoch = 0
    best_val_metrics = None

    for epoch in range(1, cfg.epochs + 1):
        if cfg.use_hetero:
            loss = train_one_epoch_hetero(
                model,
                optimizer,
                artifacts.graph_data,
                splits.train_edges,
                device,
                artifacts.num_diseases,
                artifacts.num_proteins,
            )
        else:
            loss = train_one_epoch_homo(
                model,
                optimizer,
                artifacts.graph_data.edge_index,
                splits.train_edges,
                device,
            )

        if epoch % cfg.eval_every != 0:
            continue

        if cfg.use_hetero:
            val_metrics = evaluate_hetero(
                model,
                artifacts.graph_data,
                splits.val_edges,
                device,
                artifacts.num_diseases,
                artifacts.num_proteins,
            )
        else:
            val_metrics = evaluate_homo(
                model,
                artifacts.graph_data.edge_index,
                splits.val_edges,
                device,
            )

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

    return {
        "best_val_f1": best_val,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
    }
