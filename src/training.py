from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss

from .metrics import classification_metrics
from .splits import negative_sampling_homo, negative_sampling_hetero


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


def evaluate_homo(model, edge_index: Tensor, pos_edges: Tensor, device) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pos_out = model(edge_index, pos_edges)
        neg_edges = negative_sampling_homo(model.embedding.num_embeddings, pos_edges.size(1), device)
        neg_out = model(edge_index, neg_edges)
        scores = torch.cat([pos_out, neg_out]).cpu()
        labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).cpu()
    return classification_metrics(scores, labels)


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


def evaluate_hetero(model, data, pos_edges: Tensor, device, num_diseases: int, num_proteins: int) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pos_out = model(data, pos_edges)
        neg_edges = negative_sampling_hetero(num_diseases, num_proteins, pos_edges.size(1), device)
        neg_out = model(data, neg_edges)
        scores = torch.cat([pos_out, neg_out]).cpu()
        labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).cpu()
    return classification_metrics(scores, labels)
