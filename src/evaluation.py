from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from src.metrics import classification_metrics
from src.preprocessing import negative_sampling_hetero, negative_sampling_homo


def evaluate_homo(model, edge_index: Tensor, pos_edges: Tensor, device) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pos_out = model(edge_index, pos_edges)
        neg_edges = negative_sampling_homo(model.embedding.num_embeddings, pos_edges.size(1), device)
        neg_out = model(edge_index, neg_edges)
        scores = torch.cat([pos_out, neg_out]).cpu()
        labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).cpu()
    return classification_metrics(scores, labels)


def evaluate_hetero(model, data, pos_edges: Tensor, device, num_diseases: int, num_proteins: int) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pos_out = model(data, pos_edges)
        neg_edges = negative_sampling_hetero(num_diseases, num_proteins, pos_edges.size(1), device)
        neg_out = model(data, neg_edges)
        scores = torch.cat([pos_out, neg_out]).cpu()
        labels = torch.cat([torch.ones(pos_out.size(0)), torch.zeros(neg_out.size(0))]).cpu()
    return classification_metrics(scores, labels)
