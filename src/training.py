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

class Trainer:
    def __init__(model, config, device, logger)
    
    def train_epoch(data, train_edges) -> float          # 1 epoch: pos + neg loss
    def evaluate(data, edges) -> dict[str, float]        # val/test eval
    def fit(data, splits) -> dict                        # teljes loop: train+eval+early stop+ckpt+log
    def test(data, test_edges) -> dict                   # végső teszt


