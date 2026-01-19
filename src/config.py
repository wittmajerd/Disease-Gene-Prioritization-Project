from __future__ import annotations

import argparse
import json
import yaml
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class PipelineConfig:
    use_hetero: bool = True
    add_ppi: bool = True
    model_kind: str = "sage"
    layer_count: int = 3
    layer_size: int = 256
    output_size: int = 128
    layer_sizes: list[int] | None = None
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 200
    eval_every: int = 10
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
    patience: int = 10
    ckpt_path: str | None = None
    heads: int = 4
    num_bases: int = 30
    run_name: str | None = None
    use_node2vec: bool = False
    node2vec_dim: int = 128
    node2vec_epochs: int = 20
    node2vec_lr: float = 0.01


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "run_config.yaml"


def _coerce_config(raw: dict) -> PipelineConfig:
    allowed = {f.name for f in fields(PipelineConfig)}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return PipelineConfig(**filtered)


def load_config(path: Path) -> PipelineConfig:
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return _coerce_config(data)

    data = json.loads(path.read_text(encoding="utf-8"))
    return _coerce_config(data)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Modular BioKG pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML/JSON config file")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    if config_path.exists():
        cfg = load_config(config_path)
    else:
        cfg = PipelineConfig()

    return cfg
