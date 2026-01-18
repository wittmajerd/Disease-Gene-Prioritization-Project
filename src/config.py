from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    use_hetero: bool = True
    add_ppi: bool = True
    model_kind: str = "sage"
    layer_count: int = 3
    layer_size: int = 256
    output_size: int = 128
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
    device: str = "auto"


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Modular BioKG pipeline")
    parser.add_argument("--use-hetero", action="store_true", help="Use hetero graph pipeline")
    parser.add_argument("--use-bipartite", action="store_true", help="Use bipartite graph pipeline")
    parser.add_argument("--add-ppi", dest="add_ppi", action="store_true", default=True, help="Include PPI edges")
    parser.add_argument("--no-ppi", dest="add_ppi", action="store_false", help="Disable PPI edges")
    parser.add_argument("--model-kind", type=str, default="sage")
    parser.add_argument("--layer-count", type=int, default=3)
    parser.add_argument("--layer-size", type=int, default=256)
    parser.add_argument("--output-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-bases", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()
    use_hetero = args.use_hetero or not args.use_bipartite
    return PipelineConfig(
        use_hetero=use_hetero,
        add_ppi=args.add_ppi,
        model_kind=args.model_kind,
        layer_count=args.layer_count,
        layer_size=args.layer_size,
        output_size=args.output_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        eval_every=args.eval_every,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        patience=args.patience,
        ckpt_path=args.ckpt_path,
        heads=args.heads,
        num_bases=args.num_bases,
        device=args.device,
    )
