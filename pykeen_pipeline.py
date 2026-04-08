from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

from pykeen.pipeline import pipeline
from pykeen.datasets import Dataset, EagerDataset
from pykeen.models import Model

from dataset import PrimeKGDataset


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw_text)
    elif suffix == ".json":
        data = json.loads(raw_text)
    else:
        raise ValueError("Unsupported config format. Use .yaml, .yml, or .json")

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/object")
    return data


def get_dataset(data_path: Path, dataset_cfg: dict[str, Any]) -> tuple[EagerDataset, str]:
    dataset_hash = hashlib.sha256(json.dumps(dataset_cfg, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    dataset_label = dataset_cfg.get("label", f"dataset_{dataset_hash}")

    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
    
    dataset_path = data_path / f"{dataset_label}.pkl"
    if dataset_path.exists():
        print(f"Loading dataset from {dataset_path}")
        with dataset_path.open("rb") as f:
            dataset = pickle.load(f)
    else:
        print(f"Building dataset from config")
        dataset = PrimeKGDataset(dataset_cfg)
        dataset.build_splits()
        print(f"Saving dataset to {dataset_path}")
        with dataset_path.open("wb") as f:
            pickle.dump(dataset, f)

    return dataset.get_dataset(), dataset_label


def run_pipeline(config: dict[str, Any]):
    data_path = Path(config.get("data_path", "dataset_saves"))
    dataset_cfg = config.get("dataset_config", {})
    dataset, dataset_label = get_dataset(data_path, dataset_cfg)

    model = config.get("model", "TransE")
    model_kwargs = config.get("model_kwargs", {})

    save_cfg = config.get("save", {})
    base_dir = Path(save_cfg.get("directory", "results"))
    run_name = f"{dataset_label}_{model}_{save_cfg.get("run_name", "")}"

    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_kwargs = config.get("training_kwargs", {})
    training_kwargs["checkpoint_directory"] = output_dir

    # HPO pipline param optim? ablation study
    print("Running pipeline with config:")
    # print(json.dumps(config, indent=4))
    result = pipeline(
        dataset=dataset,
        model=model,
        model_kwargs=model_kwargs,
        # 3. Loss
        loss = config.get("loss", None),
        loss_kwargs = config.get("loss_kwargs", None),
        # 4. Regularizer
        regularizer =  config.get("regularizer", None),
        regularizer_kwargs = config.get("regularizer_kwargs", None),
        # 5. Optimizer
        optimizer = config.get("optimizer", None),
        optimizer_kwargs = config.get("optimizer_kwargs", None),
        clear_optimizer = config.get("clear_optimizer", True),
        # 5.1 Learning Rate Scheduler
        lr_scheduler = config.get("lr_scheduler", None),
        lr_scheduler_kwargs = config.get("lr_scheduler_kwargs", None),
        # 6. Training Loop
        training_loop = config.get("training_loop", None),
        training_loop_kwargs = config.get("training_loop_kwargs", None),
        negative_sampler = config.get("negative_sampler", None),
        negative_sampler_kwargs = config.get("negative_sampler_kwargs", None),
        # 7. Training
        epochs = config.get("epochs", 10),
        training_kwargs = training_kwargs,
        stopper = config.get("stopper", None),
        stopper_kwargs = config.get("stopper_kwargs", None),
        # 8. Evaluation
        evaluator = config.get("evaluator", None),
        evaluator_kwargs = config.get("evaluator_kwargs", None),
        evaluation_kwargs = config.get("evaluation_kwargs", None),
        # 9. Tracking
        result_tracker = config.get("result_tracker", None),
        result_tracker_kwargs = config.get("result_tracker_kwargs", None),
        # Misc - a többi jó alapbeállításon
        random_seed = config.get("random_seed", 42),
    )
    print("Pipeline finished. Saving results...")
    result.save_to_directory(output_dir)
    with (output_dir / "result.pkl").open("wb") as f:
        pickle.dump(result, f)

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configurable PyKEEN pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline_config.yaml"),
        help="Path to YAML/JSON pipeline config",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = run_pipeline(config)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
