from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import pickle
import yaml
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from pykeen.pipeline import pipeline
from pykeen.datasets import Dataset

from dataset import PrimeKGDataset


class ConfigSafeLoader(yaml.SafeLoader):
    """Safe YAML loader with support for paths emitted by yaml.dump()."""


def _construct_path(loader: ConfigSafeLoader, node: yaml.Node) -> str:
    parts = loader.construct_sequence(node)
    return str(Path(*parts))


ConfigSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.WindowsPath",
    _construct_path,
)


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        data = yaml.load(raw_text, Loader=ConfigSafeLoader)
    elif suffix == ".json":
        data = json.loads(raw_text)
    else:
        raise ValueError("Unsupported config format. Use .yaml, .yml, or .json")

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/object")
    return data


def get_dataset(data_path: Path, dataset_cfg: dict[str, Any]) -> tuple[Dataset, str]:
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


def run_pipeline(config: dict[str, Any], random_seed: int):
    config = deepcopy(config)
    data_path = Path(config.get("data_path", "dataset_saves"))

    dataset_config = load_config(Path(config.get("dataset_config_path", "dataset_config.yaml")))
    dataset, dataset_label = get_dataset(data_path, dataset_config)

    model = config.get("model", "RotatE")
    model_kwargs = config.get("model_kwargs", {})

    run_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:8]

    save_cfg = config.get("save", {})
    base_dir = Path(save_cfg.get("directory", "results"))
    run_name = f"{dataset_label}_{model}_{run_hash}_{random_seed}"

    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config["training_kwargs"]["checkpoint_directory"] = output_dir
    config["stopper_kwargs"]["best_model_path"] = output_dir / "best_model.pth"
    config["result_tracker_kwargs"]["tags"] = [dataset_label, str(random_seed)]

    # save config for reproducibility
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

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
        training_kwargs = config.get("training_kwargs", None),
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
        random_seed = random_seed,
    )
    print("Pipeline finished. Saving results...")
    print(f"Hits@10: {result.get_metric('hits@10'):.4f}")
    print(f"Mean Reciprocal Rank: {result.get_metric('mean_reciprocal_rank'):.4f}")
    result.save_to_directory(output_dir)
    # with (output_dir / "result.pkl").open("wb") as f:
    # tries to pickle wandb module or smthing
    #     pickle.dump(result, f)

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
    try:
        configs = [
            # Path("pipeline_config.yaml"),
            # Path("pipeline_config_base.yaml"),
            # Path("pipeline_config_pseudo.yaml"),
            Path("pipeline_config_bernoulli.yaml"),
            Path("pipeline_config_nssa.yaml"),
        ]

        for config_path in configs:
            print(f"Running pipeline with config: {config_path}")
            config = load_config(config_path)
            seeds = [42]  # List of random seeds for multiple runs 123, 456, 789, 101112
            for seed in seeds:
                print(f"Running pipeline with random seed: {seed}")
                output_dir = run_pipeline(config, random_seed=seed)
                print(f"Results saved to: {output_dir}")

    except Exception as e:
        import traceback
        error_msg = f"An exception occurred during training:\n{traceback.format_exc()}"
        print(error_msg, flush=True)
    finally:
        import time
        print(datetime.now(), flush=True)
        print("Going to sleep in 30 seconds...", flush=True)
        # time.sleep(30)
        # ctypes.windll.PowrProf.SetSuspendState(False, True, False)

if __name__ == "__main__":
    main()
