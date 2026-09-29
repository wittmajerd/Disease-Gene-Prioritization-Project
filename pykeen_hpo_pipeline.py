from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pykeen.hpo import hpo_pipeline
from pykeen.datasets import Dataset
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

from dataset2 import PrimeKGDataset


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


def get_dataset(
    data_path: Path,
    dataset_config: dict[str, Any],
    keep_relations: set[str] | None = None,
    remove_relations: set[str] | None = None,
    keep_entities: set[str] | None = None,
    keep_entity_types: set[str] | None = None,
) -> tuple[Dataset, str]:
    dataset_hash = hashlib.sha256(
        json.dumps(dataset_config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    dataset_label = f"primekg_fixed_split_{dataset_config.get('random_seed')}_{dataset_hash}"
    dataset_label = "primekg_fixed_split_42_f0ac5cf4"

    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
    
    dataset_path = data_path / f"{dataset_label}.pkl"
    if dataset_path.exists():
        print(f"Loading dataset from {dataset_path}")
        with dataset_path.open("rb") as f:
            dataset = pickle.load(f)
    else:
        print("Building fixed PrimeKG split")
        dataset = PrimeKGDataset(dataset_config)
        dataset.build_splits()
        print(f"Saving dataset to {dataset_path}")
        with dataset_path.open("wb") as f:
            pickle.dump(dataset, f)

    return dataset.get_dataset(
        keep_relations=keep_relations,
        remove_relations=remove_relations,
        keep_entities=keep_entities,
        keep_entity_types=keep_entity_types,
    ), dataset_label


def run_pipeline(config: dict[str, Any], dataset_config: dict[str, Any], experiment: dict[str, Any]):
    data_path = Path(config.get("data_path", "dataset_saves"))

    # dataset_config = load_config(Path(config.get("dataset_config_path", "dataset_config.yaml")))
    # dataset, dataset_label = get_dataset(data_path, dataset_config)
    dataset, dataset_label = get_dataset(
        data_path=data_path,
        dataset_config=dataset_config,
        keep_relations=experiment.get("keep_relations"),
        remove_relations=experiment.get("remove_relations"),
        keep_entities=experiment.get("keep_entities"),
        keep_entity_types=experiment.get("keep_entity_types"),
    )

    model = config.get("model", "RotatE")

    save_cfg = config.get("save", {})
    base_dir = Path(save_cfg.get("directory", "results"))
    run_name = f"{dataset_label}_{model}_{save_cfg.get('run_name', '')}"

    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # HPO pipline param optim? ablation study
    print("Running pipeline with config:")
    # print(json.dumps(config, indent=4))
    result = hpo_pipeline(  
        dataset=dataset,
        model=model,
        model_kwargs=config.get("model_kwargs", None),
        model_kwargs_ranges=config.get("model_kwargs_ranges", None),
        # 3. Loss
        loss=config.get("loss", None),
        loss_kwargs=config.get("loss_kwargs", None),
        loss_kwargs_ranges=config.get("loss_kwargs_ranges", None),
        # 4. Regularizer
        regularizer=config.get("regularizer", None),
        regularizer_kwargs=config.get("regularizer_kwargs", None),
        regularizer_kwargs_ranges=config.get("regularizer_kwargs_ranges", None),
        # 5. Optimizer
        optimizer=config.get("optimizer", None),
        optimizer_kwargs=config.get("optimizer_kwargs", None),
        optimizer_kwargs_ranges=config.get("optimizer_kwargs_ranges", None),
        # 5.1 Learning Rate Scheduler
        lr_scheduler=config.get("lr_scheduler", None),
        lr_scheduler_kwargs=config.get("lr_scheduler_kwargs", None),
        lr_scheduler_kwargs_ranges=config.get("lr_scheduler_kwargs_ranges", None),
        # 6. Training Loop
        training_loop=config.get("training_loop", None),
        training_loop_kwargs=config.get("training_loop_kwargs", None),
        negative_sampler=config.get("negative_sampler", None),
        negative_sampler_kwargs=config.get("negative_sampler_kwargs", None),
        negative_sampler_kwargs_ranges=config.get("negative_sampler_kwargs_ranges", None),
        # 7. Training
        n_trials=config.get("n_trials", 30),
        epochs=config.get("epochs", 1000),
        training_kwargs=config.get("training_kwargs", {}),
        training_kwargs_ranges=config.get("training_kwargs_ranges", None),
        stopper=config.get("stopper", None),
        stopper_kwargs=config.get("stopper_kwargs", None),
        # 8. Evaluation
        evaluator=config.get("evaluator", None),
        evaluator_kwargs=config.get("evaluator_kwargs", None),
        evaluation_kwargs=config.get("evaluation_kwargs", None),
        # 9. Tracking
        result_tracker=config.get("result_tracker", None),
        result_tracker_kwargs=config.get("result_tracker_kwargs", None),
        study_name=f"{dataset_label}_{model}_HPO",
        load_if_exists=True,
    )
    print("Pipeline finished. Saving results...")
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
        default=Path("hpo_pipeline_config.yaml"),
        help="Path to YAML/JSON pipeline config",
    )
    return parser.parse_args()

def main() -> None:

    dataset_config = {
            "kg_path": "data/kg_filtered.csv",
            "key": "name",
            "drop_duplicates": True,
            "remove_self_loops": True,
            "inverse_relations": True,
            "split_relation": "disease_protein",
            "val_count": 8000,
            "test_count": 0,
            "random_seed": 42,
        }
    experiment ={
                "label": "disease_protein_bipartite",
                "keep_relations": {"disease_protein"},
            }

    args = parse_args()
    config = load_config(args.config)
    output_dir = run_pipeline(config, dataset_config, experiment)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
