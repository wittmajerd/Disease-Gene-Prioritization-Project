from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import glob
import hashlib
import json
import pickle
import time
import torch
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

from pykeen.pipeline import pipeline
from pykeen.datasets import Dataset

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


def run_pipeline(
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    experiment: dict[str, Any],
    random_seed: int,
):
    data_path = Path(config.get("data_path", "dataset_saves"))

    dataset, dataset_label = get_dataset(
        data_path=data_path,
        dataset_config=dataset_config,
        keep_relations=experiment.get("keep_relations"),
        remove_relations=experiment.get("remove_relations"),
        keep_entities=experiment.get("keep_entities"),
        keep_entity_types=experiment.get("keep_entity_types"),
    )

    model = config.get("model", "RotatE")
    model_kwargs = config.get("model_kwargs", {})

    run_hash = hashlib.sha256(
        json.dumps(
            {"config": config, "experiment": experiment},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:8]

    save_cfg = config.get("save", {})
    base_dir = Path(save_cfg.get("directory", "results"))
    run_name = f"{dataset_label}_{model}_{run_hash}_{random_seed}"

    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = copy.deepcopy(config)
    config.setdefault("training_kwargs", {})["checkpoint_directory"] = output_dir
    config.setdefault("stopper_kwargs", {})["best_model_path"] = output_dir / "best_model.pth"
    config.setdefault("result_tracker_kwargs", {})["tags"] = [
        dataset_label,
        experiment["label"],
        str(random_seed),
    ]

    # save config for reproducibility
    saved_config = copy.deepcopy(config)
    saved_config["training_kwargs"]["checkpoint_directory"] = str(output_dir)
    saved_config["stopper_kwargs"]["best_model_path"] = str(output_dir / "best_model.pth")
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(saved_config, f, sort_keys=False)

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
        config_path = Path("new_pipeline_config.yaml")
        config = load_config(config_path)

        dataset_config = {
            "kg_path": "primekg/kg_filtered.csv",
            "key": "name",
            "drop_duplicates": True,
            "remove_self_loops": True,
            "inverse_relations": True,
            "split_relation": "disease_protein",
            "val_count": 2000,
            "test_count": 8000,
            "random_seed": 42,
        }

        experiments = [
            # {
            #     "label": "disease_protein",
            #     "keep_relations": {"disease_protein"},
            # },
            {
                "label": "disease_protein_bipartite",
                "keep_entity_types": {"disease", "gene/protein"},
            },
            {
                "label": "disease_protein_drug",
                "keep_entity_types": {"disease", "gene/protein", "drug"},
            },
            {
                "label": "disease_protein_anatomy",
                "keep_entity_types": {"disease", "gene/protein", "anatomy"},
            },
            {
                "label": "disease_protein_bioprocess",
                "keep_entity_types": {"disease", "gene/protein", "biological_process"},
            },
            {
                "label": "disease_protein_cellcomp",
                "keep_entity_types": {"disease", "gene/protein", "cellular_component"},
            },
            {
                "label": "disease_protein_exposure",
                "keep_entity_types": {"disease", "gene/protein", "exposure"},
            },
            {
                "label": "disease_protein_molecular",
                "keep_entity_types": {"disease", "gene/protein", "molecular_function"},
            },
            {
                "label": "disease_protein_pathway",
                "keep_entity_types": {"disease", "gene/protein", "pathway"},
            },
            {
                "label": "disease_protein_phenotype",
                "keep_entity_types": {"disease", "gene/protein", "effect/phenotype"},
            },
        ]

        seeds = [42, 123, 456]
        for experiment in experiments:
            for seed in seeds:
                print(f"Running {experiment['label']} with random seed: {seed}")
                run_config = copy.deepcopy(config)
                output_dir = run_pipeline(
                    run_config,
                    dataset_config,
                    experiment,
                    seed,
                )
                print(f"Results saved to: {output_dir}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        error_msg = f"An exception occurred during training:\n{traceback.format_exc()}"
        print(error_msg, flush=True)
    finally:
        print(datetime.now(), flush=True)
        print("Going to sleep in 30 seconds...", flush=True)
        time.sleep(30)
        ctypes.windll.PowrProf.SetSuspendState(False, True, False)

if __name__ == "__main__":
    main()
