from __future__ import annotations

import argparse
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

from pykeen.pipeline import pipeline

from dataset import CustomPathDataset


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


def resolve_dataset(dataset_cfg: dict[str, Any]) -> tuple[Any, dict[str, Any], str]:
    kind = dataset_cfg.get("kind", "builtin")

    if kind == "builtin":
        name = dataset_cfg.get("name")
        if not name:
            raise ValueError("For builtin datasets, set dataset.name in config")
        return name, dataset_cfg.get("kwargs", {}), str(name)

    if kind == "custom_path":
        kwargs = dataset_cfg.get("kwargs", {})
        if "training_path" not in kwargs:
            raise ValueError(
                "For custom_path dataset, set dataset.kwargs.training_path in config"
            )
        dataset_name = dataset_cfg.get("name", "custom_path_dataset")
        return CustomPathDataset, kwargs, dataset_name

    raise ValueError(f"Unsupported dataset.kind: {kind}")


def resolve_model(model_cfg: Any) -> tuple[Any, dict[str, Any], str]:
    if isinstance(model_cfg, str):
        return model_cfg, {}, model_cfg

    if isinstance(model_cfg, dict):
        name = model_cfg.get("name")
        if not name:
            raise ValueError("If model is an object, model.name is required")
        return name, model_cfg.get("kwargs", {}), str(name)

    raise ValueError("model must be a string or an object with 'name' and optional 'kwargs'")


def run_pipeline(config: dict[str, Any]):
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", "TransE")

    dataset, dataset_kwargs, dataset_label = resolve_dataset(dataset_cfg)
    model, model_kwargs, model_label = resolve_model(model_cfg)

    pipeline_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "dataset_kwargs": dataset_kwargs,
        "model": model,
        "model_kwargs": model_kwargs,
    }

    passthrough_keys = [
        "optimizer",
        "optimizer_kwargs",
        "training_loop",
        "training_loop_kwargs",
        "negative_sampler",
        "negative_sampler_kwargs",
        "loss",
        "loss_kwargs",
        "regularizer",
        "regularizer_kwargs",
        "evaluator",
        "evaluator_kwargs",
        "training_kwargs",
        "random_seed",
        "device",
    ]

    for key in passthrough_keys:
        if key in config:
            pipeline_kwargs[key] = config[key]

    result = pipeline(**pipeline_kwargs)

    save_cfg = config.get("save", {})
    base_dir = Path(save_cfg.get("directory", "results"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = save_cfg.get("run_name", f"{dataset_label}_{model_label}_{timestamp}")

    output_dir = base_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save_to_directory(output_dir)
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
    print(f"Pipeline finished. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()





def pipeline(  # noqa: C901
    *,
    # 1. Dataset
    dataset: None | str | Dataset | type[Dataset] = None,
    dataset_kwargs: Mapping[str, Any] | None = None,
    training: Hint[CoreTriplesFactory] = None,
    testing: Hint[CoreTriplesFactory] = None,
    validation: Hint[CoreTriplesFactory] = None,
    evaluation_entity_whitelist: Collection[str] | None = None,
    evaluation_relation_whitelist: Collection[str] | None = None,
    # 2. Model
    model: None | str | Model | type[Model] = None,
    model_kwargs: Mapping[str, Any] | None = None,
    interaction: None | str | Interaction | type[Interaction] = None,
    interaction_kwargs: Mapping[str, Any] | None = None,
    dimensions: None | int | Mapping[str, int] = None,
    # 3. Loss
    loss: HintType[Loss] = None,
    loss_kwargs: Mapping[str, Any] | None = None,
    # 4. Regularizer
    regularizer: HintType[Regularizer] = None,
    regularizer_kwargs: Mapping[str, Any] | None = None,
    # 5. Optimizer
    optimizer: HintType[Optimizer] = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    clear_optimizer: bool = True,
    # 5.1 Learning Rate Scheduler
    lr_scheduler: HintType[LRScheduler] = None,
    lr_scheduler_kwargs: Mapping[str, Any] | None = None,
    # 6. Training Loop
    training_loop: HintType[TrainingLoop] = None,
    training_loop_kwargs: Mapping[str, Any] | None = None,
    negative_sampler: HintType[NegativeSampler] = None,
    negative_sampler_kwargs: Mapping[str, Any] | None = None,
    # 7. Training (ronaldo style)
    epochs: int | None = None,
    training_kwargs: Mapping[str, Any] | None = None,
    stopper: HintType[Stopper] = None,
    stopper_kwargs: Mapping[str, Any] | None = None,
    # 8. Evaluation
    evaluator: HintType[Evaluator] = None,
    evaluator_kwargs: Mapping[str, Any] | None = None,
    evaluation_kwargs: Mapping[str, Any] | None = None,
    # 9. Tracking
    result_tracker: OneOrManyHintOrType[ResultTracker] = None,
    result_tracker_kwargs: OneOrManyOptionalKwargs = None,
    # Misc
    metadata: dict[str, Any] | None = None,
    device: Hint[torch.device] = None,
    random_seed: int | None = None,
    use_testing_data: bool = True,
    evaluation_fallback: bool = False,
    filter_validation_when_testing: bool = True,
    use_tqdm: bool | None = None,
) -> PipelineResult:
    
    """Train and evaluate a model.

    :param dataset:
        The name of the dataset (a key for the :data:`pykeen.datasets.dataset_resolver`) or the
        :class:`pykeen.datasets.Dataset` instance. Alternatively, the training triples factory (``training``), testing
        triples factory (``testing``), and validation triples factory (``validation``; optional) can be specified.
    :param dataset_kwargs:
        The keyword arguments passed to the dataset upon instantiation
    :param training:
        A triples factory with training instances or path to the training file if a a dataset was not specified
    :param testing:
        A triples factory with training instances or path to the test file if a dataset was not specified
    :param validation:
        A triples factory with validation instances or path to the validation file if a dataset was not specified
    :param evaluation_entity_whitelist:
        Optional restriction of evaluation to triples containing *only* these entities. Useful if the downstream task
        is only interested in certain entities, but the relational patterns with other entities improve the entity
        embedding quality.
    :param evaluation_relation_whitelist:
        Optional restriction of evaluation to triples containing *only* these relations. Useful if the downstream task
        is only interested in certain relation, but the relational patterns with other relations improve the entity
        embedding quality.

    :param model:
        The name of the model, subclass of :class:`pykeen.models.Model`, or an instance of
        :class:`pykeen.models.Model`. Can be given as None if the ``interaction`` keyword is used.
    :param model_kwargs:
        Keyword arguments to pass to the model class on instantiation
    :param interaction: The name of the interaction class, a subclass of :class:`pykeen.nn.modules.Interaction`,
        or an instance of :class:`pykeen.nn.modules.Interaction`. Can not be given when there is also a model.
    :param interaction_kwargs:
        Keyword arguments to pass during instantiation of the interaction class. Only use with ``interaction``.
    :param dimensions:
        Dimensions to assign to the embeddings of the interaction. Only use with ``interaction``.

    :param loss:
        The name of the loss or the loss class.
    :param loss_kwargs:
        Keyword arguments to pass to the loss on instantiation

    :param regularizer:
        The name of the regularizer or the regularizer class.
    :param regularizer_kwargs:
        Keyword arguments to pass to the regularizer on instantiation

    :param optimizer:
        The name of the optimizer or the optimizer class. Defaults to :class:`torch.optim.Adagrad`.
    :param optimizer_kwargs:
        Keyword arguments to pass to the optimizer on instantiation
    :param clear_optimizer:
        Whether to delete the optimizer instance after training. As the optimizer might have additional memory
        consumption due to e.g. moments in Adam, this is the default option. If you want to continue training, you
        should set it to False, as the optimizer's internal parameter will get lost otherwise.

    :param lr_scheduler:
        The name of the lr_scheduler or the lr_scheduler class.
        Defaults to :class:`torch.optim.lr_scheduler.ExponentialLR`.
    :param lr_scheduler_kwargs:
        Keyword arguments to pass to the lr_scheduler on instantiation

    :param training_loop:
        The name of the training loop's training approach (``'slcwa'`` or ``'lcwa'``) or the training loop class.
        Defaults to :class:`pykeen.training.SLCWATrainingLoop`.
    :param training_loop_kwargs:
        Keyword arguments to pass to the training loop on instantiation
    :param negative_sampler:
        The name of the negative sampler (``'basic'`` or ``'bernoulli'``) or the negative sampler class.
        Only allowed when training with sLCWA.
        Defaults to :class:`pykeen.sampling.BasicNegativeSampler`.
    :param negative_sampler_kwargs:
        Keyword arguments to pass to the negative sampler class on instantiation

    :param epochs:
        A shortcut for setting the ``num_epochs`` key in the ``training_kwargs`` dict.
    :param training_kwargs:
        Keyword arguments to pass to the training loop's train function on call
    :param stopper:
        What kind of stopping to use. Default to no stopping, can be set to 'early'.
    :param stopper_kwargs:
        Keyword arguments to pass to the stopper upon instantiation.

    :param evaluator:
        The name of the evaluator or an evaluator class. Defaults to :class:`pykeen.evaluation.RankBasedEvaluator`.
    :param evaluator_kwargs:
        Keyword arguments to pass to the evaluator on instantiation
    :param evaluation_kwargs:
        Keyword arguments to pass to the evaluator's evaluate function on call

    :param result_tracker: Either none (will result in a Python result tracker),
        a single tracker (as either a class, instance, or string for class name), or a list
        of trackers (as either a class, instance, or string for class name
    :param result_tracker_kwargs: Either none (will use all defaults), a single dictionary
        (will be used for all trackers), or a list of dictionaries with the same length
        as the result trackers

    :param metadata:
        A JSON dictionary to store with the experiment
    :param use_testing_data:
        If true, use the testing triples. Otherwise, use the validation triples. Defaults to true - use testing triples.
    :param device: The device or device name to run on. If none is given, the device will be looked up with
        :func:`pykeen.utils.resolve_device`.
    :param random_seed: The random seed to use. If none is specified, one will be assigned before any code
        is run for reproducibility purposes. In the returned :class:`PipelineResult` instance, it can be accessed
        through :data:`PipelineResult.random_seed`.
    :param evaluation_fallback:
        If true, in cases where the evaluation failed using the GPU it will fall back to using a smaller batch size or
        in the last instance evaluate on the CPU, if even the smallest possible batch size is too big for the GPU.
    :param filter_validation_when_testing:
        If true, during the evaluating of the test dataset, validation triples are added to the set of known positive
        triples, which are filtered out when performing filtered evaluation following the approach described by
        [bordes2013]_. This should be explicitly set to false only in the scenario that you are training a single
        model using the pipeline and evaluating with the testing set, but never using the validation set for
        optimization at all. This is a very atypical scenario, so it is left as true by default to promote
        comparability to previous publications.
    :param use_tqdm:
        Globally set the usage of tqdm progress bars. Typically more useful to set to false, since the training
        loop and evaluation have it turned on by default.

    :returns: A pipeline result package.
    """