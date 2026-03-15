from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pykeen.datasets import PathDataset
from pykeen.triples import TriplesFactory

DEFAULT_DATASET_CONFIG_PATH = Path("dataset_config.yaml")


@dataclass(slots=True)
class DatasetSourceConfig:
    primekg_dir: str = "primekg"
    kg_file: str = "kg.csv"
    nodes_file: str = "nodes.csv"
    disease_features_file: str = "disease_features.csv"
    drug_features_file: str = "drug_features.csv"


@dataclass(slots=True)
class DatasetFilterConfig:
    relation_whitelist: list[str] = field(default_factory=list)
    node_type_filter_enabled: bool = False
    allowed_node_types: list[str] = field(default_factory=list)
    drop_duplicates: bool = True
    remove_self_loops: bool = False


@dataclass(slots=True)
class DatasetSplitConfig:
    train: float = 0.9
    validation: float = 0.05
    test: float = 0.05
    random_seed: int = 42


@dataclass(slots=True)
class DatasetFeatureConfig:
    enabled: bool = True
    include_node_metadata: bool = True
    include_disease_descriptions: bool = True
    include_drug_descriptions: bool = True


@dataclass(slots=True)
class DatasetExportConfig:
    enabled: bool = False
    output_dir: str = "dataset_snapshots"


@dataclass(slots=True)
class DatasetRuntimeConfig:
    max_rows: int | None = None
    verbose: bool = True


@dataclass(slots=True)
class PrimeKGDatasetConfig:
    source: DatasetSourceConfig = field(default_factory=DatasetSourceConfig)
    filters: DatasetFilterConfig = field(default_factory=DatasetFilterConfig)
    splits: DatasetSplitConfig = field(default_factory=DatasetSplitConfig)
    features: DatasetFeatureConfig = field(default_factory=DatasetFeatureConfig)
    export: DatasetExportConfig = field(default_factory=DatasetExportConfig)
    runtime: DatasetRuntimeConfig = field(default_factory=DatasetRuntimeConfig)


@dataclass(slots=True)
class FeatureRegistry:
    entity_metadata: dict[str, dict[str, Any]]
    text_features: dict[str, dict[str, str]]
    stats: dict[str, Any]


@dataclass(slots=True)
class DatasetArtifacts:
    training: TriplesFactory
    validation: TriplesFactory
    testing: TriplesFactory
    all_triples: TriplesFactory
    feature_registry: FeatureRegistry
    stats: dict[str, Any]
    snapshot_dir: Path | None = None


class PrimeKGDatasetBuilder:
    """Build a configurable PrimeKG dataset in memory with optional snapshot export."""

    HEAD_CANDIDATES = ("head", "x_index", "x_id", "source", "source_id")
    RELATION_CANDIDATES = (
        "relation",
        "display_relation",
        "relation_type",
        "edge_type",
        "predicate",
    )
    TAIL_CANDIDATES = ("tail", "y_index", "y_id", "target", "target_id")

    def __init__(self, config: PrimeKGDatasetConfig | None = None):
        self.config = config or load_dataset_config(DEFAULT_DATASET_CONFIG_PATH)

    @classmethod
    def from_config_file(
        cls,
        config_path: Path = DEFAULT_DATASET_CONFIG_PATH,
    ) -> PrimeKGDatasetBuilder:
        return cls(load_dataset_config(config_path))

    def build(self) -> DatasetArtifacts:
        self._validate_splits()
        nodes_df = self._load_nodes()
        triples_df = self._load_triples_raw()
        triples_df = self._normalize_triples_df(triples_df)

        before_count = len(triples_df)
        triples_df = self._apply_filters(triples_df, nodes_df)
        after_count = len(triples_df)

        # lehet a spliteket az alapján is érdemes lehet számolni hogy épp mennyi él van
        all_triples = self._build_triples_factory(triples_df)
        train_tf, valid_tf, test_tf = all_triples.split(
            ratios=[
                self.config.splits.train,
                self.config.splits.validation,
                self.config.splits.test,
            ],
            random_state=self.config.splits.random_seed,
        )

        feature_registry = self._build_feature_registry(nodes_df, all_triples)
        stats = self._build_stats(
            before_count=before_count,
            triples_df=triples_df,
            nodes_df=nodes_df,
        )
        stats["records_dropped"] = before_count - after_count

        artifacts = DatasetArtifacts(
            training=train_tf,
            validation=valid_tf,
            testing=test_tf,
            all_triples=all_triples,
            feature_registry=feature_registry,
            stats=stats,
        )

        if self.config.export.enabled:
            artifacts.snapshot_dir = self._export_snapshot(artifacts)

        if self.config.runtime.verbose:
            self._print_summary(artifacts)

        return artifacts

    def _validate_splits(self) -> None:
        total = self.config.splits.train + self.config.splits.validation + self.config.splits.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.6f} "
                f"({self.config.splits.train}, {self.config.splits.validation}, {self.config.splits.test})."
            )

    def _load_nodes(self) -> pd.DataFrame:
        source = self.config.source
        nodes_path = Path(source.primekg_dir) / source.nodes_file
        if not nodes_path.exists():
            raise FileNotFoundError(f"Nodes file not found: {nodes_path}")

        nodes_df = pd.read_csv(nodes_path)
        required = {"node_index", "node_type"}
        missing = required.difference(nodes_df.columns)
        if missing:
            raise ValueError(f"Missing required node columns in {nodes_path}: {sorted(missing)}")

        nodes_df["node_index"] = nodes_df["node_index"].astype(str)
        return nodes_df

    def _load_triples_raw(self) -> pd.DataFrame:
        source = self.config.source
        triples_path = Path(source.primekg_dir) / source.kg_file
        if not triples_path.exists():
            raise FileNotFoundError(f"Triples file not found: {triples_path}")

        read_kwargs: dict[str, Any] = {}
        if self.config.runtime.max_rows is not None:
            read_kwargs["nrows"] = self.config.runtime.max_rows

        return pd.read_csv(triples_path, **read_kwargs)

    def _pick_column(self, df: pd.DataFrame, candidates: tuple[str, ...], kind: str) -> str:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        raise ValueError(f"Could not find {kind} column among {candidates}. Found: {list(df.columns)}")

    def _normalize_triples_df(self, df: pd.DataFrame) -> pd.DataFrame:
        head_col = self._pick_column(df, self.HEAD_CANDIDATES, "head")
        rel_col = self._pick_column(df, self.RELATION_CANDIDATES, "relation")
        tail_col = self._pick_column(df, self.TAIL_CANDIDATES, "tail")

        triples_df = df[[head_col, rel_col, tail_col]].rename(
            columns={head_col: "head", rel_col: "relation", tail_col: "tail"}
        )
        triples_df = triples_df.dropna(subset=["head", "relation", "tail"]).copy()
        triples_df["head"] = triples_df["head"].astype(str)
        triples_df["relation"] = triples_df["relation"].astype(str)
        triples_df["tail"] = triples_df["tail"].astype(str)
        return triples_df

    def _apply_filters(self, triples_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
        filtered = triples_df

        whitelist = set(self.config.filters.relation_whitelist)
        if whitelist:
            filtered = filtered[filtered["relation"].isin(whitelist)]

        if self.config.filters.node_type_filter_enabled:
            filtered = self._filter_by_node_types(filtered, nodes_df)

        if self.config.filters.remove_self_loops:
            filtered = filtered[filtered["head"] != filtered["tail"]]

        if self.config.filters.drop_duplicates:
            filtered = filtered.drop_duplicates(subset=["head", "relation", "tail"])

        return filtered.reset_index(drop=True)

    def _filter_by_node_types(self, triples_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
        allowed = set(self.config.filters.allowed_node_types)
        if not allowed:
            return triples_df

        node_type_map = (
            nodes_df[["node_index", "node_type"]]
            .drop_duplicates(subset=["node_index"])
            .set_index("node_index")["node_type"]
            .to_dict()
        )

        head_ok = triples_df["head"].map(node_type_map).isin(allowed)
        tail_ok = triples_df["tail"].map(node_type_map).isin(allowed)
        return triples_df[head_ok & tail_ok]

    def _build_triples_factory(self, triples_df: pd.DataFrame) -> TriplesFactory:
        if triples_df.empty:
            raise ValueError("No triples left after filtering. Relax dataset filters.")

        labeled_triples = triples_df[["head", "relation", "tail"]].to_numpy(dtype=str)
        return TriplesFactory.from_labeled_triples(triples=labeled_triples)

    def _build_feature_registry(self, nodes_df: pd.DataFrame, all_triples: TriplesFactory) -> FeatureRegistry:
        if not self.config.features.enabled:
            return FeatureRegistry(entity_metadata={}, text_features={}, stats={"enabled": False})

        entity_labels = set(all_triples.entity_to_id.keys())
        metadata_map: dict[str, dict[str, Any]] = {}
        text_map: dict[str, dict[str, str]] = {}

        if self.config.features.include_node_metadata:
            for row in nodes_df.itertuples(index=False):
                key = str(getattr(row, "node_index"))
                if key not in entity_labels:
                    continue

                metadata = {
                    "node_type": getattr(row, "node_type", None),
                    "node_name": getattr(row, "node_name", None),
                    "node_source": getattr(row, "node_source", None),
                    "node_id": str(getattr(row, "node_id", "")) if hasattr(row, "node_id") else None,
                }
                metadata_map[key] = metadata

        if self.config.features.include_disease_descriptions:
            disease_text = self._read_optional_text_feature_file(
                filename=self.config.source.disease_features_file,
                field_name="disease_description",
                allowed_entities=entity_labels,
            )
            for entity, value in disease_text.items():
                text_map.setdefault(entity, {}).update(value)

        if self.config.features.include_drug_descriptions:
            drug_text = self._read_optional_text_feature_file(
                filename=self.config.source.drug_features_file,
                field_name="drug_description",
                allowed_entities=entity_labels,
            )
            for entity, value in drug_text.items():
                text_map.setdefault(entity, {}).update(value)

        stats = {
            "enabled": True,
            "entity_count": len(entity_labels),
            "metadata_coverage": len(metadata_map),
            "text_coverage": len(text_map),
        }
        return FeatureRegistry(entity_metadata=metadata_map, text_features=text_map, stats=stats)

    def _read_optional_text_feature_file(
        self,
        *,
        filename: str,
        field_name: str,
        allowed_entities: set[str],
    ) -> dict[str, dict[str, str]]:
        features_path = Path(self.config.source.primekg_dir) / filename
        if not features_path.exists():
            return {}

        feature_df = pd.read_csv(features_path)
        if "node_index" not in feature_df.columns:
            return {}

        text_col = None
        for candidate in ("description", "text", "feature", "value"):
            if candidate in feature_df.columns:
                text_col = candidate
                break

        if text_col is None:
            return {}

        output: dict[str, dict[str, str]] = {}
        for row in feature_df[["node_index", text_col]].dropna().itertuples(index=False):
            entity = str(getattr(row, "node_index"))
            if entity not in allowed_entities:
                continue
            output[entity] = {field_name: str(getattr(row, text_col))}
        return output

    def _build_stats(
        self,
        *,
        before_count: int,
        triples_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
    ) -> dict[str, Any]:
        unique_entities = pd.unique(pd.concat([triples_df["head"], triples_df["tail"]]))
        return {
            "triples_before_filtering": before_count,
            "triples_after_filtering": int(len(triples_df)),
            "unique_relations": int(triples_df["relation"].nunique()),
            "unique_entities_in_triples": int(unique_entities.size),
            "total_nodes_loaded": int(len(nodes_df)),
            "split": {
                "train": self.config.splits.train,
                "validation": self.config.splits.validation,
                "test": self.config.splits.test,
                "random_seed": self.config.splits.random_seed,
            },
        }

    def _export_snapshot(self, artifacts: DatasetArtifacts) -> Path:
        config_json = json.dumps(_config_to_dict(self.config), sort_keys=True)
        source_file = Path(self.config.source.primekg_dir) / self.config.source.kg_file
        payload = (
            f"{config_json}|{source_file.resolve()}|"
            f"{source_file.stat().st_mtime_ns}|{source_file.stat().st_size}"
        )
        config_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

        output_root = Path(self.config.export.output_dir)
        snapshot_dir = output_root / f"primekg_{config_hash}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        artifacts.training.to_path_binary(snapshot_dir / "train")
        artifacts.validation.to_path_binary(snapshot_dir / "validation")
        artifacts.testing.to_path_binary(snapshot_dir / "test")
        artifacts.all_triples.to_path_binary(snapshot_dir / "all")

        (snapshot_dir / "feature_registry.json").write_text(
            json.dumps(asdict(artifacts.feature_registry), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        (snapshot_dir / "stats.json").write_text(
            json.dumps(artifacts.stats, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        (snapshot_dir / "dataset_config.json").write_text(
            json.dumps(_config_to_dict(self.config), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        return snapshot_dir

    def _print_summary(self, artifacts: DatasetArtifacts) -> None:
        stats = artifacts.stats
        print("Dataset build summary")
        print(
            "- triples: "
            f"{stats['triples_before_filtering']} -> {stats['triples_after_filtering']} "
            f"(dropped: {stats['records_dropped']})"
        )
        print(f"- relations: {stats['unique_relations']}")
        print(f"- entities: {stats['unique_entities_in_triples']}")
        print(
            "- split sizes: "
            f"train={artifacts.training.num_triples}, "
            f"validation={artifacts.validation.num_triples}, "
            f"test={artifacts.testing.num_triples}"
        )
        if artifacts.snapshot_dir is not None:
            print(f"- snapshot: {artifacts.snapshot_dir}")


def _config_to_dict(config: PrimeKGDatasetConfig) -> dict[str, Any]:
    return asdict(config)


def load_dataset_config(config_path: Path = DEFAULT_DATASET_CONFIG_PATH) -> PrimeKGDatasetConfig:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {config_path}. "
            "Create one from dataset_config.yaml settings."
        )

    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        raw: dict[str, Any] = yaml.safe_load(text) or {}
    elif suffix == ".json":
        raw = json.loads(text)
    else:
        raise ValueError("Unsupported dataset config format. Use .yaml/.yml/.json")

    source = DatasetSourceConfig(**raw.get("source", {}))
    filters = DatasetFilterConfig(**raw.get("filters", {}))
    splits = DatasetSplitConfig(**raw.get("splits", {}))
    features = DatasetFeatureConfig(**raw.get("features", {}))
    export = DatasetExportConfig(**raw.get("export", {}))
    runtime = DatasetRuntimeConfig(**raw.get("runtime", {}))
    return PrimeKGDatasetConfig(
        source=source,
        filters=filters,
        splits=splits,
        features=features,
        export=export,
        runtime=runtime,
    )


def build_primekg_dataset(config_path: Path = DEFAULT_DATASET_CONFIG_PATH) -> DatasetArtifacts:
    """Convenience entrypoint for building dataset artifacts from config file."""

    return PrimeKGDatasetBuilder.from_config_file(config_path).build()


def load_snapshot(snapshot_dir: Path) -> DatasetArtifacts:
    """Load a previously exported dataset snapshot."""

    snapshot_dir = Path(snapshot_dir)
    training = TriplesFactory.from_path_binary(snapshot_dir / "train")
    validation = TriplesFactory.from_path_binary(snapshot_dir / "validation")
    testing = TriplesFactory.from_path_binary(snapshot_dir / "test")
    all_triples = TriplesFactory.from_path_binary(snapshot_dir / "all")

    feature_registry = FeatureRegistry(entity_metadata={}, text_features={}, stats={"enabled": False})
    feature_path = snapshot_dir / "feature_registry.json"
    if feature_path.exists():
        feature_raw = json.loads(feature_path.read_text(encoding="utf-8"))
        feature_registry = FeatureRegistry(
            entity_metadata=feature_raw.get("entity_metadata", {}),
            text_features=feature_raw.get("text_features", {}),
            stats=feature_raw.get("stats", {}),
        )

    stats: dict[str, Any] = {}
    stats_path = snapshot_dir / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

    return DatasetArtifacts(
        training=training,
        validation=validation,
        testing=testing,
        all_triples=all_triples,
        feature_registry=feature_registry,
        stats=stats,
        snapshot_dir=snapshot_dir,
    )


class CustomPathDataset(PathDataset):
    """Compatibility wrapper for existing file-based pre-split usage."""

    pass


__all__ = [
    "CustomPathDataset",
    "DatasetArtifacts",
    "FeatureRegistry",
    "PrimeKGDatasetBuilder",
    "PrimeKGDatasetConfig",
    "build_primekg_dataset",
    "load_dataset_config",
    "load_snapshot",
]
