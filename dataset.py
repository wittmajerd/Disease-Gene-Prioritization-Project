from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pykeen.triples import TriplesFactory


class PrimeKGDataset:

    def __init__(self, config):
        self.kg_path = Path(config.source.primekg_dir) / config.source.kg_file

        if config.key == "index":
            self.head = "x_index"
            self.tail = "y_index"
        if config.key == "name":
            self.head = "x_name"
            self.tail = "y_name"

        self.relation = "relation"

        self.relation_types: list[str] = config.filters.relation_types
        self.node_types: list[str] = config.filters.node_types

        self.drop_duplicates: bool = True
        self.remove_self_loops: bool = True

        self.splits: list[float] = [0.9, 0.05, 0.05]
        self.random_seed: int = 42

        self.save: bool = False
        self.output_dir: str = "dataset_snapshots"

        # features later


    def build(self):
        triples_df = pd.read_csv(self.kg_path, low_memory=False)
        triples_df = self._normalize_triples_df(triples_df)

        before_count = len(triples_df)
        triples_df = self._apply_filters(triples_df)
        after_count = len(triples_df)

        # lehet a spliteket az alapján is érdemes lehet számolni hogy épp mennyi él van
        # a triplefactory stratified splitet csinál vagy kell saját?
        # ah es az inverse is ugyan abban a setben kell legyen leakege
        train_df, valid_df, test_df = self._split_triples(triples_df)

        if triples_df.empty:
            raise ValueError("No triples left after filtering. Please adjust your filters.")

        labeled_triples = triples_df[[self.head, self.relation, self.tail]].to_numpy(dtype=str)
        self.all_triples = TriplesFactory.from_labeled_triples(triples=labeled_triples)
    
        self.train_tf, self.validation_tf, self.testing_tf = self.all_triples.split(
            ratios=self.splits,
            random_state=self.config.splits.random_seed,
        )

        # check leakage - unleak

        # beépített pykeen analysis fgv-ek pl get_relation_counts() stb
        stats = self._build_stats(
            before_count=before_count,
            triples_df=triples_df,
        )
        stats["records_dropped"] = before_count - after_count

        self.stats = stats

        if self.save:
            self._export_snapshot()

        return self.train_tf, self.validation_tf, self.testing_tf


    def _apply_filters(self, triples_df: pd.DataFrame) -> pd.DataFrame:
        triples_df = triples_df[triples_df["x_type"].isin(self.node_types) & triples_df["y_type"].isin(self.node_types)]

        triples_df = triples_df[triples_df[self.relation].isin(self.relation_types)]

        if self.remove_self_loops:
            triples_df = triples_df[triples_df[self.head] != triples_df[self.tail]]

        if self.drop_duplicates:
            triples_df = triples_df.drop_duplicates(subset=[self.head, self.relation, self.tail])

        return triples_df.reset_index(drop=True)


    def _build_stats(
        self,
        *,
        before_count: int,
        triples_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
    ) -> dict[str, Any]:
        unique_entities = pd.unique(pd.concat([triples_df[self.head], triples_df[self.tail]]))
        return {
            "triples_before_filtering": before_count,
            "triples_after_filtering": int(len(triples_df)),
            "unique_relations": int(triples_df[self.relation].nunique()),
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
        config_json = json.dumps(asdict(self.config), sort_keys=True)
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
            json.dumps(asdict(self.config), ensure_ascii=True, indent=2),
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
