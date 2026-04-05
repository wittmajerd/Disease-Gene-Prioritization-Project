from __future__ import annotations

import hashlib
import json
import yaml
from pathlib import Path
import pandas as pd
from typing import Any

from pykeen.triples import TriplesFactory


class PrimeKGDataset:
    def __init__(self, config):
        self.kg_path = config.kg_path

        if config.key == "index":
            self.head = "x_index"
            self.tail = "y_index"
        if config.key == "name":
            self.head = "x_name"
            self.tail = "y_name"

        self.relation = "relation"

        self.relation_types: list[str] = config.relation_types
        self.node_types: list[str] = config.node_types

        self.drop_duplicates: bool = True
        self.remove_self_loops: bool = True

        self.splits: list[float] = config.splits
        self.random_seed: int = config.random_seed

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
        # train_df, valid_df, test_df = self._split_triples(triples_df)

        if triples_df.empty:
            raise ValueError("No triples left after filtering. Please adjust your filters.")

        labeled_triples = triples_df[[self.head, self.relation, self.tail]].to_numpy(dtype=str)
        self.all_triples = TriplesFactory.from_labeled_triples(triples=labeled_triples)
    
        self.train_tf, self.validation_tf, self.testing_tf = self.all_triples.split(
            ratios=self.splits,
            random_state=self.random_seed,
        )

        # check leakage - unleak ha kell

        stats = self._build_stats(
            before_count=before_count,
            triples_df=triples_df,
        )

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
    ) -> dict[str, Any]:

        nodes = triples_df[self.head].unique()

        # beépített pykeen analysis fgv-ek pl get_relation_counts() stb - checkolhatjuk hogy jó e a split, filter stb

        # na ezen nagyon sokat kell javítani
        return {
            "triples_before_filtering": before_count,
            "triples_after_filtering": int(len(triples_df)),
            "records_dropped": before_count - int(len(triples_df)),
            "unique_relation_types": int(triples_df[self.relation].nunique()),
            "total_nodes": int(len(nodes)),
            "split": {
                "train": self.splits[0],
                "validation": self.splits[1],
                "test": self.splits[2],
                "random_seed": self.random_seed,
            },
        }



#     def _export_snapshot(self, artifacts: DatasetArtifacts) -> Path:
#         config_json = json.dumps(asdict(self.config), sort_keys=True)
#         source_file = Path(self.config.source.primekg_dir) / self.config.source.kg_file
#         payload = (
#             f"{config_json}|{source_file.resolve()}|"
#             f"{source_file.stat().st_mtime_ns}|{source_file.stat().st_size}"
#         )
#         config_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

#         output_root = Path(self.config.export.output_dir)
#         snapshot_dir = output_root / f"primekg_{config_hash}"
#         snapshot_dir.mkdir(parents=True, exist_ok=True)

#         artifacts.training.to_path_binary(snapshot_dir / "train")
#         artifacts.validation.to_path_binary(snapshot_dir / "validation")
#         artifacts.testing.to_path_binary(snapshot_dir / "test")
#         artifacts.all_triples.to_path_binary(snapshot_dir / "all")

#         (snapshot_dir / "feature_registry.json").write_text(
#             json.dumps(asdict(artifacts.feature_registry), ensure_ascii=True, indent=2),
#             encoding="utf-8",
#         )
#         (snapshot_dir / "stats.json").write_text(
#             json.dumps(artifacts.stats, ensure_ascii=True, indent=2),
#             encoding="utf-8",
#         )
#         (snapshot_dir / "dataset_config.json").write_text(
#             json.dumps(asdict(self.config), ensure_ascii=True, indent=2),
#             encoding="utf-8",
#         )

#         return snapshot_dir

#     def _print_summary(self, artifacts: DatasetArtifacts) -> None:
#         stats = artifacts.stats
#         print("Dataset build summary")
#         print(
#             "- triples: "
#             f"{stats['triples_before_filtering']} -> {stats['triples_after_filtering']} "
#             f"(dropped: {stats['records_dropped']})"
#         )
#         print(f"- relation types: {stats['unique_relation_types']}")
#         print(f"- nodes: {stats['total_nodes']}")
#         print(
#             "- split sizes: "
#             f"train={artifacts.training.num_triples}, "
#             f"validation={artifacts.validation.num_triples}, "
#             f"test={artifacts.testing.num_triples}"
#         )
#         if artifacts.snapshot_dir is not None:
#             print(f"- snapshot: {artifacts.snapshot_dir}")


# def load_snapshot(snapshot_dir: Path) -> DatasetArtifacts:
#     """Load a previously exported dataset snapshot."""

#     snapshot_dir = Path(snapshot_dir)
#     training = TriplesFactory.from_path_binary(snapshot_dir / "train")
#     validation = TriplesFactory.from_path_binary(snapshot_dir / "validation")
#     testing = TriplesFactory.from_path_binary(snapshot_dir / "test")
#     all_triples = TriplesFactory.from_path_binary(snapshot_dir / "all")

#     feature_registry = FeatureRegistry(entity_metadata={}, text_features={}, stats={"enabled": False})
#     feature_path = snapshot_dir / "feature_registry.json"
#     if feature_path.exists():
#         feature_raw = json.loads(feature_path.read_text(encoding="utf-8"))
#         feature_registry = FeatureRegistry(
#             entity_metadata=feature_raw.get("entity_metadata", {}),
#             text_features=feature_raw.get("text_features", {}),
#             stats=feature_raw.get("stats", {}),
#         )

#     stats: dict[str, Any] = {}
#     stats_path = snapshot_dir / "stats.json"
#     if stats_path.exists():
#         stats = json.loads(stats_path.read_text(encoding="utf-8"))

#     return DatasetArtifacts(
#         training=training,
#         validation=validation,
#         testing=testing,
#         all_triples=all_triples,
#         feature_registry=feature_registry,
#         stats=stats,
#         snapshot_dir=snapshot_dir,
#     )
