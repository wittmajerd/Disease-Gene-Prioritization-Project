from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Any

from pykeen.triples import TriplesFactory
from pykeen.datasets import EagerDataset


class PrimeKGDataset:
    def __init__(self, config):
        self.kg_path = config.get("kg_path", Path("data/kg.csv"))   # kg_filtered.csv default?

        if config.get("key") == "index":
            self.head = "x_index"
            self.tail = "y_index"
        if config.get("key") == "name":
            self.head = "x_name"
            self.tail = "y_name"

        self.relation = "relation"

        self.relation_types: list[str] | None = config.get("relation_types", None)
        self.node_types: list[str] | None = config.get("node_types", None)

        self.drop_duplicates: bool = True
        self.remove_self_loops: bool = True

        self.inverse_relations: bool = config.get("inverse_relations", True)

        self.val_count: int = config.get("val_count", 5000)
        self.test_count: int = config.get("test_count", 10000)
        self.random_seed: int = config.get("random_seed", 42)

        self.save: bool = False
        self.output_dir: str = "dataset_snapshots"

        # features later


    def build_splits(self):
        triples_df = pd.read_csv(self.kg_path, low_memory=False)

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
        self.all_triples = TriplesFactory.from_labeled_triples(triples=labeled_triples, create_inverse_triples=self.inverse_relations) # only sets train inverse
    
        train_ratio = (after_count - self.val_count - self.test_count) / after_count
        valid_ratio = self.val_count / after_count
        test_ratio = self.test_count / after_count
        self.splits = [train_ratio, valid_ratio, test_ratio]

        self.training, self.validation, self.testing = self.all_triples.split(
            ratios=self.splits,
            random_state=self.random_seed,
        )

        # akarunk validation/test.create_inverse_triples = True-t is?
        # self.training.create_inverse_triples = self.inverse_relations
        # self.validation.create_inverse_triples = self.inverse_relations
        # self.testing.create_inverse_triples = self.inverse_relations

        # self.stats = self._build_stats(
        #     before_count=before_count,
        #     triples_df=triples_df,
        # )

        print(f"Dataset built with {after_count} triples")


    def get_dataset(self):
        print(f"Train: {self.training.num_triples}, Validation: {self.validation.num_triples}, Test: {self.testing.num_triples}")
        print(f"Inverse relations - Train: {self.training.create_inverse_triples}, Validation: {self.validation.create_inverse_triples}, Test: {self.testing.create_inverse_triples}")
        return EagerDataset(training=self.training, validation=self.validation, testing=self.testing)

    def _apply_filters(self, triples_df: pd.DataFrame) -> pd.DataFrame:
        if self.node_types:
            triples_df = triples_df[triples_df["x_type"].isin(self.node_types) & triples_df["y_type"].isin(self.node_types)]

        if self.relation_types:
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
        # train_ds = Dataset().from_tf(self.training)
        # val_ds = Dataset().from_tf(self.validation)
        # test_ds = Dataset().from_tf(self.testing)

        # # get_entity_count_df(train_ds)
        # train_rel = get_relation_count_df(train_ds)
        # val_rel = get_relation_count_df(val_ds)
        # test_rel = get_relation_count_df(test_ds)

        # joined = val_rel.merge(test_rel, on=["relation_id", "relation_label"], how="outer", suffixes=("_val", "_test"))
        # joined = train_rel.merge(joined, on=["relation_id", "relation_label"], how="outer", suffixes=("_train", ""))
        # joined = joined.fillna(0)

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
