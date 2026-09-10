from __future__ import annotations

from pathlib import Path
import pandas as pd
from typing import Any

from pykeen.triples import TriplesFactory
from pykeen.datasets import EagerDataset


class PrimeKGDataset:
    def __init__(self, config):
        self.kg_path = config.get("kg_path", Path("primekg/kg_filtered.csv"))

        if config.get("key") == "index":
            self.head = "x_index"
            self.tail = "y_index"
        if config.get("key") == "name":
            self.head = "x_name"
            self.tail = "y_name"

        self.relation = "relation"

        self.drop_duplicates: bool = config.get("drop_duplicates", True)
        self.remove_self_loops: bool = config.get("remove_self_loops", True)
        self.inverse_relations: bool = config.get("inverse_relations", True)

        self.split_relation: str = config.get("split_relation", "disease_protein")
        self.val_count: int = config.get("val_count", 4000)
        self.test_count: int = config.get("test_count", 16000)
        self.random_seed: int = config.get("random_seed", 42)

        # features later


    def build_splits(self):
        triples_df = pd.read_csv(self.kg_path, low_memory=False)

        if self.remove_self_loops:
            triples_df = triples_df[triples_df[self.head] != triples_df[self.tail]]

        if self.drop_duplicates:
            triples_df = triples_df.drop_duplicates(subset=[self.head, self.relation, self.tail])

        if triples_df.empty:
            raise ValueError("No triples left after filtering. Please adjust the structural filters.")

        target_triples = triples_df[triples_df[self.relation] == self.split_relation]
        required_count = self.val_count + self.test_count
        if len(target_triples) < required_count:
            raise ValueError(
                f"Relation {self.split_relation!r} has {len(target_triples)} triples, "
                f"but {required_count} are required for validation and testing."
            )

        shuffled_target = target_triples.sample(
            frac=1,
            random_state=self.random_seed,
        )
        validation_df = shuffled_target.iloc[:self.val_count]
        testing_df = shuffled_target.iloc[self.val_count:required_count]
        training_df = triples_df[triples_df[self.relation] != self.split_relation].copy()
        training_df = pd.concat(
            [training_df, shuffled_target.iloc[required_count:]],
            ignore_index=True,
        )

        labeled_triples = triples_df[[self.head, self.relation, self.tail]].to_numpy(dtype=str)
        self.all_triples = TriplesFactory.from_labeled_triples(
            triples=labeled_triples,
            create_inverse_triples=self.inverse_relations,
        )

        def make_factory(dataframe: pd.DataFrame, *, create_inverse_triples: bool) -> TriplesFactory:
            return TriplesFactory.from_labeled_triples(
                triples=dataframe[[self.head, self.relation, self.tail]].to_numpy(dtype=str),
                entity_to_id=self.all_triples.entity_to_id,
                relation_to_id=self.all_triples.relation_to_id,
                create_inverse_triples=create_inverse_triples,
            )

        self.training = make_factory(
            training_df,
            create_inverse_triples=self.inverse_relations,
        )
        self.validation = make_factory(validation_df, create_inverse_triples=False)
        self.testing = make_factory(testing_df, create_inverse_triples=False)

        print(
            f"Dataset built with {len(triples_df)} triples: "
            f"{len(training_df)} train, {len(validation_df)} validation, "
            f"{len(testing_df)} test ({self.split_relation!r})"
        )



    def get_dataset(self, keep_relations: set = None, remove_relations: set = None, keep_entities: set = None):
        dataset = EagerDataset(
            training=self.training,
            validation=self.validation,
            testing=self.testing,
        )
        dataset = filter_dataset(
            dataset=dataset,
            keep_relations=keep_relations,
            remove_relations=remove_relations,
            keep_entities=keep_entities,
        )

        print(
            f"Train: {dataset.training.num_triples}, "
            f"Validation: {dataset.validation.num_triples}, "
            f"Test: {dataset.testing.num_triples}"
        )
        print(
            f"Inverse relations - Train: {dataset.training.create_inverse_triples}, "
            f"Validation: {dataset.validation.create_inverse_triples}, "
            f"Test: {dataset.testing.create_inverse_triples}"
        )
        return dataset




def filter_dataset(
    dataset: EagerDataset,
    keep_relations: set = None, 
    remove_relations: set = None,
    keep_entities: set = None,
) -> EagerDataset:

    training_factory = dataset.training

    if keep_relations is not None and remove_relations is not None:
        raise ValueError("Specify either keep_relations or remove_relations, not both.")

    # Build the relation selection from the original mapping.
    if keep_relations is not None:
        available_relations = set(training_factory.relation_to_id)

        relation_labels = {
            label
            for label in available_relations
            if label in keep_relations
            or (
                label.endswith("_inverse")
                and label.removesuffix("_inverse") in keep_relations
            )
        }
    elif remove_relations:
        relation_labels = {
            label
            for label in training_factory.relation_to_id
            if label not in remove_relations
            and not (
                label.endswith("_inverse")
                and label.removesuffix("_inverse") in remove_relations
            )
        }
    else:
        relation_labels = None

    relation_ids = (
        training_factory.relations_to_ids(relation_labels)
        if relation_labels is not None
        else None
    )

    entity_ids = (
        training_factory.entities_to_ids(keep_entities)
        if keep_entities is not None
        else None
    )

    # Apply exactly the same filtering to all splits.
    return EagerDataset(
        training=dataset.training.new_with_restriction(
            entities=entity_ids,
            relations=relation_ids,
        ),
        validation=dataset.validation.new_with_restriction(
            entities=entity_ids,
            relations=relation_ids,
        ),
        testing=dataset.testing.new_with_restriction(
            entities=entity_ids,
            relations=relation_ids,
        ),
    )
