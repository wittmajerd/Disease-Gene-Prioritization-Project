from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
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
        self.val_count: int = config.get("val_count", 2000)
        self.test_count: int = config.get("test_count", 8000)
        self.random_seed: int = config.get("random_seed", 42)
        self.entity_types: dict[str, set[str]] = {}

        # features later


    def build_splits(self):
        triples_df = pd.read_csv(self.kg_path, low_memory=False)

        if self.remove_self_loops:
            triples_df = triples_df[triples_df[self.head] != triples_df[self.tail]]

        if self.drop_duplicates:
            triples_df = triples_df.drop_duplicates(subset=[self.head, self.relation, self.tail])

        self.entity_types = self._build_entity_types(triples_df)

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



    def get_dataset(
        self,
        keep_relations: set = None,
        remove_relations: set = None,
        keep_entities: set = None,
        keep_entity_types: set = None,
    ):
        if keep_entities is not None and keep_entity_types is not None:
            raise ValueError("Specify either keep_entities or keep_entity_types, not both.")

        if keep_entity_types is not None and not self.entity_types:
            triples_df = pd.read_csv(self.kg_path, low_memory=False)
            self.entity_types = self._build_entity_types(triples_df)

        entity_labels = keep_entities
        if keep_entity_types is not None:
            entity_labels = {
                entity
                for entity, types in self.entity_types.items()
                if types & keep_entity_types
            }

        dataset = EagerDataset(
            training=self.training,
            validation=self.validation,
            testing=self.testing,
        )
        dataset = filter_dataset(
            dataset=dataset,
            keep_relations=keep_relations,
            remove_relations=remove_relations,
            keep_entities=entity_labels,
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

    def _build_entity_types(self, triples_df: pd.DataFrame) -> dict[str, set[str]]:
        entity_types: dict[str, set[str]] = {}
        for entity_column, type_column in (
            (self.head, "x_type"),
            (self.tail, "y_type"),
        ):
            for entity, entity_type in triples_df[[entity_column, type_column]].drop_duplicates().itertuples(index=False):
                entity_types.setdefault(str(entity), set()).add(str(entity_type))
        return entity_types




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

    entity_ids = None
    if relation_ids is not None:
        active_entity_ids = []
        for factory in (dataset.training, dataset.validation, dataset.testing):
            relation_ids_tensor = torch.as_tensor(
                relation_ids,
                dtype=factory.mapped_triples.dtype,
                device=factory.mapped_triples.device,
            )
            relation_mask = torch.isin(
                factory.mapped_triples[:, 1],
                relation_ids_tensor,
            )
            triples = factory.mapped_triples[relation_mask]
            if triples.numel():
                active_entity_ids.append(triples[:, (0, 2)].reshape(-1))
        if active_entity_ids:
            entity_ids = torch.unique(torch.cat(active_entity_ids))
        else:
            entity_ids = torch.empty(0, dtype=torch.long)

    if keep_entities is not None:
        requested_entity_ids = training_factory.entities_to_ids(keep_entities)
        requested_entity_ids = torch.as_tensor(
            requested_entity_ids,
            dtype=training_factory.mapped_triples.dtype,
            device=training_factory.mapped_triples.device,
        )
        entity_ids = (
            requested_entity_ids
            if entity_ids is None
            else entity_ids[torch.isin(entity_ids, requested_entity_ids)]
        )

    # Apply exactly the same filtering to all splits. PyKEEN preserves the
    # original entity mapping when restricting a factory, so compact it after
    # filtering to avoid retaining isolated entities in the model.
    factories = tuple(
        factory.new_with_restriction(
            entities=entity_ids,
            relations=relation_ids,
        )
        for factory in (dataset.training, dataset.validation, dataset.testing)
    )
    if entity_ids is None:
        return EagerDataset(
            training=factories[0],
            validation=factories[1],
            testing=factories[2],
        )

    active_entity_id_parts = [
        factory.mapped_triples[:, (0, 2)].reshape(-1)
        for factory in factories
        if factory.mapped_triples.numel()
    ]
    if not active_entity_id_parts:
        raise ValueError("Filtering removed all triples from the dataset.")
    active_entity_ids = torch.unique(torch.cat(active_entity_id_parts))
    entity_id_to_label = training_factory.entity_id_to_label
    compact_entity_to_id = {
        entity_id_to_label[int(entity_id)]: compact_id
        for compact_id, entity_id in enumerate(active_entity_ids.tolist())
    }
    relation_id_to_label = training_factory.relation_id_to_label
    active_relation_labels = {
        relation_id_to_label[int(relation_id)]
        for factory in factories
        for relation_id in torch.unique(factory.mapped_triples[:, 1]).tolist()
    }
    compact_relation_to_id = {
        relation_label: compact_id
        for compact_id, relation_label in enumerate(
            sorted(
                relation_label
                for relation_label in active_relation_labels
                if not relation_label.endswith("_inverse")
            )
        )
    }

    compact_factories = []
    for factory in factories:
        labeled_triples = np.asarray(
            [
                [
                    entity_id_to_label[int(head_id)],
                    relation_id_to_label[int(relation_id)],
                    entity_id_to_label[int(tail_id)],
                ]
                for head_id, relation_id, tail_id in factory.mapped_triples.tolist()
                if not relation_id_to_label[int(relation_id)].endswith("_inverse")
            ],
            dtype=str,
        )
        compact_factories.append(
            TriplesFactory.from_labeled_triples(
                triples=labeled_triples,
                entity_to_id=compact_entity_to_id,
                relation_to_id=compact_relation_to_id,
                create_inverse_triples=factory is factories[0],
            )
        )

    return EagerDataset(
        training=compact_factories[0],
        validation=compact_factories[1],
        testing=compact_factories[2],
    )
