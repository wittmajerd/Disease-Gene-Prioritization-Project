from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def normalize_kg_dataframe(
    df: pd.DataFrame,
    *,
    head_candidates: tuple[str, ...] = ("head", "x_id", "x_index", "source", "source_id"),
    relation_candidates: tuple[str, ...] = (
        "display_relation",
        "relation",
        "relation_type",
        "predicate",
        "edge_type",
    ),
    tail_candidates: tuple[str, ...] = ("tail", "y_id", "y_index", "target", "target_id"),
    head_type_candidates: tuple[str, ...] = ("head_type", "x_type", "source_type"),
    tail_type_candidates: tuple[str, ...] = ("tail_type", "y_type", "target_type"),
) -> pd.DataFrame:
    """Normalize a KG DataFrame to standard columns.

    Required output columns: head, relation, tail
    Optional output columns: head_type, tail_type
    """

    def pick_col(candidates: tuple[str, ...], label: str) -> str:
        for name in candidates:
            if name in df.columns:
                return name
        raise ValueError(f"Missing {label} column. Tried: {candidates}. Found: {list(df.columns)}")

    head_col = pick_col(head_candidates, "head")
    relation_col = pick_col(relation_candidates, "relation")
    tail_col = pick_col(tail_candidates, "tail")

    out = df[[head_col, relation_col, tail_col]].rename(
        columns={head_col: "head", relation_col: "relation", tail_col: "tail"}
    )
    out = out.dropna(subset=["head", "relation", "tail"]).copy()
    out["head"] = out["head"].astype(str)
    out["relation"] = out["relation"].astype(str)
    out["tail"] = out["tail"].astype(str)

    head_type_col = next((c for c in head_type_candidates if c in df.columns), None)
    tail_type_col = next((c for c in tail_type_candidates if c in df.columns), None)
    if head_type_col is not None:
        out["head_type"] = df.loc[out.index, head_type_col].astype("string")
    if tail_type_col is not None:
        out["tail_type"] = df.loc[out.index, tail_type_col].astype("string")

    return out.reset_index(drop=True)


def triples_factory_to_df(triples_factory: Any) -> pd.DataFrame:
    """Convert a PyKEEN TriplesFactory to a normalized triples DataFrame."""
    mapped = triples_factory.mapped_triples.cpu().numpy()
    e_map = triples_factory.entity_id_to_label
    r_map = triples_factory.relation_id_to_label

    heads = [e_map[int(i)] for i in mapped[:, 0]]
    relations = [r_map[int(i)] for i in mapped[:, 1]]
    tails = [e_map[int(i)] for i in mapped[:, 2]]

    return pd.DataFrame({"head": heads, "relation": relations, "tail": tails})


@dataclass(slots=True)
class ComponentStats:
    num_components: int
    largest_component_size: int
    smallest_component_size: int
    component_sizes: list[int]


class KGAnalyzer:
    """Standalone analyzer for triple-based knowledge graphs."""

    def __init__(
        self,
        triples_df: pd.DataFrame,
        *,
        head_col: str = "head",
        relation_col: str = "relation",
        tail_col: str = "tail",
        head_type_col: str | None = "head_type",
        tail_type_col: str | None = "tail_type",
    ):
        required = {head_col, relation_col, tail_col}
        missing = [c for c in required if c not in triples_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = triples_df[[head_col, relation_col, tail_col]].copy()
        self.df.columns = ["head", "relation", "tail"]
        self.df["head"] = self.df["head"].astype(str)
        self.df["relation"] = self.df["relation"].astype(str)
        self.df["tail"] = self.df["tail"].astype(str)

        self.has_types = False
        if head_type_col and head_type_col in triples_df.columns:
            self.df["head_type"] = triples_df[head_type_col].astype("string")
            self.has_types = True
        if tail_type_col and tail_type_col in triples_df.columns:
            self.df["tail_type"] = triples_df[tail_type_col].astype("string")
            self.has_types = True

        entities = pd.unique(pd.concat([self.df["head"], self.df["tail"]], ignore_index=True))
        self.entities = pd.Index(entities.astype(str))
        self.entity_to_idx = {e: i for i, e in enumerate(self.entities)}

        self.head_idx = self.df["head"].map(self.entity_to_idx).to_numpy(np.int64)
        self.tail_idx = self.df["tail"].map(self.entity_to_idx).to_numpy(np.int64)

    def global_stats(self) -> dict[str, Any]:
        n_nodes = len(self.entities)
        n_edges = len(self.df)
        n_rel = int(self.df["relation"].nunique())
        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

        out = {
            "total_nodes": n_nodes,
            "total_edges": n_edges,
            "unique_relations": n_rel,
            "density": float(density),
            "self_loops": int((self.df["head"] == self.df["tail"]).sum()),
            "duplicate_triples": int(self.df.duplicated(["head", "relation", "tail"]).sum()),
        }
        if self.has_types:
            out["head_types"] = int(self.df["head_type"].nunique(dropna=True)) if "head_type" in self.df else 0
            out["tail_types"] = int(self.df["tail_type"].nunique(dropna=True)) if "tail_type" in self.df else 0
        return out

    def relation_stats(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for rel, grp in self.df.groupby("relation", sort=False):
            heads = grp["head"].value_counts()
            tails = grp["tail"].value_counts()
            avg_tph = float(heads.mean()) if not heads.empty else 0.0
            avg_hpt = float(tails.mean()) if not tails.empty else 0.0
            h_label = "1" if avg_tph < 1.5 else "N"
            t_label = "1" if avg_hpt < 1.5 else "N"
            rows.append(
                {
                    "relation": rel,
                    "edge_count": int(len(grp)),
                    "unique_heads": int(heads.size),
                    "unique_tails": int(tails.size),
                    "avg_tails_per_head": avg_tph,
                    "avg_heads_per_tail": avg_hpt,
                    "cardinality": f"{h_label}-to-{t_label}",
                }
            )

        out = pd.DataFrame(rows).sort_values("edge_count", ascending=False)
        return out.reset_index(drop=True)

    def degree_stats(self) -> pd.DataFrame:
        out_deg = np.bincount(self.head_idx, minlength=len(self.entities))
        in_deg = np.bincount(self.tail_idx, minlength=len(self.entities))
        total_deg = out_deg + in_deg

        ent_df = pd.DataFrame(
            {
                "entity": self.entities,
                "in_degree": in_deg,
                "out_degree": out_deg,
                "degree": total_deg,
            }
        )

        if self.has_types and "head_type" in self.df and "tail_type" in self.df:
            type_map: dict[str, str] = {}
            head_types = self.df[["head", "head_type"]].dropna().drop_duplicates()
            tail_types = self.df[["tail", "tail_type"]].dropna().drop_duplicates()
            for row in head_types.itertuples(index=False):
                type_map[str(row.head)] = str(row.head_type)
            for row in tail_types.itertuples(index=False):
                type_map.setdefault(str(row.tail), str(row.tail_type))
            ent_df["entity_type"] = ent_df["entity"].map(type_map).fillna("unknown")

        return ent_df

    def power_law_summary(self, *, min_degree: int = 1) -> dict[str, float]:
        deg = self.degree_stats()["degree"].to_numpy(np.int64)
        deg = deg[deg >= min_degree]
        if deg.size == 0:
            return {"exponent": float("nan"), "r_squared": float("nan")}

        counts = Counter(deg.tolist())
        x = np.array(sorted(counts.keys()), dtype=np.float64)
        y = np.array([counts[int(v)] for v in x], dtype=np.float64)

        lx = np.log10(x)
        ly = np.log10(y)
        A = np.vstack([lx, np.ones_like(lx)]).T
        slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]

        pred = slope * lx + intercept
        ss_res = float(np.sum((ly - pred) ** 2))
        ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

        return {"exponent": float(-slope), "r_squared": float(r2)}

    def connected_components(self) -> ComponentStats:
        n = len(self.entities)
        parent = np.arange(n, dtype=np.int64)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[ra] = rb

        for s, t in zip(self.head_idx, self.tail_idx):
            union(int(s), int(t))

        sizes = Counter(find(i) for i in range(n))
        comp_sizes = sorted((int(v) for v in sizes.values()), reverse=True)

        return ComponentStats(
            num_components=len(comp_sizes),
            largest_component_size=comp_sizes[0] if comp_sizes else 0,
            smallest_component_size=comp_sizes[-1] if comp_sizes else 0,
            component_sizes=comp_sizes,
        )

    def connectivity_matrix(self) -> pd.DataFrame:
        if not ("head_type" in self.df and "tail_type" in self.df):
            raise ValueError("Type-aware connectivity needs head_type and tail_type columns.")

        ct = (
            self.df.dropna(subset=["head_type", "tail_type"])
            .groupby(["head_type", "tail_type"]) 
            .size()
            .rename("count")
            .reset_index()
        )
        mat = ct.pivot(index="head_type", columns="tail_type", values="count").fillna(0).astype(np.int64)
        return mat.sort_index(axis=0).sort_index(axis=1)

    def type_confusion_matrix(self, *, normalize: bool = False) -> pd.DataFrame:
        """Return type-to-type edge count matrix.

        If normalize=True, rows are normalized to probabilities.
        """
        mat = self.connectivity_matrix().astype(np.float64)
        if normalize:
            row_sum = mat.sum(axis=1).replace(0.0, np.nan)
            mat = mat.div(row_sum, axis=0).fillna(0.0)
        return mat

    def relation_symmetry_stats(self, *, min_edges: int = 1) -> pd.DataFrame:
        """Estimate symmetry per relation.

        For each relation, computes how many edges have reverse-direction counterpart
        under the same relation, i.e., (h, r, t) and (t, r, h).
        """
        rows: list[dict[str, Any]] = []
        for rel, grp in self.df.groupby("relation", sort=False):
            n_edges = int(len(grp))
            if n_edges < min_edges:
                continue

            pairs = set(zip(grp["head"].tolist(), grp["tail"].tolist()))
            if not pairs:
                continue

            reverse_hits = sum(1 for (h, t) in pairs if (t, h) in pairs)
            symmetry_ratio = reverse_hits / len(pairs)

            rows.append(
                {
                    "relation": rel,
                    "edge_count": n_edges,
                    "unique_pairs": int(len(pairs)),
                    "reverse_hits": int(reverse_hits),
                    "symmetry_ratio": float(symmetry_ratio),
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=["relation", "edge_count", "unique_pairs", "reverse_hits", "symmetry_ratio"]
            )

        out = pd.DataFrame(rows).sort_values(["symmetry_ratio", "edge_count"], ascending=[False, False])
        return out.reset_index(drop=True)

    def schema_edge_type_counts(self) -> pd.DataFrame:
        """Return schema-level type graph edges with counts.

        Output columns: head_type, tail_type, edge_count, unique_relations, relations
        """
        if not ("head_type" in self.df and "tail_type" in self.df):
            raise ValueError("Schema type graph needs head_type and tail_type columns.")

        typed = self.df.dropna(subset=["head_type", "tail_type"]).copy()
        if typed.empty:
            return pd.DataFrame(
                columns=["head_type", "tail_type", "edge_count", "unique_relations", "relations"]
            )

        grouped = typed.groupby(["head_type", "tail_type"], sort=False)
        rows = []
        for (head_type, tail_type), grp in grouped:
            rels = sorted(grp["relation"].astype(str).unique().tolist())
            rows.append(
                {
                    "head_type": str(head_type),
                    "tail_type": str(tail_type),
                    "edge_count": int(len(grp)),
                    "unique_relations": int(len(rels)),
                    "relations": rels,
                }
            )

        return pd.DataFrame(rows).sort_values("edge_count", ascending=False).reset_index(drop=True)

    def target_relation_analysis(self, relation: str, *, top_k: int = 20) -> dict[str, Any]:
        grp = self.df[self.df["relation"] == relation]
        if grp.empty:
            raise ValueError(f"Relation not found: {relation}")

        head_deg = grp["head"].value_counts()
        tail_deg = grp["tail"].value_counts()

        return {
            "relation": relation,
            "edges": int(len(grp)),
            "unique_heads": int(grp["head"].nunique()),
            "unique_tails": int(grp["tail"].nunique()),
            "head_degree_mean": float(head_deg.mean()) if not head_deg.empty else 0.0,
            "tail_degree_mean": float(tail_deg.mean()) if not tail_deg.empty else 0.0,
            "top_heads": head_deg.head(top_k).to_dict(),
            "top_tails": tail_deg.head(top_k).to_dict(),
        }

    def metapath_analysis_sampled(
        self,
        *,
        src_type: str,
        dst_type: str,
        max_hops: int = 3,
        max_start_nodes: int = 500,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        if not ("head_type" in self.df and "tail_type" in self.df):
            raise ValueError("Metapath type analysis needs head_type and tail_type columns.")

        typed = self.df.dropna(subset=["head_type", "tail_type"])
        if typed.empty:
            return pd.DataFrame(columns=["path", "length", "count"])

        # Schema-level exploration: relation-typed transitions.
        edges = typed[["head_type", "relation", "tail_type"]].drop_duplicates()
        adj: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for row in edges.itertuples(index=False):
            adj[str(row.head_type)].append((str(row.head_type), str(row.relation), str(row.tail_type)))

        paths: Counter[str] = Counter()

        def dfs(current_type: str, depth: int, acc: list[tuple[str, str, str]]) -> None:
            if depth > max_hops:
                return
            if current_type == dst_type and acc:
                label = " -> ".join(f"{s}[{r}]{d}" for s, r, d in acc)
                paths[label] += 1
            for edge in adj.get(current_type, []):
                acc.append(edge)
                dfs(edge[2], depth + 1, acc)
                acc.pop()

        dfs(src_type, 0, [])

        if not paths:
            return pd.DataFrame(columns=["path", "length", "count"])

        rows = [
            {"path": p, "length": p.count("->") + 1, "count": c}
            for p, c in paths.items()
        ]
        return pd.DataFrame(rows).sort_values(["length", "count"], ascending=[True, False]).reset_index(drop=True)

    def feature_readiness_report(
        self,
        features_df: pd.DataFrame,
        *,
        feature_key_col: str = "node_index",
    ) -> dict[str, Any]:
        if feature_key_col not in features_df.columns:
            raise ValueError(f"Feature key column missing: {feature_key_col}")

        feature_keys = set(features_df[feature_key_col].dropna().astype(str))
        graph_keys = set(self.entities.astype(str))
        overlap = graph_keys.intersection(feature_keys)

        out: dict[str, Any] = {
            "graph_entities": len(graph_keys),
            "feature_entities": len(feature_keys),
            "joinable_entities": len(overlap),
            "coverage_ratio": len(overlap) / len(graph_keys) if graph_keys else 0.0,
            "missing_feature_entities": len(graph_keys - feature_keys),
            "orphan_feature_rows": len(feature_keys - graph_keys),
        }

        if self.has_types:
            deg_df = self.degree_stats()
            if "entity_type" in deg_df.columns:
                per_type = (
                    deg_df.assign(has_feature=deg_df["entity"].isin(feature_keys))
                    .groupby("entity_type")["has_feature"]
                    .agg(["size", "sum"]) 
                    .rename(columns={"size": "entities", "sum": "joinable"})
                    .reset_index()
                )
                per_type["coverage_ratio"] = per_type["joinable"] / per_type["entities"]
                out["coverage_by_type"] = per_type.sort_values("entities", ascending=False).to_dict(
                    orient="records"
                )

        return out


# -------------------- plotting helpers --------------------


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting. Install it with: pip install matplotlib") from exc
    return plt


def plot_relation_distribution(relation_df: pd.DataFrame, *, top_n: int = 20):
    plt = _require_matplotlib()
    top = relation_df.head(top_n)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(top["relation"], top["edge_count"])
    ax.set_title(f"Top {top_n} relations by edge count")
    ax.set_ylabel("Edge count")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    return fig


def plot_degree_distribution(degree_df: pd.DataFrame):
    plt = _require_matplotlib()
    deg = degree_df["degree"].to_numpy(np.int64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(deg, bins=80)
    axes[0].set_title("Degree distribution")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Frequency")

    positive = deg[deg > 0]
    axes[1].hist(positive, bins=80, log=True)
    axes[1].set_title("Degree distribution (log-y)")
    axes[1].set_xlabel("Degree")
    axes[1].set_ylabel("Frequency (log)")
    fig.tight_layout()
    return fig


def plot_component_sizes(component_stats: ComponentStats, *, top_n: int = 30):
    plt = _require_matplotlib()
    vals = component_stats.component_sizes[:top_n]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(vals)), vals)
    ax.set_title(f"Top {top_n} component sizes")
    ax.set_xlabel("Component rank")
    ax.set_ylabel("Size")
    fig.tight_layout()
    return fig


def plot_connectivity_heatmap(connectivity_df: pd.DataFrame):
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(connectivity_df.values)
    ax.set_xticks(np.arange(connectivity_df.shape[1]))
    ax.set_xticklabels(connectivity_df.columns, rotation=60, ha="right")
    ax.set_yticks(np.arange(connectivity_df.shape[0]))
    ax.set_yticklabels(connectivity_df.index)
    ax.set_title("Type-to-type connectivity")
    fig.colorbar(im, ax=ax, label="Edge count")
    fig.tight_layout()
    return fig


def plot_type_confusion_heatmap(confusion_df: pd.DataFrame, *, title: str = "Type Confusion Matrix"):
    plt = _require_matplotlib()
    try:
        import seaborn as sns
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("seaborn is required for confusion heatmap plotting. Install it with: pip install seaborn") from exc

    fig, ax = plt.subplots(figsize=(9, 7))

    is_integer_like = np.allclose(confusion_df.values, np.round(confusion_df.values))
    fmt = ",.0f" if is_integer_like else ".3f"
    sns.heatmap(
        confusion_df,
        annot=True,
        fmt=fmt,
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Destination Type")
    ax.set_ylabel("Source Type")
    fig.tight_layout()
    return fig


def plot_schema_type_graph(
    schema_edges_df: pd.DataFrame,
    *,
    min_edge_count: int = 1,
    max_label_edges: int = 20,
):
    """Plot directed schema graph where nodes are types and edges are type pairs."""
    plt = _require_matplotlib()
    try:
        import networkx as nx
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("networkx is required for schema graph plotting. Install it with: pip install networkx") from exc

    use_df = schema_edges_df[schema_edges_df["edge_count"] >= min_edge_count].copy()
    if use_df.empty:
        raise ValueError("No schema edges left after applying min_edge_count filter.")

    g = nx.DiGraph()
    for row in use_df.itertuples(index=False):
        g.add_edge(str(row.head_type), str(row.tail_type), weight=float(row.edge_count))

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(g, seed=42, k=1.2)

    weights = np.array([data["weight"] for _, _, data in g.edges(data=True)], dtype=np.float64)
    min_w = float(weights.min())
    max_w = float(weights.max())
    width = 1.0 + 5.0 * ((weights - min_w) / (max_w - min_w + 1e-9))

    nx.draw_networkx_nodes(g, pos, node_size=2200, node_color="#9ecae1", ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(
        g,
        pos,
        width=width.tolist(),
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        edge_color="#4c72b0",
        alpha=0.85,
        ax=ax,
    )

    top_edges = use_df.sort_values("edge_count", ascending=False).head(max_label_edges)
    edge_labels = {
        (str(row.head_type), str(row.tail_type)): str(int(row.edge_count))
        for row in top_edges.itertuples(index=False)
    }
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    ax.set_title("Schema Type Graph")
    ax.axis("off")
    fig.tight_layout()
    return fig


__all__ = [
    "ComponentStats",
    "KGAnalyzer",
    "normalize_kg_dataframe",
    "triples_factory_to_df",
    "plot_component_sizes",
    "plot_connectivity_heatmap",
    "plot_schema_type_graph",
    "plot_type_confusion_heatmap",
    "plot_degree_distribution",
    "plot_relation_distribution",
]
