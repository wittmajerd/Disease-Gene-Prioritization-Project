# %%
from dataclasses import dataclass as Dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from IPython.display import display


# api-t ki kéne találni az analízishez ami nem az mint deeig volt lehet ötleteket venni de kinda szar volt az eddigi
# Mi legyen az elemzésben?

# kg.csv betöltése maradjon df mert abban benne van minden ami kell és ki is lehet számolni mindent amit kell

# nem szeretnék osztályt meg ilyenek csak szipla fgveket amik megkapnak egy df-et és abból kiszámolnak dolgokat

# statok:
# - node és edge típusok és darabszámok
# - conf mx szerű tábla a connectionokről - gráf, node centralitás
# - density, komponensek - izolált csúcsok?
# - node statistics tipus szerint
# - node degree hists
# - edge type szerinti stats
# - rel cardinality - bár minde n-n nem?
# - disease és protein részletesebb elemzése
# - metapath analysis - milyen útvonalak vannak d és p közt
# - a dis hány százalákából van él prot-ba és esetleg metapathen keresztül milyen hosszú utakon hogy százalék kapcsolódik
# - power law
# - two (3, 4...) hop reachability

# feature-ök elemzése külön, egyáltalán hogy használhatóak ezek a szövegesek

# Core graph summary: global stats, degree extraction, connected components, isolated-node reporting.
# Type-aware schema analysis: node-type stats, connectivity matrix, schema edge counts, normalized type-confusion matrix.
# Relation analysis: relation counts, density, cardinality heuristic, symmetry ratio, source and destination degree summaries.
# Target-analysis block: disease-to-protein deep dive, cold-start counts, hub proteins, direct coverage.
# Path and structure analysis: metapath enumeration, metapath reachability, power-law summary.
# Feature analysis: feature-readiness / joinability report for disease features or future feature tables.


# TODO jaaa igen ezeket a statokat úgy is érdekes lenne megnézni hogy előtte szűrünk



# DF header: relation, display_relation, x_index, x_id, x_type, x_name, x_source, y_index, y_id, y_type, y_name, y_source
# %%
# df = pd.read_csv("primekg/kg.csv", low_memory=False)


# %%

def filter_df(df, filter_type="nodes", types=None):
    df = df.copy()
    if filter_type == "nodes":
        df = df[df["x_type"].isin(types) & df["y_type"].isin(types)] if types is not None else df
    elif filter_type == "edges":
        df = df[df["relation"].isin(types)] if types is not None else df

    return df

def extract_nodes_df(df: DataFrame) -> DataFrame:
    """Build unique node table from edge list.

    Output should include at least:
    - node_id
    - node_type (optional if source data has types)

    Focus:
    - Union of head and tail ids.
    - Preserve type if available; flag conflicts if an id appears with many types.
    """

    head_types = df[["x_index", "x_type"]].rename(columns={"x_index": "node_id", "x_type": "node_type"})
    tail_types = df[["y_index", "y_type"]].rename(columns={"y_index": "node_id", "y_type": "node_type"})
    all_types = pd.concat([head_types, tail_types], ignore_index=True).drop_duplicates()

    return all_types.reset_index(drop=True)


def extract_all_nodes(df: DataFrame) -> DataFrame:
    head = df[["x_index", "x_id", "x_type", "x_name", "x_source"]].rename(columns={"x_index": "index", "x_type": "type", "x_id": "id", "x_name": "name", "x_source": "source"})
    tail = df[["y_index", "y_id", "y_type", "y_name", "y_source"]].rename(columns={"y_index": "index", "y_type": "type", "y_id": "id", "y_name": "name", "y_source": "source"})
    all_types = pd.concat([head, tail], ignore_index=True)

    return all_types.reset_index(drop=True)

def compute_degrees(df: DataFrame) -> DataFrame:
    """Compute per-node in/out/total degree table.

    Output columns:
    - node_index
    - in_degree
    - out_degree
    - total_degree
    - node_type (when available)

    Focus:
    - Reusable base table for node stats, hubs, isolates, and power-law analysis.
    """
    out_deg = df.groupby("x_index").size().rename("out_degree")
    in_deg = df.groupby("y_index").size().rename("in_degree")

    deg = pd.concat([out_deg, in_deg], axis=1).fillna(0).astype(int)
    deg["total_degree"] = deg["out_degree"] + deg["in_degree"]
    deg = deg.reset_index().rename(columns={"index": "node_id"})

    nodes = extract_nodes_df(df)
    deg = nodes[["node_id", "node_type"]].merge(
        deg,
        on="node_id",
        how="left",
    )
    for col in ["in_degree", "out_degree", "total_degree"]:
        deg[col] = deg[col].fillna(0).astype(int)

    return deg.sort_values("node_id").reset_index(drop=True)


def reverse_edge_consistency_check(df: DataFrame) -> DataFrame:
    key = df[["x_index", "y_index"]]
    reversed_key = key.rename(columns={"x_index": "y_index", "y_index": "x_index"})

    key = df[["x_index", "y_index", "relation"]].drop_duplicates()
    reversed_key = key.rename(columns={"x_index": "y_index", "y_index": "x_index"})

    paired = key.merge(
        reversed_key,
        on=["x_index", "y_index", "relation"],
        how="left",
        indicator=True,
    )
    print(paired["_merge"].unique())


def calc_global_stats(df: DataFrame) -> dict[str, Any]:
    """Compute high-level KG summary metrics.

    Should report:
    - total_nodes, total_edges
    - num_node_types, num_edge_types
    - node_types, edge_types
    - density (directed graph density)
    - self_loops count
    - duplicate_edge_rows count

    Focus:
    - Fast, deterministic summary for sanity checks.
    """
    e = df[["x_index", "y_index", "relation", "x_type", "y_type"]]
    nodes = extract_nodes_df(e)
    total_nodes = int(nodes.shape[0])
    total_edges = int(e.shape[0])
    unique_edges = int(e[["x_index", "relation", "y_index"]].drop_duplicates().shape[0])
    duplicate_edge_rows = total_edges - unique_edges
    # a source miatt van 370 duplicate ezeket majd kiszűjük
    self_loops = int((e["x_index"] == e["y_index"]).sum())

    density = 0.0
    if total_nodes > 1:
        density = total_edges / float(total_nodes * (total_nodes - 1))

    node_types = []
    if "node_type" in nodes.columns:
        node_types = sorted(nodes["node_type"].dropna().unique().tolist())

    edge_types = sorted(e["relation"].dropna().unique().tolist())

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "num_node_types": len(node_types),
        "num_edge_types": len(edge_types),
        "node_types": node_types,
        "edge_types": edge_types,
        "density": float(density),
        "self_loops": self_loops,
        "duplicate_edge_rows": int(duplicate_edge_rows),
    }

def connected_components_summary(df: DataFrame) -> dict[str, Any]:
    """Analyze weakly connected components.

    Should report:
    - num_components
    - largest_component_size
    - smallest_component_size
    - component_sizes (sorted desc)
    - isolated_node_count (size-1 components) ?

    Focus:
    - Component fragmentation and isolate diagnostics.
    """
    e = df[["x_index", "y_index", "relation", "x_type", "y_type"]]
    nodes = extract_nodes_df(e)
    node_ids = nodes["node_id"].tolist()
    node_set = set(node_ids)

    if not node_ids:
        return {
            "num_components": 0,
            "largest_component_size": 0,
            "smallest_component_size": 0,
            "component_sizes": [],
            "isolated_node_count": 0,
        }

    adj: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for h, t in e[["x_index", "y_index"]].itertuples(index=False, name=None):
        if h in node_set and t in node_set:
            adj[h].add(t)
            adj[t].add(h)

    visited: set[str] = set()
    comp_sizes: list[int] = []
    for nid in node_ids:
        if nid in visited:
            continue
        stack = [nid]
        visited.add(nid)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in adj[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        comp_sizes.append(size)

    comp_sizes.sort(reverse=True)
    return {
        "num_components": int(len(comp_sizes)),
        "largest_component_size": int(comp_sizes[0]),
        "smallest_component_size": int(comp_sizes[-1]),
        "component_sizes": comp_sizes,
        "isolated_node_count": int(sum(1 for s in comp_sizes if s == 1)),
    }

def node_type_stats(edges_df: DataFrame, degrees_df: DataFrame | None = None) -> DataFrame:
    """Compute node-degree statistics grouped by node type.

    Should include for each type:
    - node_count
    - mean/median/std/min/max degree
    - p90/p99 degree
    - isolated_nodes (degree == 0)

    Focus:
    - Heavy-tail behavior and imbalance across entity types.
    """
    deg = compute_degrees(edges_df) if degrees_df is None else degrees_df.copy()

    rows: list[dict[str, Any]] = []
    for ntype, grp in deg.groupby("node_type", dropna=False):
        td = grp["total_degree"]
        rows.append(
            {
                "node_type": str(ntype),
                "node_count": int(grp.shape[0]),
                "mean_degree": float(td.mean()),
                "median_degree": float(td.median()),
                "std_degree": float(td.std(ddof=0)),
                "min_degree": int(td.min()),
                "max_degree": int(td.max()),
                "p90_degree": float(td.quantile(0.9)),
                "p99_degree": float(td.quantile(0.99)),
                "isolated_nodes": int((td == 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("node_count", ascending=False).reset_index(drop=True)

def relation_type_stats(df: DataFrame) -> DataFrame:
    """Compute per-relation structural statistics.

    Should include:
    - relation, src_type, dst_type
    - edge_count
    - relation_density (edges / possible src-dst pairs)
    - unique_heads, unique_tails
    - avg_tails_per_head, avg_heads_per_tail
    - relation_class (1-1 / 1-N / N-1 / N-N using threshold heuristic)

    Focus:
    - Relation cardinality behavior and sparsity per relation.
    """
    e = df[["x_index", "y_index", "relation", "x_type", "y_type"]]

    rows: list[dict[str, Any]] = []
    grouped = e.groupby(["relation", "x_type", "y_type"], dropna=False)
    for (rel, src_t, dst_t), g in grouped:
        edge_count = int(g.shape[0])
        unique_heads = int(g["x_index"].nunique())
        unique_tails = int(g["y_index"].nunique())

        possible_pairs = max(unique_heads * unique_tails, 1)
        density = edge_count / float(possible_pairs)

        tails_per_head = g.groupby("x_index")["y_index"].nunique()
        heads_per_tail = g.groupby("y_index")["x_index"].nunique()
        avg_tph = float(tails_per_head.mean()) if not tails_per_head.empty else 0.0
        avg_hpt = float(heads_per_tail.mean()) if not heads_per_tail.empty else 0.0

        head_side = "1" if avg_tph < 1.5 else "N"
        tail_side = "1" if avg_hpt < 1.5 else "N"
        relation_class = f"{head_side}-{tail_side}"

        rows.append(
            {
                "relation": str(rel),
                "src_type": str(src_t),
                "dst_type": str(dst_t),
                "edge_count": edge_count,
                "relation_density": float(density),
                "unique_heads": unique_heads,
                "unique_tails": unique_tails,
                "avg_tails_per_head": avg_tph,
                "avg_heads_per_tail": avg_hpt,
                "relation_class": relation_class,
            }
        )

    return pd.DataFrame(rows).sort_values("edge_count", ascending=False).reset_index(drop=True)

# %%
def connectivity_matrix(df: DataFrame, normalize: bool = False) -> DataFrame:
    """Build source-type vs destination-type edge matrix.

    - normalize=False: raw counts
    - normalize=True: row-normalized proportions

    Focus:
    - Schema-level connectivity view and confusion-matrix-style table.
    """
    e = df[["x_index", "y_index", "relation", "x_type", "y_type"]]
    if "x_type" not in e.columns or "y_type" not in e.columns:
        raise ValueError("connectivity_matrix requires x_type/y_type or head_type/tail_type columns")

    mat = (
        e.groupby(["x_type", "y_type"]).size().rename("edge_count").reset_index().pivot(
            index="x_type", columns="y_type", values="edge_count"
        ).fillna(0)
    )
    mat = mat.astype(float if normalize else int)
    if normalize:
        row_sums = mat.sum(axis=1).replace(0, np.nan)
        mat = mat.div(row_sums, axis=0).fillna(0.0)
    return mat

# %%
def schema_edge_type_counts(df: DataFrame) -> DataFrame:
    """Summarize schema edges at (head_type, tail_type) granularity.

    Should include:
    - head_type, tail_type
    - edge_count
    - num_relations
    - relations (compact relation list)

    Focus:
    - Graph schema map for plotting and reporting.
    """
    e = df[["x_index", "y_index", "relation", "x_type", "y_type"]]
    if "x_type" not in e.columns or "y_type" not in e.columns:
        raise ValueError("schema_edge_type_counts requires x_type/y_type or head_type/tail_type columns")

    out = (
        e.groupby(["x_type", "y_type"]).agg(
            edge_count=("relation", "size"),
            num_relations=("relation", "nunique"),
            relations=("relation", lambda x: ", ".join(sorted(set(x)))),
        )
    ).reset_index()

    return out.sort_values("edge_count", ascending=False).reset_index(drop=True)

# %%
def target_relation_analysis(
    df: DataFrame,
    src_type: str = "disease",
    dst_type: str = "gene/protein",
    relation: str | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """Deep-dive on disease-protein (or any selected) target edges.

    Should report:
    - total_edges, unique_src, unique_dst
    - src_degree_stats, dst_degree_stats
    - cold_start_src_lte2, cold_start_src_zero
    - top_src_hubs, top_dst_hubs

    Focus:
    - Prioritization-target diagnostics (coverage, sparsity, hubs).
    """
    e = df[["x_index", "y_index", "relation", "x_type", "y_type"]]

    sub = e[(e["x_type"] == src_type) & (e["y_type"] == dst_type)].copy()
    display(sub.head())
    if relation is not None:
        sub = sub[sub["relation"] == relation].copy()

    src_deg = sub.groupby("x_index").size().rename("degree")
    dst_deg = sub.groupby("y_index").size().rename("degree")

    return {
        "src_type": src_type,
        "dst_type": dst_type,
        "relation": relation,
        "total_edges": int(sub.shape[0]),
        "unique_src": int(sub["x_index"].nunique()),
        "unique_dst": int(sub["y_index"].nunique()),
        # "src_degree_stats": _degree_stats_from_counts(src_deg),
        # "dst_degree_stats": _degree_stats_from_counts(dst_deg),
        "cold_start_src_lte2": int((src_deg <= 2).sum()) if not src_deg.empty else 0,
        "cold_start_src_zero": int(
            len(set(e.loc[e["x_type"] == src_type, "x_index"].unique()) - set(src_deg.index.tolist()))
        ),
        "top_src_hubs": [(k, int(v)) for k, v in src_deg.sort_values(ascending=False).head(top_k).items()],
        "top_dst_hubs": [(k, int(v)) for k, v in dst_deg.sort_values(ascending=False).head(top_k).items()],
    }

# ez most még elég szar

# %%
def metapath_schema_analysis(
    df: DataFrame,
    src_type: str = "disease",
    dst_type: str = "gene/protein",
    max_hops: int = 3,
) -> DataFrame:
    """Enumerate valid schema metapaths between two node types.

    Should include:
    - metapath_types (type sequence)
    - relation_sequence
    - hop_length

    Focus:
    - Discover message-passing routes available to multi-hop models.
    """
    e = df[["x_index", "y_index", "relation", "x_type", "y_type"]]
    if max_hops < 1:
        return pd.DataFrame(columns=["metapath_types", "relation_sequence", "hop_length"])

    schema_edges = e[["x_type", "relation", "y_type"]].drop_duplicates()
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for h_t, rel, t_t in schema_edges.itertuples(index=False, name=None):
        adjacency.setdefault(h_t, []).append((str(rel), str(t_t)))

    rows: list[dict[str, Any]] = []

    def dfs(cur_type: str, path_types: list[str], rel_seq: list[str], depth: int) -> None:
        if depth > max_hops:
            return
        if depth >= 1 and cur_type == dst_type:
            rows.append(
                {
                    "metapath_types": " -> ".join(path_types),
                    "relation_sequence": " -> ".join(rel_seq),
                    "hop_length": depth,
                }
            )
        if depth == max_hops:
            return
        for rel, nxt in adjacency.get(cur_type, []):
            dfs(nxt, path_types + [nxt], rel_seq + [rel], depth + 1)

    dfs(src_type, [src_type], [], 0)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.drop_duplicates().sort_values(["hop_length", "metapath_types"]).reset_index(drop=True)


def metapath_reachability(
    df: DataFrame,
    src_type: str = "disease",
    dst_type: str = "gene/protein",
    max_hops: int = 2,
    max_sources: int | None = 1000,
) -> DataFrame:
    """Estimate source-node reachability to destination type by hop count.

    Should return one row per hop with:
    - hop
    - reachable_sources
    - total_sources
    - reachable_fraction

    Focus:
    - Quantify how much extra neighborhood signal appears at 2/3 hops.
    """
    e = df[["x_index", "y_index", "relation", "x_type", "y_type"]]
    if "x_type" not in e.columns or "y_type" not in e.columns:
        raise ValueError("metapath_reachability requires x_type/y_type columns")
    if max_hops < 1:
        return pd.DataFrame(columns=["hop", "reachable_sources", "total_sources", "reachable_fraction"])

    src_nodes = e.loc[e["x_type"] == src_type, "x_index"].drop_duplicates().tolist()
    dst_nodes = set(e.loc[e["y_type"] == dst_type, "y_index"].drop_duplicates().tolist())

    if max_sources is not None and len(src_nodes) > max_sources:
        src_nodes = src_nodes[:max_sources]

    adjacency: dict[str, list[str]] = {}
    for h, t in e[["x_index", "y_index"]].itertuples(index=False, name=None):
        adjacency.setdefault(h, []).append(t)

    reach_by_hop = {h: 0 for h in range(1, max_hops + 1)}
    total_sources = len(src_nodes)

    for s in src_nodes:
        visited = {s}
        frontier = {s}
        first_hit_hop: int | None = None

        for hop in range(1, max_hops + 1):
            next_frontier: set[str] = set()
            for cur in frontier:
                for nxt in adjacency.get(cur, []):
                    if nxt not in visited:
                        visited.add(nxt)
                        next_frontier.add(nxt)

            if next_frontier & dst_nodes and first_hit_hop is None:
                first_hit_hop = hop

            frontier = next_frontier
            if not frontier:
                break

        if first_hit_hop is not None:
            for hop in range(first_hit_hop, max_hops + 1):
                reach_by_hop[hop] += 1

    rows = []
    for hop in range(1, max_hops + 1):
        reachable = int(reach_by_hop[hop])
        frac = float(reachable / total_sources) if total_sources > 0 else 0.0
        rows.append(
            {
                "hop": hop,
                "reachable_sources": reachable,
                "total_sources": int(total_sources),
                "reachable_fraction": frac,
            }
        )
    return pd.DataFrame(rows)


def power_law_fit(degree_series: pd.Series, min_degree: int = 1) -> dict[str, Any]:
    """Fit power-law trend on degree distribution in log-log space.

    Should report:
    - exponent (alpha)
    - fit_slope, fit_intercept
    - r_squared
    - num_points_used

    Focus:
    - Describe heavy-tail tendency, not to prove strict power-law validity.
    """
    vals = pd.Series(degree_series).dropna().astype(float)
    vals = vals[vals >= float(min_degree)]
    if vals.empty:
        return {
            "exponent": np.nan,
            "fit_slope": np.nan,
            "fit_intercept": np.nan,
            "r_squared": np.nan,
            "num_points_used": 0,
        }

    counts = vals.value_counts().sort_index()
    x = np.log10(counts.index.to_numpy(dtype=float))
    y = np.log10(counts.to_numpy(dtype=float))

    if x.size < 2:
        return {
            "exponent": np.nan,
            "fit_slope": np.nan,
            "fit_intercept": np.nan,
            "r_squared": np.nan,
            "num_points_used": int(x.size),
        }

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "exponent": float(-slope),
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "r_squared": float(r2),
        "num_points_used": int(x.size),
    }


def power_law_summary(degrees_df: DataFrame) -> DataFrame:
    """Run power-law fit globally and per node type.

    Should include:
    - scope (global or type)
    - exponent
    - r_squared
    - sample_size

    Focus:
    - Compact report table for thesis-ready interpretation.
    """
    if "total_degree" not in degrees_df.columns:
        raise ValueError("power_law_summary expects a DataFrame with a 'total_degree' column")

    rows: list[dict[str, Any]] = []
    global_fit = power_law_fit(degrees_df["total_degree"])
    rows.append(
        {
            "scope": "global",
            "exponent": global_fit["exponent"],
            "r_squared": global_fit["r_squared"],
            "sample_size": int(degrees_df.shape[0]),
            "num_points_used": global_fit["num_points_used"],
        }
    )

    if "node_type" in degrees_df.columns:
        for ntype, grp in degrees_df.groupby("node_type", dropna=False):
            fit = power_law_fit(grp["total_degree"])
            rows.append(
                {
                    "scope": str(ntype),
                    "exponent": fit["exponent"],
                    "r_squared": fit["r_squared"],
                    "sample_size": int(grp.shape[0]),
                    "num_points_used": fit["num_points_used"],
                }
            )

    return pd.DataFrame(rows)


def feature_readiness(
    edges_df: DataFrame,
    features_df: DataFrame,
    feature_key_col: str = "node_index",
    feature_type_col: str | None = None,
) -> dict[str, Any]:
    """Evaluate how well external feature rows can join to KG entities.

    Should report:
    - feature_rows
    - joinable_rows
    - coverage_ratio
    - unmatched_feature_keys
    - per_type_coverage (if type info available)

    Focus:
    - Data readiness check before model training.
    """
    nodes = extract_nodes_df(edges_df)
    if feature_key_col not in features_df.columns:
        raise ValueError(f"features_df is missing key column: {feature_key_col}")

    feature_keys = features_df[feature_key_col].dropna().astype(str)
    node_ids = set(nodes["node_id"].astype(str).tolist())

    joinable_mask = feature_keys.isin(node_ids)
    joinable_rows = int(joinable_mask.sum())
    feature_rows = int(feature_keys.shape[0])
    unmatched = sorted(feature_keys[~joinable_mask].drop_duplicates().tolist())

    out: dict[str, Any] = {
        "feature_rows": feature_rows,
        "joinable_rows": joinable_rows,
        "coverage_ratio": float(joinable_rows / feature_rows) if feature_rows > 0 else 0.0,
        "unmatched_feature_keys": unmatched,
    }

    if feature_type_col is not None and feature_type_col in features_df.columns and "node_type" in nodes.columns:
        ft = features_df[[feature_key_col, feature_type_col]].copy()
        ft[feature_key_col] = ft[feature_key_col].astype(str)

        per_type = []
        for t, grp in ft.groupby(feature_type_col, dropna=False):
            keys_t = grp[feature_key_col]
            node_ids_t = set(nodes.loc[nodes["node_type"] == str(t), "node_id"].astype(str).tolist())
            if not node_ids_t:
                cov = 0.0
                join_t = 0
            else:
                join_t = int(keys_t.isin(node_ids_t).sum())
                cov = float(join_t / max(len(keys_t), 1))
            per_type.append(
                {
                    "node_type": str(t),
                    "feature_rows": int(len(keys_t)),
                    "joinable_rows": join_t,
                    "coverage_ratio": cov,
                }
            )
        out["per_type_coverage"] = per_type

    return out

# %%

# all_nodes = extract_all_nodes(df)
# # %%
# # len, duplicates, unique idx, name stb nulls?
# # "index" = "id", "name", "source"
# deduplicated = all_nodes.drop_duplicates(subset=["index"])
# deduplicated2 = all_nodes.drop_duplicates(subset=["id", "name", "source"])

# cols = ["index", "type", "id", "name", "source"]
# d1 = deduplicated[cols].sort_values(cols).reset_index(drop=True)
# d2 = deduplicated2[cols].sort_values(cols).reset_index(drop=True)

# only_in_d1 = d1.merge(d2, on=cols, how="left", indicator=True).query('_merge == "left_only"').drop(columns=["_merge"])
# only_in_d2 = d2.merge(d1, on=cols, how="left", indicator=True).query('_merge == "left_only"').drop(columns=["_merge"])
# display(only_in_d1)
# display(only_in_d2)


# deduplicated.isna().sum() # 0


# %%
def inspect_key_consistency(df: DataFrame) -> dict[str, DataFrame]:
    all_nodes = extract_all_nodes(df)
    # deduplicated = all_nodes.drop_duplicates(subset=["index"])
    # deduplicated2 = all_nodes.drop_duplicates(subset=["id", "name", "source"])

    summary_rows = []
    for col in all_nodes.columns:
        s = all_nodes[col]
        summary_rows.append(
            {
                "column": col,
                "rows": int(len(s)),
                "nulls": int(s.isna().sum()),
                "nunique": int(s.nunique()),
                "duplicates": int(len(s) - s.nunique()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    display(summary)

    # If y_index is a key, each y_index should map to exactly one y_id and one y_name.
    idx_to_id_n = all_nodes.groupby("index", dropna=False)["id"].nunique(dropna=False).rename("id_per_index")
    idx_to_name_n = all_nodes.groupby("index", dropna=False)["name"].nunique(dropna=False).rename("name_per_index")
    idx_conflicts = pd.concat([idx_to_id_n, idx_to_name_n], axis=1).reset_index()
    idx_conflicts = idx_conflicts[(idx_conflicts["id_per_index"] > 1) | (idx_conflicts["name_per_index"] > 1)]

    # Reverse check: does one id/name appear with multiple indexes?
    id_to_idx_n = all_nodes.groupby("id", dropna=False)["index"].nunique(dropna=False).rename("index_per_id")
    id_conflicts = id_to_idx_n[id_to_idx_n > 1].reset_index()
    display(id_conflicts)

    name_to_idx_n = all_nodes.groupby("name", dropna=False)["index"].nunique(dropna=False).rename("index_per_name")
    name_conflicts = name_to_idx_n[name_to_idx_n > 1].reset_index()
    display(name_conflicts)


# x and y are interchangeble bc all rel is bidirectional

# inspect_key_consistency(df)