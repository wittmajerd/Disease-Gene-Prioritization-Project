from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd


# =====================================================================
# Helpers
# =====================================================================


def _edge_index_to_numpy(ei: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (src_array, dst_array) as int64 numpy arrays."""
    ei = ei.cpu()
    return ei[0].numpy().astype(np.int64), ei[1].numpy().astype(np.int64)


def _degree_stats(degrees: np.ndarray) -> dict[str, Any]:
    """Compute summary statistics for a degree array."""
    if len(degrees) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "p90": 0.0,
            "p99": 0.0,
        }
    return {
        "mean": float(np.mean(degrees)),
        "median": float(np.median(degrees)),
        "std": float(np.std(degrees)),
        "min": int(np.min(degrees)),
        "max": int(np.max(degrees)),
        "p90": float(np.percentile(degrees, 90)),
        "p99": float(np.percentile(degrees, 99)),
    }


# =====================================================================
# 1. Global statistics
# =====================================================================


def global_stats(raw: Any) -> dict[str, Any]:
    """Return high-level graph-wide statistics.

    Keys
    ----
    total_nodes, total_edges, num_node_types, num_edge_types,
    density, node_types, edge_types
    """
    num_nodes = _raw_num_nodes_dict(raw)
    edge_dict = _raw_edge_dict(raw)

    total_nodes = sum(num_nodes.values())
    total_edges = sum(int(ei.size(1)) for ei in edge_dict.values())
    density = total_edges / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0.0

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "num_node_types": len(num_nodes),
        "num_edge_types": len(edge_dict),
        "density": density,
        "node_types": list(num_nodes.keys()),
        "edge_types": [
            f"{s}--{r}--{d}" for (s, r, d) in edge_dict.keys()
        ],
    }


# =====================================================================
# 2. Connected-component analysis (homogeneous projection)
# =====================================================================


def connected_components(raw: Any) -> dict[str, Any]:
    """Compute weakly-connected-component stats on the full homogeneous
    projection of BioKG.

    Returns
    -------
    dict with keys: num_components, largest_component_size,
    smallest_component_size, component_sizes (sorted desc).
    """
    edge_index, total_nodes = build_full_homogeneous_graph(raw)

    # Union-Find
    parent = list(range(total_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    src, dst = _edge_index_to_numpy(edge_index)
    for s, d in zip(src, dst):
        union(int(s), int(d))

    comp_sizes: dict[int, int] = Counter(find(i) for i in range(total_nodes))
    sizes = sorted(comp_sizes.values(), reverse=True)

    return {
        "num_components": len(sizes),
        "largest_component_size": sizes[0] if sizes else 0,
        "smallest_component_size": sizes[-1] if sizes else 0,
        "component_sizes": sizes,
    }


# =====================================================================
# 3. Per node-type statistics
# =====================================================================


def _compute_node_degrees(raw: Any) -> dict[str, np.ndarray]:
    """Return {node_type: degree_array} across all relations."""
    num_nodes = _raw_num_nodes_dict(raw)
    edge_dict = _raw_edge_dict(raw)

    degrees: dict[str, np.ndarray] = {
        ntype: np.zeros(count, dtype=np.int64) for ntype, count in num_nodes.items()
    }

    for (src_type, _, dst_type), ei in edge_dict.items():
        s, d = _edge_index_to_numpy(ei)
        np.add.at(degrees[src_type], s, 1)
        if src_type != dst_type or not np.array_equal(s, d):
            np.add.at(degrees[dst_type], d, 1)

    return degrees


def node_type_stats(raw: Any) -> pd.DataFrame:
    """Per node-type summary: count, degree distribution stats, isolated
    nodes.

    Returns a DataFrame indexed by node type.
    """
    num_nodes = _raw_num_nodes_dict(raw)
    degrees = _compute_node_degrees(raw)

    rows = []
    for ntype in sorted(num_nodes.keys()):
        deg = degrees[ntype]
        stats = _degree_stats(deg)
        stats["node_type"] = ntype
        stats["count"] = num_nodes[ntype]
        stats["isolated"] = int(np.sum(deg == 0))
        rows.append(stats)

    df = pd.DataFrame(rows)
    df = df.set_index("node_type")
    col_order = ["count", "isolated", "mean", "median", "std", "min", "max", "p90", "p99"]
    return df[col_order]


def degree_distributions(raw: Any) -> dict[str, np.ndarray]:
    """Return raw degree arrays per node type (for custom plotting)."""
    return _compute_node_degrees(raw)


# =====================================================================
# 4. Per relation-type statistics
# =====================================================================


def relation_type_stats(raw: Any) -> pd.DataFrame:
    """Per edge/relation-type summary.

    Columns: src_type, dst_type, edge_count, density,
    src_degree_{mean,median,max}, dst_degree_{mean,median,max},
    relation_class, symmetry_ratio.
    """
    num_nodes = _raw_num_nodes_dict(raw)
    edge_dict = _raw_edge_dict(raw)

    rows = []
    for (src_type, rel, dst_type), ei in edge_dict.items():
        s, d = _edge_index_to_numpy(ei)
        n_src = num_nodes[src_type]
        n_dst = num_nodes[dst_type]
        n_edges = len(s)

        # Density
        density = n_edges / (n_src * n_dst) if n_src * n_dst > 0 else 0.0

        # Per-side degree arrays
        src_deg = np.bincount(s, minlength=n_src)
        dst_deg = np.bincount(d, minlength=n_dst)

        # RotatE-style relation classification
        # avg tails per head, avg heads per tail
        head_counts = Counter(s.tolist())
        tail_counts = Counter(d.tolist())
        avg_tails_per_head = np.mean(list(head_counts.values())) if head_counts else 0.0
        avg_heads_per_tail = np.mean(list(tail_counts.values())) if tail_counts else 0.0
        h_label = "1" if avg_tails_per_head < 1.5 else "N"
        t_label = "1" if avg_heads_per_tail < 1.5 else "N"
        rel_class = f"{h_label}-to-{t_label}"

        # Symmetry (only meaningful when src_type == dst_type)
        sym_ratio = np.nan
        if src_type == dst_type:
            edge_set = set(zip(s.tolist(), d.tolist()))
            reverse_hits = sum(1 for (a, b) in edge_set if (b, a) in edge_set)
            sym_ratio = reverse_hits / len(edge_set) if edge_set else 0.0

        rows.append(
            {
                "relation": rel,
                "src_type": src_type,
                "dst_type": dst_type,
                "edge_count": n_edges,
                "density": density,
                "src_deg_mean": float(np.mean(src_deg)),
                "src_deg_median": float(np.median(src_deg)),
                "src_deg_max": int(np.max(src_deg)),
                "dst_deg_mean": float(np.mean(dst_deg)),
                "dst_deg_median": float(np.median(dst_deg)),
                "dst_deg_max": int(np.max(dst_deg)),
                "avg_tails_per_head": avg_tails_per_head,
                "avg_heads_per_tail": avg_heads_per_tail,
                "relation_class": rel_class,
                "symmetry_ratio": sym_ratio,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("edge_count", ascending=False).reset_index(drop=True)
    return df


# =====================================================================
# 5. Target edge deep-dive
# =====================================================================


def target_edge_analysis(
    raw: Any,
    target_edge: tuple[str, str, str] | list[str] = ("disease", "disease-protein", "protein"),
) -> dict[str, Any]:
    """In-depth analysis of the prediction target edge type.

    Returns
    -------
    dict with keys:
        src_type, dst_type, total_edges,
        src_degree_stats, dst_degree_stats,
        src_degrees (array), dst_degrees (array),
        cold_start_src (count with degree <= 2),
        hub_dst_top20 (top-20 most connected dst nodes),
    """
    edge_dict = _raw_edge_dict(raw)
    num_nodes = _raw_num_nodes_dict(raw)

    src_type, rel, dst_type = target_edge

    # Find the matching key (handles sanitization differences)
    target_key = None
    for key in edge_dict:
        if key[0] == src_type and key[2] == dst_type and (key[1] == rel or key[1].replace("-", "_") == rel.replace("-", "_")):
            target_key = key
            break
    if target_key is None:
        raise KeyError(f"Target edge {target_edge} not found in raw data")

    ei = edge_dict[target_key]
    s, d = _edge_index_to_numpy(ei)
    n_src = num_nodes[src_type]
    n_dst = num_nodes[dst_type]

    src_deg = np.bincount(s, minlength=n_src)
    dst_deg = np.bincount(d, minlength=n_dst)

    # Hub destination nodes (e.g. proteins connected to many diseases)
    top_dst_idx = np.argsort(dst_deg)[::-1][:20]
    hub_dst = [(int(idx), int(dst_deg[idx])) for idx in top_dst_idx]

    # Cold-start source nodes (diseases with very few connections)
    cold_start_src = int(np.sum(src_deg <= 2))
    cold_start_src_zero = int(np.sum(src_deg == 0))

    return {
        "src_type": src_type,
        "dst_type": dst_type,
        "total_edges": len(s),
        "src_degree_stats": _degree_stats(src_deg),
        "dst_degree_stats": _degree_stats(dst_deg),
        "src_degrees": src_deg,
        "dst_degrees": dst_deg,
        "cold_start_src_lte2": cold_start_src,
        "cold_start_src_zero": cold_start_src_zero,
        "hub_dst_top20": hub_dst,
    }


# =====================================================================
# 6. Cross-type connectivity matrix
# =====================================================================


def connectivity_matrix(raw: Any) -> pd.DataFrame:
    """Node-type × node-type edge count matrix.

    Entry ``(i, j)`` is the total number of edges from type *i* to type *j*
    across all relation types.  Useful for a heatmap in the notebook.
    """
    num_nodes = _raw_num_nodes_dict(raw)
    edge_dict = _raw_edge_dict(raw)

    types = sorted(num_nodes.keys())
    mat = pd.DataFrame(0, index=types, columns=types, dtype=np.int64)

    for (src, _, dst), ei in edge_dict.items():
        mat.at[src, dst] = int(mat.at[src, dst]) + int(ei.size(1))  # type: ignore[arg-type]

    return mat


def connectivity_matrix_relations(raw: Any) -> pd.DataFrame:
    """Return a DataFrame listing every (src_type, relation, dst_type, count)
    tuple — for a detailed schema view with relation names.
    """
    edge_dict = _raw_edge_dict(raw)
    rows = []
    for (src, rel, dst), ei in edge_dict.items():
        rows.append(
            {"src_type": src, "relation": rel, "dst_type": dst, "count": int(ei.size(1))}
        )
    return pd.DataFrame(rows).sort_values(["src_type", "dst_type", "count"], ascending=[True, True, False]).reset_index(drop=True)


# =====================================================================
# 7. Metapath analysis
# =====================================================================


def _enumerate_metapaths(
    raw: Any,
    src_type: str,
    dst_type: str,
    max_length: int = 3,
) -> list[list[tuple[str, str, str]]]:
    """Enumerate all valid metapaths from *src_type* to *dst_type* up to
    *max_length* hops.  Returns a list of paths, where each path is a list
    of ``(src, rel, dst)`` tuples.
    """
    edge_dict = _raw_edge_dict(raw)

    # Build adjacency: for each node type, which (rel, dst_type) are reachable?
    adj: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for (s, r, d) in edge_dict.keys():
        adj[s].append((s, r, d))

    results: list[list[tuple[str, str, str]]] = []

    def dfs(current_type: str, path: list[tuple[str, str, str]], depth: int) -> None:
        if current_type == dst_type and len(path) > 0:
            results.append(list(path))
        if depth >= max_length:
            return
        for edge_key in adj[current_type]:
            _, _, next_type = edge_key
            path.append(edge_key)
            dfs(next_type, path, depth + 1)
            path.pop()

    dfs(src_type, [], 0)
    return results


def metapath_analysis(
    raw: Any,
    src_type: str = "disease",
    dst_type: str = "protein",
    max_length: int = 3,
) -> pd.DataFrame:
    """Enumerate metapaths from *src_type* to *dst_type* and report schema-level
    information.

    For each metapath, computes the number of intermediate edges and a readable
    path string.

    Parameters
    ----------
    raw : BioKG raw object
    src_type, dst_type : endpoint node types
    max_length : max number of hops

    Returns
    -------
    DataFrame with columns: path, length, edge_counts
        ``edge_counts`` lists the number of edges for each hop.
    """
    edge_dict = _raw_edge_dict(raw)
    metapaths = _enumerate_metapaths(raw, src_type, dst_type, max_length)

    rows = []
    seen = set()
    for mp in metapaths:
        path_str = " → ".join(f"{s}--[{r}]--{d}" for s, r, d in mp)
        if path_str in seen:
            continue
        seen.add(path_str)
        edge_counts = [int(edge_dict[key].size(1)) for key in mp]
        rows.append(
            {
                "path": path_str,
                "length": len(mp),
                "edge_counts": edge_counts,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("length").reset_index(drop=True)
    return df


def metapath_reachability(
    raw: Any,
    metapath: list[tuple[str, str, str]],
    max_pairs: int = 50_000,
) -> dict[str, Any]:
    """Estimate how many (src, dst) pairs are reachable via a specific metapath.

    For efficiency, only checks a random sample of source nodes when the
    source set is large.

    Parameters
    ----------
    raw : BioKG raw object
    metapath : list of (src, rel, dst) edge type tuples forming the path
    max_pairs : cap on the number of source nodes to sample

    Returns
    -------
    dict with reachable_pairs (estimated), sample_size, total_src_nodes
    """
    edge_dict = _raw_edge_dict(raw)
    num_nodes = _raw_num_nodes_dict(raw)

    src_type = metapath[0][0]
    n_src = num_nodes[src_type]

    # Build adjacency lists per hop
    hop_adj: list[dict[int, set[int]]] = []
    for key in metapath:
        ei = edge_dict[key]
        s, d = _edge_index_to_numpy(ei)
        adj: dict[int, set[int]] = defaultdict(set)
        for si, di in zip(s.tolist(), d.tolist()):
            adj[si].add(di)
        hop_adj.append(adj)

    # Sample source nodes
    if n_src > max_pairs:
        src_nodes = np.random.choice(n_src, size=max_pairs, replace=False)
    else:
        src_nodes = np.arange(n_src)

    reachable_count = 0
    for src_node in src_nodes:
        current = {int(src_node)}
        for adj in hop_adj:
            next_set: set[int] = set()
            for node in current:
                if node in adj:
                    next_set.update(adj[node])
            current = next_set
            if not current:
                break
        if current:
            reachable_count += 1

    return {
        "metapath": " → ".join(f"{s}--[{r}]--{d}" for s, r, d in metapath),
        "reachable_src_nodes": reachable_count,
        "sample_size": len(src_nodes),
        "total_src_nodes": n_src,
        "reachable_fraction": reachable_count / len(src_nodes) if len(src_nodes) > 0 else 0.0,
    }


# =====================================================================
# 8. Power-law analysis
# =====================================================================


def power_law_fit(degrees: np.ndarray, min_degree: int = 1) -> dict[str, Any]:
    """Fit a power-law exponent to a degree distribution via log-log OLS.

    Parameters
    ----------
    degrees : array of node degrees
    min_degree : minimum degree to include in the fit (avoid log(0))

    Returns
    -------
    dict with exponent (alpha), r_squared, log_degrees, log_counts
    (the last two are for plotting the log-log fit line).
    """
    mask = degrees >= min_degree
    filtered = degrees[mask]
    if len(filtered) == 0:
        return {"exponent": np.nan, "r_squared": np.nan}

    counts = Counter(filtered.tolist())
    deg_vals = np.array(sorted(counts.keys()), dtype=np.float64)
    freq_vals = np.array([counts[int(d)] for d in deg_vals], dtype=np.float64)

    log_deg = np.log10(deg_vals)
    log_freq = np.log10(freq_vals)

    # OLS on log-log
    A = np.vstack([log_deg, np.ones_like(log_deg)]).T
    result = np.linalg.lstsq(A, log_freq, rcond=None)
    slope, intercept = result[0]

    # R²
    predicted = slope * log_deg + intercept
    ss_res = np.sum((log_freq - predicted) ** 2)
    ss_tot = np.sum((log_freq - np.mean(log_freq)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "exponent": -slope,  # power-law exponent α (positive by convention)
        "r_squared": r_squared,
        "log_degrees": log_deg,
        "log_counts": log_freq,
        "fit_slope": slope,
        "fit_intercept": intercept,
    }


def power_law_summary(raw: Any) -> pd.DataFrame:
    """Fit power-law for each node type and overall.

    Returns a DataFrame with node_type, exponent, r_squared.
    """
    all_degrees = _compute_node_degrees(raw)

    rows = []
    # Per type
    for ntype in sorted(all_degrees.keys()):
        fit = power_law_fit(all_degrees[ntype])
        rows.append(
            {
                "node_type": ntype,
                "exponent": fit["exponent"],
                "r_squared": fit["r_squared"],
            }
        )

    # Overall
    combined = np.concatenate(list(all_degrees.values()))
    fit = power_law_fit(combined)
    rows.append(
        {
            "node_type": "OVERALL",
            "exponent": fit["exponent"],
            "r_squared": fit["r_squared"],
        }
    )

    return pd.DataFrame(rows).set_index("node_type")


# =====================================================================
# 9. Two-hop reachability for target edge
# =====================================================================


def two_hop_overlap(
    raw: Any,
    target_edge: tuple[str, str, str] | list[str] = ("disease", "disease-protein", "protein"),
    bridge_type: str = "protein",
    max_sample: int = 500,
) -> dict[str, Any]:
    """Check how many disease-protein pairs are reachable via 2-hop paths
    through protein-protein edges.

    For a sample of diseases, finds their direct protein neighbors, then
    checks how many *other* proteins are reachable via one protein-protein
    hop.

    Parameters
    ----------
    raw : BioKG raw object
    target_edge : the prediction target edge type
    bridge_type : intermediate node type for 2-hop paths
    max_sample : max diseases to sample

    Returns
    -------
    dict with avg_direct_neighbors, avg_2hop_reachable,
    expansion_factor, sample_size
    """
    edge_dict = _raw_edge_dict(raw)
    num_nodes = _raw_num_nodes_dict(raw)

    src_type, rel, dst_type = target_edge

    # Find target edge key
    target_key = None
    for key in edge_dict:
        if key[0] == src_type and key[2] == dst_type:
            if key[1] == rel or key[1].replace("-", "_") == rel.replace("-", "_"):
                target_key = key
                break
    if target_key is None:
        raise KeyError(f"Target edge {target_edge} not found")

    # Build disease→protein adjacency
    ei = edge_dict[target_key]
    s, d = _edge_index_to_numpy(ei)
    dis_to_prot: dict[int, set[int]] = defaultdict(set)
    for si, di in zip(s.tolist(), d.tolist()):
        dis_to_prot[si].add(di)

    # Build protein→protein adjacency (all protein-protein relations)
    prot_to_prot: dict[int, set[int]] = defaultdict(set)
    for (st, _, dt), ei_pp in edge_dict.items():
        if st == bridge_type and dt == bridge_type:
            sp, dp = _edge_index_to_numpy(ei_pp)
            for si, di in zip(sp.tolist(), dp.tolist()):
                prot_to_prot[si].add(di)
                prot_to_prot[di].add(si)  # treat as undirected

    # Sample diseases
    disease_ids = list(dis_to_prot.keys())
    if len(disease_ids) > max_sample:
        disease_ids = list(np.random.choice(disease_ids, size=max_sample, replace=False))

    direct_counts = []
    two_hop_counts = []

    for dis in disease_ids:
        direct = dis_to_prot[dis]
        direct_counts.append(len(direct))

        # 2-hop: proteins reachable via protein-protein from direct neighbors
        reachable = set()
        for prot in direct:
            if prot in prot_to_prot:
                reachable.update(prot_to_prot[prot])
        reachable -= direct  # exclude already-direct neighbors
        two_hop_counts.append(len(reachable))

    return {
        "avg_direct_neighbors": float(np.mean(direct_counts)),
        "avg_2hop_reachable": float(np.mean(two_hop_counts)),
        "expansion_factor": (
            float(np.mean(two_hop_counts)) / float(np.mean(direct_counts))
            if np.mean(direct_counts) > 0
            else 0.0
        ),
        "sample_size": len(disease_ids),
        "total_diseases_with_edges": len(dis_to_prot),
    }


# =====================================================================
# 10. Full analysis runner (convenience)
# =====================================================================


def run_full_analysis(raw: Any, target_edge: tuple[str, str, str] | list[str] | None = None) -> dict[str, Any]:
    """Run all analyses and return a dict of results.

    Useful for saving a complete snapshot to YAML/JSON.
    """
    if target_edge is None:
        target_edge = ("disease", "disease-protein", "protein")

    results: dict[str, Any] = {}

    print("Computing global stats...")
    results["global"] = global_stats(raw)

    print("Computing connected components...")
    cc = connected_components(raw)
    results["connected_components"] = {
        "num_components": cc["num_components"],
        "largest_component_size": cc["largest_component_size"],
        "smallest_component_size": cc["smallest_component_size"],
    }

    print("Computing node type stats...")
    results["node_types"] = node_type_stats(raw).to_dict(orient="index")

    print("Computing relation type stats...")
    results["relation_types"] = relation_type_stats(raw).to_dict(orient="records")

    print("Computing target edge analysis...")
    target_info = target_edge_analysis(raw, target_edge)
    results["target_edge"] = {
        k: v
        for k, v in target_info.items()
        if k not in ("src_degrees", "dst_degrees")  # skip large arrays
    }

    print("Computing connectivity matrix...")
    results["connectivity_matrix"] = connectivity_matrix(raw).to_dict()

    print("Computing metapath analysis...")
    src_type, _, dst_type = target_edge
    results["metapaths"] = metapath_analysis(raw, src_type, dst_type).to_dict(orient="records")

    print("Computing power-law summary...")
    results["power_law"] = power_law_summary(raw).to_dict(orient="index")

    print("Computing two-hop overlap...")
    results["two_hop_overlap"] = two_hop_overlap(raw, target_edge)

    print("Analysis complete.")
    return results
