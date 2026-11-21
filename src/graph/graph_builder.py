"""Directed IP communication graph (NetworkX).

Nodes are IP addresses; a directed edge u -> v exists when u initiated at least
one flow to v. Edge attributes: ``flows`` (communication frequency),
``bytes``, ``packets``, ``ports`` (distinct destination ports), ``first_seen``,
``last_seen``.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_graph(flows: pd.DataFrame) -> nx.DiGraph:
    """``flows`` uses the host-aggregator schema (src_ip, dst_ip, dst_port, start_time, bytes, packets)."""
    g = nx.DiGraph()
    if flows.empty:
        return g
    agg = flows.groupby(["src_ip", "dst_ip"], sort=False).agg(
        flows=("dst_port", "size"), bytes=("bytes", "sum"), packets=("packets", "sum"),
        ports=("dst_port", "nunique"), first_seen=("start_time", "min"), last_seen=("start_time", "max"),
    ).reset_index()
    for row in agg.itertuples(index=False):
        g.add_edge(row.src_ip, row.dst_ip, flows=int(row.flows), bytes=float(row.bytes),
                   packets=float(row.packets), ports=int(row.ports),
                   first_seen=float(row.first_seen), last_seen=float(row.last_seen))
    return g


def sampled_clustering(und: nx.Graph, nodes, max_neighbors: int = 300, seed: int = 42) -> dict:
    """Local clustering coefficient (share of neighbour pairs that are connected).

    Exact for nodes with <= ``max_neighbors`` neighbours. For larger hubs the
    coefficient is estimated on a uniform random sample of ``max_neighbors``
    neighbours (an unbiased estimate of the connected-pair share), because the
    exact computation is O(deg^2) and hub IPs here have tens of thousands of peers.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for v in nodes:
        neigh = [u for u in und[v] if u != v]
        if len(neigh) < 2:
            out[v] = 0.0
            continue
        if len(neigh) > max_neighbors:
            neigh = list(rng.choice(np.array(neigh, dtype=object), max_neighbors, replace=False))
        nset = set(neigh)
        links = sum(len(nset.intersection(und[u])) for u in neigh) / 2
        k = len(neigh)
        out[v] = links / (k * (k - 1) / 2)
    return out


def node_metrics(g: nx.DiGraph, betweenness_k: int | None = 500, seed: int = 42,
                 closeness_max_nodes: int = 20000, nodes=None) -> pd.DataFrame:
    """Centrality metrics per node (all nodes, or only ``nodes`` if given).

    Betweenness is approximated with ``k`` sampled pivot nodes on large graphs
    (exact when the graph has <= k nodes). PageRank is computed on the whole
    graph. Clustering uses :func:`sampled_clustering`. Closeness uses the
    Wasserman-Faust correction for disconnected graphs and is skipped (NaN)
    above ``closeness_max_nodes`` nodes because it is O(V*E).
    """
    if g.number_of_nodes() == 0:
        return pd.DataFrame()
    n = g.number_of_nodes()
    targets = list(g.nodes) if nodes is None else [v for v in nodes if v in g]
    k = None if betweenness_k is None or n <= betweenness_k else betweenness_k
    und = g.to_undirected(as_view=True)
    pagerank = nx.pagerank(g, weight="flows")
    betweenness = nx.betweenness_centrality(g, k=k, seed=seed, normalized=True)
    norm = 1.0 / (n - 1) if n > 1 else 0.0
    metrics = pd.DataFrame({
        "in_degree": {v: g.in_degree(v) for v in targets},
        "out_degree": {v: g.out_degree(v) for v in targets},
        "weighted_in_degree": {v: g.in_degree(v, weight="flows") for v in targets},
        "weighted_out_degree": {v: g.out_degree(v, weight="flows") for v in targets},
        "degree_centrality": {v: (g.in_degree(v) + g.out_degree(v)) * norm for v in targets},
        "betweenness_centrality": {v: betweenness[v] for v in targets},
        "pagerank": {v: pagerank[v] for v in targets},
        "clustering_coefficient": sampled_clustering(und, targets, seed=seed),
    })
    if n <= closeness_max_nodes:
        metrics["closeness_centrality"] = pd.Series({v: nx.closeness_centrality(g, v, wf_improved=True)
                                                     for v in targets})
    else:
        metrics["closeness_centrality"] = np.nan
    metrics.index.name = "ip"
    return metrics.reset_index()


def subgraph_around(g: nx.DiGraph, nodes, hops: int = 1, max_nodes: int = 300) -> nx.DiGraph:
    """Bounded neighbourhood subgraph used for plotting."""
    centre = [v for v in nodes if v in g]
    keep, frontier = set(centre), set(centre)
    for _ in range(hops):
        nxt = set()
        for v in frontier:
            nxt.update(g.successors(v))
            nxt.update(g.predecessors(v))
        frontier = nxt - keep
        keep |= nxt
    if len(keep) > max_nodes:
        ranked = sorted(keep - set(centre), key=lambda v: g.degree(v, weight="flows"), reverse=True)
        keep = set(centre) | set(ranked[: max(0, max_nodes - len(centre))])
    return g.subgraph(keep).copy()
