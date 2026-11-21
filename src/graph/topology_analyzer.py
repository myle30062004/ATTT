"""Centralized vs. peer-to-peer communication structure among suspicious hosts.

A topology is never decided from a single metric, and shared destinations are
only evidence if they are *unusual for the rest of the network*: on a real
campus network almost every host talks to the same DNS resolver and popular
web services. First version of this analyser counted such destinations as
"controllers" and labelled random groups of normal hosts as centralized
botnets (see reports/metrics/topology_simulation_*), so destinations are now
compared against a population baseline (the idea behind correlation-based
detectors such as BotMiner).

Definitions for a suspect set S and a population P of other active sources:

* popularity(d) = |{u in P : u -> d}| / |P|; d is *rare* if popularity(d) <= popular_fraction.
* controllers  = rare destinations contacted by >= ``min_shared_fraction`` of S.
* shared peers = rare destinations contacted by >= 2 suspects (excluding controllers).
* intra links  = share of suspects with at least one edge to another suspect.
* rare overlap = mean Jaccard similarity of the suspects' rare destination sets.
* concentration = share of the suspects' flows to rare destinations that go to
  the single most contacted rare destination.

Votes:
  centralized: controllers exist | concentration >= 0.2 | no peer mesh
  p2p        : peer mesh (shared peers >= max(10, 2|S|)) | intra links >= 0.3 | rare overlap >= 0.05
A verdict needs >= 2 votes and more votes than the alternative; otherwise
``no_coordinated_structure`` (no controller and no mesh) or ``insufficient_evidence``.
"""
from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np


def gini(values) -> float:
    x = np.sort(np.asarray(list(values), dtype=np.float64))
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def analyze_topology(g: nx.DiGraph, suspects: list[str], population=None, popular_fraction: float = 0.05,
                     min_shared_fraction: float = 0.5, max_pairs: int = 2000,
                     popularity_graph: nx.DiGraph | None = None) -> dict:
    """``popularity_graph`` (default ``g``) is the graph on which destination
    popularity among ``population`` is measured, e.g. the full network graph when
    ``g`` is only the suspects' neighbourhood."""
    pg = g if popularity_graph is None else popularity_graph
    suspects = [s for s in dict.fromkeys(suspects) if s in g]
    result = {"n_suspects": len(suspects), "verdict": "insufficient_evidence", "indicators": {}, "votes": {}}
    if not suspects:
        return result
    sset = set(suspects)
    if population is None:
        population = [v for v in pg if pg.out_degree(v) > 0]
    pop = {v for v in population if v in pg} - sset
    n_pop = len(pop)

    def popularity(d) -> float:
        if n_pop == 0:
            return 0.0
        if d not in pg:
            return 0.0
        return sum(1 for u in pg.predecessors(d) if u in pop) / n_pop

    dest_sets = {s: set(g.successors(s)) - {s} for s in suspects}
    all_dests = set().union(*dest_sets.values())
    pop_cache = {d: popularity(d) for d in all_dests - sset}
    rare_sets = {s: {d for d in ds if d in sset or pop_cache.get(d, 0.0) <= popular_fraction}
                 for s, ds in dest_sets.items()}
    fan_in: dict = {}
    for ds in rare_sets.values():
        for d in ds:
            if d not in sset:
                fan_in[d] = fan_in.get(d, 0) + 1
    need = max(2, int(np.ceil(min_shared_fraction * len(suspects))))
    controllers = sorted((d for d, c in fan_in.items() if c >= need), key=lambda d: -fan_in[d]) \
        if len(suspects) >= 2 else []
    shared_peers = [d for d, c in fan_in.items() if c >= 2 and d not in controllers]
    rare_flows: dict = {}
    for s in suspects:
        for d in rare_sets[s]:
            if d not in sset:
                rare_flows[d] = rare_flows.get(d, 0) + g[s][d]["flows"]
    total_rare = sum(rare_flows.values())
    concentration = max(rare_flows.values()) / total_rare if total_rare else 0.0
    intra = float(np.mean([bool(dest_sets[s] & sset) for s in suspects]))
    pairs = list(combinations(suspects, 2))[:max_pairs]
    jacc = [len(rare_sets[a] & rare_sets[b]) / max(1, len(rare_sets[a] | rare_sets[b])) for a, b in pairs]
    rare_overlap = float(np.mean(jacc)) if jacc else 0.0
    mesh = len(shared_peers) >= max(10, 2 * len(suspects))
    ind = {
        "population_sources": n_pop,
        "popular_destinations_excluded": int(sum(1 for d in all_dests - sset if pop_cache.get(d, 0) > popular_fraction)),
        "controllers": controllers[:10],
        "controller_fan_in": {d: fan_in[d] for d in controllers[:10]},
        "shared_rare_peers": len(shared_peers),
        "rare_flow_concentration": concentration,
        "rare_flow_gini": gini(rare_flows.values()),
        "suspects_with_intra_links": intra,
        "rare_destination_jaccard": rare_overlap,
        "mean_out_degree": float(np.mean([len(v) for v in dest_sets.values()])),
    }
    central = int(bool(controllers)) + int(concentration >= 0.2) + int(not mesh)
    p2p = int(mesh) + int(intra >= 0.3) + int(rare_overlap >= 0.05)
    result["indicators"] = ind
    result["votes"] = {"centralized": central, "p2p": p2p}
    if len(suspects) == 1:
        result["verdict"] = "single_host_insufficient_evidence"
    elif not controllers and not mesh:
        result["verdict"] = "no_coordinated_structure"
    elif central >= 2 and central > p2p:
        result["verdict"] = "centralized"
    elif p2p >= 2 and p2p > central:
        result["verdict"] = "p2p"
    return result
