"""CTU-13 botnet analysis (scenario 1 / capture 42, Neris) from PCAP-reconstructed flows.

    python scripts/process_pcap.py <ctu pcap> --part-dir data/processed/ctu13_s42_flows   # once
    python scripts/botnet_analysis.py

Ground truth is host-level (see config ``ctu13`` and docs/DATASET_DOCUMENTATION.md):
flows initiated by the infected host are botnet traffic, flows initiated by the
verified normal hosts are normal; every other host is *background* (unlabelled,
never used as negative ground truth).

Experiments
1. Host-window dataset: host features (+ per-window graph features) for every
   labelled internal host in 5-minute windows; chronological 70/15/15 split.
2. Detectors compared on test windows: Gaussian (multivariate, fitted on normal
   windows), Isolation Forest, Random Forest (host features), graph/beaconing
   heuristic rules (fixed a priori), Random Forest (host + graph features).
3. Beaconing analysis per (src, dst, port) for the infected and normal hosts.
4. Communication graph figures (normal host vs. infected host neighbourhood).
5. Topology identification: (a) real capture; (b) *simulation* – synthetic
   centralized / P2P bot groups injected into the real background graph, with
   known ground truth, to measure identification accuracy of the analyser.
"""
from __future__ import annotations

import ipaddress
import json
import sys
import time
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing.host_aggregator import HOST_FEATURES, flows_from_cic_frame, host_features  # noqa: E402
from src.data_processing.preprocessing import LogScaler  # noqa: E402
from src.graph.beaconing_detector import detect_beaconing, merge_bursts  # noqa: E402
from src.graph.graph_builder import build_graph, node_metrics, subgraph_around  # noqa: E402
from src.graph.topology_analyzer import analyze_topology  # noqa: E402
from src.models.gaussian_detector import MultivariateGaussianDetector, select_epsilon  # noqa: E402
from src.utils import visualization as viz  # noqa: E402
from src.utils.config import load_config, project_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metrics import detection_metrics  # noqa: E402

CFG = load_config()
SEED = CFG["project"]["seed"]
CTU = CFG["ctu13"]
FIG = project_path("figures")
METRICS = project_path("metrics")
OUT_MODELS = project_path("models") / "botnet"
LOG = get_logger("botnet", project_path("logs") / "botnet.log")
FLOW_DIR = project_path("processed") / "ctu13_s42_flows"
GRAPH_FEATURES = ["out_degree", "in_degree", "weighted_out_degree", "pagerank", "betweenness_centrality",
                  "clustering_coefficient"]
COLUMNS = ["Src IP", "Dst IP", "Dst Port", "Protocol", "Start Time", "Flow Duration", "Tot Fwd Pkts",
           "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "SYN Without ACK"]


def load_flows() -> pd.DataFrame:
    frame = pd.concat([pd.read_parquet(p, columns=COLUMNS) for p in sorted(FLOW_DIR.glob("part-*.parquet"))],
                      ignore_index=True)
    flows = flows_from_cic_frame(frame)
    net = ipaddress.ip_network(CTU["internal_network"])
    internal = {ip for ip in flows["src_ip"].unique() if _in(ip, net)}
    flows["src_internal"] = flows["src_ip"].isin(internal)
    infected, normal = set(CTU["infected_hosts"]), set(CTU["normal_hosts"])
    flows["label"] = np.select([flows["src_ip"].isin(infected), flows["src_ip"].isin(normal)],
                               ["Botnet", "Normal"], "Background")
    return flows


def _in(ip: str, net) -> bool:
    try:
        return ipaddress.ip_address(ip) in net
    except ValueError:
        return False


def host_window_dataset(flows: pd.DataFrame, window_s: float) -> pd.DataFrame:
    t0 = flows["start_time"].min()
    flows = flows.assign(window=((flows["start_time"] - t0) // window_s).astype(int))
    parts = []
    labelled = set(CTU["infected_hosts"]) | set(CTU["normal_hosts"])
    for w, g in flows.groupby("window", sort=True):
        src_sel = g[g["src_internal"]]
        hf = host_features(src_sel[src_sel["src_ip"].isin(labelled) | (src_sel["src_ip"].map(
            src_sel["src_ip"].value_counts()) >= 20)])
        if hf.empty:
            continue
        graph = build_graph(g)
        nm = node_metrics(graph, betweenness_k=32, closeness_max_nodes=0, nodes=hf["src_ip"].tolist()).set_index("ip")
        hf = hf.join(nm[GRAPH_FEATURES], on="src_ip")
        # out-degree relative to other internal hosts in the same window (graph heuristic input)
        od = hf["out_degree"]
        hf["out_degree_z"] = (od - od.median()) / (od.std() + 1e-9)
        hf["window"] = w
        hf["window_start"] = t0 + w * window_s
        parts.append(hf)
        if w % 10 == 0:
            LOG.info("  window %d: %d flows, %d hosts, graph %d nodes", w, len(g), len(hf), graph.number_of_nodes())
    data = pd.concat(parts, ignore_index=True)
    infected, normal = set(CTU["infected_hosts"]), set(CTU["normal_hosts"])
    data["label"] = np.select([data["src_ip"].isin(infected), data["src_ip"].isin(normal)],
                              ["Botnet", "Normal"], "Background")
    return data


def heuristic_rule(df: pd.DataFrame, beacon_hosts_by_window: dict) -> np.ndarray:
    """A-priori rules (not tuned on test): fan-out outlier, scanning-like failures, or beaconing."""
    fanout = df["out_degree_z"] > 3.0
    failing = (df["failed_ratio"] > 0.5) & (df["unique_dst_ips"] >= 20)
    beacon = np.array([s in beacon_hosts_by_window.get(w, set()) for s, w in zip(df["src_ip"], df["window"])])
    return (fanout | failing | beacon).astype(int)


def evaluate_detectors(data: pd.DataFrame, flows: pd.DataFrame, window_s: float) -> tuple[pd.DataFrame, dict]:
    lab = data[data["label"] != "Background"].copy()
    lab["y"] = (lab["label"] == "Botnet").astype(int)
    windows = np.sort(lab["window"].unique())
    a, b = windows[int(len(windows) * 0.70)], windows[int(len(windows) * 0.85)]
    tr, va, te = lab[lab["window"] < a], lab[(lab["window"] >= a) & (lab["window"] < b)], lab[lab["window"] >= b]
    split_info = {n: {"windows": int(d["window"].nunique()), "rows": int(len(d)), "botnet_rows": int(d["y"].sum())}
                  for n, d in (("train", tr), ("val", va), ("test", te))}
    LOG.info("host-window split: %s", split_info)
    results = {}
    # beaconing per window for heuristic
    t0 = flows["start_time"].min()
    beacon_hosts = {}
    cfgb = CFG["botnet"]
    win = ((flows["start_time"] - t0) // window_s).astype(int)
    for w, g in flows[flows["src_internal"]].groupby(win):
        bc = detect_beaconing(g, min_connections=cfgb["beacon_min_connections"], cv_threshold=cfgb["beacon_cv_threshold"],
                              min_period_s=cfgb["beacon_min_period_s"], min_span_s=60,
                              whitelist_ports=tuple(cfgb["beacon_whitelist_ports"]))
        beacon_hosts[w] = set(bc.loc[bc["is_beacon"], "src_ip"]) if not bc.empty else set()
    # Gaussian (normal windows only)
    sc = LogScaler("standard").fit(tr.loc[tr["y"] == 0, HOST_FEATURES])
    mvg = MultivariateGaussianDetector(1e-2).fit(sc.transform(tr.loc[tr["y"] == 0, HOST_FEATURES]))
    eps = select_epsilon(mvg.log_density(sc.transform(va[HOST_FEATURES])), va["y"].to_numpy())
    s = mvg.log_density(sc.transform(te[HOST_FEATURES]))
    results["Gaussian (multivariate, host features)"] = detection_metrics(te["y"], s < eps["log_epsilon"], -s)
    iso = IsolationForest(n_estimators=200, random_state=SEED).fit(sc.transform(tr.loc[tr["y"] == 0, HOST_FEATURES]))
    thr = select_epsilon(iso.score_samples(sc.transform(va[HOST_FEATURES])), va["y"].to_numpy())
    s = iso.score_samples(sc.transform(te[HOST_FEATURES]))
    results["Isolation Forest (host features)"] = detection_metrics(te["y"], s < thr["log_epsilon"], -s)
    trva = pd.concat([tr, va])
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(trva[HOST_FEATURES], trva["y"])
    p = rf.predict_proba(te[HOST_FEATURES])[:, 1]
    results["Random Forest (host features)"] = detection_metrics(te["y"], p >= 0.5, p)
    results["Graph + beaconing heuristic rules"] = detection_metrics(te["y"], heuristic_rule(te, beacon_hosts))
    feats_g = HOST_FEATURES + GRAPH_FEATURES
    rfg = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rfg.fit(trva[feats_g].fillna(0), trva["y"])
    pg = rfg.predict_proba(te[feats_g].fillna(0))[:, 1]
    results["Random Forest (host + graph features)"] = detection_metrics(te["y"], pg >= 0.5, pg)
    # behaviour on unlabelled background hosts in the test period (alert rate, NOT a false-positive rate)
    bg = data[(data["label"] == "Background") & (data["window"] >= b)]
    alert_rates = {}
    if not bg.empty:
        alert_rates = {
            "Random Forest (host features)": float((rf.predict_proba(bg[HOST_FEATURES])[:, 1] >= 0.5).mean()),
            "Random Forest (host + graph features)": float((rfg.predict_proba(bg[feats_g].fillna(0))[:, 1] >= 0.5).mean()),
            "Graph + beaconing heuristic rules": float(heuristic_rule(bg, beacon_hosts).mean()),
            "n_background_host_windows": int(len(bg)),
        }
    importance = pd.Series(rfg.feature_importances_, index=feats_g).sort_values(ascending=False)
    importance.to_csv(METRICS / "botnet_rf_graph_importance.csv", header=["importance"])
    viz.importance_plot(importance, "Host + graph features: Random Forest importance (CTU-13)",
                        FIG / "botnet_feature_importance.png")
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    rf_full = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf_full.fit(lab[HOST_FEATURES], lab["y"])
    joblib.dump({"model": rf_full, "features": HOST_FEATURES, "window_s": window_s}, OUT_MODELS / "host_model.joblib")
    table = pd.DataFrame(results).T
    return table, {"split": split_info, "background_alert_rate": alert_rates}


def beacon_analysis(flows: pd.DataFrame) -> pd.DataFrame:
    labelled = flows[flows["label"] != "Background"]
    cfgb = CFG["botnet"]
    res = detect_beaconing(labelled, min_connections=cfgb["beacon_min_connections"],
                           cv_threshold=cfgb["beacon_cv_threshold"], min_period_s=cfgb["beacon_min_period_s"],
                           whitelist_ports=tuple(cfgb["beacon_whitelist_ports"]))
    res["label"] = np.where(res["src_ip"].isin(CTU["infected_hosts"]), "Botnet", "Normal")
    res.to_csv(METRICS / "ctu13_beaconing_pairs.csv", index=False)
    return res


def beacon_timeline_plot(flows: pd.DataFrame, beacons: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=False)
    for ax, label in zip(axes, ("Botnet", "Normal")):
        cand = beacons[beacons["label"] == label]
        if cand.empty:
            ax.set_title(f"{label}: no candidate pair with enough connections", loc="left")
            continue
        top = cand.iloc[0]
        g = flows[(flows["src_ip"] == top.src_ip) & (flows["dst_ip"] == top.dst_ip) & (flows["dst_port"] == top.dst_port)]
        times = merge_bursts(np.sort(g["start_time"].to_numpy()), 1.0)
        rel = (times - times[0]) / 60
        ax.vlines(rel, 0, 1, color=viz.CATEGORICAL[0 if label == "Botnet" else 2], linewidth=1)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Minutes since first connection")
        ax.set_title(f"{label} host {top.src_ip} -> {top.dst_ip}:{int(top.dst_port)}  n={int(top.n_connections)}, "
                     f"median interval {top.interval_median:.1f}s, CV={top.cv:.2f}, beacon={bool(top.is_beacon)}",
                     loc="left", fontsize=10)
    fig.tight_layout()
    viz.save(fig, FIG / "botnet_beaconing_timeline.png")


def graph_figures(flows: pd.DataFrame) -> tuple[dict, nx.DiGraph]:
    import matplotlib.pyplot as plt
    g = build_graph(flows)
    out = {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()}
    bot = CTU["infected_hosts"][0]
    normal_counts = flows[flows["label"] == "Normal"]["src_ip"].value_counts()
    normal = normal_counts.index[0] if len(normal_counts) else None
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, host, label in ((axes[0], normal, "Normal host"), (axes[1], bot, "Infected host (Neris)")):
        if host is None or host not in g:
            ax.set_title(f"{label}: not present", loc="left")
            continue
        sub = subgraph_around(g, [host], hops=1, max_nodes=150)
        pos = nx.spring_layout(sub, seed=SEED, k=0.35)
        w = np.array([sub[u][v]["flows"] for u, v in sub.edges()], dtype=float)
        nx.draw_networkx_edges(sub, pos, ax=ax, width=0.4 + 1.6 * w / max(w.max(), 1), edge_color=viz.GRID,
                               arrows=False)
        others = [n for n in sub if n != host]
        nx.draw_networkx_nodes(sub, pos, nodelist=others, node_size=14, node_color=viz.CATEGORICAL[0], ax=ax)
        nx.draw_networkx_nodes(sub, pos, nodelist=[host], node_size=140,
                               node_color=viz.CATEGORICAL[1] if host == bot else viz.CATEGORICAL[2], ax=ax)
        ax.set_title(f"{label} {host}\n{g.out_degree(host)} destinations, {g.in_degree(host)} sources "
                     f"(150 busiest neighbours shown)", loc="left", fontsize=10)
        ax.axis("off")
        out[label] = {"ip": host, "out_degree": g.out_degree(host), "in_degree": g.in_degree(host)}
    viz.save(fig, FIG / "botnet_graph_normal_vs_infected.png")
    return out, g


def topology_simulation(g: nx.DiGraph, flows: pd.DataFrame, trials: int = 60) -> dict:
    """SIMULATION: inject bot groups with known topology into the real background graph."""
    rng = np.random.default_rng(SEED)
    internal = flows.loc[flows["src_internal"] & (flows["label"] == "Background"), "src_ip"].value_counts()
    hosts = internal[internal >= 50].index.to_numpy()
    # population baseline: all active internal sources; popularity is measured on the full graph g
    # because h only keeps the suspects' neighbourhood
    population = set(flows.loc[flows["src_internal"], "src_ip"].unique())
    rows = []
    for trial in range(trials):
        for truth in ("centralized", "p2p", "none"):
            n_bots = int(rng.integers(3, 11))
            bots = list(rng.choice(hosts, n_bots, replace=False))
            # the analyser only inspects the suspects and their successors, so copying
            # that neighbourhood of the real graph is equivalent to copying the whole graph
            neigh = set(bots).union(*(set(g.successors(b)) for b in bots if b in g))
            h = g.subgraph(neigh).copy()
            if truth == "centralized":
                cnc = f"203.0.113.{trial % 250}"
                for bot in bots:
                    h.add_edge(bot, cnc, flows=int(rng.integers(50, 400)), bytes=0, packets=0, ports=1,
                               first_seen=0, last_seen=0)
            elif truth == "p2p":
                peers = [f"198.51.{i // 250}.{i % 250}" for i in range(400)]
                for bot in bots:
                    for p in rng.choice(peers, int(rng.integers(60, 150)), replace=False):
                        h.add_edge(bot, p, flows=int(rng.integers(1, 5)), bytes=0, packets=0, ports=1,
                                   first_seen=0, last_seen=0)
                    for other in rng.choice(bots, max(1, n_bots // 2), replace=False):
                        if other != bot:
                            h.add_edge(bot, other, flows=int(rng.integers(1, 5)), bytes=0, packets=0, ports=1,
                                       first_seen=0, last_seen=0)
            verdict = analyze_topology(h, bots, population=population, popularity_graph=g)["verdict"]
            rows.append({"trial": trial, "truth": truth, "n_bots": n_bots, "verdict": verdict})
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "topology_simulation_trials.csv", index=False)
    cm = pd.crosstab(df["truth"], df["verdict"])
    cm.to_csv(METRICS / "topology_simulation_confusion.csv")
    df["correct"] = ((df["truth"] == df["verdict"]) |
                     ((df["truth"] == "none") & ~df["verdict"].isin(["centralized", "p2p"])))
    return {"trials_per_class": trials, "accuracy": float(df["correct"].mean()),
            "accuracy_by_truth": df.groupby("truth")["correct"].mean().to_dict(), "confusion": cm.to_dict()}


def topology_example_figure(g: nx.DiGraph, flows: pd.DataFrame, n_bots: int = 6) -> None:
    """SIMULATION figure: one 'none', one centralized and one P2P group injected into the real graph.

    Only suspects and their *rare* destinations (contacted by <= 5% of other internal hosts) are
    drawn, which is the evidence the topology analyser uses.
    """
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEED + 1)
    internal = flows.loc[flows["src_internal"] & (flows["label"] == "Background"), "src_ip"].value_counts()
    hosts = internal[internal >= 50].index.to_numpy()
    population = set(flows.loc[flows["src_internal"], "src_ip"].unique())
    bots = list(rng.choice(hosts, n_bots, replace=False))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, truth in zip(axes, ("none", "centralized", "p2p")):
        neigh = set(bots).union(*(set(g.successors(b)) for b in bots))
        h = g.subgraph(neigh).copy()
        if truth == "centralized":
            for b in bots:
                h.add_edge(b, "203.0.113.10", flows=200, bytes=0, packets=0, ports=1, first_seen=0, last_seen=0)
        elif truth == "p2p":
            peers = [f"198.51.100.{i}" for i in range(60)]
            for b in bots:
                for p in rng.choice(peers, 25, replace=False):
                    h.add_edge(b, p, flows=2, bytes=0, packets=0, ports=1, first_seen=0, last_seen=0)
                for o in rng.choice(bots, 3, replace=False):
                    if o != b:
                        h.add_edge(b, o, flows=2, bytes=0, packets=0, ports=1, first_seen=0, last_seen=0)
        res = analyze_topology(h, bots, population=population, popularity_graph=g)
        pop_n = len(population)

        def rare(d):
            if d in bots:
                return True
            if d not in g:
                return True
            return sum(1 for u in g.predecessors(d) if u in population) / pop_n <= 0.05

        shared = {}
        for b in bots:
            for d in h.successors(b):
                if rare(d):
                    shared[d] = shared.get(d, 0) + 1
        keep = set(bots) | {d for d, c in shared.items() if c >= 2}
        if len(keep) < len(bots) + 40:  # add some non-shared rare destinations for context
            extra = [d for d, c in shared.items() if c == 1][: 40]
            keep |= set(extra)
        sub = h.subgraph(keep)
        pos = nx.spring_layout(sub, seed=SEED, k=0.6)
        nx.draw_networkx_edges(sub, pos, ax=ax, width=0.6, edge_color=viz.GRID, arrows=False)
        cnc = [n for n in sub if n == "203.0.113.10"]
        others = [n for n in sub if n not in bots and n not in cnc]
        nx.draw_networkx_nodes(sub, pos, nodelist=others, node_size=12, node_color=viz.CATEGORICAL[0], ax=ax)
        nx.draw_networkx_nodes(sub, pos, nodelist=bots, node_size=90, node_color=viz.CATEGORICAL[1], ax=ax)
        if cnc:
            nx.draw_networkx_nodes(sub, pos, nodelist=cnc, node_size=160, node_color=viz.CATEGORICAL[2], ax=ax)
        v = res["votes"]
        ax.set_title(f"Simulated: {truth}\nverdict: {res['verdict']}\nvotes: centralized {v['centralized']}, p2p {v['p2p']}",
                     loc="left", fontsize=10)
        ax.axis("off")
    fig.text(0.01, 0.01, "orange = 6 random real internal hosts acting as bots; green = injected C&C server; "
             "blue = their rare destinations (contacted by <= 5% of other internal hosts)",
             fontsize=9, color=viz.TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    viz.save(fig, FIG / "botnet_topology_examples_simulated.png")


def topology_only() -> None:
    """Re-run only the real-graph topology analysis and the simulation; update botnet_summary.json."""
    flows = load_flows()
    g = build_graph(flows)
    internal_sources = set(flows.loc[flows["src_internal"], "src_ip"].unique())
    path = METRICS / "botnet_summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary["topology_real"] = analyze_topology(g, CTU["infected_hosts"], population=internal_sources)
    summary["topology_simulation"] = topology_simulation(g, flows)
    path.write_text(json.dumps(summary, indent=2, default=str))
    LOG.info("topology simulation: %s", summary["topology_simulation"])


def main() -> None:
    if "--only" in sys.argv and "figures" in sys.argv:
        flows = load_flows()
        _, g = graph_figures(flows)
        topology_example_figure(g, flows)
        return
    if "--only" in sys.argv and "topology" in sys.argv:
        topology_only()
        return
    t0 = time.time()
    flows = load_flows()
    LOG.info("loaded %s flows in %.0fs", f"{len(flows):,}", time.time() - t0)
    summary = {"flows_total": int(len(flows)),
               "flows_by_label": flows["label"].value_counts().to_dict(),
               "flows_by_labelled_host": flows.loc[flows["label"] != "Background", "src_ip"].value_counts().to_dict(),
               "capture_start": float(flows["start_time"].min()), "capture_end": float(flows["start_time"].max())}
    labelled = flows[flows["label"] != "Background"]
    hosts_all = host_features(labelled)
    hosts_all["label"] = np.where(hosts_all["src_ip"].isin(CTU["infected_hosts"]), "Botnet", "Normal")
    hosts_all.to_csv(METRICS / "ctu13_host_features_full_capture.csv", index=False)
    bot_flows = labelled[labelled["label"] == "Botnet"]
    summary["infected_top_dst_ports"] = bot_flows["dst_port"].value_counts().head(10).to_dict()
    summary["infected_top_destinations"] = bot_flows["dst_ip"].value_counts().head(10).to_dict()
    window = CFG["botnet"]["host_window_s"]
    LOG.info("building host-window dataset ...")
    data = host_window_dataset(flows, window)
    data.to_parquet(project_path("processed") / "ctu13_host_windows.parquet", index=False)
    table, info = evaluate_detectors(data, flows, window)
    table.to_csv(METRICS / "botnet_detector_comparison.csv")
    summary.update(info)
    LOG.info("detector comparison\n%s", table[["precision", "recall", "f1", "fpr"]].to_string())
    beacons = beacon_analysis(flows)
    beacon_timeline_plot(flows, beacons)
    summary["beacon_pairs"] = beacons.groupby("label")["is_beacon"].agg(["sum", "count"]).to_dict()
    gstats, g = graph_figures(flows)
    summary["graph"] = gstats
    labelled_ips = CTU["infected_hosts"] + CTU["normal_hosts"]
    nm = node_metrics(g, betweenness_k=64, closeness_max_nodes=0, nodes=labelled_ips).set_index("ip")
    nm.to_csv(METRICS / "ctu13_node_metrics_labelled.csv")
    internal_sources = set(flows.loc[flows["src_internal"], "src_ip"].unique())
    summary["topology_real"] = analyze_topology(g, CTU["infected_hosts"], population=internal_sources)
    summary["topology_simulation"] = topology_simulation(g, flows)
    topology_example_figure(g, flows)
    (METRICS / "botnet_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    LOG.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
