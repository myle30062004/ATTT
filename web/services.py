"""Analysis services behind the web demo.

Everything here reuses the project's own code path (pcap_parser -> FlowBuilder ->
feature_extractor -> DetectionPipeline), so the demo shows exactly what the
offline/realtime system does - no separate "demo model".
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.dataset_loader import FAMILY_OF, normalise_label
from src.data_processing.feature_extractor import CIC_FEATURE_COLUMNS
from src.detection.decision_engine import SEVERITIES, FlowVerdict
from src.detection.detection_pipeline import DetectionPipeline
from src.utils.config import PROJECT_ROOT, load_config

CFG = load_config()
WEB = Path(__file__).resolve().parent
SAMPLES_DIR = WEB / "samples"
CACHE_DIR = WEB / "cache"
CACHE_DIR.mkdir(exist_ok=True)
SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}
MAX_ALERT_FLOWS = 3000          # flagged flows turned into full alerts (explanations are per-row)
MAX_FLOW_ROWS = 400             # flow rows returned to the browser
MAX_UPLOAD_PACKETS = 3_000_000  # safety cap for uploaded PCAPs

_pipeline: DetectionPipeline | None = None
_pipe_lock = threading.Lock()


def pipeline() -> DetectionPipeline:
    global _pipeline
    with _pipe_lock:
        if _pipeline is None:
            _pipeline = DetectionPipeline(PROJECT_ROOT / "models" / "deployment", CFG)
        return _pipeline


# ------------------------------------------------------------------ samples
def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def list_samples() -> list[dict]:
    out = []
    manifest = SAMPLES_DIR / "samples.json"
    if not manifest.exists():
        return out
    for s in json.loads(manifest.read_text(encoding="utf-8")):
        path = _resolve(s["path"])
        if path.exists():
            out.append({**s, "size_mb": round(path.stat().st_size / 2**20, 1)})
    return out


def sample_path(sample_id: str) -> tuple[Path, dict]:
    for s in list_samples():
        if s["id"] == sample_id:
            return _resolve(s["path"]), s
    raise KeyError(sample_id)


# ------------------------------------------------------------------ jobs
class Job:
    def __init__(self, kind: str, name: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.name = kind, name
        self.status, self.progress, self.message = "queued", 0.0, "Đang chờ"
        self.result: dict | None = None
        self.error: str | None = None
        self.started = time.time()

    def update(self, progress: float, message: str) -> None:
        self.progress, self.message = round(min(max(progress, 0.0), 1.0), 3), message

    def to_dict(self, with_result: bool = True) -> dict:
        d = {"id": self.id, "kind": self.kind, "name": self.name, "status": self.status, "progress": self.progress,
             "message": self.message, "error": self.error, "elapsed_s": round(time.time() - self.started, 1)}
        if with_result and self.status == "done":
            d["result"] = self.result
        return d


JOBS: dict[str, Job] = {}


def start_job(kind: str, name: str, fn, *args, cache_key: str | None = None) -> Job:
    job = Job(kind, name)
    JOBS[job.id] = job
    cache = CACHE_DIR / f"{cache_key}.json" if cache_key else None
    model_mtime = (PROJECT_ROOT / "models" / "deployment" / "bundle.joblib").stat().st_mtime

    def run():
        try:
            if cache and cache.exists() and cache.stat().st_mtime > model_mtime:
                job.result = json.loads(cache.read_text(encoding="utf-8"))
                job.result["from_cache"] = True
            else:
                job.status = "running"
                job.result = fn(job, *args)
                if cache:
                    cache.write_text(json.dumps(job.result, ensure_ascii=False), encoding="utf-8")
            job.status, job.progress, job.message = "done", 1.0, "Hoàn tất"
        except Exception as exc:  # report to the UI instead of dying silently
            job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
            job.message = "Lỗi"
            traceback.print_exc()

    threading.Thread(target=run, daemon=True).start()
    return job


# ------------------------------------------------------------------ helpers
def _f(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, nd)


def _proto(p) -> str:
    return {6: "TCP", 17: "UDP", 1: "ICMP"}.get(int(p) if pd.notna(p) else -1, str(p))


def _group_alerts(alerts: list) -> list[dict]:
    """Aggregate identical alerts (same src, dst, type, module) like AlertManager suppression."""
    groups: dict[tuple, dict] = {}
    for a in alerts:
        d = a.to_dict()
        key = (d["src_ip"], d["dst_ip"], d["attack_type"], d["detection_module"])
        g = groups.get(key)
        if g is None:
            d["count"] = 1
            d["first_seen"] = d["last_seen"] = d["timestamp"]
            d["confidence"] = _f(d["confidence"])
            d["anomaly_score"] = _f(d.get("anomaly_score"), 2)
            groups[key] = d
        else:
            g["count"] += 1
            g["last_seen"] = max(g["last_seen"], d["timestamp"])
            g["first_seen"] = min(g["first_seen"], d["timestamp"])
            if SEV_RANK[d["severity"]] > SEV_RANK[g["severity"]]:
                g["severity"] = d["severity"]
    out = sorted(groups.values(), key=lambda g: (-SEV_RANK[g["severity"]], -g["count"]))
    for g in out:
        g["relevant_features"] = {k: (v if not isinstance(v, float) else _f(v)) for k, v in
                                  (g.get("relevant_features") or {}).items()}
    return out[:500]


def _timeline(times: pd.Series, labels: pd.Series, anomalies: pd.Series, bins: int = 60) -> dict:
    t = pd.to_numeric(times, errors="coerce")
    if t.notna().sum() < 2 or t.max() == t.min():
        t = pd.Series(np.arange(len(labels)), index=labels.index, dtype=float)
        unit = "flow"
    else:
        unit = "time"
    edges = np.linspace(t.min(), t.max() + 1e-9, bins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "label": labels.to_numpy(), "anom": anomalies.to_numpy().astype(int)})
    series = {lab: np.bincount(df.loc[df.label == lab, "bin"], minlength=bins).tolist()
              for lab in sorted(df.label.unique())}
    return {"unit": unit, "start": _f(edges[0], 3), "end": _f(edges[-1], 3),
            "bin_seconds": _f((edges[-1] - edges[0]) / bins, 3), "series": series,
            "anomalies": np.bincount(df.loc[df.anom == 1, "bin"], minlength=bins).tolist()}


def _ground_truth(frame: pd.DataFrame) -> pd.Series | None:
    if "Family" in frame:
        return frame["Family"].astype(str)
    if "Label" in frame:
        def fam(v):
            try:
                return FAMILY_OF.get(normalise_label(v), "?")
            except KeyError:
                return "BENIGN" if str(v).strip().upper() == "BENIGN" else "?"
        return frame["Label"].map(fam)
    return None


def _evaluate(truth: pd.Series, pred: pd.Series, classes: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    mask = truth.isin(classes)
    t, p = truth[mask].to_numpy(), pred[mask].to_numpy()
    present = [c for c in classes if (t == c).any() or (p == c).any()]
    cm = confusion_matrix(t, p, labels=present)
    prec, rec, f1, sup = precision_recall_fscore_support(t, p, labels=present, zero_division=0)
    ta, pa = t != "BENIGN", p != "BENIGN"
    benign = (~ta).sum()
    return {
        "n": int(mask.sum()), "labels": present, "confusion": cm.tolist(),
        "accuracy": _f(accuracy_score(t, p)),
        "macro_f1": _f(f1_score(t, p, labels=[c for c in present if (t == c).any()], average="macro", zero_division=0)),
        "fpr": _f((pa & ~ta).sum() / benign) if benign else None,
        "fnr": _f((~pa & ta).sum() / ta.sum()) if ta.sum() else None,
        "per_class": [{"class": c, "precision": _f(prec[i]), "recall": _f(rec[i]), "f1": _f(f1[i]),
                       "support": int(sup[i])} for i, c in enumerate(present)],
    }


# ------------------------------------------------------------------ core analysis
def analyze_frame(job: Job, frame: pd.DataFrame, source: dict) -> dict:
    pipe = pipeline()
    t0 = time.perf_counter()
    has_meta = all(c in frame for c in ("Src IP", "Dst IP", "Start Time"))
    job.update(0.55, f"Chấm điểm {len(frame):,} flow (GMM + LightGBM)")
    pipe.skipped_short_flows = 0
    scorable = pipe.scorable(frame)
    scores = pipe.score_flows(scorable)
    skipped = pipe.skipped_short_flows
    infer_s = time.perf_counter() - t0

    flagged = scores.index[scores["is_anomaly"].to_numpy() | (scores["predicted_family"] != "BENIGN").to_numpy()]
    # explanations are built row by row: keep the most confident attack flows + anomalies
    if len(flagged) > MAX_ALERT_FLOWS:
        pri = scores.loc[flagged].assign(_a=lambda d: (d.predicted_family != "BENIGN").astype(int))
        flagged = pri.sort_values(["_a", "confidence", "anomaly_score"], ascending=False).index[:MAX_ALERT_FLOWS]
    job.update(0.75, f"Sinh cảnh báo cho {len(flagged):,} flow đáng chú ý")
    sub = scorable.loc[flagged]
    if not has_meta:
        sub = sub.assign(**{"Src IP": "?", "Dst IP": "?", "Start Time": 0.0})
    flow_alerts, suspects, _ = pipe.process_flows(sub) if len(sub) else ([], set(), None)

    host_alerts, report = [], {"beacons": pd.DataFrame(), "host_scores": pd.DataFrame(), "topology": {}}
    if has_meta and len(frame):
        job.update(0.85, "Phân tích mức host: quét, beaconing, đồ thị")
        host_alerts, report = pipe.analyze_hosts(pipe.host_flows_from_frame(frame), suspects)

    job.update(0.95, "Tổng hợp kết quả")
    fam_counts = scores["predicted_family"].value_counts().to_dict()
    alerts = flow_alerts + host_alerts
    sev_counts = Counter(a.severity for a in alerts)
    type_counts = Counter(a.attack_type for a in alerts)

    # flows table: flagged first, then a few benign ones for context
    order = list(flagged[: MAX_FLOW_ROWS - 50]) + [i for i in scores.index[:200] if i not in set(flagged)][:50]
    rows = []
    for i in order:
        r, s = scorable.loc[i], scores.loc[i]
        rows.append({
            "src": str(r.get("Src IP", "")), "dst": str(r.get("Dst IP", "")),
            "sport": int(r["Src Port"]) if "Src Port" in r and pd.notna(r["Src Port"]) else None,
            "dport": int(r["Dst Port"]) if pd.notna(r.get("Dst Port")) else None,
            "proto": _proto(r.get("Protocol", 0)),
            "pkts": int(r["Tot Fwd Pkts"] + r["Tot Bwd Pkts"]),
            "bytes": int(r["TotLen Fwd Pkts"] + r["TotLen Bwd Pkts"]),
            "dur_s": _f(r["Flow Duration"] / 1e6, 3),
            "pred": s["predicted_family"], "conf": _f(s["confidence"], 3),
            "anom": bool(s["is_anomaly"]), "score": _f(s["anomaly_score"], 1),
            "truth": str(r["_truth"]) if "_truth" in r else None,
        })

    top_src = []
    if has_meta and len(scorable):
        g = scorable.assign(_pred=scores["predicted_family"], _anom=scores["is_anomaly"]).groupby("Src IP")
        agg = pd.DataFrame({"flows": g.size(), "attack_flows": g["_pred"].apply(lambda s: int((s != "BENIGN").sum())),
                            "anomalies": g["_anom"].sum(), "dsts": g["Dst IP"].nunique(),
                            "bytes": g["TotLen Fwd Pkts"].sum() + g["TotLen Bwd Pkts"].sum(),
                            "top_family": g["_pred"].agg(lambda s: s.value_counts().index[0])})
        agg = agg.sort_values(["attack_flows", "anomalies", "flows"], ascending=False).head(15)
        top_src = [{"ip": ip, **{k: (int(v) if isinstance(v, (int, np.integer, float, np.floating)) and k != "top_family"
                                     else v) for k, v in row.items()}} for ip, row in agg.iterrows()]

    beacons = report["beacons"]
    beacon_rows = []
    if isinstance(beacons, pd.DataFrame) and not beacons.empty:
        b = beacons.sort_values(["is_beacon", "beacon_score"], ascending=False).head(20)
        beacon_rows = [{"src": r.src_ip, "dst": r.dst_ip, "port": int(r.dst_port), "n": int(r.n_connections),
                        "interval": _f(r.interval_median, 1), "cv": _f(r.cv, 3), "robust_cv": _f(r.robust_cv, 3),
                        "is_beacon": bool(r.is_beacon)} for r in b.itertuples(index=False)]
    hosts = report["host_scores"]
    host_rows = []
    if isinstance(hosts, pd.DataFrame) and not hosts.empty:
        cols = [c for c in ("src_ip", "total_connections", "unique_dst_ips", "unique_dst_ports", "failed_ratio",
                            "dst_port_entropy", "botnet_score") if c in hosts]
        hs = hosts.sort_values("botnet_score" if "botnet_score" in hosts else "unique_dst_ips", ascending=False)
        host_rows = [{c: (_f(v, 3) if not isinstance(v, str) else v) for c, v in r.items()}
                     for r in hs[cols].head(15).to_dict("records")]
    topo = report.get("topology") or {}
    topo = {"verdict": topo.get("verdict"), "votes": topo.get("votes"), "reason": topo.get("reason")} if topo else {}

    truth_eval = None
    if "_truth" in scorable:
        truth_eval = _evaluate(scorable["_truth"], scores["predicted_family"], pipe.class_names)

    times = scorable["Start Time"] if "Start Time" in scorable else pd.Series(np.arange(len(scorable)),
                                                                                  index=scorable.index)
    return {
        "source": source,
        "summary": {
            "flows_total": int(len(frame)), "flows_scored": int(len(scorable)), "flows_skipped_short": int(skipped),
            "attack_flows": int((scores["predicted_family"] != "BENIGN").sum()),
            "anomalies": int(scores["is_anomaly"].sum()), "alerts": len(alerts),
            "hosts": int(frame["Src IP"].nunique()) if has_meta else None,
            "inference_ms": round(infer_s * 1000, 1),
            "us_per_flow": round(infer_s / max(len(scorable), 1) * 1e6, 1),
            "has_host_analysis": has_meta,
        },
        "family_counts": {k: int(v) for k, v in fam_counts.items()},
        "severity_counts": {s: sev_counts.get(s, 0) for s in SEVERITIES if sev_counts.get(s)},
        "type_counts": dict(type_counts.most_common()),
        "timeline": _timeline(times, scores["predicted_family"], scores["is_anomaly"]),
        "alert_groups": _group_alerts(alerts),
        "flows": rows,
        "top_sources": top_src,
        "beacons": beacon_rows,
        "hosts": host_rows,
        "topology": topo,
        "evaluation": truth_eval,
        "missing_features": sorted(pipe.missing_feature_warnings),
    }


def analyze_pcap(job: Job, path: Path, source: dict) -> dict:
    from scripts.process_pcap import pcap_to_flows

    size = path.stat().st_size
    job.update(0.02, "Đọc gói tin và dựng flow hai chiều")
    seen = {"n": 0}

    def log(msg, flush=False):  # pcap_to_flows reports every 2M packets
        seen["n"] += 2_000_000
        job.update(min(0.5, 0.05 + 0.45 * seen["n"] * 90 / max(size, 1)), msg.strip())

    t0 = time.perf_counter()
    flows = pcap_to_flows(path, max_packets=MAX_UPLOAD_PACKETS, log=log)
    stats = flows.attrs.get("parse_stats", {})
    parse_s = time.perf_counter() - t0
    result = analyze_frame(job, flows, source)
    result["summary"].update({
        "packets": stats.get("packets") or stats.get("decoded"), "parse_stats": stats,
        "parse_seconds": round(parse_s, 1),
        "capture_span_s": _f(flows["Start Time"].max() - flows["Start Time"].min(), 1) if len(flows) else 0,
    })
    return result


def read_flow_csv(path: Path) -> pd.DataFrame:
    """CICFlowMeter CSV (CSE-CIC-IDS2018 column names). Non-numeric cells -> NaN, header repeats dropped."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "Dst Port" in df:
        df = df[df["Dst Port"].astype(str) != "Dst Port"]
    for c in CIC_FEATURE_COLUMNS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "Timestamp" in df and "Start Time" not in df:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
        if ts.notna().any():
            df["Start Time"] = ts.astype("int64") / 1e9
    for c in ("Src IP", "Dst IP"):
        if c in df:
            df[c] = df[c].astype(str)
    return df.reset_index(drop=True)


def analyze_csv(job: Job, path: Path, source: dict) -> dict:
    job.update(0.1, "Đọc CSV CICFlowMeter")
    frame = read_flow_csv(path)
    missing = [c for c in pipeline().cleaner.kept_columns if c not in frame]
    if missing:
        raise ValueError(f"CSV thiếu {len(missing)} cột đặc trưng CICFlowMeter, ví dụ: {', '.join(missing[:6])}")
    truth = _ground_truth(frame)
    if truth is not None:
        frame["_truth"] = truth.to_numpy()
    return analyze_frame(job, frame, source)


# ------------------------------------------------------------------ single flow
_flow_pool: pd.DataFrame | None = None


def flow_pool() -> pd.DataFrame:
    """Small stratified pool of real test-split flows (never used for training or model selection)."""
    global _flow_pool
    if _flow_pool is None:
        cache = CACHE_DIR / "flow_pool.parquet"
        if cache.exists():
            _flow_pool = pd.read_parquet(cache)
        else:
            df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "split_test.parquet")
            parts = [g.sample(min(len(g), 400), random_state=42) for _, g in df.groupby("Family")]
            _flow_pool = pd.concat(parts).reset_index(drop=True)
            _flow_pool.to_parquet(cache, index=False)
    return _flow_pool


def random_flow(family: str | None) -> dict:
    pool = flow_pool()
    if family and family != "any":
        pool = pool[pool["Family"] == family]
    row = pool.sample(1).iloc[0]
    pipe = pipeline()
    return {"family": row["Family"], "fine_label": row.get("FineLabel"),
            "features": {c: _f(row[c], 6) for c in pipe.cleaner.kept_columns},
            "model_features": pipe.clf_features}


def predict_flow(features: dict) -> dict:
    pipe = pipeline()
    frame = pd.DataFrame([{c: float(features.get(c, 0.0) or 0.0) for c in pipe.cleaner.kept_columns}])
    scores = pipe.score_flows(frame)
    clean = scores.attrs["clean"]
    xc = clean[pipe.clf_features]
    proba = pipe.clf.predict_proba(xc)[0]
    k = int(proba.argmax())
    contrib = pipe.clf.booster_.predict(xc.to_numpy(np.float64), pred_contrib=True)[0]
    nf = len(pipe.clf_features)
    cls_contrib = contrib[k * (nf + 1): (k + 1) * (nf + 1) - 1] if len(contrib) > nf + 1 else contrib[:nf]
    top = np.argsort(-np.abs(cls_contrib))[:10]
    xg = scores.attrs["gauss_input"]
    variances = pipe.gauss.var if hasattr(pipe.gauss, "var") else np.diag(pipe.gauss.cov)
    z = ((xg - pipe.gauss.mu) / np.sqrt(variances))[0] if hasattr(pipe.gauss, "mu") else np.zeros(len(pipe.gauss_features))
    ztop = np.argsort(-np.abs(z))[:6]
    verdict = FlowVerdict(is_anomaly=bool(scores["is_anomaly"].iat[0]), anomaly_score=float(scores["anomaly_score"].iat[0]),
                          predicted_family=pipe.class_names[k], confidence=float(proba[k]))
    alert, to_host = pipe.engine.decide_flow(verdict, {"timestamp": "-", "src_ip": "?", "dst_ip": "?", "src_port": 0,
                                                        "dst_port": int(frame["Dst Port"].iat[0]), "protocol": 6})
    return {
        "probabilities": {c: _f(p) for c, p in zip(pipe.class_names, proba)},
        "predicted": pipe.class_names[k], "confidence": _f(proba[k]),
        "anomaly_score": _f(scores["anomaly_score"].iat[0], 2),
        "anomaly_threshold": _f(-pipe.gauss.log_epsilon, 2),
        "is_anomaly": bool(scores["is_anomaly"].iat[0]),
        "contributions": [{"feature": pipe.clf_features[i], "value": _f(clean[pipe.clf_features[i]].iat[0], 4),
                           "contribution": _f(cls_contrib[i])} for i in top],
        "anomaly_features": [{"feature": pipe.gauss_features[i], "z": _f(z[i], 2)} for i in ztop],
        "decision": {"alert": alert.to_dict() if alert else None, "to_host_analysis": bool(to_host)},
    }


# ------------------------------------------------------------------ results
def _read_csv(p: Path) -> list[dict] | None:
    if not p.exists():
        return None
    return json.loads(pd.read_csv(p).to_json(orient="records"))


def model_results() -> dict:
    exp, met = PROJECT_ROOT / "reports" / "experiments", PROJECT_ROOT / "reports" / "metrics"
    names = {"lr": "Logistic Regression", "dt": "Decision Tree", "rf": "Random Forest", "xgb": "XGBoost",
             "lgbm": "LightGBM", "cat": "CatBoost", "mlp": "MLP"}
    out: dict = {"validation": {}, "test": {}}
    for task in ("multiclass", "binary"):
        p = exp / f"training_summary_{task}.json"
        if p.exists():
            d = json.loads(p.read_text())
            out["validation"][task] = [{
                "model": names.get(k, k), "key": k, "train_seconds": _f(v.get("train_seconds"), 1),
                "size_mb": _f(v.get("model_size_mb"), 2), "us_per_flow": _f(v.get("val_inference_us_per_flow"), 2),
                **{m: _f(v["val_metrics"].get(m)) for m in ("accuracy", "f1_macro", "f1_weighted", "recall_macro",
                                                             "fpr", "fnr", "pr_auc", "roc_auc", "binary_view_pr_auc")}}
                for k, v in d.items() if "val_metrics" in v]
        out["test"][task] = _read_csv(met / f"model_comparison_{task}.csv")
    out["per_class_val"] = _read_csv(met / "report_val_per_class.csv")
    a = met / "anomaly_results.json"
    if a.exists():
        d = json.loads(a.read_text())
        out["anomaly"] = [{"detector": k, "roc_auc": _f(v.get("f1_optimal", {}).get("roc_auc")),
                           "pr_auc": _f(v.get("f1_optimal", {}).get("pr_auc")),
                           **{f"{op}_{m}": _f(v.get(op, {}).get(m)) for op in ("fpr_1pct", "fpr_5pct", "f1_optimal")
                              for m in ("recall", "precision", "fpr")}}
                          for k, v in d.items() if not k.startswith("_")]
        out["gmm_k"] = d.get("_gmm_component_selection")
    out["lofo"] = _read_csv(met / "unknown_attack_lofo.csv")
    out["botnet"] = _read_csv(met / "botnet_detector_comparison.csv")
    out["topology"] = _read_csv(met / "topology_simulation_confusion.csv")
    dep = met / "deployment_model_metrics.json"
    out["deployment"] = json.loads(dep.read_text()) if dep.exists() else None
    for key, fname in (("repeats", "repeats_summary.csv"), ("temporal", "temporal_vs_random_split.csv"),
                       ("ablation", "feature_ablation.csv")):
        p = exp / fname
        out[key] = _read_csv(p) if key != "repeats" else (p.read_text() if p.exists() else None)
    out["figures"] = sorted(p.name for p in (PROJECT_ROOT / "reports" / "figures").rglob("*.png"))
    return out


def model_info() -> dict:
    pipe = pipeline()
    return {"meta": pipe.meta, "class_names": pipe.class_names, "classifier_features": pipe.clf_features,
            "gaussian_features": pipe.gauss_features, "anomaly_threshold": _f(-pipe.gauss.log_epsilon, 2),
            "host_model": pipe.host_model is not None, "uncertain_threshold": pipe.engine.tau
            if hasattr(pipe.engine, "tau") else None}
