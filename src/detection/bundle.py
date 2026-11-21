"""Deployment bundle: every artifact the detection pipeline needs, saved together."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib


def save_bundle(out_dir: str | Path, *, cleaner, classifier, classifier_features: list[str], class_names: list[str],
                gaussian, gaussian_scaler, gaussian_features: list[str], importances: dict | None = None,
                classifier_scaler=None, host_model=None, host_model_features: list[str] | None = None,
                meta: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if gaussian.log_epsilon is None:
        raise ValueError("Gaussian detector has no epsilon; select it on validation data first")
    missing = [f for f in classifier_features + gaussian_features if f not in cleaner.kept_columns]
    if missing:
        raise ValueError(f"features not produced by the cleaner: {missing}")
    bundle = {"cleaner": cleaner, "classifier": classifier, "classifier_features": list(classifier_features),
              "classifier_scaler": classifier_scaler, "class_names": list(class_names), "gaussian": gaussian,
              "gaussian_scaler": gaussian_scaler, "gaussian_features": list(gaussian_features),
              "importances": dict(importances or {}), "host_model": host_model,
              "host_model_features": list(host_model_features or [])}
    joblib.dump(bundle, out / "bundle.joblib")
    info = {"created_utc": datetime.now(timezone.utc).isoformat(), "class_names": list(class_names),
            "classifier": type(classifier).__name__, "classifier_features": list(classifier_features),
            "gaussian": type(gaussian).__name__, "gaussian_features": list(gaussian_features),
            "gaussian_log_epsilon": gaussian.log_epsilon, "has_host_model": host_model is not None,
            **(meta or {})}
    (out / "bundle.json").write_text(json.dumps(info, indent=2, default=str))
    return out
