"""Classifier wrappers: RF, XGBoost, LightGBM, CatBoost, MLP, metrics."""
import numpy as np
import pytest

from src.models import catboost_model, lightgbm_model, random_forest, xgboost_model
from src.models.mlp import MLPClassifier, MLPConfig
from src.utils.metrics import classification_metrics, detection_metrics, per_class_report


@pytest.fixture(scope="module")
def imbalanced():
    rng = np.random.default_rng(0)
    n = [2000, 150, 40]                                        # imbalanced 3-class problem
    x = np.vstack([rng.normal(i * 2.5, 1.0, size=(k, 6)) for i, k in enumerate(n)]).astype(np.float32)
    y = np.concatenate([[i] * k for i, k in enumerate(n)])
    idx = rng.permutation(len(y))
    x, y = x[idx], y[idx]
    cut = int(0.7 * len(y))
    return x[:cut], y[:cut], x[cut:], y[cut:]


@pytest.mark.parametrize("module", [random_forest, xgboost_model, lightgbm_model, catboost_model])
def test_tree_models_fit_predict(module, imbalanced):
    xtr, ytr, xva, yva = imbalanced
    params = {"n_estimators": 50} if module in (random_forest, xgboost_model, lightgbm_model) else {"iterations": 50}
    if module is random_forest:
        params["class_weight"] = "balanced"
        model = module.build(params, n_jobs=2)
        module.fit(model, xtr, ytr)
    else:
        model = module.build(params, n_jobs=2, n_classes=3)
        module.fit(model, xtr, ytr, xva, yva)
    proba = model.predict_proba(xva)
    assert proba.shape == (len(yva), 3) and np.allclose(proba.sum(1), 1, atol=1e-5)
    m = classification_metrics(yva, proba.argmax(1), proba, ["BENIGN", "A", "B"])
    assert m["f1_macro"] > 0.8 and "roc_auc_ovr_macro" in m and "binary_view_pr_auc" in m


def test_feature_mismatch_raises(imbalanced):
    xtr, ytr, xva, _ = imbalanced
    model = lightgbm_model.build({"n_estimators": 10}, n_classes=3)
    lightgbm_model.fit(model, xtr, ytr)
    with pytest.raises(Exception):
        model.predict_proba(xva[:, :4])


def test_mlp_trains_saves_loads(tmp_path, imbalanced):
    xtr, ytr, xva, yva = imbalanced
    model = MLPClassifier(MLPConfig(max_epochs=8, patience=3, batch_size=128)).fit(xtr, ytr, xva, yva,
                                                                                   log=lambda *a: None)
    assert model.history.best_epoch >= 1 and len(model.history.val_loss) >= model.history.best_epoch
    acc = (model.predict(xva) == yva).mean()
    assert acc > 0.9
    model.save(tmp_path)
    loaded = MLPClassifier.load(tmp_path)
    assert np.allclose(loaded.predict_proba(xva), model.predict_proba(xva), atol=1e-6)
    with pytest.raises(RuntimeError):
        MLPClassifier().predict_proba(xva)


def test_metric_definitions():
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    p = np.array([0, 0, 0, 1, 1, 0, 2, 2])
    m = classification_metrics(y, p, None, ["BENIGN", "A", "B"])
    assert m["fpr"] == pytest.approx(0.25)                      # 1 of 4 benign flagged
    assert m["fnr"] == pytest.approx(0.25)                      # 1 of 4 attacks missed
    rep = per_class_report(y, p, ["BENIGN", "A", "B"]).set_index("class")
    assert rep.loc["B", "recall"] == 1.0 and rep.loc["A", "precision"] == 0.5
    d = detection_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert d["fpr"] == 0.5 and d["recall"] == 1.0 and d["roc_auc"] == 1.0
