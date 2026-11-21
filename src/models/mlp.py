"""Multilayer Perceptron (PyTorch) for flow classification.

Architecture: in -> 256 (ReLU, BN, Dropout) -> 128 (ReLU, BN, Dropout)
-> 64 (ReLU, Dropout) -> n_classes logits. Trained with CrossEntropyLoss
(optionally class-weighted), Adam, ReduceLROnPlateau and early stopping on
validation loss. Inputs must be scaled (see ``LogScaler``).
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden=(256, 128, 64), dropout: float = 0.3) -> None:
        super().__init__()
        h1, h2, h3 = hidden
        self.net = nn.Sequential(
            nn.Linear(n_features, h1), nn.ReLU(), nn.BatchNorm1d(h1), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.BatchNorm1d(h2), nn.Dropout(dropout),
            nn.Linear(h2, h3), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h3, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MLPConfig:
    hidden: tuple = (256, 128, 64)
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    max_epochs: int = 40
    patience: int = 6
    lr_patience: int = 2
    lr_factor: float = 0.5
    class_weighting: bool = True
    seed: int = 42


@dataclass
class TrainingHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    lr: list = field(default_factory=list)
    epoch_time_s: list = field(default_factory=list)
    best_epoch: int = -1


class MLPClassifier:
    """sklearn-style wrapper: ``fit``, ``predict``, ``predict_proba``, ``save``/``load``."""

    def __init__(self, config: MLPConfig | None = None, n_threads: int | None = None) -> None:
        self.config = config or MLPConfig()
        self.model: MLPNet | None = None
        self.history = TrainingHistory()
        self.n_classes = 0
        self.n_features = 0
        if n_threads:
            torch.set_num_threads(n_threads)

    def _loader(self, x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                           torch.from_numpy(np.asarray(y, dtype=np.int64)))
        gen = torch.Generator().manual_seed(self.config.seed)
        return DataLoader(ds, batch_size=self.config.batch_size, shuffle=shuffle, generator=gen)

    def _run_epoch(self, loader, criterion, optimizer=None) -> tuple[float, float]:
        training = optimizer is not None
        self.model.train(training)
        total_loss, correct, seen = 0.0, 0, 0
        with torch.set_grad_enabled(training):
            for xb, yb in loader:
                logits = self.model(xb)
                loss = criterion(logits, yb)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                seen += len(yb)
        return total_loss / max(seen, 1), correct / max(seen, 1)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
            log=print) -> "MLPClassifier":
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.n_features = x_train.shape[1]
        self.n_classes = int(max(y_train.max(), y_val.max()) + 1)
        self.model = MLPNet(self.n_features, self.n_classes, cfg.hidden, cfg.dropout)
        weight = None
        if cfg.class_weighting:
            counts = np.bincount(y_train, minlength=self.n_classes).astype(np.float64)
            # inverse-frequency weights, square-rooted to temper extreme ratios
            w = np.sqrt(counts.sum() / (self.n_classes * np.maximum(counts, 1)))
            weight = torch.tensor(w / w.mean(), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weight)
        eval_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=cfg.lr_factor,
                                                               patience=cfg.lr_patience)
        train_loader = self._loader(x_train, y_train, shuffle=True)
        train_eval_loader = self._loader(x_train, y_train, shuffle=False)
        val_loader = self._loader(x_val, y_val, shuffle=False)
        best_loss, best_state, waited = float("inf"), None, 0
        for epoch in range(cfg.max_epochs):
            start = time.perf_counter()
            self._run_epoch(train_loader, criterion, optimizer)
            tr_loss, tr_acc = self._run_epoch(train_eval_loader, eval_criterion)
            va_loss, va_acc = self._run_epoch(val_loader, eval_criterion)
            scheduler.step(va_loss)
            h = self.history
            h.train_loss.append(tr_loss); h.val_loss.append(va_loss)
            h.train_acc.append(tr_acc); h.val_acc.append(va_acc)
            h.lr.append(optimizer.param_groups[0]["lr"]); h.epoch_time_s.append(time.perf_counter() - start)
            log(f"epoch {epoch + 1:02d} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
                f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f} lr={h.lr[-1]:.1e} ({h.epoch_time_s[-1]:.1f}s)")
            if va_loss < best_loss - 1e-5:
                best_loss, best_state, waited = va_loss, copy.deepcopy(self.model.state_dict()), 0
                h.best_epoch = epoch + 1
            else:
                waited += 1
                if waited >= cfg.patience:
                    log(f"early stopping at epoch {epoch + 1}; best epoch {h.best_epoch}")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not trained or loaded")
        self.model.eval()
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = [torch.softmax(self.model(torch.from_numpy(x[i:i + batch_size])), dim=1).numpy()
               for i in range(0, len(x), batch_size)]
        return np.concatenate(out) if out else np.empty((0, self.n_classes), np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(1)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), directory / "mlp_weights.pt")
        meta = {"config": asdict(self.config), "n_features": self.n_features, "n_classes": self.n_classes,
                "history": asdict(self.history)}
        (directory / "mlp_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "MLPClassifier":
        meta = json.loads((directory / "mlp_meta.json").read_text())
        cfg = meta["config"]
        cfg["hidden"] = tuple(cfg["hidden"])
        obj = cls(MLPConfig(**cfg))
        obj.n_features, obj.n_classes = meta["n_features"], meta["n_classes"]
        obj.model = MLPNet(obj.n_features, obj.n_classes, obj.config.hidden, obj.config.dropout)
        obj.model.load_state_dict(torch.load(directory / "mlp_weights.pt", map_location="cpu", weights_only=True))
        obj.model.eval()
        obj.history = TrainingHistory(**meta["history"])
        return obj
