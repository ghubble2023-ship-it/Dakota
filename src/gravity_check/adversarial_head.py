"""
Optional adversarial / supervised scoring head.

Takes explainable geometric feature vectors from Gravity Check
and learns a linear separator (real=1, fake=0) to improve score
separation when labels are available.

Classical engine score is preserved; this head produces an
adjusted_score. Physics features remain the only inputs.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import json
import os
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, bundle_and_report_to_vector


class AdversarialHead:
    """
    Linear model: score = sigmoid(w · x + b)
    Trained to push real → high, fake → low.
    """

    def __init__(self):
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None
        self.trained: bool = False

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return X
        return (X - self.feature_mean) / (self.feature_std + 1e-8)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.15,
        epochs: int = 400,
        l2: float = 0.01,
    ) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape

        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        Xn = self._normalize(X)

        w = np.zeros(d)
        b = 0.0

        real_mean = Xn[y > 0.5].mean(axis=0) if (y > 0.5).any() else np.zeros(d)

        history = []
        for ep in range(epochs):
            logits = Xn @ w + b
            preds = self._sigmoid(logits)
            err = preds - y
            grad_w = (Xn.T @ err) / n + l2 * w
            grad_b = float(err.mean())
            w -= lr * grad_w
            b -= lr * grad_b

            hard = (y < 0.5) & (preds > 0.45)
            if hard.any():
                X_adv = Xn.copy()
                X_adv[hard] = X_adv[hard] + 0.05 * (real_mean - X_adv[hard])
                logits_adv = X_adv @ w + b
                preds_adv = self._sigmoid(logits_adv)
                err_adv = preds_adv - y
                w -= lr * 0.3 * ((X_adv.T @ err_adv) / n + l2 * w)

            if ep % 50 == 0 or ep == epochs - 1:
                loss = float(
                    -np.mean(y * np.log(preds + 1e-8) + (1 - y) * np.log(1 - preds + 1e-8))
                    + 0.5 * l2 * np.sum(w ** 2)
                )
                acc = float(((preds >= 0.5) == (y >= 0.5)).mean())
                history.append({"epoch": ep, "loss": loss, "acc": acc})

        self.weights = w
        self.bias = float(b)
        self.trained = True
        return {"history": history, "final_acc": history[-1]["acc"] if history else 0.0}

    def predict_score(self, x: np.ndarray) -> float:
        if not self.trained or self.weights is None:
            return float(x[-1]) if len(x) else 0.5
        xn = self._normalize(x.reshape(1, -1))
        s = float(self._sigmoid(xn @ self.weights + self.bias)[0])
        return s

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        if not self.trained or self.weights is None:
            return X[:, -1] if X.size else np.array([])
        Xn = self._normalize(X)
        return self._sigmoid(Xn @ self.weights + self.bias)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "weights": self.weights.tolist() if self.weights is not None else None,
            "bias": self.bias,
            "feature_mean": self.feature_mean.tolist() if self.feature_mean is not None else None,
            "feature_std": self.feature_std.tolist() if self.feature_std is not None else None,
            "feature_names": FEATURE_NAMES,
            "trained": self.trained,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.weights = np.array(data["weights"], dtype=np.float64) if data.get("weights") else None
        self.bias = float(data.get("bias") or 0.0)
        self.feature_mean = np.array(data["feature_mean"], dtype=np.float64) if data.get("feature_mean") else None
        self.feature_std = np.array(data["feature_std"], dtype=np.float64) if data.get("feature_std") else None
        self.trained = bool(data.get("trained"))


def train_from_runs(
    sample_paths_labels: List[Tuple[str, int]],
    model_out: str = "docs/runs/adversarial_head.json",
) -> Dict[str, Any]:
    from .run_image import analyze_image
    from .measurement_frontend import extract_from_path
    from .engine import run_gravity_check

    X_list = []
    y_list = []
    details = []

    for path, label in sample_paths_labels:
        try:
            bundle = extract_from_path(path)
            kw = bundle.to_engine_kwargs()
            if not kw.get("object_heights_px"):
                details.append({"path": path, "label": label, "error": "no objects"})
                continue
            report = run_gravity_check(**kw)
            vec = bundle_and_report_to_vector(report=report, engine_kwargs=kw)
            X_list.append(vec)
            y_list.append(label)
            details.append({
                "path": path,
                "label": label,
                "base_score": report.get("score"),
                "flags": report.get("flags"),
            })
        except Exception as e:
            details.append({"path": path, "label": label, "error": str(e)})

    if len(X_list) < 4:
        return {"error": "not enough samples with objects", "details": details}

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float64)
    head = AdversarialHead()
    train_info = head.fit(X, y)
    head.save(model_out)

    adj = head.predict_batch(X)
    real_adj = adj[y > 0.5]
    fake_adj = adj[y < 0.5]
    base = X[:, -1]
    real_base = base[y > 0.5]
    fake_base = base[y < 0.5]

    return {
        "n_samples": len(y),
        "n_real": int((y > 0.5).sum()),
        "n_fake": int((y < 0.5).sum()),
        "train": train_info,
        "base_real_avg": float(real_base.mean()) if len(real_base) else None,
        "base_fake_avg": float(fake_base.mean()) if len(fake_base) else None,
        "adj_real_avg": float(real_adj.mean()) if len(real_adj) else None,
        "adj_fake_avg": float(fake_adj.mean()) if len(fake_adj) else None,
        "model_path": model_out,
        "details": details,
    }
