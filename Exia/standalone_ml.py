from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _to_2d(x: Any) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    if x_arr.ndim != 2:
        raise ValueError("x must be 1D or 2D")
    return x_arr


def _to_1d(y: Any) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
    return y_arr


def train_test_split(
    x: Any,
    y: Any,
    test_size: float = 0.2,
    shuffle: bool = True,
    seed: int = 42,
):
    x_arr = _to_2d(x)
    y_arr = _to_1d(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")

    n = x_arr.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    test_n = max(1, int(n * test_size))
    test_idx = indices[:test_n]
    train_idx = indices[test_n:]
    return x_arr[train_idx], x_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


@dataclass
class LinearRegressionModel:
    weights: np.ndarray
    bias: float

    def predict(self, x: Any) -> np.ndarray:
        x_arr = _to_2d(x)
        return x_arr @ self.weights + self.bias


def fit_linear_regression(
    x: Any,
    y: Any,
    lr: float = 1e-2,
    epochs: int = 500,
    l2: float = 0.0,
) -> LinearRegressionModel:
    x_arr = _to_2d(x)
    y_arr = _to_1d(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    n, d = x_arr.shape
    w = np.zeros((d,), dtype=np.float32)
    b = 0.0

    for _ in range(epochs):
        pred = x_arr @ w + b
        err = pred - y_arr
        grad_w = (2.0 / n) * (x_arr.T @ err) + (2.0 * l2 * w)
        grad_b = float((2.0 / n) * err.sum())
        w -= lr * grad_w
        b -= lr * grad_b

    return LinearRegressionModel(weights=w, bias=b)


@dataclass
class LogisticRegressionModel:
    weights: np.ndarray
    bias: float

    def predict_proba(self, x: Any) -> np.ndarray:
        x_arr = _to_2d(x)
        logits = x_arr @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def predict(self, x: Any, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(x)
        return (probs >= threshold).astype(np.int32)


def fit_logistic_regression(
    x: Any,
    y: Any,
    lr: float = 5e-2,
    epochs: int = 700,
    l2: float = 0.0,
) -> LogisticRegressionModel:
    x_arr = _to_2d(x)
    y_arr = _to_1d(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    labels = np.unique(y_arr)
    if not np.all(np.isin(labels, [0.0, 1.0])):
        raise ValueError("y must be binary labels {0, 1}")

    n, d = x_arr.shape
    w = np.zeros((d,), dtype=np.float32)
    b = 0.0

    for _ in range(epochs):
        logits = x_arr @ w + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        err = probs - y_arr
        grad_w = (x_arr.T @ err) / n + 2.0 * l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b

    return LogisticRegressionModel(weights=w, bias=b)


def mse(y_true: Any, y_pred: Any) -> float:
    y_t = _to_1d(y_true)
    y_p = _to_1d(y_pred)
    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    return float(np.mean((y_t - y_p) ** 2))


def mae(y_true: Any, y_pred: Any) -> float:
    y_t = _to_1d(y_true)
    y_p = _to_1d(y_pred)
    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    return float(np.mean(np.abs(y_t - y_p)))


def r2_score(y_true: Any, y_pred: Any) -> float:
    y_t = _to_1d(y_true)
    y_p = _to_1d(y_pred)
    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - y_t.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def accuracy(y_true: Any, y_pred: Any) -> float:
    y_t = _to_1d(y_true)
    y_p = _to_1d(y_pred)
    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    return float((y_t == y_p).mean())
