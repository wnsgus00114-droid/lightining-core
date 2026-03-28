from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import torch
except Exception:
    torch = None


@dataclass
class TrainerConfig:
    epochs: int = 3
    log_every: int = 20
    device: str | None = None


class Trainer:
    def __init__(self, config: TrainerConfig | None = None):
        self.config = config or TrainerConfig()

    def fit(
        self,
        model,
        dataloader,
        optimizer,
        loss_fn: Callable,
    ) -> None:
        if torch is None:
            raise RuntimeError("fit() requires torch. Install with: pip install Exia[torch]")
        device = self.config.device or (
            "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model.to(device)
        model.train()

        step = 0
        for epoch in range(self.config.epochs):
            running = 0.0
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

                running += float(loss.item())
                step += 1
                if step % self.config.log_every == 0:
                    avg = running / self.config.log_every
                    print(f"[epoch={epoch+1}] step={step} loss={avg:.6f}")
                    running = 0.0

    def fit_linear_regression(
        self,
        x,
        y,
        lr: float = 1e-2,
    ) -> tuple[np.ndarray, float]:
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if x_arr.ndim != 2:
            raise ValueError("x must be a 2D array with shape [N, D]")
        if y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError("x and y must have the same first dimension")

        n, d = x_arr.shape
        w = np.zeros((d,), dtype=np.float32)
        b = 0.0

        for epoch in range(self.config.epochs):
            pred = x_arr @ w + b
            err = pred - y_arr
            grad_w = (2.0 / n) * (x_arr.T @ err)
            grad_b = float((2.0 / n) * err.sum())
            w -= lr * grad_w
            b -= lr * grad_b

            if (epoch + 1) % max(1, self.config.log_every) == 0:
                mse = float(np.mean(err**2))
                print(f"[standalone] epoch={epoch+1} mse={mse:.6f}")

        return w, b
