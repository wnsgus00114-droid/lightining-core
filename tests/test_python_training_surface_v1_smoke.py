#!/usr/bin/env python3
"""Python smoke: training surface v1 (multi-step, grad clip, loss scale, hooks)."""

from __future__ import annotations

import numpy as np

import lightning_core_integrated_api as lc_api


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _gradient_parity_smoke() -> None:
    rng = np.random.default_rng(20260411)
    x = (rng.standard_normal((8, 4)) * 0.3).astype(np.float32)
    y = (rng.standard_normal((8, 3)) * 0.2).astype(np.float32)
    w = (rng.standard_normal((4, 3)) * 0.1).astype(np.float32)
    b = (rng.standard_normal((1, 3)) * 0.1).astype(np.float32)

    aw = lc_api.ag_parameter(w.copy())
    ab = lc_api.ag_parameter(b.copy())
    tx = lc_api.ag_tensor(x, requires_grad=False)
    ty = lc_api.ag_tensor(y, requires_grad=False)
    pred = lc_api.ag_add(lc_api.ag_matmul(tx, aw), ab)
    loss = lc_api.ag_mse_loss(pred, ty)
    loss.backward(np.asarray([1.0], dtype=np.float32))

    if _torch_available():
        import torch

        tx_t = torch.tensor(x, dtype=torch.float32)
        ty_t = torch.tensor(y, dtype=torch.float32)
        tw = torch.tensor(w, dtype=torch.float32, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        pred_t = tx_t.matmul(tw) + tb
        loss_t = torch.mean((pred_t - ty_t) ** 2)
        loss_t.backward()
        _require(np.allclose(aw.grad, tw.grad.detach().cpu().numpy(), atol=1.0e-5, rtol=1.0e-5), "weight grad parity failed")
        _require(np.allclose(ab.grad, tb.grad.detach().cpu().numpy(), atol=1.0e-5, rtol=1.0e-5), "bias grad parity failed")


def _multistep_loss_smoke() -> None:
    _require(hasattr(lc_api, "autograd_train_loop"), "autograd_train_loop must exist")
    model = lc_api.TinyAutogradMLP(4, 8, 3, seed=20260411)
    rng = np.random.default_rng(20260411)
    x = (rng.standard_normal((32, 4)) * 0.4).astype(np.float32)
    y = (rng.standard_normal((32, 3)) * 0.3).astype(np.float32)

    calls = {"begin": 0, "after_backward": 0, "after_step": 0}

    report = lc_api.autograd_train_loop(
        model,
        x,
        y,
        steps=5,
        lr=3.0e-2,
        grad_clip_norm=1.0,
        loss_scale=4.0,
        hooks={
            "on_step_begin": lambda _: calls.__setitem__("begin", calls["begin"] + 1),
            "on_after_backward": lambda _: calls.__setitem__("after_backward", calls["after_backward"] + 1),
            "on_after_step": lambda _: calls.__setitem__("after_step", calls["after_step"] + 1),
        },
    )

    losses = list(report.get("losses", []))
    _require(len(losses) == 5, "loss history length mismatch")
    _require(all(np.isfinite(v) for v in losses), "losses must be finite")
    _require(losses[-1] <= losses[0] + 1.0e-3, "multi-step training should not diverge")
    _require(bool(report.get("loss_decreased", False)), "loss_decreased flag must be true")
    _require(calls["begin"] == 5 and calls["after_backward"] == 5 and calls["after_step"] == 5, "hook call count mismatch")


def main() -> None:
    _gradient_parity_smoke()
    _multistep_loss_smoke()
    print("python training surface v1 smoke: ok")


if __name__ == "__main__":
    main()

