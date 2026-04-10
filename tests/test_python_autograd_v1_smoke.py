#!/usr/bin/env python3
"""Python smoke: autograd bootstrap v1 (conv + attention-adjacent)."""

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


def _conv_parity() -> None:
    _require(hasattr(lc_api, "ag_conv2d"), "ag_conv2d must exist")
    rng = np.random.default_rng(20260410)
    x = (rng.standard_normal((2, 3, 6, 6)) * 0.3).astype(np.float32)
    w = (rng.standard_normal((4, 3, 3, 3)) * 0.2).astype(np.float32)
    b = (rng.standard_normal((4,)) * 0.05).astype(np.float32)
    target = (rng.standard_normal((2, 4, 6, 6)) * 0.1).astype(np.float32)

    ax = lc_api.ag_tensor(x, requires_grad=True)
    aw = lc_api.ag_parameter(w.copy())
    ab = lc_api.ag_parameter(b.copy())
    ay = lc_api.ag_tensor(target, requires_grad=False)
    pred = lc_api.ag_relu(lc_api.ag_conv2d(ax, aw, ab, stride_h=1, stride_w=1, pad_h=1, pad_w=1))
    loss = lc_api.ag_mse_loss(pred, ay)
    loss.backward()

    if _torch_available():
        import torch
        import torch.nn.functional as F

        tx = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        tw = torch.tensor(w, dtype=torch.float32, requires_grad=True)
        tb = torch.tensor(b, dtype=torch.float32, requires_grad=True)
        tt = torch.tensor(target, dtype=torch.float32)
        pred_t = F.relu(F.conv2d(tx, tw, tb, stride=(1, 1), padding=(1, 1)))
        loss_t = torch.mean((pred_t - tt) ** 2)
        loss_t.backward()
        _require(np.allclose(aw.grad, tw.grad.detach().cpu().numpy(), atol=2.0e-4, rtol=2.0e-4), "conv weight grad parity failed")
        _require(np.allclose(ab.grad, tb.grad.detach().cpu().numpy(), atol=2.0e-4, rtol=2.0e-4), "conv bias grad parity failed")


def _attention_parity() -> None:
    _require(hasattr(lc_api, "ag_attention"), "ag_attention must exist")
    rng = np.random.default_rng(20260410)
    q = (rng.standard_normal((12, 8)) * 0.2).astype(np.float32)
    k = (rng.standard_normal((12, 8)) * 0.2).astype(np.float32)
    v = (rng.standard_normal((12, 8)) * 0.2).astype(np.float32)
    target = (rng.standard_normal((12, 8)) * 0.1).astype(np.float32)

    aq = lc_api.ag_parameter(q.copy())
    ak = lc_api.ag_parameter(k.copy())
    av = lc_api.ag_parameter(v.copy())
    at = lc_api.ag_tensor(target, requires_grad=False)
    out = lc_api.ag_attention(aq, ak, av, causal=False)
    loss = lc_api.ag_mse_loss(out, at)
    loss.backward()

    if _torch_available():
        import torch

        tq = torch.tensor(q, dtype=torch.float32, requires_grad=True)
        tk = torch.tensor(k, dtype=torch.float32, requires_grad=True)
        tv = torch.tensor(v, dtype=torch.float32, requires_grad=True)
        tt = torch.tensor(target, dtype=torch.float32)
        scale = float(1.0 / np.sqrt(8.0))
        scores = tq.matmul(tk.transpose(0, 1)) * scale
        probs = torch.softmax(scores, dim=-1)
        out_t = probs.matmul(tv)
        loss_t = torch.mean((out_t - tt) ** 2)
        loss_t.backward()
        _require(np.allclose(aq.grad, tq.grad.detach().cpu().numpy(), atol=2.0e-4, rtol=2.0e-4), "attention q grad parity failed")
        _require(np.allclose(ak.grad, tk.grad.detach().cpu().numpy(), atol=2.0e-4, rtol=2.0e-4), "attention k grad parity failed")
        _require(np.allclose(av.grad, tv.grad.detach().cpu().numpy(), atol=2.0e-4, rtol=2.0e-4), "attention v grad parity failed")


def _tiny_multistep_training() -> None:
    _require(hasattr(lc_api, "TinyAutogradConvAttention"), "TinyAutogradConvAttention must exist")
    rng = np.random.default_rng(20260410)
    model = lc_api.TinyAutogradConvAttention(seed=20260410)
    x = (rng.standard_normal((1, 3, 4, 4)) * 0.2).astype(np.float32)
    y = (rng.standard_normal((16, 4)) * 0.1).astype(np.float32)

    losses = []
    params_before = [p.data.copy() for p in model.parameters()]
    for _ in range(3):
        losses.append(float(model.train_step(x, y, lr=2.0e-2)))
    _require(all(np.isfinite(v) for v in losses), "training losses must be finite")
    _require(losses[-1] <= losses[0] + 1.0e-3, "multi-step training should not diverge")
    changed = any(np.max(np.abs(p.data - pb)) > 0.0 for p, pb in zip(model.parameters(), params_before))
    _require(changed, "at least one parameter must change after training")


def main() -> None:
    np.random.seed(20260410)
    _conv_parity()
    _attention_parity()
    _tiny_multistep_training()
    print("python autograd v1 smoke: ok")


if __name__ == "__main__":
    main()

