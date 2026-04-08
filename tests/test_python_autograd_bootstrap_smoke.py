#!/usr/bin/env python3
"""Python smoke test: autograd bootstrap parity + tiny 1-step SGD."""

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


def _parity_test() -> None:
    _require(hasattr(lc_api, "ag_matmul"), "ag_matmul must exist")
    _require(hasattr(lc_api, "ag_add"), "ag_add must exist")
    _require(hasattr(lc_api, "ag_relu"), "ag_relu must exist")
    _require(hasattr(lc_api, "ag_mse_loss"), "ag_mse_loss must exist")

    rng = np.random.default_rng(20260411)
    x = (rng.standard_normal((8, 4)) * 0.5).astype(np.float32)
    y = (rng.standard_normal((8, 3)) * 0.2).astype(np.float32)
    w1 = (rng.standard_normal((4, 5)) * 0.1).astype(np.float32)
    b1 = (rng.standard_normal((1, 5)) * 0.1).astype(np.float32)
    w2 = (rng.standard_normal((5, 3)) * 0.1).astype(np.float32)
    b2 = (rng.standard_normal((1, 3)) * 0.1).astype(np.float32)

    aw1 = lc_api.ag_parameter(w1.copy())
    ab1 = lc_api.ag_parameter(b1.copy())
    aw2 = lc_api.ag_parameter(w2.copy())
    ab2 = lc_api.ag_parameter(b2.copy())
    tx = lc_api.ag_tensor(x, requires_grad=False)
    ty = lc_api.ag_tensor(y, requires_grad=False)

    pred = lc_api.ag_add(lc_api.ag_matmul(lc_api.ag_relu(lc_api.ag_add(lc_api.ag_matmul(tx, aw1), ab1)), aw2), ab2)
    loss = lc_api.ag_mse_loss(pred, ty)
    loss.backward()

    if _torch_available():
        import torch

        torch.manual_seed(20260411)
        tx_t = torch.tensor(x, dtype=torch.float32)
        ty_t = torch.tensor(y, dtype=torch.float32)
        tw1 = torch.tensor(w1, dtype=torch.float32, requires_grad=True)
        tb1 = torch.tensor(b1, dtype=torch.float32, requires_grad=True)
        tw2 = torch.tensor(w2, dtype=torch.float32, requires_grad=True)
        tb2 = torch.tensor(b2, dtype=torch.float32, requires_grad=True)

        pred_t = torch.relu(tx_t.matmul(tw1) + tb1).matmul(tw2) + tb2
        loss_t = torch.mean((pred_t - ty_t) ** 2)
        loss_t.backward()

        _require(np.allclose(aw1.grad, tw1.grad.detach().cpu().numpy(), atol=1.0e-5, rtol=1.0e-5), "w1 grad parity failed")
        _require(np.allclose(ab1.grad, tb1.grad.detach().cpu().numpy(), atol=1.0e-5, rtol=1.0e-5), "b1 grad parity failed")
        _require(np.allclose(aw2.grad, tw2.grad.detach().cpu().numpy(), atol=1.0e-5, rtol=1.0e-5), "w2 grad parity failed")
        _require(np.allclose(ab2.grad, tb2.grad.detach().cpu().numpy(), atol=1.0e-5, rtol=1.0e-5), "b2 grad parity failed")


def _tiny_training_step_test() -> None:
    _require(hasattr(lc_api, "TinyAutogradMLP"), "TinyAutogradMLP must exist")
    model = lc_api.TinyAutogradMLP(4, 6, 3, seed=20260411)
    rng = np.random.default_rng(20260411)
    x = (rng.standard_normal((16, 4)) * 0.4).astype(np.float32)
    y = (rng.standard_normal((16, 3)) * 0.3).astype(np.float32)

    pred_before = model.forward(lc_api.ag_tensor(x, requires_grad=False)).data
    loss_before = float(np.mean((pred_before - y) ** 2))
    params_before = [p.data.copy() for p in model.parameters()]

    loss_step = float(model.train_step(x, y, lr=5.0e-2))
    _require(np.isfinite(loss_step), "train_step should return finite loss")

    pred_after = model.forward(lc_api.ag_tensor(x, requires_grad=False)).data
    loss_after = float(np.mean((pred_after - y) ** 2))

    changed = any(np.max(np.abs(p.data - pb)) > 0.0 for p, pb in zip(model.parameters(), params_before))
    _require(changed, "at least one parameter must be updated by SGD step")
    _require(loss_after <= (loss_before + 1.0e-4), "1-step SGD should not increase loss significantly")


def main() -> None:
    np.random.seed(20260411)
    _parity_test()
    _tiny_training_step_test()
    print("python autograd bootstrap smoke: ok")


if __name__ == "__main__":
    main()
