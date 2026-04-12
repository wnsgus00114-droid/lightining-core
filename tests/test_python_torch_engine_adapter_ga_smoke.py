#!/usr/bin/env python3
"""Python smoke: Torch engine adapter GA telemetry + budget/coverage contract."""

from __future__ import annotations

import numpy as np

import lightning_core_integrated_api as lc_api


class _FakeDevice:
    def __init__(self, name: str):
        self.type = str(name)


class _FakeTensor:
    def __init__(self, arr: np.ndarray, *, dtype: str, device: str = "cpu"):
        self._arr = np.asarray(arr)
        self.dtype = dtype
        self.device = _FakeDevice(device)

    def detach(self):
        return self

    def to(self, device: str):
        return _FakeTensor(self._arr, dtype=self.dtype, device=device)

    def contiguous(self):
        return self

    def numpy(self):
        return np.asarray(self._arr)

    def is_contiguous(self):
        return True


class _FakeModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _FakeTorch:
    float32 = "float32"
    int64 = "int64"

    class nn:  # noqa: N801
        Module = _FakeModule

    @staticmethod
    def as_tensor(value, dtype=None, device: str = "cpu"):
        arr = np.asarray(value)
        if dtype == _FakeTorch.float32:
            arr = np.asarray(arr, dtype=np.float32)
        elif dtype == _FakeTorch.int64:
            arr = np.asarray(arr, dtype=np.int64)
        return _FakeTensor(arr, dtype=dtype if dtype is not None else _FakeTorch.float32, device=device)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    np.random.seed(20260411)
    _require(hasattr(lc_api, "torch_runner_adapter_schema"), "torch_runner_adapter_schema must exist")

    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, vocab_size=256, seed=20260411)
    x_tokens = np.asarray(np.random.randint(0, 256, size=(48,)), dtype=np.int64)
    x_t = _FakeTorch.as_tensor(x_tokens, dtype=_FakeTorch.int64, device="cpu")

    wrapper = lc_api.create_torch_module_wrapper(
        runner,
        mode="graph",
        device="cpu",
        route_policy={"conv": "auto", "attention": "auto", "graph": "torch"},
        overhead_budget_ms=5.0,
        torch_module=_FakeTorch,
    )

    y = wrapper(x_t)
    y_np = np.asarray(y.numpy(), dtype=np.float32)
    ref = np.asarray(
        runner.run(
            x_tokens,
            mode="graph",
            device="cpu",
            route_policy={"conv": "auto", "attention": "auto", "graph": "torch"},
        ),
        dtype=np.float32,
    )
    _require(np.allclose(y_np, ref, atol=3.0e-3, rtol=3.0e-3), "torch wrapper parity failed")

    telem = lc_api.get_torch_wrapper_telemetry(wrapper)
    schema = lc_api.torch_runner_adapter_schema()
    required = list(schema.get("required_telemetry_fields", []))
    for key in required:
        _require(key in telem, f"torch telemetry missing key: {key}")
    _require(bool(telem.get("reason_code_covered", False)), "reason code coverage must be true")
    _require(bool(telem.get("boundary_overhead_budget_pass", False)), "budget pass must be true")

    print("python torch engine adapter ga smoke: ok")


if __name__ == "__main__":
    main()
