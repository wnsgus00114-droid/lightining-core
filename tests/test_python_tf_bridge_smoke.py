#!/usr/bin/env python3
"""Python smoke: TensorFlow bridge prototype (installed/missing runtime paths)."""

from __future__ import annotations

import numpy as np

import lightning_core_integrated_api as lc_api


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class _FakeLayer:
    def __call__(self, inputs, *args, **kwargs):
        return self.call(inputs, *args, **kwargs)

    def call(self, inputs, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class _FakeTF:
    class keras:  # noqa: N801
        class layers:  # noqa: N801
            Layer = _FakeLayer

    @staticmethod
    def convert_to_tensor(value, dtype=None):
        return np.asarray(value, dtype=dtype if dtype is not None else np.float32)


def main() -> None:
    np.random.seed(20260411)
    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, seed=20260411)
    x = (np.random.standard_normal((48, 48)) * 0.2).astype(np.float32)
    y_ref = np.asarray(runner.run(x, mode="eager", device="cpu"), dtype=np.float32)

    # Missing-runtime path must still be callable (graceful numpy shim).
    shim = lc_api.create_tf_keras_layer_wrapper(
        runner,
        mode="graph",
        device="cpu",
        route_policy={"conv": "auto", "attention": "auto", "graph": "torch"},
        prefer_tensorflow_runtime=False,
        allow_missing_runtime=True,
        tensorflow_module=None,
    )
    y_shim = np.asarray(shim(x), dtype=np.float32)
    t_shim = lc_api.get_tf_wrapper_telemetry(shim)
    _require(bool(np.allclose(y_ref, y_shim, atol=3.0e-3, rtol=3.0e-3)), "tf shim parity failed")
    _require(str(t_shim.get("runtime", "")) == "numpy_shim", "tf shim runtime tag mismatch")
    _require(
        str(t_shim.get("boundary_reason_code", "")) == "tf_runner_graph_policy_forced_eager",
        "tf shim deterministic fallback reason mismatch",
    )

    # Installed-runtime path is tested via a fake tensorflow module to avoid hard dependency.
    layer = lc_api.create_tf_keras_layer_wrapper(
        runner,
        mode="eager",
        device="cpu",
        route_policy={"conv": "auto", "attention": "auto", "graph": "auto"},
        prefer_tensorflow_runtime=False,
        allow_missing_runtime=False,
        tensorflow_module=_FakeTF,
    )
    y_layer = np.asarray(layer(x), dtype=np.float32)
    t_layer = lc_api.get_tf_wrapper_telemetry(layer)
    _require(bool(np.allclose(y_ref, y_layer, atol=3.0e-3, rtol=3.0e-3)), "tf layer parity failed")
    _require(str(t_layer.get("runtime", "")) == "tensorflow", "tf layer runtime tag mismatch")
    _require(str(t_layer.get("boundary_reason_code", "")).strip() != "", "tf layer reason code missing")

    print("python tensorflow bridge smoke: ok")


if __name__ == "__main__":
    main()
