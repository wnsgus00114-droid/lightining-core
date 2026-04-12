#!/usr/bin/env python3
"""Python smoke: TF engine adapter beta model-level wrapper contract."""

from __future__ import annotations

import numpy as np

import lightning_core_integrated_api as lc_api


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


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    np.random.seed(20260411)
    _require(hasattr(lc_api, "create_tf_model_runner_adapter"), "create_tf_model_runner_adapter must exist")
    _require(hasattr(lc_api, "tf_runner_adapter_schema"), "tf_runner_adapter_schema must exist")

    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, vocab_size=256, seed=20260411)
    x_tokens = np.asarray(np.random.randint(0, 256, size=(48,)), dtype=np.int64)

    adapter_tf = lc_api.create_tf_model_runner_adapter(
        runner,
        mode="eager",
        device="cpu",
        route_policy={"conv": "auto", "attention": "auto", "graph": "auto"},
        prefer_tensorflow_runtime=False,
        allow_missing_runtime=False,
        tensorflow_module=_FakeTF,
    )
    y_tf = np.asarray(adapter_tf.infer(x_tokens), dtype=np.float32)
    t_tf = adapter_tf.last_telemetry()
    _require(str(t_tf.get("runtime", "")) == "tensorflow", "tensorflow runtime path expected")

    adapter_shim = lc_api.create_tf_model_runner_adapter(
        runner,
        mode="graph",
        device="cpu",
        route_policy={"conv": "auto", "attention": "auto", "graph": "torch"},
        prefer_tensorflow_runtime=False,
        allow_missing_runtime=True,
        tensorflow_module=None,
    )
    y_shim = np.asarray(adapter_shim.infer(x_tokens), dtype=np.float32)
    t_shim = adapter_shim.last_telemetry()
    _require(str(t_shim.get("runtime", "")) == "numpy_shim", "numpy shim runtime path expected")
    _require(str(t_shim.get("boundary_reason_code", "")) != "", "shim reason code must exist")

    ref = np.asarray(runner.run(x_tokens, mode="eager", device="cpu"), dtype=np.float32)
    _require(np.allclose(y_tf, ref, atol=3.0e-3, rtol=3.0e-3), "tf adapter parity failed")
    _require(np.allclose(y_shim, ref, atol=3.0e-3, rtol=3.0e-3), "tf shim adapter parity failed")

    schema = lc_api.tf_runner_adapter_schema()
    required = list(schema.get("required_telemetry_fields", []))
    for key in required:
        _require(key in t_tf, f"tf telemetry missing key: {key}")

    bench = adapter_tf.benchmark(x_tokens, warmup=1, iters=2)
    _require(str(bench.get("schema_version", "")) == "tf_runner_model_bench_v1", "benchmark schema mismatch")

    print("python tf engine adapter beta smoke: ok")


if __name__ == "__main__":
    main()
