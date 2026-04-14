#!/usr/bin/env python3
"""Python smoke: MLX bridge alpha contract."""

from __future__ import annotations

import numpy as np

import lightning_core_integrated_api as lc_api


class _FakeMLX:
    @staticmethod
    def array(value):
        return np.asarray(value, dtype=np.float32)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    np.random.seed(20260414)
    _require(hasattr(lc_api, "create_mlx_model_runner_adapter"), "mlx adapter factory missing")
    _require(hasattr(lc_api, "mlx_runner_adapter_schema"), "mlx adapter schema missing")

    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, vocab_size=256, seed=20260414)
    x_tokens = np.asarray(np.random.randint(0, 256, size=(48,)), dtype=np.int64)

    adapter_missing = lc_api.create_mlx_model_runner_adapter(
        runner,
        mode="eager",
        device="cpu",
        route_policy={"conv": "auto", "attention": "auto", "graph": "auto"},
        allow_missing_runtime=True,
        overhead_budget_ms=5.0,
        mlx_module=None,
    )
    y0 = np.asarray(adapter_missing.infer(x_tokens), dtype=np.float32)
    t0 = adapter_missing.last_telemetry()
    _require(str(t0.get("runtime", "")) in {"numpy_shim", "mlx"}, "runtime field missing")
    _require(str(t0.get("boundary_reason_code", "")) != "", "reason code missing")

    adapter_fake = lc_api.create_mlx_model_runner_adapter(
        runner,
        mode="interop",
        device="cpu",
        route_policy={"conv": "auto", "attention": "auto", "graph": "auto"},
        allow_missing_runtime=False,
        overhead_budget_ms=5.0,
        mlx_module=_FakeMLX,
    )
    y1 = np.asarray(adapter_fake.infer(x_tokens), dtype=np.float32)
    t1 = adapter_fake.last_telemetry()
    _require(str(t1.get("runtime", "")) == "mlx", "fake mlx runtime path expected")

    ref = np.asarray(runner.run(x_tokens, mode="eager", device="cpu"), dtype=np.float32)
    _require(np.allclose(y0, ref, atol=3.0e-3, rtol=3.0e-3), "mlx adapter parity failed (runtime missing)")
    _require(np.allclose(y1, ref, atol=3.0e-3, rtol=3.0e-3), "mlx adapter parity failed (fake runtime)")

    schema = lc_api.mlx_runner_adapter_schema()
    required = list(schema.get("required_telemetry_fields", []))
    for key in required:
        _require(key in t1, f"mlx telemetry missing key: {key}")

    bench = adapter_fake.benchmark(x_tokens, warmup=1, iters=2)
    _require(str(bench.get("schema_version", "")) == "mlx_runner_model_bench_v1", "benchmark schema mismatch")

    print("python mlx bridge alpha smoke: ok")


if __name__ == "__main__":
    main()
