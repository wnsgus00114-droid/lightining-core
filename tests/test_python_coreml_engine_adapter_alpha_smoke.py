#!/usr/bin/env python3
"""Python smoke: CoreML engine adapter alpha contract."""

from __future__ import annotations

import numpy as np

import lightning_core_integrated_api as lc_api


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    np.random.seed(20260414)
    _require(hasattr(lc_api, "create_coreml_model_runner_adapter"), "coreml adapter factory missing")
    _require(hasattr(lc_api, "coreml_runner_adapter_schema"), "coreml adapter schema missing")

    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, vocab_size=256, seed=20260414)
    x_tokens = np.asarray(np.random.randint(0, 256, size=(48,)), dtype=np.int64)

    adapter_missing_runtime = lc_api.create_coreml_model_runner_adapter(
        runner,
        mode="eager",
        device="cpu",
        route_policy={"conv": "auto", "attention": "auto", "graph": "auto"},
        allow_missing_runtime=True,
        enable_coreml_runtime=False,
        overhead_budget_ms=8.0,
    )
    y0 = np.asarray(adapter_missing_runtime.infer(x_tokens), dtype=np.float32)
    t0 = adapter_missing_runtime.last_telemetry()
    _require(str(t0.get("runtime", "")) in {"numpy_shim", "coreml"}, "runtime field missing")
    _require(str(t0.get("boundary_reason_code", "")) != "", "reason code missing")

    adapter_missing_model = lc_api.create_coreml_model_runner_adapter(
        runner,
        mode="graph",
        device="cpu",
        route_policy={"conv": "auto", "attention": "auto", "graph": "torch"},
        allow_missing_runtime=True,
        enable_coreml_runtime=True,
        coreml_model_path="",
        overhead_budget_ms=8.0,
    )
    y1 = np.asarray(adapter_missing_model.infer(x_tokens), dtype=np.float32)
    t1 = adapter_missing_model.last_telemetry()
    _require(str(t1.get("boundary_reason_code", "")) in {
        "coreml_model_path_missing",
        "coreml_runtime_unavailable",
        "coreml_runner_graph_policy_forced_eager",
        "coreml_runner_unknown_fallback",
    }, "unexpected coreml reason code")

    ref = np.asarray(runner.run(x_tokens, mode="eager", device="cpu"), dtype=np.float32)
    _require(np.allclose(y0, ref, atol=3.0e-3, rtol=3.0e-3), "coreml adapter parity failed (runtime missing)")
    _require(np.allclose(y1, ref, atol=3.0e-3, rtol=3.0e-3), "coreml adapter parity failed (missing model)")

    schema = lc_api.coreml_runner_adapter_schema()
    required = list(schema.get("required_telemetry_fields", []))
    for key in required:
        _require(key in t0, f"coreml telemetry missing key: {key}")

    bench = adapter_missing_runtime.benchmark(x_tokens, warmup=1, iters=2)
    _require(str(bench.get("schema_version", "")) == "coreml_runner_model_bench_v1", "benchmark schema mismatch")

    print("python coreml engine adapter alpha smoke: ok")


if __name__ == "__main__":
    main()
