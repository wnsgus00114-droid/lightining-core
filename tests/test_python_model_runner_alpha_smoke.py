#!/usr/bin/env python3
"""Python smoke: model runner beta contracts + wrapper telemetry surfaces."""

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


def main() -> None:
    np.random.seed(20260410)
    _require(hasattr(lc_api, "TinyTransformerRunner"), "TinyTransformerRunner must exist")
    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, seed=20260410)
    x = (np.random.standard_normal((48, 48)) * 0.2).astype(np.float32)

    _require(hasattr(lc_api, "runner_config_schema"), "runner_config_schema must exist")
    _require(hasattr(lc_api, "validate_runner_config"), "validate_runner_config must exist")
    schema = lc_api.runner_config_schema()
    _require(str(schema.get("schema_version", "")) != "", "runner config schema version must exist")
    cfg = lc_api.validate_runner_config(
        {
            "mode": "graph",
            "device": "cpu",
            "seed": 20260410,
            "layout": "seq_dmodel_2d",
            "dtype": "float32",
            "route_policy": {"conv": "auto", "attention": "auto", "graph": "auto"},
        },
        strict=True,
    )
    _require(bool(cfg.get("ok", False)), "runner config should validate")

    y_eager = runner.run(x, mode="eager", device="cpu")  # type: ignore[assignment]
    y_graph = runner.run(x, mode="graph", device="cpu")  # type: ignore[assignment]
    _require(np.allclose(y_eager, y_graph, atol=1.0e-4, rtol=1.0e-4), "runner eager/graph parity failed")

    # deterministic replay contract
    replay = lc_api.runner_replay_report(runner, x, mode="eager", device="cpu", repeats=3)
    _require(bool(replay.get("deterministic_replay", False)), "runner replay should be deterministic")

    # deterministic fallback contract (graph policy forcing eager)
    y_forced, meta_forced = runner.run(
        x,
        mode="graph",
        device="cpu",
        route_policy={"conv": "auto", "attention": "auto", "graph": "torch"},
        return_metadata=True,
    )
    _require(str(meta_forced.get("resolved_mode", "")) == "eager", "graph policy torch should resolve to eager")
    _require(
        str(meta_forced.get("fallback_reason_code", "")) == "runner_graph_policy_forced_eager",
        "forced graph fallback reason code mismatch",
    )
    _require(np.allclose(y_eager, y_forced, atol=3.0e-3, rtol=3.0e-3), "forced graph fallback parity failed")

    # checkpoint compatibility matrix exposure
    _require(hasattr(lc_api, "checkpoint_compatibility_matrix"), "checkpoint_compatibility_matrix must exist")
    matrix = lc_api.checkpoint_compatibility_matrix()
    rows = list(matrix.get("rows", []))
    _require(bool(rows), "checkpoint compatibility matrix rows must not be empty")

    if _torch_available():
        import torch

        y_interop = runner.run(x, mode="interop", device="cpu")
        _require(np.allclose(y_eager, y_interop, atol=3.0e-3, rtol=3.0e-3), "runner eager/interop parity failed")
        wrapper = lc_api.create_torch_module_wrapper(
            runner,
            mode="graph",
            device="cpu",
            route_policy={"conv": "auto", "attention": "auto", "graph": "torch"},
        )
        _require(wrapper is not None, "torch wrapper creation failed")
        x_t = torch.as_tensor(x, dtype=torch.float32)
        y_t = wrapper(x_t).detach().cpu().numpy()
        _require(np.allclose(y_t, y_forced, atol=3.0e-3, rtol=3.0e-3), "torch wrapper parity failed")
        telem = lc_api.get_torch_wrapper_telemetry(wrapper)
        _require(isinstance(telem, dict), "torch telemetry must be dict")
        _require(str(telem.get("boundary_reason_code", "")) != "", "torch telemetry reason code missing")
        _require(str(telem.get("runner_fallback_reason_code", "")) != "", "runner fallback reason code missing")

    tf_layer = lc_api.create_tf_keras_layer_wrapper(runner, mode="eager", device="cpu", allow_missing_runtime=True)
    _ = tf_layer(x)
    tf_telem = lc_api.get_tf_wrapper_telemetry(tf_layer)
    _require(isinstance(tf_telem, dict), "tf telemetry must be dict")
    _require(str(tf_telem.get("boundary_reason_code", "")).strip() != "", "tf telemetry reason code missing")

    print("python model runner beta smoke: ok")


if __name__ == "__main__":
    main()
