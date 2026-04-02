#!/usr/bin/env python3
"""Python smoke test: integrated API + runtime timeline observability."""

from __future__ import annotations

import numpy as np

import lightning_core as lc
import lightning_core_integrated_api as lc_api


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    np.random.seed(20260402)

    lc.runtime_trace_clear()
    lc.runtime_trace_enable(True)

    lc_api.set_backend("lightning")
    _require(lc_api.get_backend() == "lightning", "integrated backend should be 'lightning'")

    a = np.random.rand(64, 64).astype(np.float32)
    b = np.random.rand(64, 64).astype(np.float32)
    mm_out = lc_api.lightning_matmul(a, b, device="metal")
    _require(mm_out.shape == (64, 64), "lightning_matmul output shape mismatch")
    _require(mm_out.dtype == np.float32, "lightning_matmul output dtype mismatch")

    x = np.random.rand(1, 3, 8, 8).astype(np.float32)
    w = np.random.rand(16, 3, 3, 3).astype(np.float32)
    bias = np.random.rand(16).astype(np.float32)
    conv_out = lc.api.conv_relu_nchw(
        x,
        w,
        bias,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        device="metal",
    )
    _require(conv_out.shape == (1, 16, 8, 8), "conv_relu_nchw output shape mismatch")
    _require(conv_out.dtype == np.float32, "conv_relu_nchw output dtype mismatch")

    lc.runtime_trace_enable(False)
    timeline = lc.runtime_trace_timeline(
        event_sort_by="timestamp_ns",
        event_descending=False,
        group_by="op_path",
        group_sort_by="total_delta_next_ns",
        group_descending=True,
        hotspot_top_k=8,
    )

    _require(isinstance(timeline, dict), "runtime_trace_timeline must return dict")
    for key in ("events", "groups", "hotspots", "event_count", "window_ns"):
        _require(key in timeline, f"timeline missing key: {key}")
    _require(len(timeline["events"]) > 0, "timeline must contain at least one event")
    _require(len(timeline["groups"]) > 0, "timeline must contain at least one group")

    saw_path_group = False
    for row in timeline["groups"]:
        key = str(row.get("key", ""))
        if key.startswith("matmul|") or key.startswith("conv2d_nchw|"):
            saw_path_group = True
            break
    _require(saw_path_group, "timeline groups should include op_path for matmul/conv2d_nchw")

    lc.runtime_trace_clear()
    print("python integrated API + timeline smoke: ok")


if __name__ == "__main__":
    main()
