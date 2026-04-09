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
    _require(hasattr(lc, "api") and hasattr(lc.api, "set_engine"), "lc.api.set_engine must be available")
    _require(hasattr(lc.api, "get_engine"), "lc.api.get_engine must be available")
    _require(hasattr(lc, "GraphIR"), "GraphIR binding must be available")

    lc.api.set_engine("lightning")
    _require(lc.api.get_engine() == "lightning", "lc.api engine should be 'lightning'")
    _require(lc_api.get_backend() == "lightning", "helper backend should match lc.api engine")

    g = lc.GraphIR()
    ta = g.add_tensor([8, 8], dtype="float32", name="a", constant=True)
    tb = g.add_tensor([8, 8], dtype="float32", name="b", constant=True)
    tout = g.add_tensor([8, 8], dtype="float32", name="out")
    g.add_node("matmul", [ta, tb], [tout])
    plan_payload = g.plan_summary(preferred_device="metal")
    _require(isinstance(plan_payload, dict), "GraphIR.plan_summary should return dict")
    _require("summary" in plan_payload, "plan_summary should include summary")
    ps = dict(plan_payload["summary"])
    _require(int(ps.get("total_nodes", 0)) == 1, "plan_summary total_nodes mismatch")
    _require(int(ps.get("planned_dispatch_groups", 0)) >= 1, "plan_summary dispatch groups must be >=1")

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

    q = np.random.rand(48, 48).astype(np.float32)
    k = np.random.rand(48, 48).astype(np.float32)
    v = np.random.rand(48, 48).astype(np.float32)
    attn_out = lc.api.attention(q, k, v, 48, 48, False, "metal")
    _require(np.asarray(attn_out).shape == (48, 48), "lc.api.attention output shape mismatch")
    _require(np.asarray(attn_out).dtype == np.float32, "lc.api.attention output dtype mismatch")

    out_lc_eager = lc_api.lightning_conv_attention_torchstrong_nchw(
        x,
        w,
        bias,
        seq=48,
        head_dim=48,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        device="metal",
        execution_mode="eager",
    )
    out_lc_graph_req = lc_api.lightning_conv_attention_torchstrong_nchw(
        x,
        w,
        bias,
        seq=48,
        head_dim=48,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        device="metal",
        execution_mode="graph",
    )
    _require(np.asarray(out_lc_eager).dtype == np.float32, "lightning conv->attn eager dtype mismatch")
    _require(np.asarray(out_lc_graph_req).dtype == np.float32, "lightning conv->attn graph-request dtype mismatch")
    _require(
        np.allclose(np.asarray(out_lc_eager).reshape(-1), np.asarray(out_lc_graph_req).reshape(-1), atol=1e-4, rtol=1e-4),
        "lightning backend graph-request should deterministically match eager path output",
    )

    # B4 coverage: graph path should also support 3x3 conv with non-default stride/pad.
    out_lc_eager_s2 = lc_api.lightning_conv_attention_torchstrong_nchw(
        x,
        w,
        bias,
        seq=64,
        head_dim=32,
        stride_h=2,
        stride_w=2,
        pad_h=1,
        pad_w=1,
        device="metal",
        execution_mode="eager",
    )
    out_lc_graph_req_s2 = lc_api.lightning_conv_attention_torchstrong_nchw(
        x,
        w,
        bias,
        seq=64,
        head_dim=32,
        stride_h=2,
        stride_w=2,
        pad_h=1,
        pad_w=1,
        device="metal",
        execution_mode="graph",
    )
    _require(
        np.allclose(np.asarray(out_lc_eager_s2).reshape(-1), np.asarray(out_lc_graph_req_s2).reshape(-1), atol=1e-4, rtol=1e-4),
        "graph-request stride2/pad1 path should deterministically match eager output",
    )

    out_lc_eager_p0 = lc_api.lightning_conv_attention_torchstrong_nchw(
        x,
        w,
        bias,
        seq=64,
        head_dim=32,
        stride_h=1,
        stride_w=1,
        pad_h=0,
        pad_w=0,
        device="metal",
        execution_mode="eager",
    )
    out_lc_graph_req_p0 = lc_api.lightning_conv_attention_torchstrong_nchw(
        x,
        w,
        bias,
        seq=64,
        head_dim=32,
        stride_h=1,
        stride_w=1,
        pad_h=0,
        pad_w=0,
        device="metal",
        execution_mode="graph",
    )
    _require(
        np.allclose(np.asarray(out_lc_eager_p0).reshape(-1), np.asarray(out_lc_graph_req_p0).reshape(-1), atol=1e-4, rtol=1e-4),
        "graph-request stride1/pad0 path should deterministically match eager output",
    )

    route_report = lc_api.lightning_conv_attention_torchstrong_nchw_route_report(
        x,
        w,
        bias,
        seq=48,
        head_dim=48,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        device="metal",
        execution_mode="graph",
        route_policy={"graph": "torch"},
    )
    _require(str(route_report.get("requested_mode", "")) == "graph", "route report requested_mode mismatch")
    _require(str(route_report.get("resolved_mode", "")) == "eager", "graph->eager fallback mode mismatch")
    _require(
        str(route_report.get("graph_fallback_reason_code", "")) == "graph_engine_not_lightning",
        "graph fallback reason code mismatch for graph_engine=torch",
    )

    # Deterministic eager fallback contract:
    # graph mode supports conv3x3 path; conv5x5 should fallback to eager deterministically.
    w5 = np.random.rand(16, 3, 5, 5).astype(np.float32)
    out_lc_eager_5x5 = lc_api.lightning_conv_attention_torchstrong_nchw(
        x,
        w5,
        bias,
        seq=48,
        head_dim=48,
        stride_h=1,
        stride_w=1,
        pad_h=2,
        pad_w=2,
        device="metal",
        execution_mode="eager",
    )
    out_lc_graph_req_5x5 = lc_api.lightning_conv_attention_torchstrong_nchw(
        x,
        w5,
        bias,
        seq=48,
        head_dim=48,
        stride_h=1,
        stride_w=1,
        pad_h=2,
        pad_w=2,
        device="metal",
        execution_mode="graph",
    )
    _require(
        np.allclose(
            np.asarray(out_lc_eager_5x5).reshape(-1),
            np.asarray(out_lc_graph_req_5x5).reshape(-1),
            atol=1e-4,
            rtol=1e-4,
        ),
        "graph-request conv5x5 path should deterministically fallback to eager output",
    )

    # Hybrid engine policy smoke:
    # torch backend should execute conv->attn path without using lightning graph mode internals.
    try:
        import torch  # noqa: F401

        has_torch = True
    except Exception:
        has_torch = False

    if has_torch:
        lc.api.set_engine("torch")
        _require(lc.api.get_engine() == "torch", "lc.api engine should be 'torch'")
        _require(lc_api.get_backend() == "torch", "integrated backend should be 'torch'")

        out_eager = lc_api.lightning_conv_attention_torchstrong_nchw(
            x,
            w,
            bias,
            seq=48,
            head_dim=48,
            stride_h=1,
            stride_w=1,
            pad_h=1,
            pad_w=1,
            device="metal",
            execution_mode="eager",
        )
        out_graph_req = lc_api.lightning_conv_attention_torchstrong_nchw(
            x,
            w,
            bias,
            seq=48,
            head_dim=48,
            stride_h=1,
            stride_w=1,
            pad_h=1,
            pad_w=1,
            device="metal",
            execution_mode="graph",
        )
        _require(np.asarray(out_eager).dtype == np.float32, "torch conv->attn eager dtype mismatch")
        _require(np.asarray(out_graph_req).dtype == np.float32, "torch conv->attn graph-request dtype mismatch")
        _require(
            np.allclose(np.asarray(out_eager).reshape(-1), np.asarray(out_graph_req).reshape(-1), atol=1e-4, rtol=1e-4),
            "torch backend graph-request should deterministically match eager path output",
        )

        out_api_eager = lc.api.conv_attention_torchstrong_nchw(
            x, w, bias, 48, 48, 1, 1, 1, 1, "metal", "eager"
        )
        out_api_graph_req = lc.api.conv_attention_torchstrong_nchw(
            x, w, bias, 48, 48, 1, 1, 1, 1, "metal", "graph"
        )
        _require(
            np.allclose(
                np.asarray(out_api_eager).reshape(-1),
                np.asarray(out_api_graph_req).reshape(-1),
                atol=1e-4,
                rtol=1e-4,
            ),
            "lc.api torch engine graph-request should deterministically match eager path output",
        )

        mixed_conv_torch_attn_lc = lc_api.lightning_conv_attention_torchstrong_nchw(
            x,
            w,
            bias,
            seq=48,
            head_dim=48,
            stride_h=1,
            stride_w=1,
            pad_h=1,
            pad_w=1,
            device="metal",
            execution_mode="eager",
            route_policy={"conv": "torch", "attention": "lightning"},
        )
        mixed_conv_lc_attn_torch = lc_api.lightning_conv_attention_torchstrong_nchw(
            x,
            w,
            bias,
            seq=48,
            head_dim=48,
            stride_h=1,
            stride_w=1,
            pad_h=1,
            pad_w=1,
            device="metal",
            execution_mode="eager",
            route_policy={"conv": "lightning", "attention": "torch"},
        )
        _require(
            np.allclose(
                np.asarray(mixed_conv_torch_attn_lc).reshape(-1),
                np.asarray(out_lc_eager).reshape(-1),
                atol=2e-3,
                rtol=2e-3,
            ),
            "mixed route conv=torch/attn=lightning should match eager lightning numerics",
        )
        _require(
            np.allclose(
                np.asarray(mixed_conv_lc_attn_torch).reshape(-1),
                np.asarray(out_lc_eager).reshape(-1),
                atol=2e-3,
                rtol=2e-3,
            ),
            "mixed route conv=lightning/attn=torch should match eager lightning numerics",
        )

        fallback_route_policy = {"graph": "torch", "conv": "torch", "attention": "lightning"}
        mixed_eager = lc_api.lightning_conv_attention_torchstrong_nchw(
            x,
            w,
            bias,
            seq=48,
            head_dim=48,
            stride_h=1,
            stride_w=1,
            pad_h=1,
            pad_w=1,
            device="metal",
            execution_mode="eager",
            route_policy=fallback_route_policy,
        )
        mixed_graph_req = lc_api.lightning_conv_attention_torchstrong_nchw(
            x,
            w,
            bias,
            seq=48,
            head_dim=48,
            stride_h=1,
            stride_w=1,
            pad_h=1,
            pad_w=1,
            device="metal",
            execution_mode="graph",
            route_policy=fallback_route_policy,
        )
        _require(
            np.allclose(
                np.asarray(mixed_eager).reshape(-1),
                np.asarray(mixed_graph_req).reshape(-1),
                atol=2e-3,
                rtol=2e-3,
            ),
            "route_policy(graph=torch) graph-request should deterministically match eager hybrid output",
        )

    lc.api.set_engine("lightning")

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
