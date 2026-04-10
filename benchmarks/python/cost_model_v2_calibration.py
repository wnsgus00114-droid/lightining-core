#!/usr/bin/env python3
"""Cost model v2 calibration helper.

Generates a profile-signature keyed coefficient suggestion by comparing
plan-summary estimated costs and measured execution latency.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np

import lightning_core as lc


def _median_ns(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append(float(t1 - t0))
    return float(median(samples))


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    backend = str(lc.backend_name()).strip().lower()
    if backend in {"metal", "cuda", "cpu"}:
        return backend
    return "cpu"


def _build_matmul_bias_relu_case(rng: np.random.Generator):
    g = lc.GraphIR()
    a = g.add_tensor([128, 128], dtype="float32", name="a", constant=True)
    b = g.add_tensor([128, 128], dtype="float32", name="b", constant=True)
    bias = g.add_tensor([128, 128], dtype="float32", name="bias", constant=True)
    mm = g.add_tensor([128, 128], dtype="float32", name="mm")
    add = g.add_tensor([128, 128], dtype="float32", name="add")
    out = g.add_tensor([128, 128], dtype="float32", name="out")
    g.add_node("matmul", [a, b], [mm])
    g.add_node("vector_add", [mm, bias], [add])
    g.add_node("relu", [add], [out])
    feeds = {
        a: rng.random((128, 128), dtype=np.float32) * 2.0 - 1.0,
        b: rng.random((128, 128), dtype=np.float32) * 2.0 - 1.0,
        bias: rng.random((128, 128), dtype=np.float32) * 0.2 - 0.1,
    }
    return "matmul_bias_relu", g, feeds


def _build_conv_relu_case(rng: np.random.Generator):
    g = lc.GraphIR()
    x = g.add_tensor([1, 3, 16, 16], dtype="float32", name="x", constant=True)
    w = g.add_tensor([16, 3, 3, 3], dtype="float32", name="w", constant=True)
    b = g.add_tensor([16], dtype="float32", name="b", constant=True)
    mid = g.add_tensor([1, 16, 16, 16], dtype="float32", name="mid")
    out = g.add_tensor([1, 16, 16, 16], dtype="float32", name="out")
    g.add_node("conv2d_nchw3x3s1p1", [x, w, b], [mid])
    g.add_node("relu", [mid], [out])
    feeds = {
        x: rng.random((1, 3, 16, 16), dtype=np.float32) * 2.0 - 1.0,
        w: rng.random((16, 3, 3, 3), dtype=np.float32) * 0.2 - 0.1,
        b: rng.random((16,), dtype=np.float32) * 0.1 - 0.05,
    }
    return "conv_relu", g, feeds


def _build_attention_qkv_proj_case(rng: np.random.Generator):
    g = lc.GraphIR()
    x = g.add_tensor([1, 3, 8, 8], dtype="float32", name="x", constant=True)
    q = g.add_tensor([48, 48], dtype="float32", name="q")
    k = g.add_tensor([48, 48], dtype="float32", name="k")
    v = g.add_tensor([48, 48], dtype="float32", name="v")
    proj_w = g.add_tensor([48, 64], dtype="float32", name="proj_w", constant=True)
    attn_mid = g.add_tensor([48, 48], dtype="float32", name="attn_mid")
    out = g.add_tensor([48, 64], dtype="float32", name="out")
    g.add_node("qkv_pack_repeat", [x], [q, k, v])
    g.add_node("attention_forward", [q, k, v], [attn_mid])
    g.add_node("matmul", [attn_mid, proj_w], [out])
    feeds = {
        x: rng.random((1, 3, 8, 8), dtype=np.float32) * 2.0 - 1.0,
        proj_w: rng.random((48, 64), dtype=np.float32) * 0.2 - 0.1,
    }
    return "attention_qkv_proj", g, feeds


def _to_markdown(device: str, signature: str, rows: list[dict], suggested: dict) -> str:
    lines = []
    lines.append("## Cost Model v2 Calibration")
    lines.append("")
    lines.append(f"- device: `{device}`")
    lines.append(f"- profile signature: `{signature}`")
    ok_rows = [r for r in rows if str(r.get("status", "ok")) == "ok"]
    unsupported_rows = [r for r in rows if str(r.get("status", "ok")) != "ok"]
    lines.append(f"- total cases: `{len(rows)}` (ok=`{len(ok_rows)}`, unsupported=`{len(unsupported_rows)}`)")
    lines.append("")
    lines.append("| case | status | measured_ns | estimated_ns | measured/estimated |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for r in rows:
        ratio = float(r.get("ratio_measured_over_estimated", float("nan")))
        ratio_s = "n/a" if not math.isfinite(ratio) else f"{ratio:.3f}x"
        measured_ns = float(r.get("measured_ns", float("nan")))
        estimated_ns = float(r.get("estimated_ns", float("nan")))
        measured_s = "n/a" if not math.isfinite(measured_ns) else f"{measured_ns:.0f}"
        estimated_s = "n/a" if not math.isfinite(estimated_ns) else f"{estimated_ns:.0f}"
        status = str(r.get("status", "ok"))
        lines.append(f"| {r['case']} | {status} | {measured_s} | {estimated_s} | {ratio_s} |")
    if unsupported_rows:
        lines.append("")
        lines.append("### Unsupported Cases")
        for r in unsupported_rows:
            lines.append(f"- `{r['case']}`: `{r.get('reason', 'unknown')}`")
    lines.append("")
    lines.append("### Suggested Coefficients")
    for k in sorted(suggested.keys()):
        lines.append(f"- `{k}`: `{suggested[k]}`")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate Lightning Core graph cost-model v2 coefficients")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "metal", "cpu", "cuda"])
    p.add_argument("--signature", type=str, default="auto")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iters", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260410)
    p.add_argument("--json", type=Path, default=Path("benchmarks/reports/ci/cost_model_v2_calibration.json"))
    p.add_argument("--md", type=Path, default=Path("benchmarks/reports/ci/cost_model_v2_calibration.md"))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    device = _resolve_device(args.device)
    builders = [
        _build_matmul_bias_relu_case,
        _build_conv_relu_case,
        _build_attention_qkv_proj_case,
    ]

    rows: list[dict] = []
    scales = []
    for builder in builders:
        case_name, graph, feeds = builder(rng)
        try:
            plan = dict(
                graph.plan_summary(
                    preferred_device=device,
                    enable_fusion_v1=True,
                    fusion_pass_order="attention_qkv,attention,matmul,conv",
                    cost_profile_signature=args.signature,
                )
            )
            summary = dict(plan.get("summary", {}))
            estimated_ns = float(summary.get("estimated_total_cost_ns", float("nan")))
            measured_ns = _median_ns(
                lambda: graph.execute_f32(
                    feeds,
                    preferred_device=device,
                    enable_fusion_v1=True,
                    fusion_pass_order="attention_qkv,attention,matmul,conv",
                    cost_profile_signature=args.signature,
                ),
                args.warmup,
                args.iters,
            )
            ratio = measured_ns / estimated_ns if (math.isfinite(estimated_ns) and estimated_ns > 0.0) else float("nan")
            if math.isfinite(ratio) and ratio > 0.0:
                scales.append(ratio)
            rows.append(
                {
                    "case": case_name,
                    "status": "ok",
                    "measured_ns": measured_ns,
                    "estimated_ns": estimated_ns,
                    "ratio_measured_over_estimated": ratio,
                    "planner_score_model": str(summary.get("planner_score_model", "unknown")),
                    "cost_profile_signature": str(summary.get("cost_profile_signature", "default")),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "case": case_name,
                    "status": "unsupported",
                    "reason": str(exc),
                    "measured_ns": float("nan"),
                    "estimated_ns": float("nan"),
                    "ratio_measured_over_estimated": float("nan"),
                    "planner_score_model": "unsupported",
                    "cost_profile_signature": args.signature,
                }
            )

    median_scale = float(median(scales)) if scales else 1.0
    suggested = {
        "signature": args.signature,
        "scale_factor": round(median_scale, 6),
        "cost_launch_overhead_ns": round(12000.0 * median_scale, 6),
        "cost_transfer_overhead_ns": round(4000.0 * median_scale, 6),
        "cost_sync_boundary_ns": round(2500.0 * median_scale, 6),
        "cost_elementwise_per_element_ns": round(0.50 * median_scale, 9),
        "cost_matmul_per_mac_ns": round(0.0035 * median_scale, 9),
        "cost_conv_per_mac_ns": round(0.0022 * median_scale, 9),
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "cost_model_v2_calibration",
        "backend_name": str(lc.backend_name()),
        "device": device,
        "signature": args.signature,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "rows": rows,
        "ok_cases": len([r for r in rows if str(r.get("status", "ok")) == "ok"]),
        "unsupported_cases": len([r for r in rows if str(r.get("status", "ok")) != "ok"]),
        "suggested_profile": suggested,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.md.write_text(_to_markdown(device, args.signature, rows, suggested), encoding="utf-8")

    print(f"saved: {args.json}")
    print(f"saved: {args.md}")


if __name__ == "__main__":
    main()
