#!/usr/bin/env python3
"""Rule-based graph fusion pilot benchmark (v2).

This benchmark compares GraphIR execution with fusion enabled vs disabled and
emits explain reports for:
- conv+relu (conv_relu_v1)
- matmul+bias+relu (matmul_bias_relu_v1)
- attention_forward+projection(matmul) (attention_proj_v1)
- qkv_pack_repeat+attention_forward+projection(matmul) (attention_qkv_proj_v1)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np

import lightning_core as lc


def _median_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return float(median(samples))


def _fmt_ms(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.6f}"


def _fmt_ratio(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}x"


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    backend = lc.backend_name().lower()
    if backend not in {"metal", "cuda", "cpu"}:
        return "cpu"
    return backend


def _build_conv_relu_graph(*, multi_consumer: bool) -> tuple[object, dict[str, int], dict[int, np.ndarray], str, str]:
    g = lc.GraphIR()
    ids: dict[str, int] = {}
    ids["x"] = g.add_tensor([1, 3, 8, 8], dtype="float32", name="x", constant=True)
    ids["w"] = g.add_tensor([16, 3, 3, 3], dtype="float32", name="w", constant=True)
    ids["b"] = g.add_tensor([16], dtype="float32", name="b", constant=True)
    ids["mid"] = g.add_tensor([1, 16, 8, 8], dtype="float32", name="mid")
    ids["out"] = g.add_tensor([1, 16, 8, 8], dtype="float32", name="out")
    g.add_node("conv2d_nchw3x3s1p1", [ids["x"], ids["w"], ids["b"]], [ids["mid"]])
    g.add_node("relu", [ids["mid"]], [ids["out"]])

    feeds: dict[int, np.ndarray] = {}
    if multi_consumer:
        ids["bias2"] = g.add_tensor([1, 16, 8, 8], dtype="float32", name="bias2", constant=True)
        ids["side"] = g.add_tensor([1, 16, 8, 8], dtype="float32", name="side")
        g.add_node("vector_add", [ids["mid"], ids["bias2"]], [ids["side"]])

    return g, ids, feeds, "conv_relu_v1", "conv(n=1,c=3->16,h=8,w=8,k=3)+relu"


def _build_mm_bias_relu_graph(*, multi_consumer: bool) -> tuple[object, dict[str, int], dict[int, np.ndarray], str, str]:
    g = lc.GraphIR()
    ids: dict[str, int] = {}
    ids["a"] = g.add_tensor([128, 128], dtype="float32", name="a", constant=True)
    ids["b"] = g.add_tensor([128, 128], dtype="float32", name="b", constant=True)
    ids["bias"] = g.add_tensor([128, 128], dtype="float32", name="bias", constant=True)
    ids["mm"] = g.add_tensor([128, 128], dtype="float32", name="mm")
    ids["add"] = g.add_tensor([128, 128], dtype="float32", name="add")
    ids["out"] = g.add_tensor([128, 128], dtype="float32", name="out")
    g.add_node("matmul", [ids["a"], ids["b"]], [ids["mm"]])
    g.add_node("vector_add", [ids["mm"], ids["bias"]], [ids["add"]])
    g.add_node("relu", [ids["add"]], [ids["out"]])

    feeds: dict[int, np.ndarray] = {}
    if multi_consumer:
        ids["side_bias"] = g.add_tensor([128, 128], dtype="float32", name="side_bias", constant=True)
        ids["side"] = g.add_tensor([128, 128], dtype="float32", name="side")
        g.add_node("matrix_sub", [ids["mm"], ids["side_bias"]], [ids["side"]])

    return g, ids, feeds, "matmul_bias_relu_v1", "matmul(128x128x128)+bias+relu"


def _build_attention_proj_graph(*, multi_consumer: bool) -> tuple[object, dict[str, int], dict[int, np.ndarray], str, str]:
    g = lc.GraphIR()
    ids: dict[str, int] = {}
    seq = 48
    head_dim = 48
    proj_out = 64
    ids["q"] = g.add_tensor([seq, head_dim], dtype="float32", name="q", constant=True)
    ids["k"] = g.add_tensor([seq, head_dim], dtype="float32", name="k", constant=True)
    ids["v"] = g.add_tensor([seq, head_dim], dtype="float32", name="v", constant=True)
    ids["proj_w"] = g.add_tensor([head_dim, proj_out], dtype="float32", name="proj_w", constant=True)
    ids["attn_mid"] = g.add_tensor([seq, head_dim], dtype="float32", name="attn_mid")
    ids["out"] = g.add_tensor([seq, proj_out], dtype="float32", name="out")
    g.add_node("attention_forward", [ids["q"], ids["k"], ids["v"]], [ids["attn_mid"]])
    g.add_node("matmul", [ids["attn_mid"], ids["proj_w"]], [ids["out"]])

    feeds: dict[int, np.ndarray] = {}
    if multi_consumer:
        ids["side_bias"] = g.add_tensor([seq, head_dim], dtype="float32", name="side_bias", constant=True)
        ids["side"] = g.add_tensor([seq, head_dim], dtype="float32", name="side")
        g.add_node("matrix_sub", [ids["attn_mid"], ids["side_bias"]], [ids["side"]])

    return g, ids, feeds, "attention_proj_v1", "attention(seq=48,d=48)+proj(48x64)"


def _build_attention_qkv_proj_graph(*, multi_consumer: bool) -> tuple[object, dict[str, int], dict[int, np.ndarray], str, str]:
    g = lc.GraphIR()
    ids: dict[str, int] = {}
    seq = 48
    head_dim = 48
    proj_out = 64
    ids["x"] = g.add_tensor([1, 3, 8, 8], dtype="float32", name="x", constant=True)
    ids["q"] = g.add_tensor([seq, head_dim], dtype="float32", name="q")
    ids["k"] = g.add_tensor([seq, head_dim], dtype="float32", name="k")
    ids["v"] = g.add_tensor([seq, head_dim], dtype="float32", name="v")
    ids["proj_w"] = g.add_tensor([head_dim, proj_out], dtype="float32", name="proj_w", constant=True)
    ids["attn_mid"] = g.add_tensor([seq, head_dim], dtype="float32", name="attn_mid")
    ids["out"] = g.add_tensor([seq, proj_out], dtype="float32", name="out")
    g.add_node("qkv_pack_repeat", [ids["x"]], [ids["q"], ids["k"], ids["v"]])
    g.add_node("attention_forward", [ids["q"], ids["k"], ids["v"]], [ids["attn_mid"]])
    g.add_node("matmul", [ids["attn_mid"], ids["proj_w"]], [ids["out"]])

    feeds: dict[int, np.ndarray] = {}
    if multi_consumer:
        ids["side_bias"] = g.add_tensor([seq, head_dim], dtype="float32", name="side_bias", constant=True)
        ids["side"] = g.add_tensor([seq, head_dim], dtype="float32", name="side")
        g.add_node("matrix_sub", [ids["attn_mid"], ids["side_bias"]], [ids["side"]])
    return g, ids, feeds, "attention_qkv_proj_v1", "qkv_pack+attention(seq=48,d=48)+proj(48x64)"


def _run_case(
    *,
    case_name: str,
    pattern: str,
    graph_builder,
    multi_consumer: bool,
    device: str,
    warmup: int,
    iters: int,
    rng: np.random.Generator,
    cost_model_min_speedup: float,
) -> tuple[dict, dict]:
    g, ids, feeds, expected_pattern, shape_text = graph_builder(multi_consumer=multi_consumer)
    assert expected_pattern == pattern

    if pattern == "conv_relu_v1":
        feeds[ids["x"]] = rng.random((1, 3, 8, 8), dtype=np.float32)
        feeds[ids["w"]] = rng.random((16, 3, 3, 3), dtype=np.float32)
        feeds[ids["b"]] = rng.random((16,), dtype=np.float32)
        if multi_consumer:
            feeds[ids["bias2"]] = rng.random((1, 16, 8, 8), dtype=np.float32)
    elif pattern == "matmul_bias_relu_v1":
        feeds[ids["a"]] = rng.random((128, 128), dtype=np.float32) * 2.0 - 1.0
        feeds[ids["b"]] = rng.random((128, 128), dtype=np.float32) * 2.0 - 1.0
        feeds[ids["bias"]] = rng.random((128, 128), dtype=np.float32) * 0.2 - 0.1
        if multi_consumer:
            feeds[ids["side_bias"]] = rng.random((128, 128), dtype=np.float32) * 0.2 - 0.1
    elif pattern == "attention_proj_v1":
        feeds[ids["q"]] = rng.random((48, 48), dtype=np.float32) * 2.0 - 1.0
        feeds[ids["k"]] = rng.random((48, 48), dtype=np.float32) * 2.0 - 1.0
        feeds[ids["v"]] = rng.random((48, 48), dtype=np.float32) * 2.0 - 1.0
        feeds[ids["proj_w"]] = rng.random((48, 64), dtype=np.float32) * 0.2 - 0.1
        if multi_consumer:
            feeds[ids["side_bias"]] = rng.random((48, 48), dtype=np.float32) * 0.2 - 0.1
    else:
        feeds[ids["x"]] = rng.random((1, 3, 8, 8), dtype=np.float32) * 2.0 - 1.0
        feeds[ids["proj_w"]] = rng.random((48, 64), dtype=np.float32) * 0.2 - 0.1
        if multi_consumer:
            feeds[ids["side_bias"]] = rng.random((48, 48), dtype=np.float32) * 0.2 - 0.1

    def run_fused_once():
        _ = g.execute_f32(
            feeds,
            preferred_device=device,
            enable_fusion_v1=True,
            fusion_pass_order="attention_qkv,attention,matmul,conv",
            enable_fusion_cost_model_v1=True,
            fusion_cost_min_speedup=cost_model_min_speedup,
        )

    def run_unfused_once():
        _ = g.execute_f32(
            feeds,
            preferred_device=device,
            enable_fusion_v1=False,
            fusion_pass_order="attention_qkv,attention,matmul,conv",
            enable_fusion_cost_model_v1=True,
            fusion_cost_min_speedup=cost_model_min_speedup,
        )

    fused_ms = _median_ms(run_fused_once, warmup, iters)
    unfused_ms = _median_ms(run_unfused_once, warmup, iters)

    fused_out = np.asarray(
        g.execute_f32(
            feeds,
            preferred_device=device,
            enable_fusion_v1=True,
            fusion_pass_order="attention_qkv,attention,matmul,conv",
            enable_fusion_cost_model_v1=True,
            fusion_cost_min_speedup=cost_model_min_speedup,
        )["values"][ids["out"]],
        dtype=np.float32,
    ).reshape(-1)
    unfused_out = np.asarray(
        g.execute_f32(
            feeds,
            preferred_device=device,
            enable_fusion_v1=False,
            fusion_pass_order="attention_qkv,attention,matmul,conv",
            enable_fusion_cost_model_v1=True,
            fusion_cost_min_speedup=cost_model_min_speedup,
        )["values"][ids["out"]],
        dtype=np.float32,
    ).reshape(-1)
    abs_diff = np.abs(fused_out - unfused_out)
    rel_diff = abs_diff / (np.abs(unfused_out) + 1.0e-12)

    decisions_enabled = list(
        g.fusion_report(
            preferred_device=device,
            enable_fusion_v1=True,
            fusion_pass_order="attention_qkv,attention,matmul,conv",
            enable_fusion_cost_model_v1=True,
            fusion_cost_min_speedup=cost_model_min_speedup,
        )
    )
    decisions_disabled = list(
        g.fusion_report(
            preferred_device=device,
            enable_fusion_v1=False,
            fusion_pass_order="attention_qkv,attention,matmul,conv",
            enable_fusion_cost_model_v1=True,
            fusion_cost_min_speedup=cost_model_min_speedup,
        )
    )
    decisions_cost_reject = list(
        g.fusion_report(
            preferred_device=device,
            enable_fusion_v1=True,
            fusion_pass_order="attention_qkv,attention,matmul,conv",
            enable_fusion_cost_model_v1=True,
            fusion_cost_min_speedup=1000.0,
        )
    )

    decision_enabled = next((d for d in decisions_enabled if d.get("pattern") == pattern), None)
    decision_disabled = next((d for d in decisions_disabled if d.get("pattern") == pattern), None)
    decision_cost_reject = next((d for d in decisions_cost_reject if d.get("pattern") == pattern), None)

    row = {
        "suite": "fusion_pilot",
        "pattern": pattern,
        "bench": case_name,
        "shape": shape_text,
        "status": "ok",
        "device": device,
        "fused_ms": fused_ms,
        "unfused_ms": unfused_ms,
        "fused_over_unfused": (fused_ms / unfused_ms) if unfused_ms > 0 else float("nan"),
        "unfused_over_fused": (unfused_ms / fused_ms) if fused_ms > 0 else float("nan"),
        "allclose": bool(np.all(abs_diff <= (1.0e-4 + 1.0e-4 * np.abs(unfused_out)))),
        "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "max_rel_diff": float(np.max(rel_diff)) if rel_diff.size else 0.0,
        "fusion_applied": bool(decision_enabled and decision_enabled.get("fused", False)),
        "fusion_reason": str(decision_enabled.get("reason", "n/a")) if decision_enabled else "n/a",
        "fusion_disabled_reason": str(decision_disabled.get("reason", "n/a")) if decision_disabled else "n/a",
        "cost_model_reject_reason": str(decision_cost_reject.get("reason", "n/a")) if decision_cost_reject else "n/a",
        "estimated_speedup": float(decision_enabled.get("estimated_speedup", float("nan"))) if decision_enabled else float("nan"),
        "note": "",
    }

    detail = {
        "row": row,
        "fusion_report_enabled": decisions_enabled,
        "fusion_report_disabled": decisions_disabled,
        "fusion_report_cost_reject": decisions_cost_reject,
    }
    return row, detail


def _unsupported_row(pattern: str, bench: str, shape: str, device: str, reason: str) -> tuple[dict, dict]:
    row = {
        "suite": "fusion_pilot",
        "pattern": pattern,
        "bench": bench,
        "shape": shape,
        "status": "unsupported",
        "device": device,
        "fused_ms": float("nan"),
        "unfused_ms": float("nan"),
        "fused_over_unfused": float("nan"),
        "unfused_over_fused": float("nan"),
        "allclose": False,
        "max_abs_diff": float("nan"),
        "max_rel_diff": float("nan"),
        "fusion_applied": False,
        "fusion_reason": "n/a",
        "fusion_disabled_reason": "n/a",
        "cost_model_reject_reason": "n/a",
        "estimated_speedup": float("nan"),
        "note": reason,
    }
    return row, {"row": row, "error": reason}


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "suite",
        "pattern",
        "bench",
        "shape",
        "status",
        "device",
        "fused_ms",
        "unfused_ms",
        "fused_over_unfused",
        "unfused_over_fused",
        "allclose",
        "max_abs_diff",
        "max_rel_diff",
        "fusion_applied",
        "fusion_reason",
        "fusion_disabled_reason",
        "cost_model_reject_reason",
        "estimated_speedup",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _summary(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    ratios = [float(r["fused_over_unfused"]) for r in ok_rows if math.isfinite(float(r["fused_over_unfused"]))]
    by_pattern: dict[str, dict] = {}
    for pattern in sorted(set(r["pattern"] for r in rows)):
        sub = [r for r in ok_rows if r["pattern"] == pattern]
        sub_ratios = [float(r["fused_over_unfused"]) for r in sub if math.isfinite(float(r["fused_over_unfused"]))]
        by_pattern[pattern] = {
            "ok_cases": len(sub),
            "fusion_applied_cases": sum(1 for r in sub if bool(r.get("fusion_applied", False))),
            "median_fused_over_unfused": float(median(sub_ratios)) if sub_ratios else float("nan"),
        }
    return {
        "total_cases": len(rows),
        "ok_cases": len(ok_rows),
        "unsupported_cases": len(rows) - len(ok_rows),
        "fusion_applied_cases": sum(1 for r in ok_rows if bool(r.get("fusion_applied", False))),
        "allclose_ok_cases": sum(1 for r in ok_rows if bool(r.get("allclose", False))),
        "median_fused_over_unfused": float(median(ratios)) if ratios else float("nan"),
        "patterns": by_pattern,
    }


def _to_markdown(
    rows: list[dict],
    summary: dict,
    device: str,
    warmup: int,
    iters: int,
    conv_threshold: float,
    matmul_threshold: float,
    attention_threshold: float,
) -> str:
    lines = []
    lines.append("## Fusion Pilot (v2)")
    lines.append("")
    lines.append(f"- device: `{device}`")
    lines.append(f"- config: warmup={warmup}, iters={iters}")
    lines.append(f"- total cases: {summary['total_cases']} (ok={summary['ok_cases']}, unsupported={summary['unsupported_cases']})")
    lines.append(f"- fusion applied cases: {summary['fusion_applied_cases']}/{summary['ok_cases']}")
    lines.append(
        f"- median fused/unfused: {_fmt_ratio(summary['median_fused_over_unfused'])}, "
        f"perf gate thresholds: conv<={conv_threshold:.3f}x, matmul<={matmul_threshold:.3f}x, "
        f"attention<={attention_threshold:.3f}x"
    )
    lines.append("")
    lines.append("| pattern | bench | status | fused (ms) | unfused (ms) | fused/unfused | fusion_applied | reason | est speedup |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |")
    for r in rows:
        lines.append(
            f"| {r['pattern']} | {r['bench']} | {r['status']} | {_fmt_ms(float(r['fused_ms']))} | {_fmt_ms(float(r['unfused_ms']))} | "
            f"{_fmt_ratio(float(r['fused_over_unfused']))} | {r['fusion_applied']} | {r['fusion_reason']} | {_fmt_ratio(float(r['estimated_speedup']))} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Fusion pilot benchmark for conv+relu and matmul+bias+relu")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "metal", "cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--iters", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260408)
    p.add_argument("--cost-model-min-speedup", type=float, default=1.01)
    p.add_argument("--csv", type=Path, default=Path("benchmarks/reports/ci/fusion_pilot.csv"))
    p.add_argument("--json", type=Path, default=Path("benchmarks/reports/ci/fusion_pilot.json"))
    p.add_argument("--md", type=Path, default=Path("benchmarks/reports/ci/fusion_pilot.md"))
    p.add_argument(
        "--max-fused-over-unfused",
        type=float,
        default=float("nan"),
        help="Legacy/global threshold. When finite, overrides both per-pattern thresholds.",
    )
    p.add_argument(
        "--max-fused-over-unfused-conv",
        type=float,
        default=1.15,
        help="Fail when conv eligible fused case exceeds this ratio.",
    )
    p.add_argument(
        "--max-fused-over-unfused-matmul",
        type=float,
        default=1.90,
        help="Fail when matmul eligible fused case exceeds this ratio.",
    )
    p.add_argument(
        "--max-fused-over-unfused-attention",
        type=float,
        default=2.20,
        help="Fail when attention eligible fused case exceeds this ratio.",
    )
    p.add_argument(
        "--fail-on-empty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero when no case succeeds.",
    )
    args = p.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = _resolve_device(args.device)

    rows: list[dict] = []
    details: list[dict] = []
    cases = [
        ("conv_relu_eligible", "conv_relu_v1", _build_conv_relu_graph, False),
        ("conv_relu_multi_consumer", "conv_relu_v1", _build_conv_relu_graph, True),
        ("matmul_bias_relu_eligible", "matmul_bias_relu_v1", _build_mm_bias_relu_graph, False),
        ("matmul_bias_relu_multi_consumer", "matmul_bias_relu_v1", _build_mm_bias_relu_graph, True),
        ("attention_proj_eligible", "attention_proj_v1", _build_attention_proj_graph, False),
        ("attention_proj_multi_consumer", "attention_proj_v1", _build_attention_proj_graph, True),
        ("attention_qkv_proj_eligible", "attention_qkv_proj_v1", _build_attention_qkv_proj_graph, False),
        ("attention_qkv_proj_multi_consumer", "attention_qkv_proj_v1", _build_attention_qkv_proj_graph, True),
    ]

    for case_name, pattern, builder, multi_consumer in cases:
        if pattern == "conv_relu_v1":
            shape = "conv(n=1,c=3->16,h=8,w=8,k=3)+relu"
        elif pattern == "matmul_bias_relu_v1":
            shape = "matmul(128x128x128)+bias+relu"
        elif pattern == "attention_proj_v1":
            shape = "attention(seq=48,d=48)+proj(48x64)"
        else:
            shape = "qkv_pack+attention(seq=48,d=48)+proj(48x64)"
        try:
            row, detail = _run_case(
                case_name=case_name,
                pattern=pattern,
                graph_builder=builder,
                multi_consumer=multi_consumer,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                rng=rng,
                cost_model_min_speedup=float(args.cost_model_min_speedup),
            )
        except Exception as exc:
            row, detail = _unsupported_row(pattern, case_name, shape, device, str(exc))
        rows.append(row)
        details.append(detail)

    summary = _summary(rows)
    conv_threshold = float(args.max_fused_over_unfused_conv)
    matmul_threshold = float(args.max_fused_over_unfused_matmul)
    attention_threshold = float(args.max_fused_over_unfused_attention)
    if math.isfinite(float(args.max_fused_over_unfused)):
        conv_threshold = float(args.max_fused_over_unfused)
        matmul_threshold = float(args.max_fused_over_unfused)
        attention_threshold = float(args.max_fused_over_unfused)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "fusion_pilot",
        "backend_name": lc.backend_name(),
        "device": device,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "cost_model_min_speedup": args.cost_model_min_speedup,
        "max_fused_over_unfused": args.max_fused_over_unfused,
        "max_fused_over_unfused_conv": conv_threshold,
        "max_fused_over_unfused_matmul": matmul_threshold,
        "max_fused_over_unfused_attention": attention_threshold,
        "summary": summary,
        "rows": rows,
        "details": details,
    }

    _write_csv(args.csv, rows)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(
        _to_markdown(
            rows,
            summary,
            device,
            args.warmup,
            args.iters,
            conv_threshold,
            matmul_threshold,
            attention_threshold,
        ),
        encoding="utf-8",
    )

    print(f"saved: {args.csv}")
    print(f"saved: {args.json}")
    print(f"saved: {args.md}")

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        if args.fail_on_empty:
            raise SystemExit(2)
        return
    if any(not bool(r.get("allclose", False)) for r in ok_rows):
        raise SystemExit(3)

    eligible_rows = [
        r
        for r in ok_rows
        if r["bench"] in {
            "conv_relu_eligible",
            "matmul_bias_relu_eligible",
            "attention_proj_eligible",
            "attention_qkv_proj_eligible",
        }
    ]
    for r in eligible_rows:
        ratio = float(r.get("fused_over_unfused", float("nan")))
        if not bool(r.get("fusion_applied", False)) or not math.isfinite(ratio):
            continue
        if r.get("pattern") == "conv_relu_v1":
            threshold = conv_threshold
        elif r.get("pattern") == "matmul_bias_relu_v1":
            threshold = matmul_threshold
        else:
            threshold = attention_threshold
        if ratio > threshold:
            raise SystemExit(4)


if __name__ == "__main__":
    main()
