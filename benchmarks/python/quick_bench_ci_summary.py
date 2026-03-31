#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _to_float(raw: str) -> float:
    try:
        return float(raw)
    except Exception:
        return float("nan")


def _fmt_ms(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.6f}"


def _fmt_ratio(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.2f}x"


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_summary(rows: list[dict], regress_threshold: float) -> dict:
    speedups = [_to_float(r.get("speedup_torch_over_lc", "")) for r in rows]
    finite_speedups = [v for v in speedups if math.isfinite(v)]
    regressions = [r for r in rows if math.isfinite(_to_float(r.get("speedup_torch_over_lc", ""))) and _to_float(r.get("speedup_torch_over_lc", "")) < regress_threshold]

    if finite_speedups:
        speedup_stats = {
            "count": len(finite_speedups),
            "min": min(finite_speedups),
            "median": statistics.median(finite_speedups),
            "mean": statistics.mean(finite_speedups),
            "max": max(finite_speedups),
        }
    else:
        speedup_stats = {"count": 0, "min": float("nan"), "median": float("nan"), "mean": float("nan"), "max": float("nan")}

    return {
        "rows": len(rows),
        "torch_mps_rows": len(finite_speedups),
        "regress_threshold": regress_threshold,
        "regression_count": len(regressions),
        "speedup_stats": speedup_stats,
        "regressions": [
            {
                "bench": r.get("bench", ""),
                "shape": r.get("shape", ""),
                "lightning_core_ms": _to_float(r.get("lightning_core_ms", "")),
                "torch_mps_ms": _to_float(r.get("torch_mps_ms", "")),
                "speedup_torch_over_lc": _to_float(r.get("speedup_torch_over_lc", "")),
            }
            for r in regressions
        ],
    }


def to_markdown(rows: list[dict], summary: dict) -> str:
    stats = summary["speedup_stats"]
    lines = []
    lines.append("## Quick Bench CI Summary")
    lines.append("")
    lines.append(f"- total rows: {summary['rows']}")
    lines.append(f"- rows with Torch MPS comparable data: {summary['torch_mps_rows']}")
    lines.append(f"- regression threshold (`torch/lc`): < {summary['regress_threshold']:.2f}x")
    lines.append(f"- regression rows: {summary['regression_count']}")
    lines.append(f"- speedup stats (`torch/lc`): min={_fmt_ratio(stats['min'])}, median={_fmt_ratio(stats['median'])}, mean={_fmt_ratio(stats['mean'])}, max={_fmt_ratio(stats['max'])}")
    lines.append("")
    lines.append("| bench | shape | LC (ms) | Torch MPS (ms) | torch/lc |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for r in rows:
        lc_ms = _to_float(r.get("lightning_core_ms", ""))
        mps_ms = _to_float(r.get("torch_mps_ms", ""))
        ratio = _to_float(r.get("speedup_torch_over_lc", ""))
        lines.append(
            f"| {r.get('bench', '')} | {r.get('shape', '')} | {_fmt_ms(lc_ms)} | {_fmt_ms(mps_ms)} | {_fmt_ratio(ratio)} |"
        )
    lines.append("")
    if summary["regression_count"] > 0:
        lines.append("### Regression Candidates")
        for item in summary["regressions"]:
            lines.append(
                f"- {item['bench']} / {item['shape']} "
                f"(LC={_fmt_ms(item['lightning_core_ms'])}ms, TorchMPS={_fmt_ms(item['torch_mps_ms'])}ms, "
                f"torch/lc={_fmt_ratio(item['speedup_torch_over_lc'])})"
            )
    else:
        lines.append("No regression candidates under the configured threshold.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown/json summary from quick_bench.csv")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path from quick_bench.py")
    parser.add_argument("--json", type=Path, required=True, help="Output JSON summary path")
    parser.add_argument("--md", type=Path, required=True, help="Output Markdown summary path")
    parser.add_argument("--regress-threshold", type=float, default=1.0, help="Rows below this torch/lc threshold are flagged")
    args = parser.parse_args()

    rows = load_rows(args.input)
    summary = build_summary(rows, regress_threshold=args.regress_threshold)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_text = to_markdown(rows, summary)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(md_text, encoding="utf-8")

    print(f"saved: {args.json}")
    print(f"saved: {args.md}")
    print(f"rows={summary['rows']} mps_rows={summary['torch_mps_rows']} regressions={summary['regression_count']}")


if __name__ == "__main__":
    main()
