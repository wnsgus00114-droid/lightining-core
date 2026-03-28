from __future__ import annotations

import json
import math
import statistics
from datetime import date
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent.parent
BENCH_DIR = WORKSPACE_ROOT / "benchmark_results"
REPORT_DIR = THIS_DIR / "reports" / str(date.today())


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _series_stats(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "avg": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "min": float("nan"),
        }
    return {
        "count": len(values),
        "avg": statistics.mean(values),
        "median": statistics.median(values),
        "max": max(values),
        "min": min(values),
    }


def _normalize_rows() -> list[dict]:
    ml = _load_json(BENCH_DIR / "ml_all_bench.json")
    dl = _load_json(BENCH_DIR / "large_gemm_auto_sweep.json")
    ai = _load_json(BENCH_DIR / "ai_model_all_bench.json")

    rows: list[dict] = []

    for src in ml.get("rows", []):
        rows.append(
            {
                "suite": "ml",
                "bench": src["bench"],
                "shape": src["shape"],
                "lc_ms": float(src["lightning_core_ms"]),
                "torch_mps_ms": float(src["torch_mps_ms"]),
                "exia_ms": float(src["exia_standalone_ms"]),
            }
        )

    for src in dl.get("best_rows_per_shape", []):
        rows.append(
            {
                "suite": "dl",
                "bench": "large_gemm_best_policy",
                "shape": src["shape"],
                "lc_ms": float(src["lc_best_ms"]),
                "torch_mps_ms": float(src["torch_mps_ms"]),
                "exia_ms": float(src["exia_standalone_ms"]),
            }
        )

    for src in ai.get("rows", []):
        rows.append(
            {
                "suite": "ai",
                "bench": src["bench"],
                "shape": src["shape"],
                "lc_ms": float(src["lightning_core_ms"]),
                "torch_mps_ms": float(src["torch_mps_ms"]),
                "exia_ms": float(src["exia_standalone_ms"]),
            }
        )

    return rows


def _build_pair_stats(rows: list[dict], key: str) -> dict:
    valid = [r for r in rows if _is_finite(r["lc_ms"]) and _is_finite(r[key]) and r["lc_ms"] > 0.0]

    speedup = [r[key] / r["lc_ms"] for r in valid]
    gap_ms = [r[key] - r["lc_ms"] for r in valid]
    abs_gap_ms = [abs(v) for v in gap_ms]

    max_abs_row = None
    if valid:
        max_abs_row = max(valid, key=lambda r: abs(r[key] - r["lc_ms"]))

    return {
        "speedup_ratio": _series_stats(speedup),
        "gap_ms": _series_stats(gap_ms),
        "abs_gap_ms": _series_stats(abs_gap_ms),
        "max_abs_gap_case": (
            {
                "suite": max_abs_row["suite"],
                "bench": max_abs_row["bench"],
                "shape": max_abs_row["shape"],
                "lc_ms": max_abs_row["lc_ms"],
                "other_ms": max_abs_row[key],
                "abs_gap_ms": abs(max_abs_row[key] - max_abs_row["lc_ms"]),
                "ratio": max_abs_row[key] / max_abs_row["lc_ms"],
            }
            if max_abs_row is not None
            else None
        ),
    }


def _build_summary(rows: list[dict]) -> dict:
    by_suite: dict[str, dict] = {}
    for suite in ["ml", "dl", "ai"]:
        srows = [r for r in rows if r["suite"] == suite]
        by_suite[suite] = {
            "count": len(srows),
            "torch_mps_vs_lc": _build_pair_stats(srows, "torch_mps_ms"),
            "exia_vs_lc": _build_pair_stats(srows, "exia_ms"),
        }

    overall = {
        "count": len(rows),
        "torch_mps_vs_lc": _build_pair_stats(rows, "torch_mps_ms"),
        "exia_vs_lc": _build_pair_stats(rows, "exia_ms"),
    }

    return {
        "generated_at": date.today().isoformat(),
        "source_dir": str(BENCH_DIR),
        "overall": overall,
        "by_suite": by_suite,
    }


def _fmt(v: float) -> str:
    if not _is_finite(v):
        return "nan"
    return f"{v:.4f}"


def _to_markdown(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# Cross-Suite Benchmark Summary")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")

    def add_block(title: str, block: dict) -> None:
        lines.append(f"## {title}")
        lines.append("")
        t = block["torch_mps_vs_lc"]
        e = block["exia_vs_lc"]

        lines.append("- Torch MPS vs LC")
        lines.append(
            f"  - speedup ratio avg/median/max: {_fmt(t['speedup_ratio']['avg'])} / {_fmt(t['speedup_ratio']['median'])} / {_fmt(t['speedup_ratio']['max'])}"
        )
        lines.append(
            f"  - gap(ms) avg/median/max_abs: {_fmt(t['gap_ms']['avg'])} / {_fmt(t['gap_ms']['median'])} / {_fmt(t['abs_gap_ms']['max'])}"
        )

        lines.append("- Exia standalone vs LC")
        lines.append(
            f"  - speedup ratio avg/median/max: {_fmt(e['speedup_ratio']['avg'])} / {_fmt(e['speedup_ratio']['median'])} / {_fmt(e['speedup_ratio']['max'])}"
        )
        lines.append(
            f"  - gap(ms) avg/median/max_abs: {_fmt(e['gap_ms']['avg'])} / {_fmt(e['gap_ms']['median'])} / {_fmt(e['abs_gap_ms']['max'])}"
        )
        lines.append("")

    add_block("Overall", summary["overall"])
    for suite in ["ml", "dl", "ai"]:
        add_block(f"Suite: {suite.upper()}", summary["by_suite"][suite])

    return "\n".join(lines) + "\n"


def main() -> None:
    rows = _normalize_rows()
    summary = _build_summary(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "cross_suite_summary.json"
    md_path = REPORT_DIR / "cross_suite_summary.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(_to_markdown(summary))

    print(f"saved: {json_path}")
    print(f"saved: {md_path}")


if __name__ == "__main__":
    main()
