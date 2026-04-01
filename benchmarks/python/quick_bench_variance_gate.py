#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _to_float(raw: str) -> float:
    try:
        return float(raw)
    except Exception:
        return float("nan")


def _fmt_ms(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.6f}"


def _fmt_pct(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.2f}%"


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_runs_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_idx",
        "suite",
        "bench",
        "shape",
        "lightning_core_ms",
        "torch_mps_ms",
        "speedup_torch_over_lc",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def run_quick_bench(
    *,
    run_idx: int,
    bench_script: Path,
    repo_root: Path,
    device: str,
    warmup: int,
    iters: int,
    seed: int,
    work_dir: Path,
) -> tuple[list[dict], dict]:
    run_csv = work_dir / f"quick_bench_run_{run_idx:02d}.csv"
    run_log = work_dir / f"quick_bench_run_{run_idx:02d}.log"

    cmd = [
        sys.executable,
        str(bench_script),
        "--device",
        device,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--seed",
        str(seed),
        "--out",
        str(run_csv),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    log_text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    run_log.write_text(log_text, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(
            f"quick_bench failed at run_idx={run_idx} exit={proc.returncode}. "
            f"See log: {run_log}"
        )

    rows = load_rows(run_csv)
    for row in rows:
        row["run_idx"] = str(run_idx)
    run_meta = {
        "run_idx": run_idx,
        "csv": str(run_csv),
        "log": str(run_log),
        "elapsed_ms": elapsed_ms,
        "command": cmd,
        "row_count": len(rows),
    }
    return rows, run_meta


def build_summary(
    rows: list[dict],
    run_count: int,
    threshold_pct: float,
    device: str,
    warmup: int,
    iters: int,
    seed: int,
    enforce_per_case: bool,
    trim_runs: int,
) -> dict:
    grouped: dict[tuple[str, str], list[float]] = {}
    run_totals: dict[str, float] = {}
    for row in rows:
        key = (row.get("bench", ""), row.get("shape", ""))
        lc_ms = _to_float(row.get("lightning_core_ms", ""))
        grouped.setdefault(key, []).append(lc_ms)
        run_idx = row.get("run_idx", "")
        if run_idx != "" and math.isfinite(lc_ms):
            run_totals[run_idx] = run_totals.get(run_idx, 0.0) + lc_ms

    case_stats = []
    for (bench, shape), vals in grouped.items():
        finite_vals = [v for v in vals if math.isfinite(v)]
        mean_ms = statistics.mean(finite_vals) if finite_vals else float("nan")
        stdev_ms = statistics.pstdev(finite_vals) if len(finite_vals) >= 2 else 0.0
        cv_pct = (stdev_ms / mean_ms * 100.0) if math.isfinite(mean_ms) and mean_ms > 0 else float("nan")
        spread_pct = (
            ((max(finite_vals) - min(finite_vals)) / mean_ms * 100.0)
            if len(finite_vals) >= 2 and math.isfinite(mean_ms) and mean_ms > 0
            else float("nan")
        )
        threshold_exceeded = math.isfinite(cv_pct) and cv_pct > threshold_pct
        case_stats.append(
            {
                "bench": bench,
                "shape": shape,
                "samples": len(finite_vals),
                "mean_lc_ms": mean_ms,
                "stdev_lc_ms": stdev_ms,
                "cv_pct": cv_pct,
                "spread_pct": spread_pct,
                "min_lc_ms": min(finite_vals) if finite_vals else float("nan"),
                "max_lc_ms": max(finite_vals) if finite_vals else float("nan"),
                "threshold_exceeded": threshold_exceeded,
            }
        )

    case_stats.sort(key=lambda x: x["cv_pct"] if math.isfinite(x["cv_pct"]) else -1.0, reverse=True)
    failed_cases = [c for c in case_stats if c["threshold_exceeded"]]
    max_cv = max((c["cv_pct"] for c in case_stats if math.isfinite(c["cv_pct"])), default=float("nan"))
    run_total_rows = []
    run_total_values = []
    for run_idx in sorted(run_totals.keys(), key=lambda x: int(x)):
        v = run_totals[run_idx]
        run_total_rows.append({"run_idx": int(run_idx), "total_lc_ms": v})
        run_total_values.append(v)

    if len(run_total_values) >= 2:
        suite_total_mean_ms = statistics.mean(run_total_values)
        suite_total_stdev_ms = statistics.pstdev(run_total_values)
        suite_total_cv_pct = (
            (suite_total_stdev_ms / suite_total_mean_ms * 100.0)
            if suite_total_mean_ms > 0
            else float("nan")
        )
    else:
        suite_total_mean_ms = float("nan")
        suite_total_stdev_ms = float("nan")
        suite_total_cv_pct = float("nan")

    sorted_totals = sorted(run_total_values)
    if trim_runs > 0 and len(sorted_totals) > (2 * trim_runs):
        trimmed_totals = sorted_totals[trim_runs : len(sorted_totals) - trim_runs]
    else:
        trimmed_totals = sorted_totals

    if len(trimmed_totals) >= 2:
        suite_total_trimmed_mean_ms = statistics.mean(trimmed_totals)
        suite_total_trimmed_stdev_ms = statistics.pstdev(trimmed_totals)
        suite_total_trimmed_cv_pct = (
            (suite_total_trimmed_stdev_ms / suite_total_trimmed_mean_ms * 100.0)
            if suite_total_trimmed_mean_ms > 0
            else float("nan")
        )
    else:
        suite_total_trimmed_mean_ms = float("nan")
        suite_total_trimmed_stdev_ms = float("nan")
        suite_total_trimmed_cv_pct = float("nan")

    overall_pass = (
        math.isfinite(suite_total_trimmed_cv_pct)
        and suite_total_trimmed_cv_pct <= threshold_pct
    )
    if enforce_per_case:
        overall_pass = overall_pass and len(failed_cases) == 0

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": {
            "overall_pass": overall_pass,
            "mode": "suite_total_trimmed_cv",
            "variance_threshold_pct": threshold_pct,
            "enforce_per_case": enforce_per_case,
            "trim_runs": trim_runs,
            "suite_total_mean_ms": suite_total_mean_ms,
            "suite_total_stdev_ms": suite_total_stdev_ms,
            "suite_total_cv_pct": suite_total_cv_pct,
            "suite_total_trimmed_mean_ms": suite_total_trimmed_mean_ms,
            "suite_total_trimmed_stdev_ms": suite_total_trimmed_stdev_ms,
            "suite_total_trimmed_cv_pct": suite_total_trimmed_cv_pct,
            "per_case_failed_count": len(failed_cases),
            "max_cv_pct": max_cv,
            "run_count": run_count,
            "device": device,
            "warmup": warmup,
            "iters": iters,
            "seed": seed,
        },
        "run_totals": run_total_rows,
        "case_count": len(case_stats),
        "cases": case_stats,
        "failed_cases": failed_cases,
    }


def to_markdown(summary: dict, run_meta: list[dict]) -> str:
    gate = summary["gate"]
    lines = []
    lines.append("## Quick Bench Variance Gate")
    lines.append("")
    lines.append(f"- gate status: {'PASS' if gate['overall_pass'] else 'FAIL'}")
    lines.append(f"- gate mode: {gate['mode']}")
    lines.append(f"- variance threshold (CV): <= {gate['variance_threshold_pct']:.2f}%")
    lines.append(
        f"- suite total LC latency (raw): mean={_fmt_ms(gate['suite_total_mean_ms'])}ms, "
        f"stdev={_fmt_ms(gate['suite_total_stdev_ms'])}ms, CV={_fmt_pct(gate['suite_total_cv_pct'])}"
    )
    lines.append(
        f"- suite total LC latency (trimmed): mean={_fmt_ms(gate['suite_total_trimmed_mean_ms'])}ms, "
        f"stdev={_fmt_ms(gate['suite_total_trimmed_stdev_ms'])}ms, "
        f"CV={_fmt_pct(gate['suite_total_trimmed_cv_pct'])} (trim_runs={gate['trim_runs']})"
    )
    lines.append(
        f"- per-case threshold exceeded: {gate['per_case_failed_count']} "
        f"(enforce_per_case={gate['enforce_per_case']})"
    )
    lines.append(f"- max per-case observed CV: {_fmt_pct(gate['max_cv_pct'])}")
    lines.append(
        f"- config: runs={gate['run_count']}, device={gate['device']}, "
        f"warmup={gate['warmup']}, iters={gate['iters']}, seed={gate['seed']}"
    )
    lines.append("")
    lines.append("| bench | shape | samples | mean LC (ms) | stdev (ms) | CV (%) | spread (%) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for c in summary["cases"]:
        lines.append(
            f"| {c['bench']} | {c['shape']} | {c['samples']} | "
            f"{_fmt_ms(c['mean_lc_ms'])} | {_fmt_ms(c['stdev_lc_ms'])} | "
            f"{_fmt_pct(c['cv_pct'])} | {_fmt_pct(c['spread_pct'])} |"
        )

    lines.append("")
    if summary["failed_cases"]:
        lines.append("### Threshold Exceeded")
        for c in summary["failed_cases"]:
            lines.append(
                f"- {c['bench']} / {c['shape']} (CV={_fmt_pct(c['cv_pct'])}, "
                f"mean={_fmt_ms(c['mean_lc_ms'])}ms)"
            )
    else:
        lines.append("All cases are within the configured variance threshold.")

    lines.append("")
    lines.append("### Per-run Metadata")
    for meta in run_meta:
        lines.append(
            f"- run {meta['run_idx']}: rows={meta['row_count']}, elapsed={meta['elapsed_ms']:.2f}ms, "
            f"csv=`{meta['csv']}`, log=`{meta['log']}`"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repeated quick_bench and enforce per-case LC variance threshold."
    )
    parser.add_argument("--runs-csv", type=Path, required=True, help="Combined per-run CSV output path")
    parser.add_argument("--json", type=Path, required=True, help="Variance summary JSON output path")
    parser.add_argument("--md", type=Path, required=True, help="Variance summary Markdown output path")
    parser.add_argument("--repeats", type=int, default=5, help="How many repeated quick_bench runs")
    parser.add_argument("--variance-threshold-pct", type=float, default=2.0, help="Fail when per-case CV exceeds this threshold")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per quick_bench run")
    parser.add_argument("--iters", type=int, default=40, help="Timed iterations per quick_bench run")
    parser.add_argument("--seed", type=int, default=20260401, help="Fixed seed used for every repeated run")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "metal", "cpu"], help="Device passed into quick_bench.py")
    parser.add_argument("--trim-runs", type=int, default=1, help="Trim this many min/max suite-total runs before CV gate")
    parser.add_argument(
        "--enforce-per-case",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When enabled, also requires every case CV to satisfy threshold.",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, exits non-zero if threshold is exceeded.",
    )
    args = parser.parse_args()

    if args.repeats < 2:
        raise ValueError("--repeats must be >= 2 for variance estimation")
    if args.variance_threshold_pct <= 0:
        raise ValueError("--variance-threshold-pct must be > 0")
    if args.trim_runs < 0:
        raise ValueError("--trim-runs must be >= 0")
    if args.trim_runs > 0 and args.repeats <= (2 * args.trim_runs):
        raise ValueError("--repeats must be > 2 * --trim-runs")

    repo_root = Path(__file__).resolve().parents[2]
    bench_script = Path(__file__).resolve().parent / "quick_bench.py"
    work_dir = args.runs_csv.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    run_meta: list[dict] = []
    for run_idx in range(args.repeats):
        rows, meta = run_quick_bench(
            run_idx=run_idx,
            bench_script=bench_script,
            repo_root=repo_root,
            device=args.device,
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed,
            work_dir=work_dir,
        )
        all_rows.extend(rows)
        run_meta.append(meta)

    write_runs_csv(args.runs_csv, all_rows)
    summary = build_summary(
        all_rows,
        run_count=args.repeats,
        threshold_pct=args.variance_threshold_pct,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
        enforce_per_case=args.enforce_per_case,
        trim_runs=args.trim_runs,
    )
    summary["run_meta"] = run_meta
    summary["runs_csv"] = str(args.runs_csv)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(to_markdown(summary, run_meta), encoding="utf-8")

    print(f"saved: {args.runs_csv}")
    print(f"saved: {args.json}")
    print(f"saved: {args.md}")
    print(
        f"overall_pass={summary['gate']['overall_pass']} "
        f"suite_total_raw_cv_pct={summary['gate']['suite_total_cv_pct']} "
        f"suite_total_trimmed_cv_pct={summary['gate']['suite_total_trimmed_cv_pct']} "
        f"per_case_failed_count={summary['gate']['per_case_failed_count']} "
        f"max_case_cv_pct={summary['gate']['max_cv_pct']}"
    )

    if args.fail_on_threshold and not summary["gate"]["overall_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
