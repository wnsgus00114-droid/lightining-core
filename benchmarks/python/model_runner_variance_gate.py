#!/usr/bin/env python3
"""Runner benchmark stability gate (CV <= threshold)."""

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


def _to_float(v) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _fmt_ms(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:.6f}"


def _fmt_pct(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:.2f}%"


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be object: {path}")
    return payload


def _write_runs_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_idx",
        "mode",
        "input_kind",
        "status",
        "latency_ms",
        "mode_over_eager",
        "fallback_reason_code",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _run_once(*, run_idx: int, script: Path, repo_root: Path, device: str, warmup: int, iters: int, seed: int, out_dir: Path) -> tuple[list[dict], dict]:
    csv_path = out_dir / f"runner_bench_run_{run_idx:02d}.csv"
    json_path = out_dir / f"runner_bench_run_{run_idx:02d}.json"
    md_path = out_dir / f"runner_bench_run_{run_idx:02d}.md"
    log_path = out_dir / f"runner_bench_run_{run_idx:02d}.log"
    cmd = [
        sys.executable,
        str(script),
        "--device",
        str(device),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--seed",
        str(seed),
        "--require-replay-match",
        "--validate-artifact-schema",
        "--csv",
        str(csv_path),
        "--json",
        str(json_path),
        "--md",
        str(md_path),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    log_path.write_text((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"model_runner_alpha_bench failed (run={run_idx}): {log_path}")
    payload = _load_json(json_path)
    run_rows = list(payload.get("rows", []))
    for r in run_rows:
        r["run_idx"] = int(run_idx)
    meta = {
        "run_idx": int(run_idx),
        "elapsed_ms": float(elapsed_ms),
        "row_count": len(run_rows),
        "json": str(json_path),
        "csv": str(csv_path),
        "md": str(md_path),
        "log": str(log_path),
        "command": cmd,
    }
    return run_rows, meta


def _build_summary(rows: list[dict], run_meta: list[dict], *, threshold_pct: float, trim_runs: int, device: str, warmup: int, iters: int, seed: int) -> dict:
    grouped: dict[tuple[str, str], list[float]] = {}
    per_run_total: dict[int, float] = {}
    for r in rows:
        mode = str(r.get("mode", ""))
        input_kind = str(r.get("input_kind", ""))
        lat = _to_float(r.get("latency_ms"))
        grouped.setdefault((mode, input_kind), []).append(lat)
        run_idx = int(r.get("run_idx", -1))
        if run_idx >= 0 and math.isfinite(lat):
            per_run_total[run_idx] = per_run_total.get(run_idx, 0.0) + lat

    cases: list[dict] = []
    for (mode, input_kind), vals in grouped.items():
        finite_vals = [v for v in vals if math.isfinite(v)]
        mean_ms = statistics.mean(finite_vals) if finite_vals else float("nan")
        stdev_ms = statistics.pstdev(finite_vals) if len(finite_vals) >= 2 else 0.0
        cv_pct = (stdev_ms / mean_ms * 100.0) if math.isfinite(mean_ms) and mean_ms > 0.0 else float("nan")
        cases.append(
            {
                "mode": mode,
                "input_kind": input_kind,
                "samples": len(finite_vals),
                "mean_ms": float(mean_ms),
                "stdev_ms": float(stdev_ms),
                "cv_pct": float(cv_pct),
                "threshold_exceeded": bool(math.isfinite(cv_pct) and cv_pct > float(threshold_pct)),
            }
        )
    cases.sort(key=lambda x: x["cv_pct"] if math.isfinite(x["cv_pct"]) else -1.0, reverse=True)

    totals = [v for _, v in sorted(per_run_total.items(), key=lambda x: x[0])]
    if len(totals) >= 2:
        total_mean = statistics.mean(totals)
        total_stdev = statistics.pstdev(totals)
        total_cv = (total_stdev / total_mean * 100.0) if total_mean > 0 else float("nan")
    else:
        total_mean = float("nan")
        total_stdev = float("nan")
        total_cv = float("nan")

    sorted_totals = sorted(totals)
    trim_n = max(0, int(trim_runs))
    if trim_n > 0 and len(sorted_totals) > (2 * trim_n):
        trimmed = sorted_totals[trim_n : len(sorted_totals) - trim_n]
    else:
        trimmed = sorted_totals

    if len(trimmed) >= 2:
        trim_mean = statistics.mean(trimmed)
        trim_stdev = statistics.pstdev(trimmed)
        trim_cv = (trim_stdev / trim_mean * 100.0) if trim_mean > 0 else float("nan")
    else:
        trim_mean = float("nan")
        trim_stdev = float("nan")
        trim_cv = float("nan")

    failed_cases = [c for c in cases if bool(c.get("threshold_exceeded", False))]
    gate_pass = bool(math.isfinite(trim_cv) and trim_cv <= float(threshold_pct))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": {
            "overall_pass": bool(gate_pass),
            "threshold_cv_pct": float(threshold_pct),
            "run_count": int(len(run_meta)),
            "device": str(device),
            "warmup": int(warmup),
            "iters": int(iters),
            "seed": int(seed),
            "trim_runs": int(trim_n),
            "suite_total_mean_ms": float(total_mean),
            "suite_total_stdev_ms": float(total_stdev),
            "suite_total_cv_pct": float(total_cv),
            "suite_total_trimmed_mean_ms": float(trim_mean),
            "suite_total_trimmed_stdev_ms": float(trim_stdev),
            "suite_total_trimmed_cv_pct": float(trim_cv),
            "failed_case_count": int(len(failed_cases)),
        },
        "case_count": len(cases),
        "cases": cases,
        "failed_cases": failed_cases,
        "run_meta": run_meta,
    }


def _render_md(summary: dict) -> str:
    gate = dict(summary.get("gate", {}))
    lines = [
        "## Model Runner Variance Gate",
        "",
        f"- status: {'PASS' if gate.get('overall_pass', False) else 'FAIL'}",
        f"- threshold (CV): <= {gate.get('threshold_cv_pct', float('nan')):.2f}%",
        (
            f"- suite total (trimmed): mean={_fmt_ms(_to_float(gate.get('suite_total_trimmed_mean_ms')))}ms, "
            f"stdev={_fmt_ms(_to_float(gate.get('suite_total_trimmed_stdev_ms')))}ms, "
            f"CV={_fmt_pct(_to_float(gate.get('suite_total_trimmed_cv_pct')))}"
        ),
        "",
        "| mode | input_kind | samples | mean_ms | stdev_ms | cv_pct |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for c in list(summary.get("cases", [])):
        lines.append(
            f"| {c.get('mode')} | {c.get('input_kind')} | {int(c.get('samples', 0))} | "
            f"{_fmt_ms(_to_float(c.get('mean_ms')))} | {_fmt_ms(_to_float(c.get('stdev_ms')))} | {_fmt_pct(_to_float(c.get('cv_pct')))} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-csv", type=Path, required=True)
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--md", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--variance-threshold-pct", type=float, default=2.0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260411)
    p.add_argument("--trim-runs", type=int, default=1)
    p.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "benchmarks/python/model_runner_alpha_bench.py"
    if not script.exists():
        raise FileNotFoundError(f"missing benchmark script: {script}")

    work_dir = args.json.parent
    work_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    run_meta: list[dict] = []
    for idx in range(max(1, int(args.repeats))):
        rows, meta = _run_once(
            run_idx=idx,
            script=script,
            repo_root=repo_root,
            device=str(args.device),
            warmup=int(args.warmup),
            iters=int(args.iters),
            seed=int(args.seed),
            out_dir=work_dir,
        )
        all_rows.extend(rows)
        run_meta.append(meta)

    _write_runs_csv(args.runs_csv, all_rows)
    summary = _build_summary(
        all_rows,
        run_meta,
        threshold_pct=float(args.variance_threshold_pct),
        trim_runs=int(args.trim_runs),
        device=str(args.device),
        warmup=int(args.warmup),
        iters=int(args.iters),
        seed=int(args.seed),
    )
    args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.md.write_text(_render_md(summary), encoding="utf-8")

    print(f"saved: {args.runs_csv}")
    print(f"saved: {args.json}")
    print(f"saved: {args.md}")

    if args.require_pass and (not bool(summary.get("gate", {}).get("overall_pass", False))):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
