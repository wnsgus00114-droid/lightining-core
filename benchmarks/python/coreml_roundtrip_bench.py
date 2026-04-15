#!/usr/bin/env python3
"""CoreML round-trip beta benchmark.

Round-trip here means: runner state export (v2 checkpoint) -> reload -> run parity,
plus optional CoreML runtime benchmark evidence when model path is provided.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import lightning_core as lc
import lightning_core_integrated_api as lc_api


def _save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "metal", "cpu", "cuda"])
    p.add_argument("--mode", type=str, default="eager", choices=["eager", "graph", "interop"])
    p.add_argument("--seed", type=int, default=20260415)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--coreml-model-path", type=str, default="")
    p.add_argument("--bundle-dir", type=Path, default=Path("benchmarks/reports/ci/coreml_roundtrip_bundle"))
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/reports/ci"))
    p.add_argument("--csv", type=str, default="coreml_roundtrip.csv")
    p.add_argument("--json", type=str, default="coreml_roundtrip.json")
    p.add_argument("--md", type=str, default="coreml_roundtrip.md")
    p.add_argument("--require-parity", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()

    np.random.seed(args.seed)
    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, vocab_size=256, seed=args.seed)

    token_ids = np.asarray(np.random.randint(0, 256, size=(48,)), dtype=np.int64)
    report = lc_api.coreml_roundtrip_beta_report(
        runner,
        token_ids,
        bundle_dir=args.bundle_dir,
        coreml_model_path=str(args.coreml_model_path).strip() or None,
        mode=args.mode,
        device=args.device,
        route_policy={"conv": "auto", "attention": "auto", "graph": "auto"},
        warmup=args.warmup,
        iters=args.iters,
    )

    row = {
        "suite": "coreml_roundtrip",
        "status": str(report.get("status", "unknown")),
        "roundtrip_reason_code": str(report.get("roundtrip_reason_code", "")),
        "mode": str(args.mode),
        "device": str(args.device),
        "baseline_median_ms": float(report.get("baseline_median_ms", float("nan"))),
        "reload_median_ms": float(report.get("reload_median_ms", float("nan"))),
        "max_abs_diff": float(report.get("max_abs_diff", float("nan"))),
        "parity_pass": bool(report.get("parity_pass", False)),
        "coreml_runtime_available": bool(report.get("coreml_runtime_available", False)),
        "coreml_benchmark_status": str(report.get("coreml_benchmark_status", "not_run")),
        "coreml_benchmark_avg_ms": float(report.get("coreml_benchmark_avg_ms", float("nan"))),
        "bundle_dir": str(report.get("bundle_dir", "")),
        "checkpoint_path": str(report.get("checkpoint_path", "")),
        "manifest_path": str(report.get("manifest_path", "")),
    }

    fields = list(row.keys())
    csv_path = args.out_dir / args.csv
    json_path = args.out_dir / args.json
    md_path = args.out_dir / args.md

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "coreml_roundtrip",
        "artifact_schema_version": "coreml_roundtrip_artifact_v1",
        "backend_name": str(lc.backend_name()),
        "device": str(args.device),
        "mode": str(args.mode),
        "seed": int(args.seed),
        "schema": lc_api.coreml_roundtrip_schema() if hasattr(lc_api, "coreml_roundtrip_schema") else {},
        "rows": [row],
        "report": report,
    }

    _save_csv(csv_path, [row], fields)
    _save_json(json_path, payload)
    md_path.write_text(
        "\n".join(
            [
                "## CoreML Round-trip Beta",
                "",
                f"- status: `{row['status']}`",
                f"- reason: `{row['roundtrip_reason_code']}`",
                f"- parity_pass: `{row['parity_pass']}`",
                f"- coreml_runtime_available: `{row['coreml_runtime_available']}`",
                f"- coreml_benchmark_status: `{row['coreml_benchmark_status']}`",
                f"- bundle_dir: `{row['bundle_dir']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")
    print(f"saved: {md_path}")
    if args.require_parity and not bool(row["parity_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
