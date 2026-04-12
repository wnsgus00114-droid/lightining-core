#!/usr/bin/env python3
"""Model Runner benchmark with beta contracts (schema/replay/checkpoint matrix)."""

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
import lightning_core_integrated_api as lc_api

_DEFAULT_ROW_FIELDS = [
    "suite",
    "mode",
    "status",
    "device",
    "layout",
    "dtype",
    "seed",
    "input_kind",
    "seq_len",
    "d_model",
    "d_ff",
    "vocab_size",
    "logits_dim",
    "latency_ms",
    "mode_over_eager",
    "allclose_vs_eager",
    "replay_deterministic",
    "replay_max_abs_diff",
    "requested_mode",
    "resolved_mode",
    "resolved_engine",
    "fallback_reason_code",
    "runner_contract_freeze_id",
    "runner_contract_schema_hash",
    "route_policy_json",
    "note",
]
_ARTIFACT_SCHEMA = (
    lc_api.runner_artifact_schema()
    if hasattr(lc_api, "runner_artifact_schema")
    else {"schema_version": "model_runner_artifact_v3", "row_fields": _DEFAULT_ROW_FIELDS}
)
ROW_FIELDS = list(_ARTIFACT_SCHEMA.get("row_fields", _DEFAULT_ROW_FIELDS))
ARTIFACT_SCHEMA_VERSION = str(_ARTIFACT_SCHEMA.get("schema_version", "model_runner_artifact_v3"))


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


def _ratio(numer: float, denom: float) -> float:
    if denom <= 0.0 or (not math.isfinite(numer)) or (not math.isfinite(denom)):
        return float("nan")
    return numer / denom


def _save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _markdown(payload: dict) -> str:
    rows = list(payload.get("rows", []))
    matrix_rows = list(payload.get("checkpoint_compatibility_matrix", {}).get("rows", []))
    lines = []
    lines.append("## Model Runner Benchmark (Beta Contracts)")
    lines.append("")
    lines.append(f"- backend: `{payload.get('backend_name', 'unknown')}`")
    lines.append(f"- device: `{payload.get('device', 'unknown')}`")
    lines.append(f"- artifact schema: `{payload.get('artifact_schema_version', 'unknown')}`")
    lines.append(f"- runner contract freeze: `{payload.get('runner_contract_manifest', {}).get('freeze_id', 'unknown')}`")
    lines.append(f"- replay iters: `{payload.get('replay_iters', 0)}`")
    lines.append("")
    lines.append("| mode | input_kind | status | latency_ms | vs_eager | allclose_vs_eager | replay_deterministic | fallback_reason_code |")
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- | --- |")
    for row in rows:
        lat = row.get("latency_ms", float("nan"))
        lat_s = "n/a" if not math.isfinite(float(lat)) else f"{float(lat):.6f}"
        ratio = row.get("mode_over_eager", float("nan"))
        ratio_s = "n/a" if not math.isfinite(float(ratio)) else f"{float(ratio):.3f}x"
        lines.append(
            f"| {row.get('mode')} | {row.get('input_kind', 'n/a')} | {row.get('status')} | {lat_s} | {ratio_s} | "
            f"{bool(row.get('allclose_vs_eager', False))} | {bool(row.get('replay_deterministic', False))} | "
            f"{row.get('fallback_reason_code', 'none')} |"
        )
    if matrix_rows:
        lines.append("")
        lines.append("### Checkpoint Compatibility Matrix")
        lines.append("")
        lines.append("| format | save_supported | load_checkpoint | load_model_checkpoint |")
        lines.append("| --- | --- | --- | --- |")
        for r in matrix_rows:
            lines.append(
                f"| {r.get('format')} | {bool(r.get('save_supported', False))} | "
                f"{bool(r.get('load_checkpoint_supported', False))} | {bool(r.get('load_model_checkpoint_supported', False))} |"
            )
    runner_matrix_rows = list(payload.get("runner_checkpoint_compatibility_matrix", {}).get("rows", []))
    if runner_matrix_rows:
        lines.append("")
        lines.append("### Runner Checkpoint Compatibility Matrix (v2)")
        lines.append("")
        lines.append("| format | save_runner_checkpoint | load_runner_checkpoint | optimizer_state |")
        lines.append("| --- | --- | --- | --- |")
        for r in runner_matrix_rows:
            lines.append(
                f"| {r.get('format')} | {bool(r.get('save_runner_checkpoint_supported', False))} | "
                f"{bool(r.get('load_runner_checkpoint_supported', False))} | {bool(r.get('load_optimizer_state_supported', False))} |"
            )
    return "\n".join(lines)


def _validate_rows(rows: list[dict]) -> None:
    for i, row in enumerate(rows):
        missing = [f for f in ROW_FIELDS if f not in row]
        if missing:
            raise ValueError(f"row[{i}] missing required fields: {missing}")


def _validate_payload(payload: dict) -> None:
    required_top = {
        "generated_at_utc",
        "suite",
        "artifact_schema_version",
        "artifact_schema",
        "backend_name",
        "device",
        "warmup",
        "iters",
        "replay_iters",
        "seed",
        "runner_config_schema",
        "runner_contract_manifest",
        "checkpoint_compatibility_matrix",
        "runner_checkpoint_compatibility_matrix",
        "rows",
    }
    missing = sorted([k for k in required_top if k not in payload])
    if missing:
        raise ValueError(f"payload missing required top-level fields: {missing}")
    _validate_rows(list(payload.get("rows", [])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "metal", "cpu", "cuda"])
    p.add_argument("--layout", type=str, default="seq_dmodel_2d", choices=["seq_dmodel_2d"])
    p.add_argument("--dtype", type=str, default="float32", choices=["float32"])
    p.add_argument("--input-kind", type=str, default="token_ids", choices=["token_ids", "embedding_features"])
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--iters", type=int, default=24)
    p.add_argument("--replay-iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260410)
    p.add_argument("--csv", type=Path, default=Path("benchmarks/reports/ci/model_runner_alpha.csv"))
    p.add_argument("--json", type=Path, default=Path("benchmarks/reports/ci/model_runner_alpha.json"))
    p.add_argument("--md", type=Path, default=Path("benchmarks/reports/ci/model_runner_alpha.md"))
    p.add_argument("--require-interop", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--require-replay-match", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--validate-artifact-schema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--parity-atol", type=float, default=1.0e-4)
    p.add_argument("--parity-rtol", type=float, default=1.0e-4)
    args = p.parse_args()

    np.random.seed(args.seed)
    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, vocab_size=max(8, int(args.vocab_size)), seed=args.seed)
    if args.input_kind == "token_ids":
        x = np.asarray(np.random.randint(0, int(runner.vocab_size), size=(runner.seq_len,)), dtype=np.int64)
    else:
        x = (np.random.standard_normal((runner.seq_len, runner.d_model)) * 0.2).astype(np.float32)

    mode_rows: list[dict] = []
    modes = ["eager", "graph", "interop"]
    eager_ref: np.ndarray | None = None
    eager_ms = float("nan")
    for mode in modes:
        try:
            cfg = lc_api.validate_runner_config(
                {
                    "mode": mode,
                    "device": args.device,
                    "seed": args.seed,
                    "layout": args.layout,
                    "dtype": args.dtype,
                    "route_policy": {"conv": "auto", "attention": "auto", "graph": "auto"},
                },
                strict=True,
            )
            fn = lambda m=mode: runner.run(x, mode=m, device=args.device)
            lat = _median_ms(fn, args.warmup, args.iters)
            out, run_meta = runner.run(x, mode=mode, device=args.device, return_metadata=True)
            out = np.asarray(out, dtype=np.float32)
            if mode == "eager":
                eager_ref = out
                eager_ms = lat
            atol = float(args.parity_atol)
            rtol = float(args.parity_rtol)
            if mode == "interop":
                atol = max(atol, 3.0e-3)
                rtol = max(rtol, 3.0e-3)
            allclose = bool(np.allclose(out, eager_ref, atol=atol, rtol=rtol)) if eager_ref is not None else True
            replay = lc_api.runner_replay_report(
                runner,
                x,
                mode=mode,
                device=args.device,
                route_policy=cfg["normalized"]["route_policy"],
                repeats=args.replay_iters,
            )
            mode_rows.append(
                {
                    "suite": "model_runner_alpha",
                    "mode": mode,
                    "status": "ok",
                    "device": args.device,
                    "layout": args.layout,
                    "dtype": args.dtype,
                    "seed": int(args.seed),
                    "input_kind": str(run_meta.get("input_kind", args.input_kind)),
                    "seq_len": int(run_meta.get("seq_len", runner.seq_len)),
                    "d_model": int(run_meta.get("d_model", runner.d_model)),
                    "d_ff": int(run_meta.get("d_ff", runner.d_ff)),
                    "vocab_size": int(run_meta.get("vocab_size", runner.vocab_size)),
                    "logits_dim": int(run_meta.get("logits_dim", runner.vocab_size)),
                    "latency_ms": float(lat),
                    "mode_over_eager": _ratio(float(lat), float(eager_ms)),
                    "allclose_vs_eager": allclose,
                    "replay_deterministic": bool(replay.get("deterministic_replay", False)),
                    "replay_max_abs_diff": float(replay.get("max_abs_replay_diff", float("nan"))),
                    "requested_mode": str(run_meta.get("requested_mode", mode)),
                    "resolved_mode": str(run_meta.get("resolved_mode", mode)),
                    "resolved_engine": str(run_meta.get("resolved_engine", "unknown")),
                    "fallback_reason_code": str(run_meta.get("fallback_reason_code", "none")),
                    "runner_contract_freeze_id": str(run_meta.get("runner_contract_freeze_id", "")),
                    "runner_contract_schema_hash": str(run_meta.get("runner_contract_schema_hash", "")),
                    "route_policy_json": json.dumps(
                        cfg["normalized"]["route_policy"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
                    ),
                    "note": str(run_meta.get("fallback_message", "")),
                }
            )
        except Exception as exc:
            mode_rows.append(
                {
                    "suite": "model_runner_alpha",
                    "mode": mode,
                    "status": "unsupported",
                    "device": args.device,
                    "layout": args.layout,
                    "dtype": args.dtype,
                    "seed": int(args.seed),
                    "input_kind": str(args.input_kind),
                    "seq_len": int(runner.seq_len),
                    "d_model": int(runner.d_model),
                    "d_ff": int(runner.d_ff),
                    "vocab_size": int(runner.vocab_size),
                    "logits_dim": int(runner.vocab_size),
                    "latency_ms": float("nan"),
                    "mode_over_eager": float("nan"),
                    "allclose_vs_eager": False,
                    "replay_deterministic": False,
                    "replay_max_abs_diff": float("nan"),
                    "requested_mode": mode,
                    "resolved_mode": "unsupported",
                    "resolved_engine": "unsupported",
                    "fallback_reason_code": "unsupported",
                    "runner_contract_freeze_id": str(
                        lc_api.runner_contract_manifest().get("freeze_id", "v0.4.0-rc0")
                    )
                    if hasattr(lc_api, "runner_contract_manifest")
                    else "v0.4.0-rc0",
                    "runner_contract_schema_hash": str(
                        lc_api.runner_contract_manifest().get("schema_hash_sha256", "")
                    )
                    if hasattr(lc_api, "runner_contract_manifest")
                    else "",
                    "route_policy_json": json.dumps(
                        {"conv": "auto", "attention": "auto", "graph": "auto"},
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "note": str(exc),
                }
            )

    if args.require_interop:
        interop_ok = [r for r in mode_rows if r["mode"] == "interop" and r["status"] == "ok"]
        if not interop_ok:
            raise SystemExit(3)
    if args.require_replay_match:
        bad = [r for r in mode_rows if r["status"] == "ok" and not bool(r.get("replay_deterministic", False))]
        if bad:
            raise SystemExit(4)

    if args.validate_artifact_schema:
        _validate_rows(mode_rows)

    _save_csv(args.csv, mode_rows, ROW_FIELDS)
    cfg_schema = lc_api.runner_config_schema() if hasattr(lc_api, "runner_config_schema") else {}
    contract_manifest = lc_api.runner_contract_manifest() if hasattr(lc_api, "runner_contract_manifest") else {}
    ckpt_matrix = lc_api.checkpoint_compatibility_matrix() if hasattr(lc_api, "checkpoint_compatibility_matrix") else {}
    runner_ckpt_matrix = (
        lc_api.runner_checkpoint_compatibility_matrix()
        if hasattr(lc_api, "runner_checkpoint_compatibility_matrix")
        else {}
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "model_runner_alpha",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_schema": dict(_ARTIFACT_SCHEMA),
        "backend_name": str(lc.backend_name()),
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        "replay_iters": args.replay_iters,
        "seed": args.seed,
        "runner_config_schema": cfg_schema,
        "runner_contract_manifest": contract_manifest,
        "checkpoint_compatibility_matrix": ckpt_matrix,
        "runner_checkpoint_compatibility_matrix": runner_ckpt_matrix,
        "rows": mode_rows,
    }
    if args.validate_artifact_schema:
        _validate_payload(payload)
    _save_json(args.json, payload)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(_markdown(payload), encoding="utf-8")
    print(f"saved: {args.csv}")
    print(f"saved: {args.json}")
    print(f"saved: {args.md}")


if __name__ == "__main__":
    main()
