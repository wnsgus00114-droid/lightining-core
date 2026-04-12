#!/usr/bin/env python3
"""lc-run CLI: inference/training/benchmark with repro-pack manifests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

import lightning_core as lc
import lightning_core_integrated_api as lc_api


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _pkg_ver(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "n/a"


def _median_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(max(0, int(warmup))):
        fn()
    vals: list[float] = []
    for _ in range(max(1, int(iters))):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        vals.append((t1 - t0) / 1e6)
    return float(median(vals))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _repro_pack(*, command: list[str], output_paths: list[Path], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for p in output_paths:
        exists = p.exists()
        files.append(
            {
                "path": str(p),
                "exists": bool(exists),
                "size_bytes": p.stat().st_size if exists else 0,
                "sha256": _sha256(p) if exists else "",
            }
        )
    return {
        "schema_version": "lc_run_repro_pack_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "cwd": os.getcwd(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": {
            "lightning-core": _pkg_ver("lightning-core"),
            "numpy": _pkg_ver("numpy"),
            "torch": _pkg_ver("torch"),
            "tensorflow": _pkg_ver("tensorflow"),
        },
        "runtime": {
            "backend_name": lc.backend_name(),
            "metal_available": bool(lc.metal_available()),
            "cuda_available": bool(lc.cuda_available()),
        },
        "artifacts": files,
        "extra": dict(extra or {}),
    }


def _default_repro_path(out_json: Path) -> Path:
    return out_json.with_name(f"{out_json.stem}.repro.json")


def _build_runner(args) -> lc_api.TinyTransformerRunner:
    return lc_api.TinyTransformerRunner(
        seq_len=int(args.seq_len),
        d_model=int(args.d_model),
        d_ff=int(args.d_ff),
        vocab_size=int(args.vocab_size),
        seed=int(args.seed),
    )


def _cmd_infer(args) -> dict[str, Any]:
    runner = _build_runner(args)
    np.random.seed(int(args.seed))
    if str(args.input_kind) == "token_ids":
        x = np.asarray(np.random.randint(0, int(args.vocab_size), size=(int(args.seq_len),)), dtype=np.int64)
    else:
        x = (np.random.standard_normal((int(args.seq_len), int(args.d_model))) * 0.2).astype(np.float32)

    y, meta = runner.run(
        x,
        mode=str(args.mode),
        device=str(args.device),
        route_policy={"conv": "auto", "attention": "auto", "graph": "auto"},
        return_metadata=True,
    )
    y_arr = np.asarray(y, dtype=np.float32)
    return {
        "schema_version": "lc_run_infer_v1",
        "command": "infer",
        "config": {
            "mode": str(args.mode),
            "device": str(args.device),
            "seed": int(args.seed),
            "seq_len": int(args.seq_len),
            "d_model": int(args.d_model),
            "d_ff": int(args.d_ff),
            "vocab_size": int(args.vocab_size),
            "input_kind": str(args.input_kind),
        },
        "output": {
            "shape": list(y_arr.shape),
            "dtype": str(y_arr.dtype),
            "mean": float(np.mean(y_arr)),
            "std": float(np.std(y_arr)),
            "min": float(np.min(y_arr)),
            "max": float(np.max(y_arr)),
        },
        "run_meta": meta,
    }


def _cmd_train(args) -> dict[str, Any]:
    np.random.seed(int(args.seed))
    x = (np.random.standard_normal((int(args.batch), int(args.d_in))) * 0.25).astype(np.float32)
    y = (np.random.standard_normal((int(args.batch), int(args.d_out))) * 0.15).astype(np.float32)
    model = lc_api.TinyAutogradMLP(int(args.d_in), int(args.d_hidden), int(args.d_out), seed=int(args.seed))
    report = lc_api.autograd_train_loop(
        model,
        x,
        y,
        steps=int(args.steps),
        lr=float(args.lr),
        grad_clip_norm=float(args.grad_clip_norm) if args.grad_clip_norm is not None else None,
        loss_scale=float(args.loss_scale),
    )
    return {
        "schema_version": "lc_run_train_v1",
        "command": "train",
        "config": {
            "steps": int(args.steps),
            "lr": float(args.lr),
            "grad_clip_norm": None if args.grad_clip_norm is None else float(args.grad_clip_norm),
            "loss_scale": float(args.loss_scale),
            "seed": int(args.seed),
            "batch": int(args.batch),
            "d_in": int(args.d_in),
            "d_hidden": int(args.d_hidden),
            "d_out": int(args.d_out),
        },
        "report": report,
    }


def _cmd_bench(args) -> dict[str, Any]:
    runner = _build_runner(args)
    np.random.seed(int(args.seed))
    if str(args.input_kind) == "token_ids":
        x = np.asarray(np.random.randint(0, int(args.vocab_size), size=(int(args.seq_len),)), dtype=np.int64)
    else:
        x = (np.random.standard_normal((int(args.seq_len), int(args.d_model))) * 0.2).astype(np.float32)

    rows: list[dict[str, Any]] = []
    eager_ms = float("nan")
    for mode in ["eager", "graph", "interop"]:
        fn = lambda m=mode: runner.run(x, mode=m, device=str(args.device))
        lat = _median_ms(fn, int(args.warmup), int(args.iters))
        y, meta = runner.run(x, mode=mode, device=str(args.device), return_metadata=True)
        y_arr = np.asarray(y, dtype=np.float32)
        if mode == "eager":
            eager_ms = float(lat)
            eager_ref = np.asarray(y_arr, dtype=np.float32)
            allclose = True
        else:
            allclose = bool(np.allclose(y_arr, eager_ref, atol=3.0e-3, rtol=3.0e-3))
        rows.append(
            {
                "mode": mode,
                "latency_ms": float(lat),
                "mode_over_eager": float(lat / eager_ms) if eager_ms > 0 else float("nan"),
                "allclose_vs_eager": bool(allclose),
                "fallback_reason_code": str(meta.get("fallback_reason_code", "none")),
                "resolved_mode": str(meta.get("resolved_mode", mode)),
                "resolved_engine": str(meta.get("resolved_engine", "unknown")),
            }
        )

    return {
        "schema_version": "lc_run_bench_v1",
        "command": "bench",
        "config": {
            "device": str(args.device),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "seed": int(args.seed),
            "seq_len": int(args.seq_len),
            "d_model": int(args.d_model),
            "d_ff": int(args.d_ff),
            "vocab_size": int(args.vocab_size),
            "input_kind": str(args.input_kind),
        },
        "rows": rows,
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--seed", type=int, default=20260411)
    base.add_argument("--out-json", type=Path, required=True)
    base.add_argument("--repro-pack", type=Path, default=None)

    runner_base = argparse.ArgumentParser(add_help=False)
    runner_base.add_argument("--device", type=str, default="cpu", choices=["auto", "metal", "cpu", "cuda"])
    runner_base.add_argument("--seq-len", type=int, default=48)
    runner_base.add_argument("--d-model", type=int, default=48)
    runner_base.add_argument("--d-ff", type=int, default=128)
    runner_base.add_argument("--vocab-size", type=int, default=256)
    runner_base.add_argument("--input-kind", type=str, default="token_ids", choices=["token_ids", "embedding_features"])

    infer_p = sub.add_parser("infer", parents=[base, runner_base])
    infer_p.add_argument("--mode", type=str, default="graph", choices=["eager", "graph", "interop"])

    train_p = sub.add_parser("train", parents=[base])
    train_p.add_argument("--steps", type=int, default=5)
    train_p.add_argument("--lr", type=float, default=3.0e-2)
    train_p.add_argument("--grad-clip-norm", type=float, default=1.0)
    train_p.add_argument("--loss-scale", type=float, default=2.0)
    train_p.add_argument("--batch", type=int, default=32)
    train_p.add_argument("--d-in", type=int, default=8)
    train_p.add_argument("--d-hidden", type=int, default=16)
    train_p.add_argument("--d-out", type=int, default=4)

    bench_p = sub.add_parser("bench", parents=[base, runner_base])
    bench_p.add_argument("--warmup", type=int, default=4)
    bench_p.add_argument("--iters", type=int, default=20)

    return p


def main() -> int:
    parser = _parser()
    args = parser.parse_args()

    if args.cmd == "infer":
        payload = _cmd_infer(args)
    elif args.cmd == "train":
        payload = _cmd_train(args)
    else:
        payload = _cmd_bench(args)

    _write_json(args.out_json, payload)
    repro_path = args.repro_pack if args.repro_pack is not None else _default_repro_path(args.out_json)
    repro = _repro_pack(command=[sys.executable, *sys.argv], output_paths=[args.out_json], extra={"cmd": args.cmd})
    _write_json(repro_path, repro)

    print(f"saved: {args.out_json}")
    print(f"saved: {repro_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
