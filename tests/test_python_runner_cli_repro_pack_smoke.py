#!/usr/bin/env python3
"""Python smoke: lc-run CLI emits infer/train/bench outputs + repro packs."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _run(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    _require(proc.returncode == 0, f"command failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cli = repo_root / "python/lc_run.py"
    _require(cli.exists(), "lc_run.py must exist")

    with tempfile.TemporaryDirectory(prefix="lc_run_cli_smoke_") as td:
        out_dir = Path(td)

        infer_json = out_dir / "infer.json"
        train_json = out_dir / "train.json"
        bench_json = out_dir / "bench.json"

        _run(
            [
                sys.executable,
                str(cli),
                "infer",
                "--mode",
                "graph",
                "--device",
                "cpu",
                "--out-json",
                str(infer_json),
            ],
            cwd=repo_root,
        )
        _run(
            [
                sys.executable,
                str(cli),
                "train",
                "--steps",
                "3",
                "--lr",
                "0.03",
                "--out-json",
                str(train_json),
            ],
            cwd=repo_root,
        )
        _run(
            [
                sys.executable,
                str(cli),
                "bench",
                "--device",
                "cpu",
                "--warmup",
                "1",
                "--iters",
                "2",
                "--out-json",
                str(bench_json),
            ],
            cwd=repo_root,
        )

        for path in (infer_json, train_json, bench_json):
            _require(path.exists(), f"missing output json: {path}")
            repro = path.with_name(f"{path.stem}.repro.json")
            _require(repro.exists(), f"missing repro pack: {repro}")

            payload = _load(path)
            repro_payload = _load(repro)
            _require(str(repro_payload.get("schema_version", "")) == "lc_run_repro_pack_v1", "repro schema mismatch")
            _require(bool(repro_payload.get("artifacts", [])), "repro artifacts must not be empty")
            _require(str(payload.get("command", "")) in {"infer", "train", "bench"}, "command field mismatch")

    print("python runner cli repro pack smoke: ok")


if __name__ == "__main__":
    main()
