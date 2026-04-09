#!/usr/bin/env python3
"""Generate one-shot Phase B baseline artifact."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import lightning_core as lc


def _device() -> str:
    name = str(lc.backend_name()).lower()
    if name in {"metal", "cpu", "cuda"}:
        return name
    return "cpu"


def _plan_cache_probe(device: str) -> dict:
    g = lc.GraphIR()
    a = g.add_tensor([64, 64], dtype="float32", name="a", constant=True)
    b = g.add_tensor([64, 64], dtype="float32", name="b", constant=True)
    mm = g.add_tensor([64, 64], dtype="float32", name="mm")
    c = g.add_tensor([64, 64], dtype="float32", name="c", constant=True)
    o = g.add_tensor([64, 64], dtype="float32", name="o")
    g.add_node("matmul", [a, b], [mm])
    g.add_node("vector_add", [mm, c], [o])

    g.clear_plan_cache()
    s1 = dict(g.plan_summary(preferred_device=device, enable_plan_cache=True)).get("summary", {})
    s2 = dict(g.plan_summary(preferred_device=device, enable_plan_cache=True)).get("summary", {})
    stats = dict(g.plan_cache_stats())
    return {
        "first_summary": s1,
        "second_summary": s2,
        "stats": stats,
    }


def _validation_probe() -> dict:
    g = lc.GraphIR()
    a = g.add_tensor([4], dtype="float32", name="a", constant=True)
    b = g.add_tensor([4], dtype="float32", name="b", constant=True)
    o = g.add_tensor([2, 2], dtype="float32", name="o")
    g.add_node("matmul", [a, b], [o], attributes={"unknown_attr": 1})
    return dict(g.validate_report())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract-json", type=Path, default=Path("docs/phase_b_graph_contract.json"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/reports/baselines/phase_b_v0_2_0_rc0_baseline.json"),
    )
    args = p.parse_args()

    contract = json.loads(args.contract_json.read_text(encoding="utf-8"))
    device = _device()

    payload = {
        "artifact_kind": "phase_b_baseline",
        "contract_version": contract.get("contract_version", "unknown"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "backend_name": lc.backend_name(),
            "metal_available": bool(lc.metal_available()),
            "cuda_available": bool(lc.cuda_available()),
        },
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "graph_validation_probe": _validation_probe(),
        "plan_cache_probe": _plan_cache_probe(device),
        "meta": {
            "command": "python benchmarks/python/phase_b_baseline_artifact.py",
            "device": device,
            "note": "one-shot baseline evidence for Phase B contract freeze",
            "generated_ns": time.time_ns(),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
