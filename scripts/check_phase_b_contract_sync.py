#!/usr/bin/env python3
"""Check Phase B contract constants are wired into CI workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require_contains(text: str, needle: str, *, file_label: str) -> None:
    if needle not in text:
        raise SystemExit(f"missing token in {file_label}: {needle}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--contract-json", type=Path, default=Path("docs/phase_b_graph_contract.json"))
    p.add_argument(
        "--workflow",
        action="append",
        default=[
            ".github/workflows/benchmark-artifacts.yml",
            ".github/workflows/python-wheel-publish.yml",
        ],
    )
    args = p.parse_args()

    root = args.repo_root.resolve()
    contract = json.loads((root / args.contract_json).read_text(encoding="utf-8"))
    graph_ab = dict(contract.get("ci_constants", {}).get("graph_eager_ab", {}))
    fusion = dict(contract.get("ci_constants", {}).get("fusion_pilot", {}))

    expected_literals = {
        "LC_GRAPH_AB_WARMUP": str(graph_ab.get("warmup")),
        "LC_GRAPH_AB_ITERS": str(graph_ab.get("iters")),
        "LC_GRAPH_AB_TRACE_ITERS": str(graph_ab.get("trace_iters")),
        "LC_GRAPH_AB_FUSION_COST_MIN_SPEEDUP": str(graph_ab.get("fusion_cost_min_speedup")),
        "LC_GRAPH_AB_MIN_HOST_DISPATCH_REDUCTION_RATE_PCT": str(graph_ab.get("min_host_dispatch_reduction_rate_pct")),
        "LC_FUSION_MAX_CONV": str(fusion.get("max_fused_over_unfused_conv")),
        "LC_FUSION_MAX_MATMUL": str(fusion.get("max_fused_over_unfused_matmul")),
        "LC_FUSION_MAX_ATTENTION": str(fusion.get("max_fused_over_unfused_attention")),
    }

    for wf_rel in args.workflow:
        wf_path = (root / wf_rel).resolve()
        text = wf_path.read_text(encoding="utf-8")
        file_label = str(wf_path)

        _require_contains(text, "docs/phase_b_graph_contract.json", file_label=file_label)
        _require_contains(text, "LC_GRAPH_AB_MIN_HOST_DISPATCH_REDUCTION_RATE_PCT", file_label=file_label)
        _require_contains(text, "LC_FUSION_MAX_ATTENTION", file_label=file_label)

        for env_name, value in expected_literals.items():
            _require_contains(text, env_name, file_label=file_label)
            _require_contains(text, value, file_label=file_label)

    print(
        json.dumps(
            {
                "status": "ok",
                "contract": str((root / args.contract_json).resolve()),
                "workflows_checked": [str((root / w).resolve()) for w in args.workflow],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
