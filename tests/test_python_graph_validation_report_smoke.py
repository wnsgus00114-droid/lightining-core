#!/usr/bin/env python3
"""Python smoke: Graph validation report schema (pass-v2 + reason codes)."""

from __future__ import annotations

import lightning_core as lc


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    passes = list(lc.graph_validation_passes())
    expected = {
        "schema_contract",
        "topology",
        "alias_lifetime",
        "layout_flow",
        "backend_capability",
    }
    _require(set(passes) == expected, f"validation passes mismatch: {passes}")

    g = lc.GraphIR()
    ta = g.add_tensor([4], dtype="float32", name="a", constant=True)
    tb = g.add_tensor([4], dtype="float32", name="b", constant=True)
    to = g.add_tensor([2, 2], dtype="float32", name="out")
    g.add_node("matmul", [ta, tb], [to], attributes={"unknown_attr": 1})

    report = dict(g.validate_report())
    _require(not bool(report.get("ok", True)), "invalid graph should fail validate_report")
    issues = list(report.get("issues", []))
    _require(len(issues) > 0, "issues should be non-empty")

    saw_rank = False
    saw_attr = False
    for issue in issues:
        _require("pass" in issue, "issue.pass missing")
        _require("status" in issue, "issue.status missing")
        _require("reason_code" in issue, "issue.reason_code missing")
        _require("message" in issue, "issue.message missing")
        if issue["reason_code"] == "rank_mismatch":
            saw_rank = True
        if issue["reason_code"] == "attribute_unsupported":
            saw_attr = True
    _require(saw_rank, "rank_mismatch reason_code missing")
    _require(saw_attr, "attribute_unsupported reason_code missing")

    # Plan cache summary structure should be stable.
    pg = lc.GraphIR()
    a = pg.add_tensor([2, 2], dtype="float32", name="a", constant=True)
    b = pg.add_tensor([2, 2], dtype="float32", name="b", constant=True)
    o = pg.add_tensor([2, 2], dtype="float32", name="o")
    pg.add_node("matmul", [a, b], [o])
    pg.clear_plan_cache()

    s1 = dict(pg.plan_summary(preferred_device="cpu", enable_plan_cache=True)).get("summary", {})
    s2 = dict(pg.plan_summary(preferred_device="cpu", enable_plan_cache=True)).get("summary", {})
    _require("plan_cache_hit" in s1 and "plan_cache_hit" in s2, "plan cache fields missing")
    _require(bool(s1.get("plan_cache_hit", False)) is False, "first plan should miss cache")
    _require(bool(s2.get("plan_cache_hit", False)) is True, "second plan should hit cache")

    stats = dict(pg.plan_cache_stats())
    _require(int(stats.get("hits", 0)) >= 1, "plan cache stats hits missing")
    _require(int(stats.get("misses", 0)) >= 1, "plan cache stats misses missing")
    for key in (
        "planner_score_model",
        "cost_profile_signature",
        "estimated_total_cost_ns",
        "estimated_compute_cost_ns",
        "estimated_launch_cost_ns",
        "estimated_boundary_cost_ns",
    ):
        _require(key in s2, f"{key} missing in plan summary")

    # Fusion pass-manager fields should be deterministic and exposed.
    fg = lc.GraphIR()
    a = fg.add_tensor([8, 8], dtype="float32", name="a", constant=True)
    b = fg.add_tensor([8, 8], dtype="float32", name="b", constant=True)
    bias = fg.add_tensor([8, 8], dtype="float32", name="bias", constant=True)
    mm = fg.add_tensor([8, 8], dtype="float32", name="mm")
    add = fg.add_tensor([8, 8], dtype="float32", name="add")
    out = fg.add_tensor([8, 8], dtype="float32", name="out")
    fg.add_node("matmul", [a, b], [mm])
    fg.add_node("vector_add", [mm, bias], [add])
    fg.add_node("relu", [add], [out])
    d1 = list(
        fg.fusion_report(
            preferred_device="cpu",
            fusion_pass_order="matmul,conv,attention,attention_qkv",
            fusion_cost_min_speedup=1.0,
        )
    )
    d2 = list(
        fg.fusion_report(
            preferred_device="cpu",
            fusion_pass_order="matmul,conv,attention,attention_qkv",
            fusion_cost_min_speedup=1.0,
        )
    )
    _require(len(d1) == len(d2), "fusion_report length should be deterministic")
    for x, y in zip(d1, d2):
        _require(x.get("pass_id") == y.get("pass_id"), "fusion_report pass_id mismatch")
        _require(x.get("pass_order") == y.get("pass_order"), "fusion_report pass_order mismatch")
        _require(x.get("reason") == y.get("reason"), "fusion_report reason mismatch")

    print("python graph validation report smoke: ok")


if __name__ == "__main__":
    main()
