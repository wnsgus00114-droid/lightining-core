#!/usr/bin/env python3
"""Python smoke: operator onboarding kit copy-paste execution."""

from __future__ import annotations

import numpy as np

import lightning_core as lc


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    schema_fn = getattr(lc, "graph_operator_schemas", None)
    if not callable(schema_fn):
        schema_fn = getattr(lc, "graph_registry_schemas", None)
    _require(callable(schema_fn), "graph operator schema API must be available")
    schemas = list(schema_fn())
    names = {str(s.get("name", "")) for s in schemas}
    _require("conv2d_nchw3x3" in names, "operator schema conv2d_nchw3x3 must be registered")
    _require("qkv_pack_repeat" in names, "operator schema qkv_pack_repeat must be registered")

    g = lc.GraphIR()
    ta = g.add_tensor([4, 4], dtype="float32", name="a", constant=True)
    tb = g.add_tensor([4, 4], dtype="float32", name="b", constant=True)
    tout = g.add_tensor([4, 4], dtype="float32", name="out")
    g.add_node("vector_add", [ta, tb], [tout])

    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    b = np.ones((4, 4), dtype=np.float32)
    result = dict(g.execute_f32({ta: a, tb: b}, preferred_device="cpu"))
    values = dict(result.get("values", {}))
    _require(tout in values, "vector_add output tensor missing")
    out = np.asarray(values[tout], dtype=np.float32).reshape(4, 4)
    _require(np.allclose(out, a + b, atol=1.0e-5, rtol=1.0e-5), "vector_add onboarding smoke mismatch")

    summary = dict(result.get("summary", {}))
    _require(int(summary.get("planned_dispatch_groups", 0)) >= 1, "planned_dispatch_groups must be >= 1")
    print("python operator onboarding smoke: ok")


if __name__ == "__main__":
    main()
