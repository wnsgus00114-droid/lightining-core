# Operator Onboarding Kit

Add one new graph operator with deterministic validation, planner visibility, and CI coverage.

## 1) Edit Checklist (Template)

1. Schema registry:
   - file: `include/lightning_core/core/graph.hpp`
   - add `OpKind` enum + `opKindName()`
   - register schema in `OperatorRegistry` (rank/dtype/layout/attrs)
2. Validation contract:
   - file: `include/lightning_core/core/graph.hpp`
   - add shape/layout/attribute checks in graph validation path
   - return deterministic reason codes on violations
3. Execution path:
   - file: `include/lightning_core/core/graph.hpp`
   - add `executeNodeTyped` switch case
   - keep CPU baseline behavior deterministic
4. Python name mapping:
   - file: `python/bindings/bind_graph.cpp`
   - map Python op string to the new `OpKind`
5. Tests:
   - file: `tests/test_graph_ir.cpp` (C++ correctness + planner/validation checks)
   - file: `tests/test_python_operator_onboarding_smoke.py` (copy-paste smoke)
6. Bench evidence:
   - file: `benchmarks/python/graph_eager_ab_bench.py` (if op participates in chain dispatch)
   - include fallback reason visibility in CSV/JSON/MD row fields

## 2) Minimal Contract Rules

- Unsupported shape/layout/dtype must fail with deterministic reason codes.
- Graph-request failures must preserve eager-equivalent numerics through deterministic fallback.
- Add at least one positive (supported) and one negative (unsupported) test case.
- Ensure plan summary reports remain stable (`planned_dispatch_groups`, `fallback_reason_code`).

## 3) Copy-Paste Smoke Example

Use this exact snippet to validate local onboarding flow:

```python
import numpy as np
import lightning_core as lc

g = lc.GraphIR()
ta = g.add_tensor([4, 4], dtype="float32", name="a", constant=True)
tb = g.add_tensor([4, 4], dtype="float32", name="b", constant=True)
tout = g.add_tensor([4, 4], dtype="float32", name="out")
g.add_node("vector_add", [ta, tb], [tout])

a = np.arange(16, dtype=np.float32).reshape(4, 4)
b = np.ones((4, 4), dtype=np.float32)
result = g.execute_f32({ta: a, tb: b}, preferred_device="cpu")
out = np.asarray(result["values"][tout], dtype=np.float32).reshape(4, 4)

assert np.allclose(out, a + b, atol=1e-5, rtol=1e-5)
print("operator onboarding smoke: ok")
```

## 4) CI Gate Targets

- `ci-contract-tests.yml`:
  - C++ contract subset (`test_graph_ir`) must pass
  - Python onboarding smoke must pass
- `benchmark-artifacts.yml`:
  - graph/eager artifact must include fixed planner/fallback evidence columns

## 5) Fast Validation Commands

```bash
cmake --build build-v021 -j 8
ctest --test-dir build-v021 --output-on-failure -R test_graph_ir
python tests/test_python_operator_onboarding_smoke.py
python benchmarks/python/graph_eager_ab_bench.py --device auto --warmup 2 --iters 8 --trace-iters 4
```
