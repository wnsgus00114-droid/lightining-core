# Lightning Core Contributor Path

This guide is for contributors who want to modify code, run validation, and submit changes safely.

## 1) Development setup

```bash
python3 -m pip install -e .
cmake -S . -B build -DCJ_ENABLE_METAL=ON -DCJ_BUILD_TESTS=ON -DCJ_BUILD_BENCHMARKS=ON -DCJ_BUILD_PYTHON=ON -DCJ_BUILD_EXAMPLES=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## 2) Codebase map

- Core implementation: src/
- Canonical internal headers: include/lightning_core/core/
- Public wrappers: include/lightning_core/
- Legacy compatibility shims: include/cudajun/
- Python bindings: python/bindings/
- Tests: tests/
- Benchmarks: benchmarks/

## 3) Naming and compatibility policy

- New code should use lightning_core naming.
- Keep compatibility shims working for existing integrations.
- Do not remove include/cudajun/ forwarding headers without explicit migration plan.

## 4) Python binding policy

- Keep binding modules split by concern:
  - bind_tensor
  - bind_ops
  - bind_attention
  - bind_runtime
- Prefer exposing numpy-friendly APIs for Python ergonomics.
- For resident sessions, keep in-place *_into methods available.

## 5) Validation checklist before commit

1. Build succeeds with tests/benchmarks/python enabled.
2. ctest passes.
3. At least one python import/runtime smoke check succeeds.
4. README/docs links still resolve.

## 6) Documentation policy

- Keep README short and beginner-first.
- Put advanced details in docs/advanced.md.
- Update docs/index.md when adding/removing major guides.

## 7) Roadmap and large refactors

For staged work and migration direction, see:

- ROADMAP.md

## 8) Repository rename operations

If GitHub repo rename is pending, keep current live URL.
When rename becomes available:

```bash
./scripts/sync_remote_after_repo_rename.sh --dry-run
./scripts/sync_remote_after_repo_rename.sh
```
