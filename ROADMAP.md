# Lightning Core Roadmap

This roadmap tracks known architecture gaps and compatibility work.

## P0: Branding and Surface Consistency

- Unify public naming across package, docs, targets, and namespaces.
- Keep compatibility wrappers for existing `cudajun` users during migration.
- Ensure metadata and repository URLs always match the canonical repo.

## P1: Documentation and Expectation Management

- Keep README scope explicit: optimization runtime prototype, not full framework.
- Separate "implemented now" vs "planned" in docs.
- Standardize comment/doc tone for external contributors.

## P1: Backend Abstraction Cleanup

- Remove CUDA-centric internal identifiers where backend-neutral semantics are intended.
- Clarify memory model contracts for each backend.
- Reduce hidden backend-specific behavior under generic runtime names.
- Gradually migrate internal namespace naming from `cudajun` to canonical core naming while preserving compatibility.

## P2: Tensor Core Expansion

- Add dtype/layout/stride metadata.
- Add contiguous and view/slice semantics.
- Add stricter shape/lifetime validation and richer error paths.
- Split responsibilities over time: storage, metadata, view rules, and reshape/slice validation layers.

## P2: Ops Layer Modularization

- Split monolithic ops surfaces into:
  - `ops/vector`
  - `ops/matrix`
  - `ops/policy`
  - `ops/session`
- Preserve high-level helper APIs while reducing header bloat.

## P2: Python API Expansion

- Expose attention and selected matmul/session APIs.
- Expose policy controls with safe defaults.
- Keep minimal onboarding path for first-time users.
- Keep bindings modular (`bind_tensor`, `bind_ops`, `bind_attention`, `bind_runtime`) for maintainability.
- Improve numpy/buffer-oriented in-place paths for resident sessions.

## P2: Test Depth

- Add resident-loop stability tests.
- Add fuzz/edge-case shape tests.
- Add policy-combination and memory-lifetime regression tests.

## P3: Build Matrix Evolution

- Current mode is macOS-first by design.
- Add optional CPU-only CI profiles for Linux/Windows to prevent accidental lock-in.
- Keep platform expansion behind explicit build options.
