# Lightning Core Roadmap

Version context: v0.5.6 (2026-04-14)

## 1) North Star

Lightning Core started as a macOS Apple-Silicon performance runtime. The long-term target is a general-purpose ML runtime framework with:

- backend-pluggable execution (Metal first, CPU stable, CUDA/other backends as plugins),
- graph-aware optimization and fusion,
- predictable performance with explicit policy control,
- a simple user API for model-level execution,
- strong reproducibility and benchmark discipline.

## 2) Non-Negotiables

- Keep macOS/Metal as first-class, never treated as a second-tier backend.
- Keep KWU-1.0 licensing and attribution model.
- Keep benchmark transparency: all claims linked to runnable source and CSV/JSON outputs.
- Keep API ergonomics improvements performance-safe by default.
- Keep deprecation policy explicit; no silent API behavior changes.

## 3) Current Baseline (v0.5.6)

- Public package on PyPI/TestPyPI.
- C++ core + Python bindings for runtime/tensor/ops/attention/integrated APIs.
- Resident execution and policy-based IO control.
- Public benchmark source and quick benchmark script.
- Legacy `cudajun` forwarding headers removed; canonical namespace is `lightning_core`.
- `lc.api` engine bridge (`set_engine/get_engine`) is stabilized on one API surface for `lightning/torch/tf/coreml/mlx/auto`.
- Graph plan summary API + fixed host-dispatch evidence fields are available in graph/eager benchmark artifacts.
- Fusion pass-3 includes `attention_forward + projection(matmul)` rule-based pattern with explain report coverage.
- Checkpoint IO v1.2 integrity includes model-level save/load helpers, tensor/manifest hash validation, and structured diagnostics.
- Autograd bootstrap v1 covers `matmul/add/relu` plus `conv2d` and attention-adjacent backward paths with tiny multi-step training smoke.
- Model Runner Alpha (`eager/graph/interop`) emits reproducible CSV/JSON/MD artifacts through a single benchmark entry.
- Interop boundary hardening adds route-policy boundary switch/copy/overhead telemetry and standardized reason-code gates.
- Phase B0 baseline contract is frozen in `docs/phase_b_graph_contract.json` and synchronized to CI constants.
- Operator registry contracts now enforce rank/layout/dtype/shape/attribute validation with deterministic reason codes.
- Validation pass pack v2 emits pass-scoped diagnostics (`schema_contract/topology/alias_lifetime/layout_flow/backend_capability`) for C++/Python graph reports.
- Planner v3 includes graph-hash/device/sync-policy plan-cache with hit/miss telemetry and benchmark artifact exposure.
- Phase B exit audit evidence (`phase_b_exit_audit`) is wired into CI/release artifacts with ROADMAP 11.2 metric checks and candidate-bundle manifest output.
- Phase C exit audit evidence (`phase_c_exit_audit`) is wired into CI/release artifacts with `v0.3.0-rc0` criteria lock constants.

## 4) Evolution Plan (Mac Runtime -> General Framework)

| Phase | Target Window | Theme | Primary Outcome |
| --- | --- | --- | --- |
| Phase A | 2026 Q2 | Runtime Core Hardening | Stable backend abstraction and memory contracts |
| Phase B | 2026 Q3 | Graph + Operator Framework | Operator registry + graph execution planner |
| Phase C | 2026 Q4 | Compiler and Fusion Layer | Pattern fusion and execution-plan lowering |
| Phase D | 2027 H1 | Model Runner Layer | Tiny transformer runner + training/inference loop |
| Phase E | 2027 H2 | Ecosystem Interop | CoreML/MLX/PyTorch bridge and export/import paths |
| Phase F | 2028 | 1.0 Readiness | API stability, compatibility policy, production-grade CI |

## 5) Detailed Phase Roadmap

## Phase A (2026 Q2): Runtime Core Hardening

Objectives:

- formalize backend interface boundaries,
- lock tensor semantics across backends,
- improve scheduling and sync observability.

Deliverables:

- `BackendDriver` interface split into compute, memory, sync, and profiling surfaces,
- unified memory contract documentation (ownership, lifetime, aliasing, async visibility),
- runtime trace events for upload/dispatch/download/sync,
- deterministic fallback rules when backend feature is unavailable.

Exit criteria:

- same API behavior on Metal/CPU for validated ops,
- zero undefined behavior in tensor lifetime tests,
- reproducible per-op trace logs for benchmark runs.

## Phase B (2026 Q3): Graph + Operator Framework

Objectives:

- move from call-by-call execution to graph-aware planning,
- support operator extensibility without monolithic code growth.

Deliverables:

- operator registry with typed schemas (shape constraints, dtype constraints, backend capability flags),
- minimal graph IR (nodes, edges, tensor metadata, control dependencies),
- execution planner that groups ops by backend and sync boundaries,
- graph validation passes (shape, aliasing, dead tensor, unsupported path).

Exit criteria:

- graph execution parity with eager path on reference workloads,
- measurable reduction in host round-trips for chained workloads,
- plugin-style operator onboarding documentation + templates.

## Phase C (2026 Q4): Compiler and Fusion Layer

Objectives:

- reduce launch overhead and memory traffic through programmatic fusion,
- provide explainable optimization decisions.

Deliverables:

- rule-based fusion pass for known patterns (`matmul+bias+act`, `conv+act`, attention subgraphs),
- cost model v1 with launch-overhead and transfer-overhead terms,
- per-graph optimization report (what fused, what not fused, why),
- fallback lowering when fusion preconditions are not met.

Exit criteria:

- documented fusion correctness tests for each pattern,
- throughput/latency gain in benchmark suites for at least three fusion families,
- no regression beyond agreed tolerance for unfused paths.

## Phase D (2027 H1): Model Runner Layer

Objectives:

- provide a practical model-level entry point,
- prove end-to-end usability beyond operator microbenchmarks.

Deliverables:

- tiny transformer runner (token embedding -> attention/ffn blocks -> logits),
- training loop skeleton (forward/backward/update hooks where supported),
- checkpoint IO format v1,
- CLI runner for reproducible local experiments.

Exit criteria:

- documented quick-start for end-to-end model run,
- stable runner benchmark profile with reproducible outputs,
- API examples covering both inference and training-style loops.

## Phase E (2027 H2): Ecosystem Interoperability

Objectives:

- connect Lightning Core with existing ecosystem tools,
- reduce lock-in and improve adoption.

Deliverables:

- CoreML export pathway for supported graph subset,
- MLX interoperability adapter for tensor exchange,
- PyTorch bridge for selected ops and benchmark parity harness,
- model import adapters (subset-based, explicit capability table).

Exit criteria:

- at least one documented round-trip workflow (import -> optimize -> export),
- interoperability tests in CI for supported subsets,
- clear unsupported-op diagnostics.

## Phase F (2028): 1.0 Readiness

Objectives:

- transition from research-grade runtime to stable framework baseline,
- set long-term compatibility guarantees.

Deliverables:

- semantic versioning policy with deprecation schedule,
- LTS-style support matrix for Python/macOS/Apple Silicon generations,
- hardened CI matrix and release checklist automation,
- full API reference generation and versioned docs site.

Exit criteria:

- zero critical known correctness bugs in 1.0 scope,
- public migration guides for all breaking changes,
- signed, reproducible benchmark report for release candidate.

## 6) Cross-Cutting Workstreams

## 6.1 Backend Strategy

- Metal remains default high-performance backend.
- CPU path remains correctness and fallback baseline.
- CUDA/other backends evolve as optional plugin layers, not core assumptions.
- Backend feature matrix maintained per release (supported ops, dtypes, limits).

## 6.2 Kernel Strategy

- maintain tuned kernels for hotspot shapes,
- add auto-tuning profile persistence by backend/device family,
- unify kernel metadata for planner and cost model,
- enforce kernel validation against reference numerics.

## 6.3 API and UX Strategy

- maintain low-level explicit APIs (`*_into`, resident/session, policy flags),
- add high-level stable APIs for common model flows,
- keep clear distinction between eager micro APIs and graph/model APIs,
- provide copy-paste runnable examples for every major feature.

## 6.4 Quality and Reliability Strategy

- correctness tests, stress tests, and regression benchmarks run in CI,
- publish coverage trend and benchmark trend dashboards,
- maintain hardware/software tested matrix with explicit dates,
- enforce release gates on benchmark reproducibility checks.

## 6.5 Documentation Strategy

- host docs via GitHub Pages or ReadTheDocs,
- auto-generate C++/Python API references,
- keep benchmark methodology and reproducibility guides versioned,
- maintain migration notes for any behavioral/API change.

## 7) Benchmark and Quality Gates per Release

Every minor release should satisfy:

- benchmark artifacts published (CSV/JSON + command lines),
- no untracked performance claim in README,
- API compatibility checks vs previous minor release,
- validated install paths (PyPI/TestPyPI/source),
- CI status green for required workflows.

## 8) Risks and Mitigation

Risk: framework scope expansion hurts macOS-first performance focus.
Mitigation: keep macOS-first benchmark gates mandatory for every phase.

Risk: backend abstraction causes lowest-common-denominator design.
Mitigation: backend capability descriptors + explicit fast-path contracts.

Risk: documentation debt increases as API layers grow.
Mitigation: release checklist includes docs and migration updates as hard gate.

Risk: ecosystem interoperability introduces fragile adapters.
Mitigation: subset-first adapters with strict compatibility tables and tests.

## 9) Public Milestone Tracking

Planned milestone tags:

- `M-A` Runtime Core Hardening
- `M-B` Graph and Operator Framework
- `M-C` Compiler and Fusion
- `M-D` Model Runner
- `M-E` Interoperability
- `M-F` 1.0 Readiness

Each milestone tracks:

- scope,
- acceptance criteria,
- benchmark gates,
- migration impact,
- documentation tasks.

## 10) Immediate Next 90 Days

- [completed] ship operator registry v1 and minimal graph IR prototype.
- [completed] add CI-visible benchmark summary artifact for every main push.
- [completed] add tested environment matrix table to README with concrete device/OS entries.
- [completed] finalize backend abstraction interfaces and docs.
- [completed] harden docs site MVP and expand toward generated API references.

### Next Execution Queue (2026-04-01 Replan)

1. [completed] Backend abstraction split completion (M-A, runtime/api/docs)
   - deliverable: explicit compute/memory/sync/profiler interface boundaries and backend adapter wiring.
   - acceptance: public docs/examples updated and parity tests pass without behavior drift.
2. [completed] Benchmark stability gate (M-A, ci/benchmark)
   - deliverable: variance check (`<=2%`) in release-tag benchmark evidence pipeline.
   - acceptance: gate emits pass/fail summary with reproducible seed/workload metadata.
3. [completed] Graph-mode benchmark adoption pass (M-B, graph/benchmark)
   - deliverable: run graph/eager A/B in shipped pipeline benchmarks and publish host-dispatch delta.
   - acceptance: CSV/JSON include graph metrics and explicit fallback counters.
4. [completed] Generated API reference pipeline (M-A docs + M-B docs)
   - deliverable: docs site build includes generated C++/Python API reference pages.
   - acceptance: docs workflow publishes reference pages and link checks stay green.
5. [completed] Parity + contract coverage expansion (M-A test/runtime)
   - deliverable: extend Metal/CPU parity to graph path and runtime sync policy scenarios.
   - acceptance: CI contract suite blocks merge on regression in shape/layout/lifetime semantics.
6. [completed] macOS-first generic engine mode + Torch bridge (M-A python/runtime)
   - deliverable: integrated API engine selector (`lightning`/`torch`/`auto`) with documented fallback behavior.
   - acceptance: same API surface runs with both engines and benchmark docs separate pure-LC vs interop paths.
7. [completed] Engine split benchmark report hardening (M-A benchmark/docs)
   - deliverable: dedicated pure-LC and interop benchmark artifacts from the same API entrypoints.
   - acceptance: README and release evidence clearly separate pure runtime speed from interop bridge overhead.
8. [completed] v0.1.18 performance round (M-A runtime/benchmark/ci)
   - deliverable: timeline(`group_by=op_path`) guided tiny conv/conv->attn bottleneck optimization and hotspot shape-set coverage in release evidence.
   - acceptance: targeted bottleneck shapes are included in engine-split release artifacts and trace-hotspot fields are emitted per path.
9. [completed] v0.1.19 runtime contract freeze pass (M-A tensor/test/docs/ci)
   - deliverable: stronger tensor shape/layout/lifetime/alias regression tests + frozen contract matrix docs.
   - acceptance: contract-regression subset and Python tensor-contract smoke are explicit hard gates in CI.
10. [completed] v0.1.20 graph execution foundation pass-1 (M-B graph/benchmark)
   - deliverable: capability-aware planner path selection with backend/sync-boundary grouping and fixed host-dispatch reduction fields in graph/eager artifacts.
   - acceptance: graph/eager CSV/JSON/MD always emit host-dispatch reduction metrics and planner grouping tests cover capability-aligned backend selection.
11. [completed] v0.1.21 graph parity + deterministic eager fallback hardening (M-B graph/test/python)
   - deliverable: deterministic graph-request fallback contract for unsupported integrated graph path and expanded regression tests.
   - acceptance: CI smoke blocks numerical drift on graph-request fallback and graph parity stays within tolerance.
12. [completed] v0.1.22 fusion pilot pass-1 (M-C graph/benchmark/ci)
   - deliverable: rule-based `conv+relu` fusion v1 in GraphIR execution, fusion decision report API, and fusion pilot benchmark artifacts.
   - acceptance: fusion correctness and performance gate run in benchmark/release artifact workflows with explicit fallback-reason reporting.
13. [completed] v0.1.23 fusion pass-2 (M-C graph/benchmark/test)
   - deliverable: `matmul+bias+relu` rule-based fusion v1 with decision report + regression coverage.
   - acceptance: fusion report includes both patterns and dedicated correctness/perf gate rows.
14. [completed] v0.1.24 graph planner pass-2 + host-dispatch hard gate (M-B graph/benchmark/ci)
   - deliverable: capability-aware planner score pass and release `graph_eager_ab` hard gate.
   - acceptance: release artifact pipeline enforces minimum host-dispatch reduction rate and publishes graph/eager evidence.
15. [completed] v0.1.25 fusion cost model v1 (M-C graph/docs)
   - deliverable: fusion decision includes estimated unfused/fused cost + speedup and cost-model reject reason.
   - acceptance: Python fusion report is timeline-friendly and explains fused vs fallback decisions.
16. [completed] v0.1.26 engine federation stabilization (M-A python/benchmark/ci/docs)
   - deliverable: `lightning/torch/auto` policy reflected consistently in API wrappers, split benchmarks, and release gates.
   - acceptance: pure-LC vs interop evidence remains structurally separated and coverage-checked in benchmark script.
17. [completed] v0.1.27 checkpoint IO v1 (M-D python/test/docs)
   - deliverable: `save_checkpoint/load_checkpoint` with `state_dict/load_state_dict` support for integrated Python blocks.
   - acceptance: checkpoint round-trip smoke test blocks regressions in CI.
18. [completed] v0.1.28 graph execution foundation pass-2 (M-B graph/benchmark/docs)
   - deliverable: graph plan summary API (`backend/sync/fallback` grouping stats) and fixed host-dispatch evidence fields in graph/eager artifacts.
   - acceptance: graph/eager CSV/JSON/MD always exposes host-dispatch reduction evidence source and planner-summary counters.
19. [completed] v0.1.29 fusion pass-3 attention pattern (M-C graph/benchmark/test)
   - deliverable: rule-based `attention_forward + projection(matmul)` fusion (`attention_proj_v1`) with fallback reasons and cost-model explain fields.
   - acceptance: correctness + perf gate rows pass in fusion pilot benchmark and fallback reasons are reported deterministically.
20. [completed] v0.1.30 deterministic fallback + numerical stress hardening (M-A test/ci/graph)
   - deliverable: randomized boundary shape/dtype/layout regression smoke and expanded graph-request fallback contract checks.
   - acceptance: CI hard gate blocks numerical drift and boundary regressions through dedicated stress smoke scripts.
21. [completed] v0.1.31 checkpoint IO v1.1 model-level expansion (M-D python/test/docs)
   - deliverable: model-level checkpoint helpers (`save_model_checkpoint/load_model_checkpoint`) with compatibility metadata.
   - acceptance: forward-compat smoke verifies v1 checkpoints load through v1.1 model loader.
22. [completed] v0.1.32 autograd bootstrap v0 (M-D python/test)
   - deliverable: `matmul/add/relu` backward graph + tiny MLP 1-step SGD training helper.
   - acceptance: Torch gradient parity smoke and tiny training-step regression test pass in CI.
23. [completed] v0.2.0-rc0 B0 baseline freeze (M-B docs/ci/benchmark)
   - deliverable: freeze graph contracts (support scope/fallback reason codes/numerical tolerance/perf gate constants) in source-of-truth manifest.
   - acceptance: docs/CI sync checker passes and one-shot baseline artifact is generated.
24. [completed] v0.2.0 operator registry v2 contract hardening (M-B graph/test)
   - deliverable: schema-level rank/layout/dtype/shape/attribute enforcement with standardized reason-code surfaces.
   - acceptance: invalid graph boundary tests deterministically report machine-readable reason codes.
25. [completed] v0.2.1 validation pass pack v2 (M-B graph/python/test)
   - deliverable: pass-split graph validation reports (topology/alias-lifetime/layout-flow/backend-capability) with stable report structure.
   - acceptance: C++ regression + Python smoke catch pass/reason regressions in CI.
26. [completed] v0.2.2 planner v3 + plan cache (M-B graph/benchmark/ci)
   - deliverable: graph-hash/device/sync-policy keyed cache with hit-rate telemetry in plan summary and graph/eager artifact outputs.
   - acceptance: host-dispatch evidence remains fixed in artifacts and plan-cache hit/miss fields are always present.
27. [completed] v0.2.7 phase B exit audit + release-candidate evidence bundle (M-B benchmark/ci/docs)
   - deliverable: release/CI runs generate `phase_b_exit_audit` (JSON/MD) and candidate-bundle manifest with fixed evidence file hashing.
   - acceptance: ROADMAP 11.2 metrics (dispatch/chained-latency/adoption) are emitted every run and release tag gate checks README/ROADMAP version sync.

### Next Execution Queue (2026-04-09 Phase C Kickoff Replan)

28. [completed] v0.2.8 release gate stabilization pass (M-B benchmark/ci)
   - deliverable: chained-latency gate applicability is tied to chain-level dispatch-reduction evidence to prevent false failures on non-applicable paths.
   - acceptance: benchmark artifact workflow publishes fixed applicability metrics in CSV/JSON/MD without false chained-latency gate failures.
29. [completed] v0.2.9 fusion pass manager v1 (M-C graph/docs/test)
   - deliverable: explicit fusion pass manager ordering (`conv`, `matmul`, `attention`) with per-pass enable flags and deterministic explain output.
   - acceptance: pass-order regression tests and explain-report snapshots are stable across repeated runs.
30. [completed] v0.2.10 attention fusion pass-4 subset (M-C graph/benchmark)
   - deliverable: rule-based subset for `qk^T -> softmax -> v` (or equivalent safe decomposition) with deterministic fallback reason taxonomy.
   - acceptance: correctness parity within tolerance + dedicated perf gate row + fallback reason coverage in fusion artifacts.
31. [completed] v0.2.11 cost model v2 calibration (M-C benchmark/runtime)
   - deliverable: backend/device-calibrated launch/transfer coefficients, persisted by runtime profile signature.
   - acceptance: cost-model decisions are reproducible and beat/unmatch baseline within documented confidence bounds.
32. [completed] v0.2.12 planner-cost co-optimization (M-B/M-C graph)
   - deliverable: planner integrates cost-model signals when selecting backend/sync/fusion boundaries for chain workloads.
   - acceptance: host-dispatch reduction and chained-latency gates improve or remain within regression budget on representative shape sets.
33. [completed] v0.2.13 checkpoint IO v1.2 integrity hardening (M-D python/test)
   - deliverable: checkpoint metadata now includes tensor hash/manifest hash/format version/signature with `validate_checkpoint` and conversion diagnostics APIs.
   - acceptance: corrupted/mismatched checkpoints fail with structured reason codes (`checkpoint_*`) and CI smoke asserts hard failures.
34. [completed] v0.2.14 autograd bootstrap v1 expansion (M-D python/ops/test)
   - deliverable: backward coverage extended with `ag_conv2d` and `ag_attention`, plus tiny multi-step conv->attention training helper.
   - acceptance: Torch gradient parity smoke covers conv and attention-adjacent primitives for supported subsets.
35. [completed] v0.2.15 model runner alpha (M-D model/docs)
   - deliverable: `TinyTransformerRunner` with `eager/graph/interop` mode switch and one-command benchmark (`model_runner_alpha_bench.py`).
   - acceptance: reproducible CSV/JSON/MD artifacts are emitted for runner modes.
36. [completed] v0.2.16 interop boundary hardening (M-E python/benchmark)
   - deliverable: strict route-policy validator + boundary switch/copy/overhead telemetry and standardized interop boundary reason codes.
   - acceptance: interop benchmark now emits boundary metrics and supports max-overhead budget gate.
37. [completed] v0.2.17 phase C exit audit + v0.3.0-rc0 criteria lock (M-C/M-F docs/ci)
   - deliverable: `phase_c_exit_audit.py` + `docs/phase_c_engine_contract.json` with CI/release wiring for Phase C audit bundles.
   - acceptance: docs/CI/release metadata sync checks include Phase C contract lock paths and hard-gate audit flow.

### Next Execution Queue (2026-04-11 Post-v0.3.4 Transition)

38. [completed] v0.3.0 release metadata sync hardening (M-A docs/release)
   - deliverable: single-source version sync (`pyproject` -> README/ROADMAP/contracts/release notes) + sync report generation.
   - acceptance: release-tag CI hard-fails on version drift and emits `version_sync_report.{json,md}` evidence.
39. [completed] v0.3.1 model runner beta contracts (M-D model/test)
   - deliverable: runner config schema validator, deterministic replay report API, checkpoint compatibility matrix, fixed artifact schema.
   - acceptance: runner replay/schema smoke and CI benchmark artifacts enforce stable CSV/JSON/MD contract.
40. [completed] v0.3.2 torch bridge beta (M-E python/interop)
   - deliverable: `nn.Module` wrapper with explicit `route_policy`, always-on boundary telemetry, deterministic fallback reason codes.
   - acceptance: torch wrapper parity smoke plus interop reason-coverage and overhead-budget gates are wired into CI/release benchmarks.
41. [completed] v0.3.3 tensorflow bridge prototype (M-E python/interop)
   - deliverable: minimal `keras.Layer`-style wrapper, deterministic TF fallback reason mapping, and missing-runtime graceful numpy shim path.
   - acceptance: TF smoke covers both installed/fake-runtime and runtime-missing paths; TF interop benchmark artifacts emit fixed schema.
42. [completed] v0.3.4 interop boundary budget v2 (M-E benchmark/ci)
   - deliverable: upload/switch/copy/sync boundary decomposition, zero-copy vs fallback-copy split metrics, and per-boundary budget gates.
   - acceptance: interop benchmark + CI/release workflows enforce component budgets and zero-copy-fallback reason coverage.
43. [completed] Fusion/Cost explain coverage hardening (M-C graph/benchmark)
   - deliverable: normalized reason-code fields (`fusion_reason_code`, `fusion_disabled_reason_code`, `cost_model_reject_reason_code`) for all fusion rows.
   - acceptance: explain reason-code coverage is emitted and gateable in fusion artifacts and Phase-C audit.
44. [completed] Phase C performance evidence expansion (M-C benchmark/audit)
   - deliverable: audit now emits conv/attention/ffn E2E improvement evidence fields and dispatch overhead p95 trend snapshot fields.
   - acceptance: release/CI audit bundles expose these metrics in fixed JSON/MD schema for longitudinal tracking.
45. [completed] Phase C exit audit bundle hardening (M-F docs/ci)
   - deliverable: `phase_c_exit_audit` adds required-artifact manifest hash, TF interop input support, and expanded interop boundary metrics.
   - acceptance: bundle carries deterministic manifest-hash evidence and remains hard-gate ready for release candidates.
46. [completed] v0.4.0-rc0 runner contract freeze (M-D model/ci)
   - deliverable: runner input schema (`mode/device/seed/layout/dtype/route_policy`) frozen with hash/freeze-id manifest and artifact schema v3.
   - acceptance: schema-drift smoke fails on missing/unknown fields and CI runs freeze hard-gate tests.
47. [completed] v0.4.1 tiny transformer runner beta (M-D model/benchmark)
   - deliverable: single API path now supports `embedding->attention->ffn->logits` for `eager/graph/interop` with identical interface.
   - acceptance: one-command benchmark emits reproducible artifact v3 and parity smoke passes for runner modes.
48. [completed] v0.4.2 training surface v1 (M-D autograd/test)
   - deliverable: multi-step training loop with optimizer step, grad-clip norm, loss-scale, and per-step hooks.
   - acceptance: gradient parity smoke + loss-decrease multi-step smoke pass in Python hard-gate tests.
49. [completed] v0.4.3 checkpoint IO v2 runner manifest (M-D checkpoint/test)
   - deliverable: model/runner graph-state + optimizer-state + metadata manifest persisted in `lc_checkpoint_v2` with compat matrix.
   - acceptance: cross-version load smoke (`v1/v1.1/v1.2/v2`) passes and runner checkpoint compatibility matrix is published.
50. [completed] v0.4.4 Torch engine adapter GA (M-E torch/benchmark)
   - deliverable: `nn.Module`-style runner wrapper formalized with always-on boundary telemetry, budget fields, and reason-code coverage fields.
   - acceptance: torch-runner adapter artifacts enforce 100% reason coverage and boundary budget pass-rate gates.
51. [completed] v0.4.5 TensorFlow engine adapter beta (M-E tf/benchmark)
   - deliverable: `keras.Layer` runner path expanded to model-level adapter API with tensorflow/numpy-shim dual-runtime path.
   - acceptance: TF runner adapter smoke + artifact schema regression gate pass for installed/missing runtime paths.
52. [completed] v0.4.6 runner CLI + repro pack (M-D cli/docs)
   - deliverable: `lc-run` CLI supports `infer/train/bench` and always emits reproducibility manifest (`command/env/artifact hash`).
   - acceptance: quickstart copy-paste command paths generate deterministic JSON + repro-pack artifacts.
53. [completed] v0.4.7 phase D exit audit (M-F audit/ci/docs)
   - deliverable: phase-D audit bundle with quickstart <=50-line check, inference/training example presence, and runner variance (`CV<=2%`) hard gate.
   - acceptance: `phase_d_exit_audit` JSON/MD/bundle artifacts are generated in CI/release workflows with contract-sync checks.
54. [completed] v0.4.8 stabilization patch (M-A runtime/docs/test)
   - deliverable: tune-cache v2 header/version management for matmul/attention + docs-pages metadata sync + local artifact cleanup policy.
   - acceptance: worktree-local artifact policy is codified in `.gitignore` and tune-cache format regression smoke (`test_python_tune_cache_format_smoke.py`) is wired into CI.
55. [completed] v0.5.0 phase E contract freeze (M-E docs/ci)
   - deliverable: `docs/engine_federation_contract.json` promoted to source-of-truth with engine/reason/budget constants and CI sync checker.
   - acceptance: workflow contract checks hard-fail on docs/CI drift via `check_phase_e_contract_sync.py`.
56. [completed] v0.5.1 CoreML engine adapter alpha (M-E python/benchmark)
   - deliverable: CoreML runtime bridge exposed to Python runner adapter with deterministic fallback telemetry/reason codes.
   - acceptance: CoreML adapter benchmark emits fixed CSV/JSON/MD schema and smoke tests validate runtime-missing/model-missing deterministic paths.
57. [completed] v0.5.2 MLX bridge alpha (M-E python/benchmark)
   - deliverable: MLX runner adapter with runtime-optional bridge path and reason-coded fallback/copy telemetry.
   - acceptance: MLX benchmark + smoke tests enforce reason-code coverage and artifact schema stability.
58. [completed] v0.5.3 engine federation policy v2 (M-E runtime/benchmark)
   - deliverable: unified federation schema (`lightning/torch/tf/coreml/mlx/auto`) and bridge telemetry contract surfaces in `lc.api`.
   - acceptance: route-policy schema/bridge schema/version surfaces are queryable and benchmarked across adapters.
59. [completed] v0.5.4 import/export compatibility matrix automation (M-E docs/tooling)
   - deliverable: auto-generated import/export capability matrix (`README + docs`) from `docs/import_export_compatibility_matrix.json`.
   - acceptance: CI/docs workflows enforce matrix sync (`generate_import_export_matrix_docs.py --check`) and link checks pass.
60. [completed] v0.5.5 interop perf gate 강화 (M-E benchmark/ci)
   - deliverable: `interop_perf_gate_v2.py` aggregates bridge-level reason coverage + overhead decomposition into fixed release evidence.
   - acceptance: release/CI workflows hard-gate reason/explain coverage and boundary overhead budgets with structured JSON/MD outputs.
61. [completed] v0.5.6 phase E exit audit (M-F audit/ci/docs)
   - deliverable: `phase_e_exit_audit.py` + candidate bundle manifest with matrix-sync/docs-sync/artifact-hash evidence lock.
   - acceptance: Phase E exit audit is wired into benchmark/release workflows as a hard gate.

Progress update history is auto-generated from:

- `docs/roadmap_updates.json`

<!-- AUTO-ROADMAP-HISTORY:BEGIN -->

### Progress History (Auto-generated)

- Total tracked updates: `82`
- Source of truth: `docs/roadmap_updates.json`
- Quick add command:
  `python scripts/generate_roadmap_history.py --add --date YYYY-MM-DD --milestone M-A --area runtime --title "your update"`

**Date Summary**

| Date | Updates | Milestones | Highlights |
| --- | --- | --- | --- |
| 2026-04-11 | 4 | M-E, M-C, M-A | Completed v0.3.4 interop boundary budget v2 with upload/switch/copy/sync decomposition and per-component budget gates. / Completed v0.3.3 TensorFlow bridge prototype with deterministic fallback mapping and graceful missing-runtime shim. / ... (+2 more) |
| 2026-04-09 | 14 | M-C, M-B, M-A | Published detailed Phase C kickoff roadmap queue (v0.2.8-v0.2.17) with deliverables and acceptance gates. / Completed v0.2.8 release-gate stabilization by applying chained-latency applicability only to chain-dispatch-reduced cases. / ... (+12 more) |
| 2026-04-08 | 19 | M-D, M-C, M-B, M-A | Completed v0.1.32 autograd bootstrap v0 (matmul/add/relu backward + tiny 1-step SGD) with Torch gradient parity smoke. / Completed v0.1.31 checkpoint IO v1.1 model-level save/load helpers with v1 forward-compat smoke coverage. / ... (+17 more) |
| 2026-04-07 | 6 | M-A | Optimized tiny conv->attn integrated path using op_path timeline bottleneck guidance and tiny-chain CPU preference heuristic. / Finalized lc.api engine bridge (lightning/torch/auto) with same-surface engine switching / ... (+4 more) |
| 2026-04-02 | 5 | M-B, M-A | Completed v0.1.15 generated API reference pipeline (Python/C++) in docs build and removed API index placeholder entries. / Expanded graph-path contract coverage: sync policy(auto/always/never), fallback/device-change boundary checks, and shape/layout/lifetime regression guards. / ... (+3 more) |
| 2026-04-01 | 16 | M-B, M-A | Completed generated API reference pipeline with auto-built Python/C++ reference pages and docs link-check gate in CI/docs workflows. / Added graph/eager A/B benchmark script with runtime host-dispatch delta and fallback counters, plus CI artifact publishing. / ... (+14 more) |
| 2026-03-31 | 6 | M-A | Shipped docs site MVP with mkdocs and docs-pages workflow. / Re-tuned tiny one-shot conv CPU crossover default to `CJ_CONV2D_CPU_CROSSOVER_MACS=260000` via threshold sweep. / ... (+4 more) |
| 2026-03-30 | 9 | M-B, M-A | Added operator registry v1 and minimal Graph IR prototype. / Added graph validation report passes and grouped planner options with sync-boundary/fallback segmentation. / ... (+7 more) |
| 2026-03-29 | 2 | M-A | Split docs into quickstart/advanced/index and improved package/release guidance. / Added large GEMM auto sweep, tuned policy profiles, and cross-suite summary artifacts. |
| 2026-03-28 | 1 | M-A | Initial macOS package and release workflow launch. |

**Detailed Timeline**

#### 2026-04-11 (4 updates)

- [completed] [M-E] [benchmark] Completed v0.3.4 interop boundary budget v2 with upload/switch/copy/sync decomposition and per-component budget gates. (`local`)
- [completed] [M-E] [python] Completed v0.3.3 TensorFlow bridge prototype with deterministic fallback mapping and graceful missing-runtime shim. (`local`)
- [completed] [M-C] [graph] Hardened fusion explain coverage with normalized reason-code fields and CI/release gate enforcement. (`local`)
- [completed] [M-A] [release] Bumped public baseline to v0.3.4 and synced release metadata across README/ROADMAP/contracts/release notes. (`local`)

#### 2026-04-09 (14 updates)

- [completed] [M-C] [docs] Published detailed Phase C kickoff roadmap queue (v0.2.8-v0.2.17) with deliverables and acceptance gates. (`local`)
- [completed] [M-B] [benchmark] Completed v0.2.8 release-gate stabilization by applying chained-latency applicability only to chain-dispatch-reduced cases. (`local`)
- [completed] [M-B] [benchmark] Completed v0.2.7 Phase B exit audit with release-candidate artifact bundle and ROADMAP 11.2 metric gate wiring. (`local`)
- [completed] [M-B] [docs] Completed v0.2.6 docs + operator onboarding kit with copy-paste smoke gate and MkDocs navigation integration. (`local`)
- [completed] [M-B] [benchmark] Completed v0.2.5 benchmark/release gate strengthening with fixed planner/fallback evidence fields plus chained latency/unsupported ratio gates. (`local`)
- [completed] [M-B] [python] Completed v0.2.4 hybrid execution policy formalization with subgraph-level route_policy (conv/attention/graph engines) and deterministic graph fallback reason codes. (`local`)
- [completed] [M-B] [graph] Completed v0.2.3 integrated conv->attn graph coverage expansion (conv2d_nchw3x3 attrs + qkv_pack_repeat path) with widened shape-set benchmark cases. (`local`)
- [completed] [M-B] [graph] Completed v0.2.2 planner v3 + plan cache with cache hit-rate telemetry and fixed graph/eager dispatch evidence artifacts. (`local`)
- [completed] [M-B] [graph] Completed v0.2.1 validation pass pack v2 with pass-scoped topology/alias-lifetime/layout-flow/backend-capability reports. (`local`)
- [completed] [M-B] [docs] Completed v0.2.0-rc0 B0 contract baseline freeze with docs/CI constant sync and baseline artifact generation. (`local`)
- [completed] [M-B] [graph] Completed v0.2.0 Operator Registry v2 contracts with rank/layout/dtype/shape/attribute validation and deterministic reason codes. (`local`)
- [completed] [M-A] [release] Bumped public baseline to v0.2.8 (post-gate-fix) and aligned README/ROADMAP/pyproject version metadata. (`local`)
- [completed] [M-A] [release] Bumped public baseline to v0.2.7 and aligned README/ROADMAP/pyproject version metadata. (`local`)
- [completed] [M-A] [release] Bumped public baseline to v0.2.2 and aligned README/ROADMAP/pyproject version metadata. (`local`)

#### 2026-04-08 (19 updates)

- [completed] [M-D] [test] Completed v0.1.32 autograd bootstrap v0 (matmul/add/relu backward + tiny 1-step SGD) with Torch gradient parity smoke. (`local`)
- [completed] [M-D] [python] Completed v0.1.31 checkpoint IO v1.1 model-level save/load helpers with v1 forward-compat smoke coverage. (`local`)
- [completed] [M-D] [python] Completed v0.1.27 checkpoint IO v1 with save/load helpers and integrated block state_dict/load_state_dict paths.
- [completed] [M-C] [graph] Completed v0.1.29 fusion pass-3 with attention_forward+projection(matmul) rule-based pattern, explain report, and benchmark gate coverage. (`local`)
- [completed] [M-C] [graph] Completed v0.1.25 fusion cost model v1 with estimated fused/unfused costs, speedup fields, and cost-model reject reasons.
- [completed] [M-C] [graph] Completed v0.1.23 fusion pass-2 with matmul+bias+relu v1 pattern, expanded fusion report coverage, and dedicated benchmark rows.
- [completed] [M-C] [graph] Completed v0.1.22 fusion pilot pass-1 with conv+relu v1 rule-based fusion, fusion report API, and benchmark/release artifact gate wiring.
- [completed] [M-B] [graph] Completed v0.1.28 graph plan summary API and fixed host-dispatch evidence fields in graph/eager artifacts. (`local`)
- [completed] [M-B] [graph] Completed v0.1.24 graph planner pass-2 with capability-aware scoring and release host-dispatch hard-gate wiring.
- [completed] [M-B] [graph] Completed v0.1.21 graph parity + deterministic eager fallback hardening (integrated graph-request fallback contract + CI smoke coverage).
- [completed] [M-B] [graph] Completed v0.1.20 graph execution foundation pass-1: capability-aware planner grouping and fixed host-dispatch reduction metrics in graph/eager artifacts. (`local`)
- [completed] [M-A] [test] Completed v0.1.30 deterministic fallback and numerical stress hardening with randomized boundary shape/dtype/layout CI smoke. (`local`)
- [completed] [M-A] [benchmark] Completed v0.1.26 engine federation stabilization with engine split coverage checks and release evidence consistency.
- [completed] [M-A] [test] Completed v0.1.19 runtime contract freeze with strengthened tensor shape/layout/lifetime/alias regression tests and explicit CI hard gates. (`local`)
- [completed] [M-A] [release] Bumped public baseline to v0.1.32 and aligned README/ROADMAP/pyproject version metadata. (`local`)
- [completed] [M-A] [release] Bumped public baseline to v0.1.27 and aligned README/ROADMAP/pyproject version metadata.
- [completed] [M-A] [release] Bumped public baseline to v0.1.22 and aligned README/ROADMAP/pyproject version metadata.
- [completed] [M-A] [release] Bumped public baseline to v0.1.20 and aligned README/ROADMAP/pyproject version metadata. (`local`)
- [completed] [M-A] [release] Bumped public baseline to v0.1.19 and aligned README/ROADMAP/pyproject version metadata. (`local`)

#### 2026-04-07 (6 updates)

- [completed] [M-A] [runtime] Optimized tiny conv->attn integrated path using op_path timeline bottleneck guidance and tiny-chain CPU preference heuristic. (`local`)
- [completed] [M-A] [python] Finalized lc.api engine bridge (lightning/torch/auto) with same-surface engine switching (`local`)
- [completed] [M-A] [ci] Extended release/CI engine-split benchmark evidence with trace-iters and hotspot shape-set reporting for tiny conv and conv->attn cases. (`local`)
- [completed] [M-A] [release] Bumped public baseline to v0.1.18 and aligned README/ROADMAP/pyproject version metadata. (`local`)
- [completed] [M-A] [release] Bumped public baseline to v0.1.17 and aligned README/ROADMAP/version metadata (`local`)
- [completed] [M-A] [benchmark] Added engine_split_bench and release/CI evidence split for pure-LC vs interop (`local`)

#### 2026-04-02 (5 updates)

- [completed] [M-B] [docs] Completed v0.1.15 generated API reference pipeline (Python/C++) in docs build and removed API index placeholder entries.
- [completed] [M-A] [test] Expanded graph-path contract coverage: sync policy(auto/always/never), fallback/device-change boundary checks, and shape/layout/lifetime regression guards.
- [completed] [M-A] [release] Bumped release baseline to v0.1.16 for parity + contract coverage expansion.
- [completed] [M-A] [release] Bumped release baseline to v0.1.15 for generated API reference pipeline completion.
- [completed] [M-A] [ci] Added CI Python smoke gate for integrated API + runtime timeline (tests/test_python_integrated_timeline_smoke.py).

#### 2026-04-01 (16 updates)

- [completed] [M-B] [docs] Completed generated API reference pipeline with auto-built Python/C++ reference pages and docs link-check gate in CI/docs workflows.
- [completed] [M-B] [benchmark] Added graph/eager A/B benchmark script with runtime host-dispatch delta and fallback counters, plus CI artifact publishing.
- [completed] [M-A] [python] Moved integrated API helper into package distribution and install path (wheel/editable).
- [completed] [M-A] [test] Expanded contract coverage with graph planned-path Metal/CPU parity checks and runtime sync-policy trace-detail assertions.
- [completed] [M-A] [python] Completed macOS-first generic engine mode + Torch bridge milestone with documented pure-LC vs interop benchmark separation.
- [completed] [M-A] [runtime] Completed backend interface split contracts across C++/C/Python and added integrated engine selector (lightning/torch/auto).
- [completed] [M-A] [release] Bumped to v0.1.9 and updated release baseline/docs.
- [completed] [M-A] [release] Bumped to v0.1.8 and aligned README roadmap baseline. (`d486d05`)
- [completed] [M-A] [release] Bumped to v0.1.11 with release benchmark stability gate (suite-total CV<=2%) and updated release baseline/docs.
- [completed] [M-A] [release] Bumped to v0.1.10 after removing workspace-level duplicate helper file and verifying package-only import path.
- [completed] [M-A] [release] Bumped release baseline to v0.1.14 after completing roadmap queue items 1-6.
- [completed] [M-A] [docs] Automated README/docs capability and tested-environment matrix generation.
- [completed] [M-A] [ci] Added release-tag benchmark gate with standardized evidence bundle (csv/json/summary/command/environment/manifest) and PyPI publish dependency.
- [completed] [M-A] [ci] Added release benchmark stability gate with fixed-seed repeated quick-bench runs and CV<=2% enforcement.
- [completed] [M-A] [runtime] Added op-dispatch trace metadata (`op`, `requested_device`, `selected_device`, `fallback`) for timeline bottleneck analysis. (`74d31a1`)
- [completed] [M-A] [python] Added `runtime_trace_timeline` API with sorting/grouping/hotspot extraction. (`c0a56c2`)

#### 2026-03-31 (6 updates)

- [completed] [M-A] [docs] Shipped docs site MVP with mkdocs and docs-pages workflow. (`bc8adf7`)
- [completed] [M-A] [conv] Re-tuned tiny one-shot conv CPU crossover default to `CJ_CONV2D_CPU_CROSSOVER_MACS=260000` via threshold sweep. (`3f32888`)
- [completed] [M-A] [integrated] Enabled shape-keyed graph/session caching on conv->attn path to reduce per-call rebuild overhead. (`3f32888`)
- [completed] [M-A] [ci] Added quick benchmark artifact workflow and summary publishing. (`7bab6cb`)
- [completed] [M-A] [ci] Added contract-test quality gate workflow with CMake + CTest. (`e873de2`)
- [completed] [M-A] [test] Added Metal/CPU backend parity coverage for matmul/vector/matrix/attention/conv. (`cfd1272`)

#### 2026-03-30 (9 updates)

- [completed] [M-B] [graph] Added operator registry v1 and minimal Graph IR prototype. (`8c4993b`)
- [completed] [M-B] [graph] Added graph validation report passes and grouped planner options with sync-boundary/fallback segmentation. (`7bd57a3`)
- [completed] [M-B] [graph] Added graph execution path for matmul/vector/matrix/attention/conv and integrated graph/eager A/B toggle. (`567b633`)
- [completed] [M-A] [api] Removed legacy `cudajun` forwarding headers and unified canonical `lightning_core` include surface. (`60eff32`)
- [completed] [M-A] [docs] Expanded README roadmap details and bumped docs/version narrative to v0.1.5. (`dfc49d1`)
- [completed] [M-A] [tensor] Added tensor semantics contract validators (shape/stride/layout/view bounds) and Python exposure. (`f1932f6`)
- [completed] [M-A] [runtime] Added runtime trace observability baseline and Python bindings. (`fe9912d`)
- [completed] [M-A] [runtime] Added explicit runtime sync policy API across C++/C/Python. (`468069c`)
- [completed] [M-A] [runtime] Added backend capability contract surfaces across C++/C/Python. (`b75551d`)

#### 2026-03-29 (2 updates)

- [completed] [M-A] [docs] Split docs into quickstart/advanced/index and improved package/release guidance. (`a25b1ed`)
- [completed] [M-A] [benchmark] Added large GEMM auto sweep, tuned policy profiles, and cross-suite summary artifacts. (`7be1280`)

#### 2026-03-28 (1 updates)

- [completed] [M-A] [release] Initial macOS package and release workflow launch. (`8b1d456`)

<!-- AUTO-ROADMAP-HISTORY:END -->

## 11) Release-Train Detail (v0.5.6 -> v1.0)

## 11.1 2026 Q2 (v0.1.32 ~ v0.2.0): Runtime Contracts [completed]

Planned scope:

- split runtime backend driver into capability-based interfaces,
- introduce explicit stream/sync policy object in public runtime API,
- freeze tensor metadata contract (shape/stride/layout/dtype ownership rules),
- expose first profiler timeline API to Python,
- stabilize integrated engine selector for macOS-first generic usage (lightning/torch/auto).

Quality gates:

- no new public API without docs + example,
- per-op parity tests across Metal and CPU in CI,
- benchmark scripts publish CSV/JSON artifacts on every release tag,
- release benchmark gate enforces suite-total LC variance (`CV <= 2%`) on fixed seed/workload repeats and publishes per-case CV diagnostics.

Success metrics:

- <= 2% variance across repeated microbench runs on the same device (fixed seed/workload),
- zero known tensor lifetime bugs in open issue tracker for two consecutive releases,
- documented capability matrix published in README and docs site.

## 11.2 2026 Q3 (v0.2.x): Graph Execution Foundation [completed]

Planned scope:

- operator registry with schema validation and backend support flags,
- graph IR for dependency-aware scheduling,
- compile-time and runtime validation passes with human-readable diagnostics,
- graph/eager execution toggle for A/B verification,
- define hybrid execution policy between LC graph path and Torch engine fallback.

Quality gates:

- graph mode must match eager numerical outputs within configured tolerance,
- every fused/optimized graph path must keep a deterministic fallback path,
- unsupported op diagnostics include actionable remediation message.

Success metrics:

- >= 25% reduction in host dispatch calls for representative pipeline benchmarks,
- >= 15% latency reduction on chained workloads dominated by launch overhead,
- graph mode adoption in all shipped benchmark pipelines.

## 11.3 2026 Q4 (v0.3.x): Fusion and Cost Model [entry criteria locked]

Planned scope:

- rule-based fusion engine + pass manager,
- cost model v1 (launch cost, transfer cost, occupancy heuristic),
- optimization explain report (`what`, `why`, `fallback reason`),
- persistent tuning cache format v2 with compatibility versioning.

Quality gates:

- each fusion pattern gets dedicated correctness + perf regression tests,
- no fallback correctness drift when fusion is disabled,
- tuning cache corruption fallback path tested and documented.

Success metrics:

- >= 20% end-to-end speedup in at least 3 benchmark families (conv/attn/ffn pipelines),
- optimization report coverage for 100% graph executions,
- p95 dispatch overhead reduced release-over-release on target devices.

## 11.4 2027 H1 (v0.4.x): Model Runner and Training Surface

Planned scope:

- lightweight model runner with reusable block abstraction,
- tiny transformer reference model (inference first, training hooks second),
- optimizer/checkpoint interfaces and format compatibility policy,
- minimal data loader abstraction for reproducible experiments.

Quality gates:

- one-command end-to-end runnable examples,
- checkpoint load/save forward compatibility tests,
- runner-level benchmark and memory profile published per release.

Success metrics:

- users can run end-to-end sample model with <50 lines of Python,
- stable throughput variance envelope defined and published,
- docs-driven onboarding validation (new user follows quickstart without internal guidance).

## 11.5 2027 H2 (v0.5.x): Interoperability, Ecosystem Bridges, and Engine Federation

Planned scope:

- CoreML export for validated op subset,
- MLX tensor bridge and layout conversion utilities,
- PyTorch interop runner and engine federation hooks for hybrid execution experiments,
- import/export compatibility tables generated automatically.

Quality gates:

- each bridge path has integration tests in CI,
- unsupported conversions fail with deterministic structured errors,
- benchmark docs separate pure-LC numbers from interop overhead numbers.

Success metrics:

- at least one stable round-trip (import -> optimize -> export -> run),
- interop smoke tests green across supported macOS/SoC matrix,
- first external user reports successful interop workflow without maintainer patching.

## 11.6 2028 (v1.0): Framework Stabilization

Planned scope:

- semantic versioning + LTS support policy,
- production-grade observability (trace, metrics, debug bundle),
- deterministic build + benchmark reproducibility package,
- API freeze for 1.0 surface with migration guides.

Quality gates:

- zero P0 correctness issues in release candidate cycle,
- release checklist fully automated in CI/CD,
- signed benchmark report for release artifacts.

Success metrics:

- stable API adoption across at least two independent application repos,
- measured regression budget maintained within defined threshold,
- migration pain index reduced through versioned docs and upgrade tooling.

## 12) Architecture Expansion Blueprint

## 12.1 Backend Plugin Model

- `BackendDescriptor`: declares device family, supported ops/dtypes/layout constraints.
- `KernelRegistry`: backend-local kernel metadata + launch capability query.
- `MemoryAdapter`: unified buffer semantics (owned, borrowed, shared, host-visible).
- `SyncAdapter`: explicit event/fence/wait API for deterministic boundary control.
- `ProfileAdapter`: backend trace hooks normalized to common event schema.

Design rule:

- common API is stable, backend-specific optimization knobs are opt-in extension fields.

## 12.2 Graph and Compiler Stack

- `Front IR`: user-facing op graph with readable semantics.
- `Mid IR`: optimization-ready graph with canonicalized layouts and explicit dependencies.
- `Low IR`: backend-lowerable plan (kernel choice, launch groups, sync points).
- pass pipeline: validation -> canonicalization -> fusion -> scheduling -> lowering.
- explanation pipeline: pass-by-pass diff summary for auditability.

## 12.3 Runtime + Model Layer Separation

- runtime core remains operator and scheduling focused,
- model runner layer composes reusable blocks (attn, ffn, norm, embedding),
- training helpers stay modular so inference-only users avoid overhead,
- extension hooks allow custom blocks without editing runtime internals.

## 13) Productization Milestones

## 13.1 Documentation Productization

- docs hosting with version switcher and release-aligned snapshots,
- generated C++/Python API reference with source links,
- benchmark reproducibility guide with exact commands and environment capture,
- architecture decision records (ADR) for major changes.

## 13.2 Reliability Productization

- tested environment matrix (macOS + M1/M2/M3/M4 variants),
- benchmark regression dashboard with historical trend lines,
- flaky-test quarantine and weekly stabilization policy,
- release blocker triage protocol with SLA targets.

## 13.3 Community Productization

- contribution guide with `good first issue` track,
- template issues for perf regression, correctness bug, and feature request,
- monthly roadmap update notes tied to milestone tags,
- transparent benchmark archive for each release.

## 14) "Mac-First to General Framework" Guardrails

- never trade away Metal fast-path quality for abstract architecture elegance,
- require every new abstraction to prove no regression on macOS benchmark gates,
- separate "portable baseline path" and "mac-optimized path" explicitly in docs,
- promote portability by additive backend plugins, not by weakening core contracts,
- keep benchmark claims reproducible with public scripts and raw artifacts.
