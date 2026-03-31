# Lightning Core Roadmap

Version context: v0.1.7 (2026-03-31)

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

## 3) Current Baseline (v0.1.7)

- Public package on PyPI/TestPyPI.
- C++ core + Python bindings for runtime/tensor/ops/attention/integrated APIs.
- Resident execution and policy-based IO control.
- Public benchmark source and quick benchmark script.
- Legacy `cudajun` forwarding headers removed; canonical namespace is `lightning_core`.

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

- finalize backend abstraction interfaces and docs,
- ship operator registry v1 and minimal graph IR prototype,
- add CI-visible benchmark summary artifact for every main push,
- harden docs site MVP and expand toward generated API references,
- add tested environment matrix table to README with concrete device/OS entries.

Progress update (2026-03-31):

- completed: runtime trace observability API baseline (C++ runtime + Python binding + runtime smoke test).
- completed: explicit runtime sync policy object (C++/C/Python APIs + policy-based synchronize path).
- completed: backend capability contract surfaces (compute/memory/sync/profiling descriptors via C++/C/Python APIs).
- completed: tensor semantics contract checks (shape/stride/layout/view-bounds validators with Python exposure + tests).
- completed: operator registry v1 + minimal graph IR prototype (C++ API, Python bindings, validation/planning test).
- completed: graph validation report passes + grouped planner options (device fallback segmentation + sync-boundary grouping).
- completed: graph execution path for matmul/vector/matrix/attention/conv (C++ + Python) and integrated conv->attn graph/eager execution toggle for A/B verification.
- completed: integrated graph validation diagnostics with actionable hints and `conv_attention_torchstrong_nchw_ab_report` API for eager-vs-graph parity/speed checks.
- completed: attention Python bindings now reuse shape/device session cache for repeated `attention2d`/`attention_forward` calls (lower binding-path overhead).
- completed: integrated conv->attn graph path now caches shape-keyed GraphIR sessions to remove per-call graph rebuild overhead.
- completed: tiny one-shot conv crossover (`CJ_CONV2D_CPU_CROSSOVER_MACS`) routes small Metal convs to CPU when end-to-end latency is lower.
- completed: tiny one-shot conv crossover default re-tuned to `260000` MACs using threshold sweep (`100000..300000`) while keeping benchmark win coverage (`kernel/pipeline/ml` losing rows = 0 in validation run).
- completed: CI-visible benchmark summary artifact workflow now runs `benchmarks/python/quick_bench.py` on every `main/master` push and uploads `csv/log/json/md` artifacts with GitHub step summary.
- completed: CI correctness gate workflow now runs CMake configure/build + CTest on every `main/master` push and pull request.
- completed: per-op Metal/CPU parity coverage is now enforced in CI via `test_backend_parity` (matmul/vector/matrix/attention/conv).
- completed: Python runtime timeline API (`runtime_trace_timeline`) now provides trace sorting, bottleneck grouping, and hotspot extraction for immediate runtime-path diagnosis.
- completed: docs site MVP pipeline is implemented (`mkdocs.yml` + `docs-pages.yml`) with `quickstart`/`advanced`/`api_index` navigation; deploy auto-runs when repository Pages is enabled for GitHub Actions.

## 11) Release-Train Detail (v0.1.7 -> v1.0)

## 11.1 2026 Q2 (v0.1.7 ~ v0.1.9): Runtime Contracts

Planned scope:

- split runtime backend driver into capability-based interfaces,
- introduce explicit stream/sync policy object in public runtime API,
- freeze tensor metadata contract (shape/stride/layout/dtype ownership rules),
- expose first profiler timeline API to Python.

Quality gates:

- no new public API without docs + example,
- per-op parity tests across Metal and CPU in CI,
- benchmark scripts publish CSV/JSON artifacts on every release tag.

Success metrics:

- <= 2% variance across repeated microbench runs on the same device (fixed seed/workload),
- zero known tensor lifetime bugs in open issue tracker for two consecutive releases,
- documented capability matrix published in README and docs site.

## 11.2 2026 Q3 (v0.2.x): Graph Execution Foundation

Planned scope:

- operator registry with schema validation and backend support flags,
- graph IR for dependency-aware scheduling,
- compile-time and runtime validation passes with human-readable diagnostics,
- graph/eager execution toggle for A/B verification.

Quality gates:

- graph mode must match eager numerical outputs within configured tolerance,
- every fused/optimized graph path must keep a deterministic fallback path,
- unsupported op diagnostics include actionable remediation message.

Success metrics:

- >= 25% reduction in host dispatch calls for representative pipeline benchmarks,
- >= 15% latency reduction on chained workloads dominated by launch overhead,
- graph mode adoption in all shipped benchmark pipelines.

## 11.3 2026 Q4 (v0.3.x): Fusion and Cost Model

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

## 11.5 2027 H2 (v0.5.x): Interoperability and Ecosystem Bridges

Planned scope:

- CoreML export for validated op subset,
- MLX tensor bridge and layout conversion utilities,
- PyTorch interop runner for hybrid execution experiments,
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
