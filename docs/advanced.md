# Lightning Core Advanced Guide

This guide contains advanced/optional topics that are not required for first-time setup.

## Advanced Build Matrix

- macOS is the primary target (Metal + CPU fallback)
- CUDA is currently disabled by default in this repository flow

## Python Packaging and Release

### Wheel publish workflow

Workflow file:

- [.github/workflows/python-wheel-publish.yml](../.github/workflows/python-wheel-publish.yml)

Behavior:

- push/pull_request: build macOS wheel and sdist artifacts
- v* tag: publish to PyPI (release only)
- workflow_dispatch: manual publish target (none/testpypi)

Safety rules in workflow:

- PyPI publish is tag-only to avoid accidental release from main
- twine check validates built distributions before publish
- skip-existing avoids hard failures on rerun with already-uploaded files

### Trusted publishing checklist

PyPI/TestPyPI trusted publishing must be configured before release publishing.

### Post-release quick check

- PyPI page: confirm new version appears on https://pypi.org/project/lightning-core/
- Install check: `python -m pip install -U lightning-core && python -c "import lightning_core; print(lightning_core.backend_name())"`
- Tag check: `git fetch --tags && git tag -l | grep '^v' | tail -n 5`

### Rename migration note

Repository rename is complete. Current live URL is:

- https://github.com/wnsgus00114-droid/lightning-core

Helper script:

```bash
./scripts/sync_remote_after_repo_rename.sh --dry-run
./scripts/sync_remote_after_repo_rename.sh
```

The script checks target repo availability with git ls-remote and safely skips when rename is not ready.

## Docs Generation and Link Gate

The docs workflow now regenerates API references and validates markdown links before `mkdocs build`.

```bash
python3 scripts/generate_capability_docs.py
python3 scripts/generate_roadmap_history.py
python3 scripts/generate_api_reference_docs.py
python3 scripts/check_docs_links.py README.md docs
```

Generated API reference outputs:

- [docs/reference/python_api_generated.md](reference/python_api_generated.md)
- [docs/reference/cpp_api_generated.md](reference/cpp_api_generated.md)

## Benchmark Suite

Run all benchmark binaries:

```bash
./build/benchmarks/bench_vector_add
./build/benchmarks/bench_attention
./build/benchmarks/bench_matmul
./build/benchmarks/bench_matrix_ops
./build/benchmarks/bench_transformer
./build/benchmarks/bench_lstm_rnn
./build/benchmarks/bench_cnn_dnn
./build/benchmarks/bench_vlm
```

Graph/eager A/B benchmark (Python, with host-dispatch/fallback counters):

```bash
python benchmarks/python/graph_eager_ab_bench.py \
  --device auto \
  --warmup 6 \
  --iters 24 \
  --trace-iters 8 \
  --csv benchmarks/reports/ci/graph_eager_ab.csv \
  --json benchmarks/reports/ci/graph_eager_ab.json \
  --md benchmarks/reports/ci/graph_eager_ab.md
```

Engine split benchmark policy (pure-LC vs interop):

```python
import lightning_core_integrated_api as lc_api

lc_api.set_backend("lightning")  # pure-LC
# ... run integrated API benchmark rows for LC path

lc_api.set_backend("torch")      # interop
# ... run the same API rows for Torch bridge path
```

When publishing results, keep the two paths in separate sections/tables.

### Attention benchmark parameters

```bash
export CJ_ATTN_SEQ=512
export CJ_ATTN_DIM=64
export CJ_ATTN_ITERS=20
./build/benchmarks/bench_attention
```

### Attention sweep

```bash
export CJ_ATTN_SWEEP=1
export CJ_ATTN_WARMUP=4
export CJ_ATTN_ITERS=10
export CJ_ATTN_BATCH=2
./build/benchmarks/bench_attention
```

Generated CSV:

- [build/benchmarks/attention_shape_sweep.csv](../build/benchmarks/attention_shape_sweep.csv)

### Vector add crossover sweep

```bash
export CJ_BENCH_SWEEP=1
./build/benchmarks/bench_vector_add
```

Generated files:

- [build/benchmarks/vector_add_crossover.csv](../build/benchmarks/vector_add_crossover.csv)
- [build/benchmarks/vector_add_crossover_hint.env](../build/benchmarks/vector_add_crossover_hint.env)

Apply measured crossover:

```bash
source build/benchmarks/vector_add_crossover_hint.env
./build/benchmarks/bench_vector_add
```

### Matrix ops sweep

```bash
./benchmarks/sweep_matrix_ops.sh
```

Generated CSV:

- [build/benchmarks/matrix_ops_sweep.csv](../build/benchmarks/matrix_ops_sweep.csv)

## Resident Sessions and Policy APIs

Recommended resident flow on Metal:

- start: upload once (no download/sync)
- run: reuse resident buffers (no upload/download/sync)
- finish: download + sync once

Policy helpers:

- ops::makeMetalResidentStartPolicy
- ops::makeMetalResidentRunPolicy
- ops::makeMetalResidentFinishPolicy
- ops::makeMetalElemwiseResidentStartPolicy
- ops::makeMetalElemwiseResidentRunPolicy
- ops::makeMetalElemwiseResidentFinishPolicy

## Model-Family Wrapper Notes

Model-family examples (Transformer/LSTM/RNN/DNN/CNN/GCN/GAT/VLM) are advanced wrapper demonstrations.

They are not end-to-end framework model implementations.

## Runtime Profile and Tuning

Runtime profile autoload lookup order:

- CJ_RUNTIME_PROFILE_ENV_FILE (if set)
- [build/benchmarks/model_runtime_profile.env](../build/benchmarks/model_runtime_profile.env)
- [../build/benchmarks/model_runtime_profile.env](../build/benchmarks/model_runtime_profile.env)

Disable autoload:

```bash
export CJ_RUNTIME_PROFILE_AUTOLOAD=0
```

Set custom profile file:

```bash
export CJ_RUNTIME_PROFILE_ENV_FILE=/absolute/path/to/model_runtime_profile.env
```

Generate merged profile env:

```bash
bash benchmarks/generate_model_profile_env.sh
source build/benchmarks/model_runtime_profile.env
```

## Runtime Trace Timeline (Python)

You can profile runtime-level bottlenecks directly from Python.

```python
import lightning_core as lc
import numpy as np

lc.runtime_trace_clear()
lc.runtime_trace_enable(True)

a = np.random.rand(512, 512).astype(np.float32)
b = np.random.rand(512, 512).astype(np.float32)
for _ in range(20):
    lc.matmul2d(a, b, "metal")

lc.runtime_trace_enable(False)

report = lc.runtime_trace_timeline(
    event_sort_by="timestamp_ns",
    event_descending=False,
    group_by="op_path",
    group_sort_by="total_delta_next_ns",
    group_descending=True,
    hotspot_top_k=8,
)

print("window_ns:", report["window_ns"])
print("top groups:", report["groups"][:3])
print("hotspots:", report["hotspots"][:5])
```

Interpretation:

- `groups`: aggregated by op/path (`op|selected_device|direct_or_fallback`) to show where time is concentrated.
- `hotspots`: top single runtime events by `delta_next_ns` (time until next event).
- `events`: full timeline rows for manual inspection/export.

## Namespace/Compatibility

Canonical internal headers:

- [include/lightning_core/core/](../include/lightning_core/core/)

Compatibility headers:

- Legacy `include/cudajun/` forwarding headers have been removed. Use `include/lightning_core/`.

Public wrappers:

- [include/lightning_core/](../include/lightning_core/)

This keeps the public surface consistent around `lightning_core`.
