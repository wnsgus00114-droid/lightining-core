# Lightning Core

[![PyPI version](https://img.shields.io/pypi/v/lightning-core.svg)](https://pypi.org/project/lightning-core/)
[![PyPI downloads](https://img.shields.io/pypi/dm/lightning-core.svg)](https://pypi.org/project/lightning-core/)

Lightning Core is a macOS-first CUDA-style runtime focused on custom attention training/inference paths.

## Why This Project Exists

Deep learning tooling is often CUDA-centric, while many local development environments are macOS + Apple Silicon.
This project was created to provide a practical, CUDA-style runtime experience on macOS with a Metal backend,
so custom attention and tensor/ops experiments can be built, profiled, and tuned without changing the entire workflow.

In short, Lightning Core targets the gap between:

- familiar CUDA-like execution mental model
- native Metal execution path on macOS
- fast iteration loop with both C++ and Python APIs

## Why Lightning Core

| Point | What it gives you |
| --- | --- |
| Metal-first runtime path | Optimized execution flow on Apple Silicon/macOS |
| Attention fastpath focus | Practical tuning path for custom attention workloads |
| Resident session policy | Reduced upload/download/sync overhead in repeated runs |
| C++ + Python workflow | Native core with pybind11 module for quick iteration |

## Core Architecture

```mermaid
flowchart LR
	A[Python API: lightning_core] --> B[pybind11 bindings]
	C[C++ API: lightning_core/*] --> D[Runtime Core]
	B --> D
	D --> E[Tensor and Ops]
	E --> F[Metal Backend]
	E --> G[CPU Fallback]
	D --> H[Benchmark and Profile Artifacts]
```

## Performance Checkpoints

| Checkpoint | Command | Artifact |
| --- | --- | --- |
| Attention shape sweep | `./build/benchmarks/bench_attention` with sweep env vars | `build/benchmarks/attention_shape_sweep.csv` |
| Vector add crossover | `./build/benchmarks/bench_vector_add` with sweep mode | `build/benchmarks/vector_add_crossover.csv` |
| Matrix ops sweep | `./benchmarks/sweep_matrix_ops.sh` | `build/benchmarks/matrix_ops_sweep.csv` |

See [docs/advanced.md](docs/advanced.md) for full benchmark and tuning flow.

## Latest Benchmark Snapshot (2026-03-29, local Apple Silicon)

Torch(MPS) vs Lightning Core direct operator comparison:

| Bench | Shape | Lightning Core ms | Torch MPS ms | Speedup (Torch/LCore) |
| --- | --- | ---: | ---: | ---: |
| vector_add | n=4096 | 0.0008 | 0.3401 | 421.89x |
| vector_add | n=16384 | 0.0031 | 0.3042 | 99.45x |
| vector_add | n=65536 | 0.0083 | 0.1926 | 23.33x |
| vector_add | n=262144 | 0.0285 | 0.2101 | 7.38x |
| vector_add | n=1048576 | 0.1087 | 0.2559 | 2.35x |
| matmul | m=256,k=256,n=256 | 0.2373 | 0.1943 | 0.82x |
| matmul | m=512,k=512,n=512 | 0.2957 | 0.2617 | 0.89x |
| matmul | m=1024,k=1024,n=1024 | 0.9845 | 1.2519 | 1.27x |
| matmul | m=2048,k=2048,n=2048 | 5.7157 | 4.7575 | 0.83x |

Lightning Core native benchmark sweep highlights (from build artifacts):

- Attention best speedup: 14.35x at seq=2048, dim=64
- Transformer best speedup: 15.32x at seq=1024, dim=64
- Matrix sub best speedup: 3.45x
- Matrix div best speedup: 3.81x
- Vector add crossover (metal recommended from): n=65536

```mermaid
xychart-beta
	title "Lightning Core Native Sweep Speedups"
	x-axis ["Attention", "Transformer", "MatSub", "MatDiv"]
	y-axis "Speedup (CPU/Metal)" 0 --> 20
	bar [14.35, 15.32, 3.45, 3.81]
```

Why use Lightning Core:

- Strong low-latency vector path and custom-op runtime control on Apple Silicon.
- Dense matmul competitiveness improved with a 1024-shape crossover win vs Torch MPS in this run.
- C++ benchmark/tuning pipeline with concrete artifact outputs.
- Easy hybrid usage with higher-level frameworks when dense GEMM-heavy paths are preferable there.

Raw artifacts:

- benchmarks/reports/2026-03-29/torch_mps_vs_lightning_core.csv
- benchmarks/reports/2026-03-29/torch_mps_vs_lightning_core.json
- benchmarks/reports/2026-03-29/lightning_core_sweep_summary.json

## Install and Use

Install from PyPI:

```bash
python -m pip install -U lightning-core
```

Install from source:

```bash
git clone https://github.com/wnsgus00114-droid/lightning-core.git
cd lightning-core
python -m pip install .
```

Quick Python usage:

```python
import numpy as np
import lightning_core as lc

print(lc.backend_name())
a = np.arange(8, dtype=np.float32)
b = np.arange(8, dtype=np.float32)
out = lc.vector_add(a, b, "metal")
print(out)
```

## Quick Start (Beginner)

Documentation entrypoint:

- docs/index.md

Use this path first:

1. Install and import-check
2. Build and run one C API example
3. Run tests

```bash
python3 -m pip install .
python -c "import lightning_core; print(lightning_core.backend_name())"

cmake -S . -B build -DCJ_ENABLE_METAL=ON -DCJ_BUILD_TESTS=ON -DCJ_BUILD_PYTHON=ON -DCJ_BUILD_EXAMPLES=ON
cmake --build build -j

cmake --build build --target lightning_core_c_api_example -j
./build/lightning_core_c_api_example

ctest --test-dir build --output-on-failure
```

Detailed beginner guide:

- docs/quickstart.md

## Scope (Current)

This project is an optimization-focused runtime prototype, not a full deep learning framework.

- Core focus: runtime, attention path, selected matrix/vector ops
- Model-family wrappers are advanced policy/fastpath helpers, not full model implementations
- API and internals are still actively evolving

## Identity and Naming

- Public package/module: lightning-core / lightning_core
- Public C++ include path/namespace: lightning_core/* and lightning_core::...
- Internal canonical headers: include/lightning_core/core/*
- Legacy include/cudajun/* remains as compatibility shim

## Advanced Topics

For advanced usage and operations, see:

- docs/advanced.md

For contributor workflow and coding conventions, see:

- docs/contributor.md

Includes:

- benchmark sweeps and generated artifacts
- resident session and policy tuning
- model-family wrapper examples and caveats
- runtime profile/env tuning
- release and publishing workflow notes
- repository rename transition operations

## Build Targets

Useful targets:

- library: lightning_core::lightning_core
- python module: lightning_core
- c api example: lightning_core_c_api_example

## Repository

Current GitHub repository URL:

- https://github.com/wnsgus00114-droid/lightning-core

If you still have an older local clone URL, use:

```bash
./scripts/sync_remote_after_repo_rename.sh --dry-run
./scripts/sync_remote_after_repo_rename.sh
```

The script automatically checks target repository availability and skips safely when rename is not ready.

## Project Layout

- include/lightning_core: public wrappers
- include/lightning_core/core: canonical internal headers
- include/cudajun: compatibility shims for legacy integrations
- src: runtime + tensor + ops implementation
- tests: C++ unit tests
- benchmarks: benchmark binaries and sweep scripts
- python: pybind11 bindings
- docs: split docs (index/quickstart/advanced/contributor)

## Author

- JunHyeon Beak (Kwangwoon University)

## License

This project is licensed under the Kwangwoon University License 1.0 (KWU-1.0).

See [LICENSE](LICENSE).
