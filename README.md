# Lightning Core

`Lightning Core` is a macOS-first CUDA-style runtime focused on custom attention training/inference paths.

## Identity and Naming

- Repository/package public name: `lightning-core` / `lightning_core`
- Public C++ include path/namespace: `lightning_core/*` and `lightning_core::...`
- Legacy compatibility shim remains available internally for older integrations

Public-facing docs and examples now use only `lightning_core` naming.

## Goals

- CUDA-like runtime API on macOS (Metal-first)
- macOS build support (Apple Silicon and Intel Mac)
- Attention-first custom runtime path (forward + train step)
- Device abstraction for CPU / Metal / CUDA
- Apple Silicon path: Metal-first execution with CPU exception fallback

## Non-goals (for now)

- Full binary compatibility with every CUDA API symbol
- Reproducing all PyTorch kernels

## Build Matrix

- macOS: Metal + CPU path (CUDA backend disabled)

`Lightning Core` now targets macOS only and uses Metal as the primary GPU path.

## Apple Silicon Strategy

- Inference default: Metal, then CUDA, then CPU
- Training default: Metal (for custom kernels/grad flow), then CUDA, then CPU
- Public API still uses CUDA-like device selection semantics

## Attention-Only Custom Path

- Custom forward: scaled dot-product attention
- Custom train step: forward + MSE gradient + SGD update on V
- CPU path includes SIMD dot-product optimization (ARM NEON)
- Metal path uses native custom kernels (scores -> softmax -> output -> grad/update)
- CPU path is kept as exception/fallback path when device execution is unavailable

## Current Scope (Important)

This project is currently an optimization-focused runtime prototype, not a full deep learning framework.

- Implemented core focus: runtime, attention path, and selected matrix/vector ops
- Model-family wrappers (Transformer/LSTM/CNN/GCN/GAT/VLM) are policy/fastpath helpers, not full model implementations
- Tensor API is intentionally minimal today (shape + host transfer + basic ops) and is still evolving
- Python API now includes runtime visibility + basic attention forward in addition to tensor helpers

If you need full framework-level features (autograd graph, broad op coverage, rich tensor views/layout APIs), treat this repository as experimental groundwork rather than a drop-in replacement.

## Staged Implementation Status

The following stage-1 upgrades are implemented:

- Tensor view/slice/layout metadata layer
- Python matmul/session/policy first-pass exposure
- Public API naming integration stage-2 wrappers (`lightning_core` include path expansion)

## Build

```bash
cmake -S . -B build -DCJ_ENABLE_METAL=ON -DCJ_BUILD_TESTS=ON -DCJ_BUILD_BENCHMARKS=ON -DCJ_BUILD_PYTHON=ON -DCJ_BUILD_EXAMPLES=ON
cmake --build build -j
```

On macOS, CUDA is disabled and Metal/CPU paths are used automatically.

## CUDA-Style C API (Lightning Core)

Use the new C header:

```c
#include <lightning_core/lightning_core.h>

void* dptr = NULL;
if (lcMalloc(&dptr, 1024) != LC_SUCCESS) {
	return;
}

float src[256] = {0.0f};
lcMemcpy(dptr, src, sizeof(src), LC_MEMCPY_HOST_TO_DEVICE);
lcDeviceSynchronize();
lcFree(dptr);
```

Core runtime calls:

- `lcMalloc` / `lcFree`
- `lcMemcpy`
- `lcDeviceSynchronize`
- `lcGetDeviceCount`
- `lcGetPreferredDeviceForInference` / `lcGetPreferredDeviceForTraining`
- `lcBackendName` / `lcGetErrorString`
- `lcGetMemoryModel` / `lcGetMemoryModelName`

Memory model note:

- `native-device`: backend provides true device memory semantics
- `host-managed-compat`: compatibility mode for device-like API surface (current macOS Metal runtime path)

C example source:

- `examples/lightning_core_c_api.c`

Compile/run example:

```bash
cmake --build build --target lightning_core_c_api_example -j
./build/lightning_core_c_api_example
```

## Python Packaging (pip)

This repository now supports local pip installation using scikit-build:

```bash
pip install .
```

Quick import check:

```bash
python -c "import lightning_core; print(lightning_core.backend_name())"
```

Python API highlights:

- `Tensor` / `Tensor64` now expose `shape`, `strides`, `rank`, `is_contiguous`, `dtype`, `reshape`
- `Tensor` / `Tensor64` also expose `view(...)` and `slice(axis, start, end)` returning tensor view metadata objects
- `Tensor` / `Tensor64` expose concrete view data access: `to_host_view(view)`, `slice_copy(...)`, `read_strided(...)`
- Runtime helpers: `backend_name`, `cuda_available`, `metal_available`, `memory_model_name`
- Attention helper: `attention_forward(q, k, v, seq_len, head_dim, causal=False, device='metal')`
- Matmul helpers: `matmul(...)`, `matmul_with_policy(...)`
- Policy/session helpers: `MatMulIoPolicy`, `MatrixElementwiseIoPolicy`, `VectorAddIoPolicy`, `AttentionIoPolicy`
- Session helpers: `MatMulMetalResidentSession`, `MatrixElemwiseMetalResidentSession`, `VectorAddMetalResidentSession`, `AttentionSession`

## Install for macOS Users (Terminal)

Production install (after PyPI publish is configured):

```bash
python3 -m pip install lightning-core
```

Immediate install from GitHub (works now, no PyPI needed):

```bash
python3 -m pip install "git+https://github.com/wnsgus00114-droid/lightning-core.git@main"
```

Install a specific release tag:

```bash
python3 -m pip install "git+https://github.com/wnsgus00114-droid/lightning-core.git@v0.1.1"
```

## GitHub Actions: Wheel Build and Publish

Workflow file:

- `.github/workflows/python-wheel-publish.yml`

Behavior:

- `push`/`pull_request`: build macOS wheel and sdist, then upload `dist/*` artifacts.
- `v*` tag push: run build job and publish artifacts to PyPI.
- `workflow_dispatch`: optional manual publish target (`none`, `testpypi`, `pypi`).

Release example:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Manual TestPyPI dry-run:

1. Run GitHub Actions workflow `Build and Publish Python Wheel`.
2. Select `publish_target=testpypi`.
3. After publish, verify install:

```bash
python -m pip install -i https://test.pypi.org/simple/ lightning-core
```

Note:

- PyPI publishing in this workflow uses GitHub OIDC trusted publishing. Configure your PyPI project to trust this repository/workflow.

Trusted Publishing setup checklist (PyPI):

1. Create the `lightning-core` project on PyPI (once).
2. In PyPI project settings, open `Publishing` -> `Add a new pending publisher`.
3. Set:
	- Owner: your GitHub org/user
	- Repository name: your repo name
	- Workflow name: `python-wheel-publish.yml`
	- Environment name: `pypi`
4. Save, then push a version tag (`v*`) to trigger publish.

Trusted Publishing setup checklist (TestPyPI):

1. Create the `lightning-core` project on TestPyPI (once).
2. In TestPyPI project settings, open `Publishing` -> `Add a new pending publisher`.
3. Set:
	- Owner: your GitHub org/user
	- Repository name: your repo name
	- Workflow name: `python-wheel-publish.yml`
	- Environment name: `testpypi`
4. Save, then run workflow_dispatch with `publish_target=testpypi`.

GitHub repository recommendation:

- Add protection rules/reviewers for the `pypi` and `testpypi` environments before first publish.

## C++ Namespace Migration

To use the new namespace without rewriting existing implementation internals, include Lightning Core wrappers:

- `include/lightning_core/runtime.hpp`
- `include/lightning_core/tensor.hpp`
- `include/lightning_core/ops.hpp`
- `include/lightning_core/attention.hpp`
- `include/lightning_core/model_customization.hpp`

Example:

```cpp
#include "lightning_core/tensor.hpp"
lightning_core::Tensor t(shape, lightning_core::Device::kMetal);
```

Migration status:

- Step 1 (done): public wrapper headers with `lightning_core` alias
- Step 2 (done): expanded wrapper coverage for model headers and umbrella include
- Step 3 (done): public docs/examples now use only `lightning_core` naming
- Compatibility shim remains for legacy code paths

Detailed technical roadmap is tracked in `ROADMAP.md`.

Stage-2 naming integration status:

- Added lightning_core model wrapper headers under `include/lightning_core/models/*`
- Added umbrella header `include/lightning_core/lightning_core.hpp`
- Existing legacy namespace remains supported for compatibility

Ops modularization status:

- `ops.hpp` is being split incrementally.
- Policy helpers are now in `include/lightning_core/ops/policy.hpp` and re-exported via `ops.hpp`.

## Test

```bash
ctest --test-dir build --output-on-failure
```

## Benchmark

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

Attention benchmark parameters:

```bash
export CJ_ATTN_SEQ=512
export CJ_ATTN_DIM=64
export CJ_ATTN_ITERS=20
./build/benchmarks/bench_attention
```

Attention shape sweep + split metrics:

```bash
export CJ_ATTN_SWEEP=1
export CJ_ATTN_WARMUP=4
export CJ_ATTN_ITERS=10
export CJ_ATTN_BATCH=2
./build/benchmarks/bench_attention
```

Generated CSV:

- `build/benchmarks/attention_shape_sweep.csv`

Vector add crossover sweep:

```bash
export CJ_BENCH_SWEEP=1
./build/benchmarks/bench_vector_add
```

Generated files:

- `build/benchmarks/vector_add_crossover.csv`
- `build/benchmarks/vector_add_crossover_hint.env`

Metric semantics:

- `*_e2e_ms`: end-to-end latency including host transfer/sync.
- `*_enqueue_only_ms`: host-side command enqueue time (not a full GPU completion latency).

You can apply the measured one-shot crossover to runtime policy:

```bash
source build/benchmarks/vector_add_crossover_hint.env
./build/benchmarks/bench_vector_add
```

If you must change crossover without process restart, enable dynamic refresh:

```bash
export CJ_VECTORADD_CPU_CROSSOVER_DYNAMIC=1
export CJ_VECTORADD_CPU_CROSSOVER_N=131072
```

For CoreML inference benchmark, provide model path:

```bash
export CJ_COREML_MODEL=/absolute/path/to/model.mlmodelc
./build/benchmarks/bench_vector_add
```

The benchmark now prints:

- custom backend timings (CPU/Metal)
- MPSGraph add timing (existing MPS baseline)
- MPSGraph train-step timing
- CoreML inference timing (if model path is provided)

## Basic Matrix Ops On Metal

- `ops::matMul<T>(A, B, C, M, K, N, Device::kMetal)` is available for float32.
- `A` shape: `M x K`, `B` shape: `K x N`, `C` shape: `M x N`.
- If Metal path is unavailable, runtime falls back to CPU.
- `ops::matrixSub<T>(A, B, C, rows, cols, Device::kMetal)` and `ops::matrixDiv<T>(...)` are available for float32.

Recommended performance policy on macOS (Metal resident flow):

1. Start step: upload once, no download, no sync.
2. Run step: no upload, no download, no sync.
3. Finish step: no upload, download + sync once.

Use policy helpers:

- `ops::makeMetalResidentStartPolicy()`
- `ops::makeMetalResidentRunPolicy()`
- `ops::makeMetalResidentFinishPolicy()`
- `ops::makeMetalElemwiseResidentStartPolicy()`
- `ops::makeMetalElemwiseResidentRunPolicy()`
- `ops::makeMetalElemwiseResidentFinishPolicy()`

High-level resident wrappers (no manual policy setup):

```cpp
// matmul resident flow on Metal
ops::matMulMetalResidentStart<float>(a, b, c, m, k, n);
for (int i = 0; i < steps; ++i) {
	ops::matMulMetalResidentRun<float>(a, b, c, m, k, n);
}
ops::matMulMetalResidentFinish<float>(a, b, c, m, k, n);

// matrix elementwise resident flow on Metal
ops::matrixSubMetalResidentStart<float>(a, b, out, rows, cols);
ops::matrixSubMetalResidentRun<float>(a, b, out, rows, cols);
ops::matrixSubMetalResidentFinish<float>(a, b, out, rows, cols);
```

Resident session wrappers (recommended in production loops):

```cpp
ops::MatMulMetalResidentSession<float> mm(m, k, n);
mm.start(a, b, c);
for (int i = 0; i < steps; ++i) {
	mm.run(a, b, c);
}
mm.finish(a, b, c);

ops::MatrixElemwiseMetalResidentSession<float> me(rows, cols);
me.subStart(a, b, out);
me.subRun(a, b, out);
me.subFinish(a, b, out);

ops::VectorAddMetalResidentSession<float> va(n);
va.start(a, b, out);
va.run(a, b, out);
va.finish(a, b, out);
```

Attention forward postprocess (fused in Metal forward command buffer):

```cpp
lightning_core::AttentionIoPolicy p;
p.output_scale = 0.5f;
p.output_bias = 0.1f;
p.output_relu = true;
lightning_core::attentionForwardWithPolicy(q, k, v, out, cfg, lightning_core::runtime::Device::kMetal, p);
```

Model-family customization presets (Transformer/LSTM/RNN/DNN/CNN/GCN/GAT/VLM):

```cpp
#include "lightning_core/model_customization.hpp"

using namespace lightning_core;

// 1) pick an aggressive preset for your model family
auto custom = makeAggressiveCustomization(
	ModelFamily::kTransformer,
	ExecutionMode::kTraining,
	/*seq_len=*/2048,
	/*hidden_dim=*/128);

// 2) apply loop-stage policies to existing high-performance paths
auto mm_start = ops::makeMetalResidentStartPolicy();
auto mm_run = makeMatMulPolicyForLoop(custom, LoopStage::kRun);
auto mm_finish = makeMatMulPolicyForLoop(custom, LoopStage::kFinish);

auto attn_run = makeAttentionPolicyForLoop(custom, LoopStage::kRun);
auto attn_finish = makeAttentionPolicyForLoop(custom, LoopStage::kFinish);

// 3) optional routing hint for vector one-shot workloads
// export CJ_VECTORADD_CPU_CROSSOVER_N=<custom.vector_oneshot_crossover_n>
```

Preset intent:

- Transformer/VLM: long-sequence attention + matmul loops (resident + enqueue-oriented).
- LSTM/RNN: recurrent small-step loops (smaller vector crossover, resident loop policy).
- DNN/CNN: dense matmul/elemwise loops (resident scheduling).
- GCN/GAT: graph mini-batch loops (resident scheduling + medium crossover).

Transformer fast-path wrapper:

```cpp
#include "lightning_core/models/transformer_fastpath.hpp"

lightning_core::models::TransformerFastPathConfig cfg;
cfg.seq_len = 1024;
cfg.head_dim = 64;
cfg.causal = true;
cfg.training = true;

lightning_core::models::TransformerFastPath fast(cfg, lightning_core::runtime::Device::kMetal);

// attention loop stage example
fast.attentionForward(q, k, v, out, lightning_core::LoopStage::kStart);
for (int step = 0; step < steps; ++step) {
	fast.attentionForward(q, k, v, out, lightning_core::LoopStage::kRun);
}
fast.attentionForward(q, k, v, out, lightning_core::LoopStage::kFinish);
```

LSTM/RNN fast-path wrapper:

```cpp
#include "lightning_core/models/lstm_rnn_fastpath.hpp"

lightning_core::models::LstmRnnFastPathConfig rnn_cfg;
rnn_cfg.input_dim = 512;
rnn_cfg.hidden_dim = 512;
rnn_cfg.training = true;
rnn_cfg.lstm_mode = true;

lightning_core::models::LstmRnnFastPath rnn(rnn_cfg, lightning_core::runtime::Device::kMetal);
rnn.projectInput(x, w_x, h_x, batch, lightning_core::LoopStage::kRun);
rnn.projectHidden(h, w_h, h_h, batch, lightning_core::LoopStage::kRun);
rnn.fuseRecurrent(h_x, h_h, h_out, batch * rnn_cfg.hidden_dim, lightning_core::LoopStage::kRun);
```

DNN/CNN fast-path wrapper:

```cpp
#include "lightning_core/models/dnn_cnn_fastpath.hpp"

lightning_core::models::DnnCnnFastPathConfig cnn_cfg;
cnn_cfg.in_dim = 1024;
cnn_cfg.out_dim = 512;
cnn_cfg.training = true;
cnn_cfg.cnn_mode = true;

lightning_core::models::DnnCnnFastPath cnn(cnn_cfg, lightning_core::runtime::Device::kMetal);
cnn.denseProject(x, w, proj, batch, lightning_core::LoopStage::kRun);
cnn.residualSub(proj, skip, sub, batch, cnn_cfg.out_dim, lightning_core::LoopStage::kRun);
cnn.channelNormDiv(sub, norm, out, batch, cnn_cfg.out_dim, lightning_core::LoopStage::kRun);
```

GCN/GAT sparse-friendly policy layer:

```cpp
#include "lightning_core/models/graph_fastpath.hpp"

auto gcn = lightning_core::models::makeGraphSparseFriendlyPolicy(
	lightning_core::ModelFamily::kGcn,
	lightning_core::ExecutionMode::kTraining,
	/*num_nodes=*/131072,
	/*num_edges=*/1048576,
	/*feature_dim=*/128);

auto proj_run = lightning_core::models::makeGraphProjectionPolicy(gcn, lightning_core::LoopStage::kRun);
auto update_run = lightning_core::models::makeGraphUpdatePolicy(gcn, lightning_core::LoopStage::kRun);
```

VLM fast-path wrapper (vision projection + text projection + fusion attention):

```cpp
#include "lightning_core/models/vlm_fastpath.hpp"

lightning_core::models::VlmFastPathConfig vcfg;
vcfg.image_tokens = 256;
vcfg.text_tokens = 128;
vcfg.vision_dim = 1024;
vcfg.text_dim = 768;
vcfg.fused_dim = 64;
vcfg.training = true;

lightning_core::models::VlmFastPath vlm(vcfg, lightning_core::runtime::Device::kMetal);
vlm.projectVision(image_tokens, w_vision, image_proj, lightning_core::LoopStage::kRun);
vlm.projectText(text_tokens, w_text, text_proj, lightning_core::LoopStage::kRun);
vlm.runCrossAttentionFast(image_proj, text_proj, out_text, lightning_core::LoopStage::kRun);
vlm.runFusionAttention(q, k, v, out, lightning_core::LoopStage::kRun);
```

VLM patch embedding integration:

```cpp
// image_nhwc: [H, W, C], patchified internally then projected.
vlm.patchEmbedFromImage(image_nhwc, image_h, image_w, w_patch, image_proj, lightning_core::LoopStage::kRun);
```

Graph policy cache persistence:

```bash
export CJ_GRAPH_POLICY_CACHE_FILE=build/benchmarks/graph_policy_cache.csv
```

Model benchmark sweeps:

```bash
export CJ_TR_SWEEP=1
./build/benchmarks/bench_transformer

export CJ_LSTM_SWEEP=1
./build/benchmarks/bench_lstm_rnn

export CJ_CNN_SWEEP=1
./build/benchmarks/bench_cnn_dnn

export CJ_VLM_SWEEP=1
./build/benchmarks/bench_vlm
```

Optional VLM tuning env:

```bash
export CJ_VLM_PATCH_SIZE=16
export CJ_VLM_IMAGE_CHANNELS=3
export CJ_VLM_ONLINE_RETUNE_EVERY=64
./build/benchmarks/bench_vlm
```

Generated CSV:

- `build/benchmarks/transformer_shape_sweep.csv`
- `build/benchmarks/lstm_rnn_shape_sweep.csv`
- `build/benchmarks/cnn_dnn_shape_sweep.csv`
- `build/benchmarks/vlm_shape_sweep.csv`

Generated profile env hints:

- `build/benchmarks/transformer_runtime_profile.env`
- `build/benchmarks/lstm_rnn_runtime_profile.env`
- `build/benchmarks/cnn_dnn_runtime_profile.env`
- `build/benchmarks/vlm_runtime_profile.env`

Generate one merged runtime profile env:

```bash
bash benchmarks/generate_model_profile_env.sh
source build/benchmarks/model_runtime_profile.env
```

Runtime autoload of profile env:

- On first runtime usage, `lightning_core::runtime` tries to load tuning vars from:
	- `CJ_RUNTIME_PROFILE_ENV_FILE` (if set), or
	- `build/benchmarks/model_runtime_profile.env`, or
	- `../build/benchmarks/model_runtime_profile.env`
- Existing environment variables are never overwritten.

Disable autoload:

```bash
export CJ_RUNTIME_PROFILE_AUTOLOAD=0
```

Use a custom profile file:

```bash
export CJ_RUNTIME_PROFILE_ENV_FILE=/absolute/path/to/model_runtime_profile.env
```

Shape sweep automation for matrix ops:

```bash
./benchmarks/sweep_matrix_ops.sh
```

Generated CSV:

- `build/benchmarks/matrix_ops_sweep.csv`

## Python Binding

The python module target is built as `lightning_core` via pybind11 if pybind11 is available.

## Project Layout

- `include/lightning_core`: public headers
- legacy compatibility headers/shims are preserved for existing integrations
- `src`: runtime + tensor + ops implementation
- `tests`: C++ unit tests
- `benchmarks`: benchmarking binaries
- `python`: pybind11 module build
