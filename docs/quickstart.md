# Lightning Core Quickstart

This guide is for first-time users.

## 1) Install

From repository root:

```bash
python3 -m pip install .
python -c "import lightning_core; print(lightning_core.backend_name())"
```

From GitHub:

```bash
python3 -m pip install "git+https://github.com/wnsgus00114-droid/lightning-core.git@main"
```

## 2) Build once

```bash
cmake -S . -B build -DCJ_ENABLE_METAL=ON -DCJ_BUILD_TESTS=ON -DCJ_BUILD_PYTHON=ON -DCJ_BUILD_EXAMPLES=ON
cmake --build build -j
```

## 3) Run one C API sample

```bash
cmake --build build --target lightning_core_c_api_example -j
./build/lightning_core_c_api_example
```

## 4) Run tests

```bash
ctest --test-dir build --output-on-failure
```

## 5) Minimal C++ usage

```cpp
#include "lightning_core/lightning_core.hpp"

std::vector<std::int64_t> shape{8};
lightning_core::Tensor t(shape, lightning_core::Device::kMetal);
```

## 6) Minimal Python usage

```python
import numpy as np
import lightning_core as lc

print(lc.backend_name())
a = np.arange(8, dtype=np.float32)
b = np.arange(8, dtype=np.float32)
out = lc.vector_add(a, b, "metal")
print(out[:4])
```

## 7) Scope reminder

Lightning Core is currently an optimization-focused runtime prototype, not a full deep learning framework.

- Core focus: runtime, attention path, selected matrix/vector ops
- Model-family wrappers are advanced policy/fastpath helpers, not full model implementations
- Public API is active and still evolving

## 8) Next

For deeper tuning, benchmarking, model-family wrappers, and release workflows, see [docs/advanced.md](advanced.md).

## 9) End-to-End Runner (<=50 lines)

```python
import numpy as np
import lightning_core_integrated_api as lc_api

seed = 20260411
np.random.seed(seed)

# Inference: single API for eager/graph/interop
runner = lc_api.TinyTransformerRunner(
    seq_len=48,
    d_model=48,
    d_ff=128,
    vocab_size=256,
    seed=seed,
)
tokens = np.random.randint(0, 256, size=(48,), dtype=np.int64)
logits_graph, meta = runner.run(tokens, mode="graph", device="cpu", return_metadata=True)
logits_eager = runner.run(tokens, mode="eager", device="cpu")
print("graph fallback:", meta["fallback_reason_code"])
print("parity:", np.allclose(logits_graph, logits_eager, atol=1e-4, rtol=1e-4))

# Training: multi-step + grad clip + loss scale
train_model = lc_api.TinyAutogradMLP(8, 16, 4, seed=seed)
x = np.random.randn(32, 8).astype(np.float32)
y = np.random.randn(32, 4).astype(np.float32)
report = lc_api.autograd_train_loop(
    train_model,
    x,
    y,
    steps=5,
    lr=3e-2,
    grad_clip_norm=1.0,
    loss_scale=2.0,
)
print("losses:", report["losses"])
```
