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
python3 -m pip install "git+https://github.com/wnsgus00114-droid/lightining-core.git@main"
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

For deeper tuning, benchmarking, model-family wrappers, and release workflows, see docs/advanced.md.
