# Lightning Core API Index

This page is the quick index for public API surfaces.

## Python API (PyPI package)

Primary import:

```python
import lightning_core as lc
```

Main Python surfaces:

- Runtime helpers: `backend_name`, `runtime_capability`, `runtime_trace_*`, `runtime_sync_*`
- Runtime timeline helper: `runtime_trace_timeline` (sorting/grouping/hotspot summary in Python)
- Tensor ops: `vector_add`, `matrix_add`, `matrix_sub`, `matmul2d`, `conv2d_nchw`
- Attention: `attention2d`, `attention_forward`, `AttentionSession`
- Integrated API namespace: `lc.api.*` (clean operation names for conv/attention/integrated paths)

Related binding sources:

- [python/bindings/pybind_module.cpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/python/bindings/pybind_module.cpp)
- [python/bindings/bind_runtime.cpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/python/bindings/bind_runtime.cpp)
- [python/bindings/bind_ops.cpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/python/bindings/bind_ops.cpp)
- [python/bindings/bind_attention.cpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/python/bindings/bind_attention.cpp)
- [python/bindings/bind_integrated.cpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/python/bindings/bind_integrated.cpp)

## C++ Public Headers

Public wrapper headers:

- [include/lightning_core/lightning_core.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.hpp)
- [include/lightning_core/runtime.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/runtime.hpp)
- [include/lightning_core/attention.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/attention.hpp)
- [include/lightning_core/ops.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/ops.hpp)

Canonical core headers:

- [include/lightning_core/core/runtime.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/runtime.hpp)
- [include/lightning_core/core/ops.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/ops.hpp)
- [include/lightning_core/core/attention.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/attention.hpp)
- [include/lightning_core/core/graph.hpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/core/graph.hpp)

## C API

Public C API header:

- [include/lightning_core/lightning_core.h](https://github.com/wnsgus00114-droid/lightning-core/blob/main/include/lightning_core/lightning_core.h)

Implementation:

- [src/lightning_core_c_api.cpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/src/lightning_core_c_api.cpp)

## Notes

- This is an index-style page for quick navigation.
- Generated API reference (Sphinx/Breathe style) is planned in roadmap and not yet the default documentation path.
