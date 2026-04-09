# Python API Reference (Generated)

Generated from pybind binding sources in `python/bindings/`.

| Metric | Count |
| --- | --- |
| Binding files | 6 |
| Submodules | 1 |
| Module functions | 59 |
| Classes | 7 |

Regenerate:
`python scripts/generate_api_reference_docs.py`

## `bind_attention.cpp`

Source: [python/bindings/bind_attention.cpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/python/bindings/bind_attention.cpp)

Module-level functions:
- `attention2d`
- `attention2d_into`
- `attention2d_session`
- `attention_forward`
- `attention_forward_with_policy`
- `clear_attention2d_session_cache`

Classes:
- `AttentionIoPolicy`
- `AttentionSession`
  - methods: `forward`, `forward_into`, `forward_with_policy`, `forward_with_policy_into`, `set_default_policy`, `train_step`, `train_step_with_policy`

## `bind_graph.cpp`

Source: [python/bindings/bind_graph.cpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/python/bindings/bind_graph.cpp)

Module-level functions:
- `graph_registry_schema`
- `graph_registry_schemas`
- `graph_registry_size`
- `graph_validation_passes`

Classes:
- `GraphIR`
  - methods: `add_node`, `add_tensor`, `clear_plan_cache`, `execute_f32`, `execute_f64`, `fusion_report`, `nodes`, `num_nodes`, `num_tensors`, `plan`, `plan_cache_stats`, `plan_groups`, `plan_summary`, `tensors`, `validate`, `validate_report`

## `bind_integrated.cpp`

Source: [python/bindings/bind_integrated.cpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/python/bindings/bind_integrated.cpp)

Submodules:
- `api`

Module-level functions:
- `attention`
- `attention_into`
- `clear_attention_session_cache`
- `clear_integrated_attention_session_cache`
- `conv_attention_torchstrong_nchw`
- `conv_attention_torchstrong_nchw_ab_report`
- `conv_attention_torchstrong_nchw_into`
- `conv_relu_nchw`
- `conv_relu_nchw_into`
- `lightning_attention`
- `lightning_attention_into`
- `lightning_conv_attention_torchstrong_nchw`
- `lightning_conv_attention_torchstrong_nchw_ab_report`
- `lightning_conv_attention_torchstrong_nchw_into`
- `lightning_conv_relu_nchw`
- `lightning_conv_relu_nchw_into`

## `bind_ops.cpp`

Source: [python/bindings/bind_ops.cpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/python/bindings/bind_ops.cpp)

Module-level functions:
- `conv2d_nchw`
- `conv2d_nchw_into`
- `conv2d_nchw_metal_resident`
- `matmul`
- `matmul2d`
- `matmul2d_into`
- `matmul2d_resident_session`
- `matmul_np`
- `matmul_np_into`
- `matmul_np_into_with_policy`
- `matmul_np_with_policy`
- `matmul_reset_tuning`
- `matmul_with_policy`
- `matrix_sub`
- `vector_add`

Classes:
- `Conv2dMetalResidentSession`
  - methods: `add`, `div`, `div_finish`, `div_finish_into`, `div_run`, `div_run_into`, `div_start`, `div_start_into`, `finish`, `finish_into`, `run`, `run_batch_sync_into`, `run_into`, `start`, `start_into`, `sub`, `sub_finish`, `sub_finish_into`, `sub_run`, `sub_run_into`, `sub_start`, `sub_start_into`, `sync_into`
- `MatMulIoPolicy`
- `MatrixElementwiseIoPolicy`
- `VectorAddIoPolicy`
  - methods: `finish`, `finish_into`, `run`, `run_batch_sync`, `run_batch_sync_cached_no_download`, `run_batch_sync_into`, `run_batch_sync_no_download_into`, `run_into`, `start`, `start_into`, `sync`, `sync_into`

## `bind_runtime.cpp`

Source: [python/bindings/bind_runtime.cpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/python/bindings/bind_runtime.cpp)

Module-level functions:
- `backend_name`
- `cuda_available`
- `memory_model_name`
- `metal_available`
- `runtime_active_backend_capabilities`
- `runtime_active_backend_interfaces`
- `runtime_backend_capabilities`
- `runtime_backend_interfaces`
- `runtime_sync_apply`
- `runtime_sync_apply_default`
- `runtime_sync_policy_get`
- `runtime_sync_policy_set`
- `runtime_trace_capacity`
- `runtime_trace_clear`
- `runtime_trace_enable`
- `runtime_trace_enabled`
- `runtime_trace_events`
- `runtime_trace_timeline`

## `bind_tensor.cpp`

Source: [python/bindings/bind_tensor.cpp](https://github.com/wnsgus00114-droid/lightning-core/blob/main/python/bindings/bind_tensor.cpp)
