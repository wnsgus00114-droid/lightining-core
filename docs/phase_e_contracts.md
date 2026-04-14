# Phase E Engine Federation Contracts

This page is generated from `docs/engine_federation_contract.json`.

<!-- AUTO-PHASE-E-CONTRACT:BEGIN -->

### Phase E Engine Federation Contract (Auto-generated)

- Contract version: `phase_e_v0.5.x_lock`
- As-of date: `2026-04-14`
- Source of truth: `docs/engine_federation_contract.json`
- CI sync checker: `python scripts/check_phase_e_contract_sync.py`

#### Engine Values

| Engine |
| --- |
| lightning |
| torch |
| tf |
| coreml |
| mlx |
| auto |

#### Route Policy Keys

| Key |
| --- |
| conv |
| attention |
| graph |

#### Bridge Reason Codes

| Torch |
| --- |
| none |
| interop_torch_tensor_boundary_copy |

| TensorFlow |
| --- |
| none |
| tf_runtime_unavailable |
| tf_tensor_boundary_copy |
| tf_runner_graph_policy_forced_eager |
| tf_runner_graph_execute_failed |
| tf_runner_interop_policy_forced_lightning |
| tf_runner_unknown_fallback |

| CoreML |
| --- |
| none |
| coreml_runtime_unavailable |
| coreml_model_path_missing |
| coreml_inference_failed |
| coreml_tensor_boundary_copy |
| coreml_runner_graph_policy_forced_eager |
| coreml_runner_unknown_fallback |

| MLX |
| --- |
| none |
| mlx_runtime_unavailable |
| mlx_tensor_boundary_copy |
| mlx_bridge_execute_failed |
| mlx_runner_graph_policy_forced_eager |
| mlx_runner_unknown_fallback |

#### CI Constants (Phase E Exit Audit)

| Key | Value |
| --- | --- |
| max_boundary_overhead_ms | 6.0 |
| min_coreml_reason_coverage_pct | 100.0 |
| min_federation_reason_coverage_pct | 100.0 |
| min_mlx_reason_coverage_pct | 100.0 |
| min_perf_explain_coverage_pct | 100.0 |
| min_tf_reason_coverage_pct | 100.0 |
| min_torch_reason_coverage_pct | 100.0 |
| require_import_export_matrix_sync | True |
| require_roundtrip_artifacts | True |

<!-- AUTO-PHASE-E-CONTRACT:END -->
