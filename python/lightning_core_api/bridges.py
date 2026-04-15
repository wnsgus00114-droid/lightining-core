"""Bridge and federation helpers extracted from lightning_core_integrated_api.

This module intentionally re-exports functions from the legacy monolithic module
so we can split implementation incrementally without breaking existing imports.
"""

from __future__ import annotations

from lightning_core_integrated_api import (  # noqa: F401
    coreml_roundtrip_beta_report,
    coreml_roundtrip_schema,
    create_coreml_model_runner_adapter,
    create_mlx_model_runner_adapter,
    create_tf_keras_layer_wrapper,
    create_tf_model_runner_adapter,
    create_torch_module_wrapper,
    engine_federation_policy_schema,
    engine_federation_policy_v3_schema,
    get_coreml_wrapper_telemetry,
    get_mlx_wrapper_telemetry,
    get_tf_wrapper_telemetry,
    get_torch_wrapper_telemetry,
    import_export_compatibility_matrix,
    resolve_engine_federation_policy_v3,
)
