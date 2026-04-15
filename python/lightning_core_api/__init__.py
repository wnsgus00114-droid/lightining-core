"""Modular compatibility facade for lightning_core_integrated_api.

This package is the first extraction layer for the large integrated API module.
Public behavior stays backward-compatible while giving users stable submodule imports.
"""

from .bridges import (  # noqa: F401
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
from .checkpoint import (  # noqa: F401
    checkpoint_compatibility_matrix,
    checkpoint_conversion_diagnostics,
    load_checkpoint,
    load_model_checkpoint,
    load_runner_checkpoint,
    runner_checkpoint_compatibility_matrix,
    save_checkpoint,
    save_model_checkpoint,
    save_runner_checkpoint_v2,
    validate_checkpoint,
    validate_checkpoint_conversion,
)
