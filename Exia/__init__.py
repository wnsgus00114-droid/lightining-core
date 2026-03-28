from .tensor import tensor, as_numpy, default_device
from .backend import (
    set_backend,
    get_backend,
    configure_backend_from_env,
    has_torch,
    has_transformers,
    has_lightning_core,
)
from .nn import (
    Module,
    Sequential,
    Linear,
    Conv2d,
    ReLU,
    GELU,
    Dropout,
    LayerNorm,
    Embedding,
    CrossEntropyLoss,
)
from .optim import AdamW, SGD
from .training import Trainer, TrainerConfig
from .hub import load_hf_pipeline, load_hf_model_and_tokenizer
from .lightning_backend import lightning_vector_add
from .standalone_ml import (
    train_test_split,
    LinearRegressionModel,
    LogisticRegressionModel,
    fit_linear_regression,
    fit_logistic_regression,
    mse,
    mae,
    r2_score,
    accuracy,
)

__all__ = [
    "tensor",
    "as_numpy",
    "default_device",
    "set_backend",
    "get_backend",
    "configure_backend_from_env",
    "has_torch",
    "has_transformers",
    "has_lightning_core",
    "Module",
    "Sequential",
    "Linear",
    "Conv2d",
    "ReLU",
    "GELU",
    "Dropout",
    "LayerNorm",
    "Embedding",
    "CrossEntropyLoss",
    "AdamW",
    "SGD",
    "Trainer",
    "TrainerConfig",
    "train_test_split",
    "LinearRegressionModel",
    "LogisticRegressionModel",
    "fit_linear_regression",
    "fit_logistic_regression",
    "mse",
    "mae",
    "r2_score",
    "accuracy",
    "load_hf_pipeline",
    "load_hf_model_and_tokenizer",
    "lightning_vector_add",
]
