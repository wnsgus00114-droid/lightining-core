from __future__ import annotations

import importlib.util
import os
import warnings


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def has_torch() -> bool:
    return _has_module("torch")


def has_transformers() -> bool:
    return _has_module("transformers")


def has_lightning_core() -> bool:
    return _has_module("lightning_core")


_BACKEND = "lightning"
_BACKEND_ENV_KEY = "EXIA_BACKEND"


def _normalize_backend(name: str) -> str:
    return name.strip().lower()


def configure_backend_from_env(*, strict: bool = False) -> str:
    global _BACKEND

    requested = os.getenv(_BACKEND_ENV_KEY)
    if requested is None or requested.strip() == "":
        return _BACKEND

    normalized = _normalize_backend(requested)
    if normalized not in {"lightning", "torch"}:
        message = (
            f"{_BACKEND_ENV_KEY}={requested!r} is invalid. "
            "Supported values are 'lightning' or 'torch'."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message + " Falling back to 'lightning'.", RuntimeWarning)
        _BACKEND = "lightning"
        return _BACKEND

    if normalized == "torch" and not has_torch():
        message = (
            f"{_BACKEND_ENV_KEY}='torch' was requested but torch is not installed. "
            "Install with: pip install Exia[torch]"
        )
        if strict:
            raise RuntimeError(message)
        warnings.warn(message + " Falling back to 'lightning'.", RuntimeWarning)
        _BACKEND = "lightning"
        return _BACKEND

    _BACKEND = normalized
    return _BACKEND


def set_backend(name: str) -> None:
    global _BACKEND
    normalized = _normalize_backend(name)
    if normalized not in {"lightning", "torch"}:
        raise ValueError("backend must be either 'lightning' or 'torch'")
    if normalized == "torch" and not has_torch():
        raise RuntimeError("torch backend requested but torch is not installed. Install with: pip install Exia[torch]")
    _BACKEND = normalized


def get_backend() -> str:
    return _BACKEND


configure_backend_from_env(strict=False)
