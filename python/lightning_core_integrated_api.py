from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from threading import Lock
import time
from typing import Any

import numpy as np

import lightning_core as lc

_SESSION_CACHE_LIMIT = 32
_ATTN_SESSION_CACHE: OrderedDict[tuple[int, int, bool, str], Any] = OrderedDict()
_CACHE_LOCK = Lock()
_BACKEND_LOCK = Lock()
_BACKEND_ENGINE = "lightning"
_VALID_ENGINES = {"lightning", "torch", "auto"}
_VALID_ROUTE_ENGINES = {"lightning", "torch", "auto"}
_VALID_ROUTE_POLICY_KEYS = {"conv", "attention", "graph"}
_LC_API_BRIDGE_LOCK = Lock()
_LC_API_BRIDGE_INSTALLED = False
_LC_API_DIRECT_EXPORTS: dict[str, Any] = {}
_CHECKPOINT_META_KEY = "__lc_checkpoint_meta__"
_CHECKPOINT_FORMAT = "lc_checkpoint_v1"
_CHECKPOINT_FORMAT_V11 = "lc_checkpoint_v1_1"
_CHECKPOINT_FORMAT_V12 = "lc_checkpoint_v1_2"
_CHECKPOINT_INTEGRITY_SIGNATURE = "lc_integrity_sha256_v1"


class CheckpointValidationError(ValueError):
    """Structured checkpoint validation error."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"{code}: {message}")
        self.code = str(code)
        self.details = details or {}


class RoutePolicyValidationError(ValueError):
    """Structured route-policy boundary validation error."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"{code}: {message}")
        self.code = str(code)
        self.details = details or {}


def _import_torch():
    try:
        import torch
        import torch.nn.functional as F

        return torch, F
    except Exception:
        return None, None


def _torch_device_for(device: str) -> str:
    torch, _ = _import_torch()
    if torch is None:
        return "cpu"
    if device == "metal":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _resolve_engine(device: str) -> str:
    with _BACKEND_LOCK:
        configured = _BACKEND_ENGINE
    if configured == "lightning":
        return "lightning"
    if configured == "torch":
        return "torch"

    # auto mode: prefer lightning-core when the requested device is natively available.
    if device == "metal" and hasattr(lc, "metal_available") and lc.metal_available():
        return "lightning"
    if device == "cuda" and hasattr(lc, "cuda_available") and lc.cuda_available():
        return "lightning"
    if device == "cpu":
        return "lightning"

    torch, _ = _import_torch()
    return "torch" if torch is not None else "lightning"


def _normalize_route_engine(value: Any, *, field: str) -> str:
    token = str(value).strip().lower()
    if token not in _VALID_ROUTE_ENGINES:
        options = ", ".join(sorted(_VALID_ROUTE_ENGINES))
        raise ValueError(f"{field} route must be one of: {options}")
    return token


def _normalize_route_policy(route_policy: dict[str, Any] | None) -> dict[str, str]:
    normalized = {"conv": "auto", "attention": "auto", "graph": "auto"}
    if route_policy is None:
        return normalized
    if not isinstance(route_policy, dict):
        raise TypeError("route_policy must be a dict with optional keys: conv, attention, graph")
    for key, value in route_policy.items():
        key_s = str(key).strip().lower()
        if key_s not in _VALID_ROUTE_POLICY_KEYS:
            allowed = ", ".join(sorted(_VALID_ROUTE_POLICY_KEYS))
            raise ValueError(f"unsupported route_policy key '{key}'; allowed keys: {allowed}")
        normalized[key_s] = _normalize_route_engine(value, field=f"route_policy.{key_s}")
    return normalized


def validate_route_policy(route_policy: Any, *, strict: bool = False) -> dict[str, Any]:
    """Validate route policy contract with structured reason codes."""

    def _fail(code: str, message: str, details: dict[str, Any]) -> dict[str, Any]:
        out = {
            "ok": False,
            "code": str(code),
            "message": str(message),
            "details": dict(details),
            "normalized": {"conv": "auto", "attention": "auto", "graph": "auto"},
        }
        if strict:
            raise RoutePolicyValidationError(code, message, details)
        return out

    if route_policy is None:
        return {
            "ok": True,
            "code": "ok",
            "message": "route_policy omitted; using defaults",
            "details": {},
            "normalized": {"conv": "auto", "attention": "auto", "graph": "auto"},
        }
    if not isinstance(route_policy, dict):
        return _fail(
            "interop_route_policy_invalid_type",
            "route_policy must be a dict with optional keys: conv, attention, graph",
            {"type": type(route_policy).__name__},
        )
    for key in route_policy.keys():
        key_s = str(key).strip().lower()
        if key_s not in _VALID_ROUTE_POLICY_KEYS:
            return _fail(
                "interop_route_policy_invalid_key",
                f"unsupported route_policy key '{key_s}'",
                {"allowed_keys": sorted(_VALID_ROUTE_POLICY_KEYS), "received_key": key_s},
            )
    normalized: dict[str, str] = {"conv": "auto", "attention": "auto", "graph": "auto"}
    for key, value in route_policy.items():
        key_s = str(key).strip().lower()
        token = str(value).strip().lower()
        if token not in _VALID_ROUTE_ENGINES:
            return _fail(
                "interop_route_policy_invalid_value",
                f"{key_s} route must be one of: {', '.join(sorted(_VALID_ROUTE_ENGINES))}",
                {"key": key_s, "value": token},
            )
        normalized[key_s] = token
    return {
        "ok": True,
        "code": "ok",
        "message": "route policy validated",
        "details": {},
        "normalized": normalized,
    }


def _resolve_engine_preference(device: str, preference: str) -> tuple[str, str]:
    pref = _normalize_route_engine(preference, field="route preference")
    if pref == "auto":
        return _resolve_engine(device), "auto"
    if pref == "torch":
        torch, _ = _import_torch()
        if torch is None:
            return "lightning", "torch_unavailable_fallback"
    return pref, "requested"


def _graph_conv_attention_shape_supported(
    *,
    x_arr: np.ndarray,
    w_arr: np.ndarray,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
) -> bool:
    if stride_h <= 0 or stride_w <= 0:
        return False
    if pad_h < 0 or pad_w < 0:
        return False
    if x_arr.ndim != 4 or w_arr.ndim != 4:
        return False
    if w_arr.shape[2] != 3 or w_arr.shape[3] != 3:
        return False
    in_h = int(x_arr.shape[2])
    in_w = int(x_arr.shape[3])
    return (in_h + (2 * int(pad_h)) >= 3) and (in_w + (2 * int(pad_w)) >= 3)


def _env_flag_enabled(name: str) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return False
    c0 = raw.strip().lower()[0]
    return c0 in {"1", "t", "y"}


def _conv2d_cpu_crossover_macs() -> int:
    raw = os.environ.get("CJ_CONV2D_CPU_CROSSOVER_MACS", "").strip()
    if raw.isdigit():
        return int(raw)
    return 260000


def _should_prefer_tiny_cpu_conv_attn_chain(
    *,
    device: str,
    batch: int,
    in_channels: int,
    in_h: int,
    in_w: int,
    out_channels: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    seq_len: int,
    head_dim: int,
) -> bool:
    if _env_flag_enabled("LC_CONV_ATTN_TINY_CHAIN_DISABLE"):
        return False
    if device != "metal":
        return False
    if kernel_h != 3 or kernel_w != 3 or stride_h != 1 or stride_w != 1 or pad_h != 1 or pad_w != 1:
        return False
    if batch > 64 or in_channels > 16 or out_channels > 32:
        return False
    out_h = (in_h + 2 * pad_h - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kernel_w) // stride_w + 1
    conv_macs = batch * out_h * out_w * out_channels * in_channels * kernel_h * kernel_w
    if conv_macs > _conv2d_cpu_crossover_macs():
        return False
    attn_work = int(seq_len) * int(head_dim)
    return 0 < attn_work <= 12288


def _as_f32_c(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == np.float32 and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_out_f32_c_no_copy(out: Any) -> np.ndarray:
    arr = np.asarray(out)
    if arr.dtype != np.float32 or not arr.flags.c_contiguous:
        raise ValueError("out must be float32 C-contiguous to avoid copy-back")
    return arr


def set_backend(name: str) -> None:
    name_l = str(name).strip().lower()
    if name_l not in _VALID_ENGINES:
        raise ValueError("backend engine must be one of: lightning, torch, auto")
    if name_l == "torch":
        torch, _ = _import_torch()
        if torch is None:
            raise RuntimeError("torch backend requested but torch is not installed")
    global _BACKEND_ENGINE
    with _BACKEND_LOCK:
        _BACKEND_ENGINE = name_l


def get_backend() -> str:
    with _BACKEND_LOCK:
        return _BACKEND_ENGINE


def set_engine(name: str) -> None:
    set_backend(name)


def get_engine() -> str:
    return get_backend()


def tensor(x: Any) -> np.ndarray:
    return _as_f32_c(x)


def _json_dumps_stable(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_entry_hash(arr: np.ndarray) -> str:
    data = np.asarray(arr)
    header = {
        "dtype": str(data.dtype),
        "shape": [int(x) for x in data.shape],
        "nbytes": int(data.nbytes),
    }
    h = hashlib.sha256()
    h.update(_json_dumps_stable(header).encode("utf-8"))
    h.update(np.ascontiguousarray(data).view(np.uint8).tobytes())
    return h.hexdigest()


def _build_tensor_manifest(flat_state: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for key in sorted(flat_state.keys()):
        arr = np.asarray(flat_state[key])
        manifest[str(key)] = {
            "dtype": str(arr.dtype),
            "shape": [int(x) for x in arr.shape],
            "nbytes": int(arr.nbytes),
            "sha256": _tensor_entry_hash(arr),
        }
    return manifest


def _manifest_hash(manifest: dict[str, dict[str, Any]]) -> str:
    return _sha256_hex(_json_dumps_stable(manifest).encode("utf-8"))


def _checkpoint_meta_array(payload: dict[str, Any]) -> np.ndarray:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return np.frombuffer(raw, dtype=np.uint8)


def _checkpoint_meta_from_array(arr: np.ndarray) -> dict[str, Any]:
    raw = bytes(np.asarray(arr, dtype=np.uint8).tolist())
    if not raw:
        return {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _checkpoint_state_to_array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    arr = np.asarray(value)
    if arr.dtype == object:
        raise ValueError("checkpoint state value must not be object dtype")
    return np.ascontiguousarray(arr)


def _flatten_checkpoint_state(state: dict[str, Any], *, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in state.items():
        key = str(k)
        if not key:
            raise ValueError("checkpoint key must be non-empty")
        full_key = f"{prefix}.{key}" if prefix else key
        if full_key == _CHECKPOINT_META_KEY:
            raise ValueError(f"checkpoint key '{_CHECKPOINT_META_KEY}' is reserved")
        if isinstance(v, dict):
            nested = _flatten_checkpoint_state(v, prefix=full_key)
            for nk, nv in nested.items():
                if nk in out:
                    raise ValueError(f"duplicate checkpoint key: {nk}")
                out[nk] = nv
        else:
            if full_key in out:
                raise ValueError(f"duplicate checkpoint key: {full_key}")
            out[full_key] = _checkpoint_state_to_array(v)
    return out


def _strip_state_prefix(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    p = str(prefix).rstrip(".") + "."
    out: dict[str, Any] = {}
    for k, v in state.items():
        key = str(k)
        if key.startswith(p):
            out[key[len(p) :]] = v
    return out


def _checkpoint_integrity_meta(
    *,
    format_name: str,
    flat_state: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
    model_class: str = "",
    compat_read: list[str] | None = None,
) -> dict[str, Any]:
    tensor_manifest = _build_tensor_manifest(flat_state)
    tensor_hashes = {k: v["sha256"] for k, v in tensor_manifest.items()}
    manifest_hash = _manifest_hash(tensor_manifest)
    payload = {
        "format": str(format_name),
        "format_version": "1.2",
        "compat_read": list(compat_read or [_CHECKPOINT_FORMAT, _CHECKPOINT_FORMAT_V11, _CHECKPOINT_FORMAT_V12]),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": get_backend(),
        "model_class": str(model_class),
        "num_tensors": len(flat_state),
        "keys": sorted(flat_state.keys()),
        "integrity_signature": _CHECKPOINT_INTEGRITY_SIGNATURE,
        "manifest_hash": manifest_hash,
        "tensor_hashes": tensor_hashes,
        "tensor_manifest": tensor_manifest,
        "user_metadata": metadata or {},
    }
    return payload


def _checkpoint_fail(
    *,
    path: str,
    meta: dict[str, Any],
    code: str,
    message: str,
    details: dict[str, Any],
    strict: bool,
) -> dict[str, Any]:
    out = {
        "path": path,
        "ok": False,
        "code": str(code),
        "message": str(message),
        "details": dict(details),
        "meta": meta,
    }
    if strict:
        raise CheckpointValidationError(code, message, out)
    return out


def validate_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_formats: list[str] | tuple[str, ...] | None = None,
    strict: bool = False,
    require_signature: bool = True,
) -> dict[str, Any]:
    """Validate checkpoint integrity and compatibility contract."""
    load_path = os.fspath(path)
    try:
        with np.load(load_path, allow_pickle=False) as data:
            files = list(data.files)
            if _CHECKPOINT_META_KEY not in files:
                return _checkpoint_fail(
                    path=load_path,
                    meta={},
                    code="checkpoint_meta_missing",
                    message="checkpoint metadata key is missing",
                    details={"meta_key": _CHECKPOINT_META_KEY},
                    strict=strict,
                )
            meta = _checkpoint_meta_from_array(np.asarray(data[_CHECKPOINT_META_KEY]))
            if not isinstance(meta, dict) or not meta:
                return _checkpoint_fail(
                    path=load_path,
                    meta={},
                    code="checkpoint_meta_invalid",
                    message="checkpoint metadata is empty or invalid JSON object",
                    details={},
                    strict=strict,
                )
            fmt = str(meta.get("format", ""))
            if expected_formats is None:
                expected = {_CHECKPOINT_FORMAT, _CHECKPOINT_FORMAT_V11, _CHECKPOINT_FORMAT_V12}
            else:
                expected = {str(x) for x in expected_formats}
            if fmt and fmt not in expected:
                return _checkpoint_fail(
                    path=load_path,
                    meta=meta,
                    code="checkpoint_format_unsupported",
                    message=f"unsupported checkpoint format: {fmt}",
                    details={"expected_formats": sorted(expected), "observed_format": fmt},
                    strict=strict,
                )

            if require_signature:
                sig = str(meta.get("integrity_signature", ""))
                if sig and sig != _CHECKPOINT_INTEGRITY_SIGNATURE:
                    return _checkpoint_fail(
                        path=load_path,
                        meta=meta,
                        code="checkpoint_signature_mismatch",
                        message="checkpoint integrity signature mismatch",
                        details={"expected_signature": _CHECKPOINT_INTEGRITY_SIGNATURE, "observed_signature": sig},
                        strict=strict,
                    )

            state: dict[str, np.ndarray] = {}
            for name in files:
                if name == _CHECKPOINT_META_KEY:
                    continue
                state[name] = np.asarray(data[name])
    except CheckpointValidationError:
        raise
    except Exception as exc:
        return _checkpoint_fail(
            path=load_path,
            meta={},
            code="checkpoint_read_error",
            message=str(exc),
            details={"exception_type": exc.__class__.__name__},
            strict=strict,
        )

    observed_num = len(state)
    expected_num = int(meta.get("num_tensors", observed_num))
    if expected_num != observed_num:
        return _checkpoint_fail(
            path=load_path,
            meta=meta,
            code="checkpoint_num_tensors_mismatch",
            message="num_tensors in metadata does not match state payload",
            details={"meta_num_tensors": expected_num, "state_num_tensors": observed_num},
            strict=strict,
        )

    observed_manifest = _build_tensor_manifest(state)
    observed_hash = _manifest_hash(observed_manifest)
    meta_manifest_hash = str(meta.get("manifest_hash", ""))
    if meta_manifest_hash and meta_manifest_hash != observed_hash:
        return _checkpoint_fail(
            path=load_path,
            meta=meta,
            code="checkpoint_manifest_hash_mismatch",
            message="manifest hash mismatch",
            details={"meta_manifest_hash": meta_manifest_hash, "observed_manifest_hash": observed_hash},
            strict=strict,
        )

    meta_tensor_hashes = dict(meta.get("tensor_hashes", {})) if isinstance(meta.get("tensor_hashes", {}), dict) else {}
    if meta_tensor_hashes:
        for key in sorted(observed_manifest.keys()):
            observed_tensor_hash = str(observed_manifest[key].get("sha256", ""))
            meta_tensor_hash = str(meta_tensor_hashes.get(key, ""))
            if not meta_tensor_hash:
                return _checkpoint_fail(
                    path=load_path,
                    meta=meta,
                    code="checkpoint_manifest_missing_tensor",
                    message=f"tensor hash missing for key '{key}'",
                    details={"key": key},
                    strict=strict,
                )
            if meta_tensor_hash != observed_tensor_hash:
                return _checkpoint_fail(
                    path=load_path,
                    meta=meta,
                    code="checkpoint_tensor_hash_mismatch",
                    message=f"tensor hash mismatch for key '{key}'",
                    details={"key": key, "meta_hash": meta_tensor_hash, "observed_hash": observed_tensor_hash},
                    strict=strict,
                )

    return {
        "path": load_path,
        "ok": True,
        "code": "ok",
        "message": "checkpoint validation passed",
        "meta": meta,
        "details": {
            "state_num_tensors": observed_num,
            "observed_manifest_hash": observed_hash,
            "expected_formats": sorted(expected),
        },
    }


def checkpoint_conversion_diagnostics(
    state_or_obj: Any,
    *,
    source_format: str = "auto",
) -> dict[str, Any]:
    """Diagnose conversion readiness from external framework state payloads."""
    if hasattr(state_or_obj, "state_dict") and callable(getattr(state_or_obj, "state_dict")):
        state_obj = state_or_obj.state_dict()  # type: ignore[call-arg]
    else:
        state_obj = state_or_obj
    if not isinstance(state_obj, dict):
        raise TypeError("state_or_obj must be dict or object implementing state_dict()")

    src = str(source_format).strip().lower()
    if src == "auto":
        src = "generic_state_dict"
        for value in state_obj.values():
            module_name = getattr(type(value), "__module__", "").lower()
            if module_name.startswith("torch"):
                src = "torch_state_dict"
                break
            if module_name.startswith("tensorflow"):
                src = "tf_checkpoint"
                break

    supported: dict[str, np.ndarray] = {}
    unsupported: list[dict[str, Any]] = []
    for key, value in state_obj.items():
        key_s = str(key)
        try:
            supported[key_s] = _checkpoint_state_to_array(value)
        except Exception as exc:
            unsupported.append(
                {
                    "key": key_s,
                    "reason_code": "conversion_value_unsupported",
                    "message": str(exc),
                    "value_type": type(value).__name__,
                }
            )

    manifest = _build_tensor_manifest(supported)
    return {
        "source_format": src,
        "integrity_signature": _CHECKPOINT_INTEGRITY_SIGNATURE,
        "total_entries": len(state_obj),
        "convertible_entries": len(supported),
        "unsupported_entries": unsupported,
        "convertible": len(unsupported) == 0,
        "manifest_hash": _manifest_hash(manifest),
        "tensor_manifest": manifest,
    }


def validate_checkpoint_conversion(
    state_or_obj: Any,
    *,
    source_format: str = "auto",
    strict: bool = False,
) -> dict[str, Any]:
    out = checkpoint_conversion_diagnostics(state_or_obj, source_format=source_format)
    if strict and not bool(out.get("convertible", False)):
        raise CheckpointValidationError(
            "checkpoint_conversion_not_convertible",
            "state payload contains unsupported entries",
            {"unsupported_entries": out.get("unsupported_entries", [])},
        )
    return out


def save_checkpoint(
    path: str | os.PathLike[str],
    state_or_obj: Any,
    *,
    metadata: dict[str, Any] | None = None,
    compressed: bool = True,
) -> str:
    """Save checkpoint state to .npz (format: lc_checkpoint_v1_2)."""
    if hasattr(state_or_obj, "state_dict") and callable(getattr(state_or_obj, "state_dict")):
        state_obj = state_or_obj.state_dict()  # type: ignore[call-arg]
    else:
        state_obj = state_or_obj
    if not isinstance(state_obj, dict):
        raise TypeError("state_or_obj must be dict or object implementing state_dict()")

    flat_state = _flatten_checkpoint_state(state_obj)
    meta_payload = _checkpoint_integrity_meta(
        format_name=_CHECKPOINT_FORMAT_V12,
        flat_state=flat_state,
        metadata=metadata,
    )

    save_dict: dict[str, np.ndarray] = dict(flat_state)
    save_dict[_CHECKPOINT_META_KEY] = _checkpoint_meta_array(meta_payload)

    save_path = os.fspath(path)
    if compressed:
        np.savez_compressed(save_path, **save_dict)
    else:
        np.savez(save_path, **save_dict)
    return save_path


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    into: Any | None = None,
    strict: bool = True,
    validate: bool = True,
    expected_formats: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load checkpoint from .npz and optionally apply to target via load_state_dict()."""
    load_path = os.fspath(path)
    state: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    validation: dict[str, Any] | None = None
    if validate:
        validation = validate_checkpoint(load_path, expected_formats=expected_formats, strict=True)
    with np.load(load_path, allow_pickle=False) as data:
        files = list(data.files)
        if _CHECKPOINT_META_KEY in files:
            meta = _checkpoint_meta_from_array(np.asarray(data[_CHECKPOINT_META_KEY]))
        for name in files:
            if name == _CHECKPOINT_META_KEY:
                continue
            state[name] = np.asarray(data[name])

    if into is not None:
        load_fn = getattr(into, "load_state_dict", None)
        if not callable(load_fn):
            raise TypeError("into must implement load_state_dict(state, strict=...)")
        load_fn(state, strict=bool(strict))

    return {"path": load_path, "meta": meta, "state": state, "validation": validation}


def save_model_checkpoint(
    path: str | os.PathLike[str],
    model: Any,
    *,
    metadata: dict[str, Any] | None = None,
    compressed: bool = True,
) -> str:
    """Save model-level checkpoint (format: lc_checkpoint_v1_2)."""
    state_fn = getattr(model, "state_dict", None)
    if not callable(state_fn):
        raise TypeError("model must implement state_dict()")

    model_state = state_fn()
    if not isinstance(model_state, dict):
        raise TypeError("model.state_dict() must return dict")

    flat_state = _flatten_checkpoint_state(model_state, prefix="model")
    meta_payload = _checkpoint_integrity_meta(
        format_name=_CHECKPOINT_FORMAT_V12,
        flat_state=flat_state,
        metadata=metadata,
        model_class=model.__class__.__name__,
        compat_read=[_CHECKPOINT_FORMAT, _CHECKPOINT_FORMAT_V11, _CHECKPOINT_FORMAT_V12],
    )
    save_dict: dict[str, np.ndarray] = dict(flat_state)
    save_dict[_CHECKPOINT_META_KEY] = _checkpoint_meta_array(meta_payload)

    save_path = os.fspath(path)
    if compressed:
        np.savez_compressed(save_path, **save_dict)
    else:
        np.savez(save_path, **save_dict)
    return save_path


def load_model_checkpoint(
    path: str | os.PathLike[str],
    *,
    into: Any | None = None,
    strict: bool = True,
    validate: bool = True,
) -> dict[str, Any]:
    """Load model-level checkpoint with forward-compat support for v1."""
    payload = load_checkpoint(path, into=None, strict=strict, validate=validate)
    meta = dict(payload.get("meta", {}))
    state = dict(payload.get("state", {}))
    fmt = str(meta.get("format", ""))
    if fmt and fmt not in {_CHECKPOINT_FORMAT, _CHECKPOINT_FORMAT_V11, _CHECKPOINT_FORMAT_V12}:
        raise ValueError(f"unsupported checkpoint format: {fmt}")

    model_state = _strip_state_prefix(state, "model")
    # Forward-compat: old v1 checkpoints may store top-level state without
    # model.* prefix.
    if not model_state:
        model_state = state

    if into is not None:
        load_fn = getattr(into, "load_state_dict", None)
        if not callable(load_fn):
            raise TypeError("into must implement load_state_dict(state, strict=...)")
        load_fn(model_state, strict=bool(strict))

    return {
        "path": payload.get("path"),
        "meta": meta,
        "state": state,
        "model_state": model_state,
    }


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = np.random.randn(self.in_features, self.out_features).astype(np.float32) * 0.02
        self.bias = np.zeros((self.out_features,), dtype=np.float32) if bias else None

    def __call__(self, x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.in_features:
            raise ValueError("Linear input must be [batch, in_features]")
        out = lightning_matmul(arr, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out

    def state_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        key = (str(prefix).rstrip(".") + ".") if prefix else ""
        out: dict[str, np.ndarray] = {
            f"{key}weight": np.ascontiguousarray(self.weight),
            f"{key}in_features": np.asarray([self.in_features], dtype=np.int32),
            f"{key}out_features": np.asarray([self.out_features], dtype=np.int32),
            f"{key}has_bias": np.asarray([1 if self.bias is not None else 0], dtype=np.int32),
        }
        if self.bias is not None:
            out[f"{key}bias"] = np.ascontiguousarray(self.bias)
        return out

    def load_state_dict(self, state: dict[str, Any], strict: bool = True, prefix: str = "") -> None:
        key = (str(prefix).rstrip(".") + ".") if prefix else ""
        w_key = f"{key}weight"
        b_key = f"{key}bias"
        hb_key = f"{key}has_bias"

        if w_key not in state:
            raise KeyError(f"missing key: {w_key}")
        w = np.asarray(state[w_key], dtype=np.float32)
        if w.shape != (self.in_features, self.out_features):
            raise ValueError(f"weight shape mismatch: expected {(self.in_features, self.out_features)} got {tuple(w.shape)}")
        self.weight = np.ascontiguousarray(w)

        has_bias = self.bias is not None
        if hb_key in state:
            hb = np.asarray(state[hb_key]).reshape(-1)
            if hb.size > 0:
                has_bias = bool(int(hb[0]))

        if has_bias:
            if b_key not in state:
                if strict:
                    raise KeyError(f"missing key: {b_key}")
                self.bias = np.zeros((self.out_features,), dtype=np.float32)
            else:
                b = np.asarray(state[b_key], dtype=np.float32).reshape(-1)
                if b.shape != (self.out_features,):
                    raise ValueError(f"bias shape mismatch: expected {(self.out_features,)} got {tuple(b.shape)}")
                self.bias = np.ascontiguousarray(b)
        else:
            self.bias = None


class TinyMLPModel:
    """Tiny model-runner style block for checkpoint/model-level smoke."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        self.d_in = int(d_in)
        self.d_hidden = int(d_hidden)
        self.d_out = int(d_out)
        self.fc1 = Linear(self.d_in, self.d_hidden, bias=True)
        self.fc2 = Linear(self.d_hidden, self.d_out, bias=True)

    def __call__(self, x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.d_in:
            raise ValueError("TinyMLPModel input must be [batch, d_in]")
        h = self.fc1(arr)
        h = np.maximum(h, 0.0).astype(np.float32, copy=False)
        return self.fc2(h)

    def state_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        key = (str(prefix).rstrip(".") + ".") if prefix else ""
        out: dict[str, np.ndarray] = {
            f"{key}d_in": np.asarray([self.d_in], dtype=np.int32),
            f"{key}d_hidden": np.asarray([self.d_hidden], dtype=np.int32),
            f"{key}d_out": np.asarray([self.d_out], dtype=np.int32),
            f"{key}format_version": np.asarray([11], dtype=np.int32),
        }
        out.update(self.fc1.state_dict(prefix=f"{key}fc1"))
        out.update(self.fc2.state_dict(prefix=f"{key}fc2"))
        return out

    def load_state_dict(self, state: dict[str, Any], strict: bool = True, prefix: str = "") -> None:
        key = (str(prefix).rstrip(".") + ".") if prefix else ""
        for name in ("d_in", "d_hidden", "d_out"):
            sk = f"{key}{name}"
            if sk in state:
                val = np.asarray(state[sk]).reshape(-1)
                if val.size > 0:
                    setattr(self, name, int(val[0]))
            elif strict:
                raise KeyError(f"missing key: {sk}")

        self.fc1 = Linear(self.d_in, self.d_hidden, bias=True)
        self.fc2 = Linear(self.d_hidden, self.d_out, bias=True)
        self.fc1.load_state_dict(state, strict=strict, prefix=f"{key}fc1")
        self.fc2.load_state_dict(state, strict=strict, prefix=f"{key}fc2")


class _EngineScope:
    def __init__(self, name: str):
        self._name = str(name)
        self._prev = get_backend()

    def __enter__(self) -> "_EngineScope":
        set_backend(self._name)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        set_backend(self._prev)
        return False


class TinyTransformerRunner:
    """Model-runner alpha: tiny transformer-ish block with graph/eager/interop modes."""

    def __init__(self, *, seq_len: int = 48, d_model: int = 48, d_ff: int = 128, seed: int = 20260410):
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        rng = np.random.default_rng(int(seed))
        self.wq = (rng.standard_normal((self.d_model, self.d_model)) * 0.05).astype(np.float32)
        self.wk = (rng.standard_normal((self.d_model, self.d_model)) * 0.05).astype(np.float32)
        self.wv = (rng.standard_normal((self.d_model, self.d_model)) * 0.05).astype(np.float32)
        self.wo = (rng.standard_normal((self.d_model, self.d_model)) * 0.05).astype(np.float32)
        self.w1 = (rng.standard_normal((self.d_model, self.d_ff)) * 0.04).astype(np.float32)
        self.b1 = np.zeros((1, self.d_ff), dtype=np.float32)
        self.w2 = (rng.standard_normal((self.d_ff, self.d_model)) * 0.04).astype(np.float32)
        self.b2 = np.zeros((1, self.d_model), dtype=np.float32)
        self._graph_bundle: dict[str, Any] | None = None

    def state_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        key = (str(prefix).rstrip(".") + ".") if prefix else ""
        return {
            f"{key}seq_len": np.asarray([self.seq_len], dtype=np.int32),
            f"{key}d_model": np.asarray([self.d_model], dtype=np.int32),
            f"{key}d_ff": np.asarray([self.d_ff], dtype=np.int32),
            f"{key}wq": np.ascontiguousarray(self.wq),
            f"{key}wk": np.ascontiguousarray(self.wk),
            f"{key}wv": np.ascontiguousarray(self.wv),
            f"{key}wo": np.ascontiguousarray(self.wo),
            f"{key}w1": np.ascontiguousarray(self.w1),
            f"{key}b1": np.ascontiguousarray(self.b1),
            f"{key}w2": np.ascontiguousarray(self.w2),
            f"{key}b2": np.ascontiguousarray(self.b2),
        }

    def load_state_dict(self, state: dict[str, Any], strict: bool = True, prefix: str = "") -> None:
        key = (str(prefix).rstrip(".") + ".") if prefix else ""
        mapping = {
            "wq": (self.d_model, self.d_model),
            "wk": (self.d_model, self.d_model),
            "wv": (self.d_model, self.d_model),
            "wo": (self.d_model, self.d_model),
            "w1": (self.d_model, self.d_ff),
            "b1": (1, self.d_ff),
            "w2": (self.d_ff, self.d_model),
            "b2": (1, self.d_model),
        }
        for field in ("seq_len", "d_model", "d_ff"):
            fk = f"{key}{field}"
            if fk in state:
                vv = np.asarray(state[fk]).reshape(-1)
                if vv.size > 0:
                    setattr(self, field, int(vv[0]))
            elif strict:
                raise KeyError(f"missing key: {fk}")
        for name, shape in mapping.items():
            sk = f"{key}{name}"
            if sk not in state:
                if strict:
                    raise KeyError(f"missing key: {sk}")
                continue
            arr = np.asarray(state[sk], dtype=np.float32)
            if tuple(arr.shape) != tuple(shape):
                raise ValueError(f"{name} shape mismatch: expected {shape} got {tuple(arr.shape)}")
            setattr(self, name, np.ascontiguousarray(arr))
        self._graph_bundle = None

    def _forward_eager(self, x_arr: np.ndarray, *, device: str) -> np.ndarray:
        q = lightning_matmul(x_arr, self.wq, device=device)
        k = lightning_matmul(x_arr, self.wk, device=device)
        v = lightning_matmul(x_arr, self.wv, device=device)
        attn = lightning_attention(q, k, v, seq=self.seq_len, head_dim=self.d_model, device=device, causal=False)
        o = lightning_matmul(attn, self.wo, device=device)
        h = lightning_matmul(o, self.w1, device=device) + self.b1
        h = np.maximum(h, 0.0).astype(np.float32, copy=False)
        y = lightning_matmul(h, self.w2, device=device) + self.b2
        return np.asarray(y, dtype=np.float32)

    def _build_graph_bundle(self) -> dict[str, Any]:
        if self._graph_bundle is not None:
            return self._graph_bundle
        if not hasattr(lc, "GraphIR"):
            raise RuntimeError("GraphIR is unavailable")
        g = lc.GraphIR()
        ids: dict[str, int] = {}
        ids["x"] = g.add_tensor([self.seq_len, self.d_model], dtype="float32", name="x", constant=True)
        ids["wq"] = g.add_tensor([self.d_model, self.d_model], dtype="float32", name="wq", constant=True)
        ids["wk"] = g.add_tensor([self.d_model, self.d_model], dtype="float32", name="wk", constant=True)
        ids["wv"] = g.add_tensor([self.d_model, self.d_model], dtype="float32", name="wv", constant=True)
        ids["wo"] = g.add_tensor([self.d_model, self.d_model], dtype="float32", name="wo", constant=True)
        ids["w1"] = g.add_tensor([self.d_model, self.d_ff], dtype="float32", name="w1", constant=True)
        ids["b1"] = g.add_tensor([self.seq_len, self.d_ff], dtype="float32", name="b1", constant=True)
        ids["w2"] = g.add_tensor([self.d_ff, self.d_model], dtype="float32", name="w2", constant=True)
        ids["b2"] = g.add_tensor([self.seq_len, self.d_model], dtype="float32", name="b2", constant=True)

        ids["q"] = g.add_tensor([self.seq_len, self.d_model], dtype="float32", name="q")
        ids["k"] = g.add_tensor([self.seq_len, self.d_model], dtype="float32", name="k")
        ids["v"] = g.add_tensor([self.seq_len, self.d_model], dtype="float32", name="v")
        ids["attn"] = g.add_tensor([self.seq_len, self.d_model], dtype="float32", name="attn")
        ids["o"] = g.add_tensor([self.seq_len, self.d_model], dtype="float32", name="o")
        ids["ff1"] = g.add_tensor([self.seq_len, self.d_ff], dtype="float32", name="ff1")
        ids["ff1b"] = g.add_tensor([self.seq_len, self.d_ff], dtype="float32", name="ff1b")
        ids["ff1r"] = g.add_tensor([self.seq_len, self.d_ff], dtype="float32", name="ff1r")
        ids["ff2"] = g.add_tensor([self.seq_len, self.d_model], dtype="float32", name="ff2")
        ids["out"] = g.add_tensor([self.seq_len, self.d_model], dtype="float32", name="out")

        g.add_node("matmul", [ids["x"], ids["wq"]], [ids["q"]])
        g.add_node("matmul", [ids["x"], ids["wk"]], [ids["k"]])
        g.add_node("matmul", [ids["x"], ids["wv"]], [ids["v"]])
        g.add_node("attention_forward", [ids["q"], ids["k"], ids["v"]], [ids["attn"]])
        g.add_node("matmul", [ids["attn"], ids["wo"]], [ids["o"]])
        g.add_node("matmul", [ids["o"], ids["w1"]], [ids["ff1"]])
        g.add_node("vector_add", [ids["ff1"], ids["b1"]], [ids["ff1b"]])
        g.add_node("relu", [ids["ff1b"]], [ids["ff1r"]])
        g.add_node("matmul", [ids["ff1r"], ids["w2"]], [ids["ff2"]])
        g.add_node("vector_add", [ids["ff2"], ids["b2"]], [ids["out"]])

        self._graph_bundle = {"graph": g, "ids": ids}
        return self._graph_bundle

    def _forward_graph(self, x_arr: np.ndarray, *, device: str) -> np.ndarray:
        bundle = self._build_graph_bundle()
        g = bundle["graph"]
        ids = bundle["ids"]
        b1_full = np.broadcast_to(self.b1, (self.seq_len, self.d_ff)).astype(np.float32, copy=False)
        b2_full = np.broadcast_to(self.b2, (self.seq_len, self.d_model)).astype(np.float32, copy=False)
        feeds = {
            ids["x"]: x_arr,
            ids["wq"]: self.wq,
            ids["wk"]: self.wk,
            ids["wv"]: self.wv,
            ids["wo"]: self.wo,
            ids["w1"]: self.w1,
            ids["b1"]: b1_full,
            ids["w2"]: self.w2,
            ids["b2"]: b2_full,
        }
        out = g.execute_f32(
            feeds,
            preferred_device=device,
            enable_fusion_v1=True,
            fusion_pass_order="attention_qkv,attention,matmul,conv",
        )
        values = dict(out.get("values", {}))
        if ids["out"] not in values:
            raise RuntimeError("graph output missing")
        return np.asarray(values[ids["out"]], dtype=np.float32).reshape(self.seq_len, self.d_model)

    def run(self, x: Any, *, mode: str = "eager", device: str = "auto") -> np.ndarray:
        mode_s = str(mode).strip().lower()
        if mode_s not in {"eager", "graph", "interop"}:
            raise ValueError("mode must be one of: eager, graph, interop")
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.shape != (self.seq_len, self.d_model):
            raise ValueError(f"input shape must be {(self.seq_len, self.d_model)}")
        if mode_s == "graph":
            with _EngineScope("lightning"):
                return self._forward_graph(x_arr, device=device)
        if mode_s == "interop":
            with _EngineScope("torch"):
                return self._forward_eager(x_arr, device=device)
        return self._forward_eager(x_arr, device=device)


def create_torch_module_wrapper(
    runner: TinyTransformerRunner,
    *,
    mode: str = "eager",
    device: str = "auto",
):
    torch, _ = _import_torch()
    if torch is None:
        raise RuntimeError("torch is not available")

    class LightningCoreTorchModule(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self, model_runner: TinyTransformerRunner):
            super().__init__()
            self.model_runner = model_runner

        def forward(self, x):  # type: ignore[override]
            x_np = x.detach().to("cpu").contiguous().numpy().astype(np.float32, copy=False)
            y_np = self.model_runner.run(x_np, mode=mode, device=device)
            return torch.as_tensor(y_np, dtype=x.dtype, device=x.device)

    return LightningCoreTorchModule(runner)


def create_tf_keras_layer_wrapper(
    runner: TinyTransformerRunner,
    *,
    mode: str = "eager",
    device: str = "auto",
):
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        raise RuntimeError("tensorflow is not available") from exc

    class LightningCoreKerasLayer(tf.keras.layers.Layer):  # type: ignore[name-defined]
        def __init__(self, model_runner: TinyTransformerRunner):
            super().__init__()
            self.model_runner = model_runner

        def call(self, inputs, *args, **kwargs):  # type: ignore[override]
            x_np = np.asarray(inputs, dtype=np.float32)
            y_np = self.model_runner.run(x_np, mode=mode, device=device)
            return tf.convert_to_tensor(y_np, dtype=inputs.dtype)

    return LightningCoreKerasLayer(runner)


def _sum_to_shape(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    g = np.asarray(grad, dtype=np.float32)
    target = tuple(int(x) for x in shape)
    if g.shape == target:
        return g

    while g.ndim > len(target):
        g = g.sum(axis=0)
    for axis, dim in enumerate(target):
        if dim == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(target)


class AutoTensor:
    """Minimal autograd tensor for matmul/add/relu training bootstrap."""

    def __init__(
        self,
        data: Any,
        *,
        requires_grad: bool = False,
        parents: tuple["AutoTensor", ...] = (),
        op: str = "",
    ):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = bool(requires_grad)
        self.grad: np.ndarray | None = None
        self._parents = tuple(parents)
        self._op = str(op)
        self._backward = lambda: None

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, grad: Any | None = None) -> None:
        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be provided for non-scalar tensor")
            upstream = np.ones_like(self.data, dtype=np.float32)
        else:
            upstream = np.asarray(grad, dtype=np.float32)
            if upstream.shape != self.data.shape:
                raise ValueError("backward grad shape mismatch")

        topo: list[AutoTensor] = []
        visited: set[int] = set()

        def _build(v: AutoTensor) -> None:
            key = id(v)
            if key in visited:
                return
            visited.add(key)
            for p in v._parents:
                _build(p)
            topo.append(v)

        _build(self)
        self.grad = upstream
        for node in reversed(topo):
            node._backward()


def ag_tensor(x: Any, *, requires_grad: bool = False) -> AutoTensor:
    return AutoTensor(x, requires_grad=requires_grad)


def ag_parameter(x: Any) -> AutoTensor:
    return AutoTensor(x, requires_grad=True)


def _as_tensor(x: Any) -> AutoTensor:
    if isinstance(x, AutoTensor):
        return x
    return ag_tensor(x, requires_grad=False)


def ag_matmul(a: Any, b: Any) -> AutoTensor:
    ta = _as_tensor(a)
    tb = _as_tensor(b)
    out = AutoTensor(
        ta.data @ tb.data,
        requires_grad=(ta.requires_grad or tb.requires_grad),
        parents=(ta, tb),
        op="matmul",
    )

    def _backward() -> None:
        if out.grad is None:
            return
        if ta.requires_grad:
            ga = out.grad @ tb.data.T
            ta.grad = ga if ta.grad is None else (ta.grad + ga)
        if tb.requires_grad:
            gb = ta.data.T @ out.grad
            tb.grad = gb if tb.grad is None else (tb.grad + gb)

    out._backward = _backward
    return out


def ag_add(a: Any, b: Any) -> AutoTensor:
    ta = _as_tensor(a)
    tb = _as_tensor(b)
    out = AutoTensor(
        ta.data + tb.data,
        requires_grad=(ta.requires_grad or tb.requires_grad),
        parents=(ta, tb),
        op="add",
    )

    def _backward() -> None:
        if out.grad is None:
            return
        if ta.requires_grad:
            ga = _sum_to_shape(out.grad, ta.data.shape)
            ta.grad = ga if ta.grad is None else (ta.grad + ga)
        if tb.requires_grad:
            gb = _sum_to_shape(out.grad, tb.data.shape)
            tb.grad = gb if tb.grad is None else (tb.grad + gb)

    out._backward = _backward
    return out


def ag_relu(x: Any) -> AutoTensor:
    tx = _as_tensor(x)
    out_data = np.maximum(tx.data, 0.0).astype(np.float32, copy=False)
    out = AutoTensor(out_data, requires_grad=tx.requires_grad, parents=(tx,), op="relu")

    def _backward() -> None:
        if out.grad is None or not tx.requires_grad:
            return
        gx = out.grad * (tx.data > 0.0).astype(np.float32)
        tx.grad = gx if tx.grad is None else (tx.grad + gx)

    out._backward = _backward
    return out


def ag_reshape(x: Any, shape: tuple[int, ...] | list[int]) -> AutoTensor:
    tx = _as_tensor(x)
    target = tuple(int(v) for v in shape)
    out = AutoTensor(np.asarray(tx.data.reshape(target), dtype=np.float32), requires_grad=tx.requires_grad, parents=(tx,), op="reshape")

    def _backward() -> None:
        if out.grad is None or not tx.requires_grad:
            return
        gx = np.asarray(out.grad, dtype=np.float32).reshape(tx.data.shape)
        tx.grad = gx if tx.grad is None else (tx.grad + gx)

    out._backward = _backward
    return out


def _conv2d_forward_nchw_numpy(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    *,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
) -> np.ndarray:
    n, ic, ih, iw = [int(v) for v in x.shape]
    oc, w_ic, kh, kw = [int(v) for v in w.shape]
    if ic != w_ic:
        raise ValueError("conv2d channel mismatch")
    oh = (ih + 2 * int(pad_h) - kh) // int(stride_h) + 1
    ow = (iw + 2 * int(pad_w) - kw) // int(stride_w) + 1
    x_pad = np.pad(x, ((0, 0), (0, 0), (int(pad_h), int(pad_h)), (int(pad_w), int(pad_w))), mode="constant")
    y = np.zeros((n, oc, oh, ow), dtype=np.float32)
    for bn in range(n):
        for oc_i in range(oc):
            for oy in range(oh):
                iy0 = oy * int(stride_h)
                for ox in range(ow):
                    ix0 = ox * int(stride_w)
                    region = x_pad[bn, :, iy0 : iy0 + kh, ix0 : ix0 + kw]
                    y[bn, oc_i, oy, ox] = float(np.sum(region * w[oc_i]))
            if b is not None:
                y[bn, oc_i, :, :] += float(b[oc_i])
    return y


def _conv2d_backward_nchw_numpy(
    x: np.ndarray,
    w: np.ndarray,
    grad_out: np.ndarray,
    *,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, ic, ih, iw = [int(v) for v in x.shape]
    oc, w_ic, kh, kw = [int(v) for v in w.shape]
    if ic != w_ic:
        raise ValueError("conv2d channel mismatch")
    _, _, oh, ow = [int(v) for v in grad_out.shape]
    x_pad = np.pad(x, ((0, 0), (0, 0), (int(pad_h), int(pad_h)), (int(pad_w), int(pad_w))), mode="constant")
    gx_pad = np.zeros_like(x_pad, dtype=np.float32)
    gw = np.zeros_like(w, dtype=np.float32)
    gb = grad_out.sum(axis=(0, 2, 3)).astype(np.float32, copy=False)
    for bn in range(n):
        for oc_i in range(oc):
            for oy in range(oh):
                iy0 = oy * int(stride_h)
                for ox in range(ow):
                    ix0 = ox * int(stride_w)
                    go = float(grad_out[bn, oc_i, oy, ox])
                    region = x_pad[bn, :, iy0 : iy0 + kh, ix0 : ix0 + kw]
                    gw[oc_i] += go * region
                    gx_pad[bn, :, iy0 : iy0 + kh, ix0 : ix0 + kw] += go * w[oc_i]
    if int(pad_h) == 0 and int(pad_w) == 0:
        gx = gx_pad
    else:
        gx = gx_pad[:, :, int(pad_h) : int(pad_h) + ih, int(pad_w) : int(pad_w) + iw]
    return gx.astype(np.float32, copy=False), gw.astype(np.float32, copy=False), gb


def ag_conv2d(
    x: Any,
    weight: Any,
    bias: Any | None = None,
    *,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0,
) -> AutoTensor:
    tx = _as_tensor(x)
    tw = _as_tensor(weight)
    tb = _as_tensor(bias) if bias is not None else None
    x_arr = np.asarray(tx.data, dtype=np.float32)
    w_arr = np.asarray(tw.data, dtype=np.float32)
    b_arr = None if tb is None else np.asarray(tb.data, dtype=np.float32).reshape(-1)
    if x_arr.ndim != 4 or w_arr.ndim != 4:
        raise ValueError("ag_conv2d expects x:[N,C,H,W], weight:[O,C,KH,KW]")
    if b_arr is not None and b_arr.shape[0] != w_arr.shape[0]:
        raise ValueError("bias shape mismatch")

    out_data = _conv2d_forward_nchw_numpy(
        x_arr,
        w_arr,
        b_arr,
        stride_h=int(stride_h),
        stride_w=int(stride_w),
        pad_h=int(pad_h),
        pad_w=int(pad_w),
    )
    parents = (tx, tw) if tb is None else (tx, tw, tb)
    out = AutoTensor(
        out_data,
        requires_grad=bool(tx.requires_grad or tw.requires_grad or (tb is not None and tb.requires_grad)),
        parents=parents,
        op="conv2d",
    )

    def _backward() -> None:
        if out.grad is None:
            return
        gx, gw, gb = _conv2d_backward_nchw_numpy(
            x_arr,
            w_arr,
            np.asarray(out.grad, dtype=np.float32),
            stride_h=int(stride_h),
            stride_w=int(stride_w),
            pad_h=int(pad_h),
            pad_w=int(pad_w),
        )
        if tx.requires_grad:
            tx.grad = gx if tx.grad is None else (tx.grad + gx)
        if tw.requires_grad:
            tw.grad = gw if tw.grad is None else (tw.grad + gw)
        if tb is not None and tb.requires_grad:
            gb_r = gb.reshape(tb.data.shape)
            tb.grad = gb_r if tb.grad is None else (tb.grad + gb_r)

    out._backward = _backward
    return out


def ag_attention(q: Any, k: Any, v: Any, *, causal: bool = False) -> AutoTensor:
    tq = _as_tensor(q)
    tk = _as_tensor(k)
    tv = _as_tensor(v)
    q_arr = np.asarray(tq.data, dtype=np.float32)
    k_arr = np.asarray(tk.data, dtype=np.float32)
    v_arr = np.asarray(tv.data, dtype=np.float32)
    if q_arr.ndim != 2 or k_arr.ndim != 2 or v_arr.ndim != 2:
        raise ValueError("ag_attention expects q/k/v rank-2 tensors [seq, head_dim]")
    if q_arr.shape != k_arr.shape or q_arr.shape != v_arr.shape:
        raise ValueError("ag_attention q/k/v shape mismatch")
    seq, dim = int(q_arr.shape[0]), int(q_arr.shape[1])
    scale = 1.0 / float(np.sqrt(max(dim, 1)))
    scores = (q_arr @ k_arr.T) * scale
    causal_mask: np.ndarray | None = None
    if causal:
        causal_mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
        scores = scores.copy()
        scores[causal_mask] = -1.0e9
    max_per_row = np.max(scores, axis=-1, keepdims=True)
    shifted = scores - max_per_row
    exp_shifted = np.exp(shifted).astype(np.float32, copy=False)
    probs = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
    out_data = probs @ v_arr
    out = AutoTensor(
        out_data.astype(np.float32, copy=False),
        requires_grad=bool(tq.requires_grad or tk.requires_grad or tv.requires_grad),
        parents=(tq, tk, tv),
        op="attention",
    )

    def _backward() -> None:
        if out.grad is None:
            return
        dout = np.asarray(out.grad, dtype=np.float32)
        dprobs = dout @ v_arr.T
        dv = probs.T @ dout
        dot = np.sum(dprobs * probs, axis=-1, keepdims=True)
        dscores = probs * (dprobs - dot)
        if causal_mask is not None:
            dscores = dscores.copy()
            dscores[causal_mask] = 0.0
        dq = (dscores @ k_arr) * scale
        dk = (dscores.T @ q_arr) * scale
        if tq.requires_grad:
            tq.grad = dq if tq.grad is None else (tq.grad + dq)
        if tk.requires_grad:
            tk.grad = dk if tk.grad is None else (tk.grad + dk)
        if tv.requires_grad:
            tv.grad = dv if tv.grad is None else (tv.grad + dv)

    out._backward = _backward
    return out


def ag_mse_loss(pred: Any, target: Any) -> AutoTensor:
    tp = _as_tensor(pred)
    tt = _as_tensor(target)
    if tp.data.shape != tt.data.shape:
        raise ValueError("mse loss shape mismatch")
    diff = tp.data - tt.data
    value = np.asarray([float(np.mean(diff * diff))], dtype=np.float32)
    out = AutoTensor(value, requires_grad=tp.requires_grad, parents=(tp,), op="mse_loss")

    def _backward() -> None:
        if out.grad is None or not tp.requires_grad:
            return
        scale = float(out.grad.reshape(-1)[0]) * (2.0 / float(tp.data.size))
        gp = scale * diff
        tp.grad = gp if tp.grad is None else (tp.grad + gp)

    out._backward = _backward
    return out


def ag_zero_grad(params: list[AutoTensor]) -> None:
    for p in params:
        p.zero_grad()


def ag_sgd_step(params: list[AutoTensor], *, lr: float) -> None:
    lr_f = float(lr)
    for p in params:
        if p.grad is None:
            continue
        p.data = np.asarray(p.data - lr_f * p.grad, dtype=np.float32)


class TinyAutogradMLP:
    """Tiny 2-layer MLP with manual autograd graph (matmul/add/relu)."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, *, seed: int = 20260408):
        self.d_in = int(d_in)
        self.d_hidden = int(d_hidden)
        self.d_out = int(d_out)
        rng = np.random.default_rng(int(seed))
        self.w1 = ag_parameter((rng.standard_normal((self.d_in, self.d_hidden)) * 0.02).astype(np.float32))
        self.b1 = ag_parameter(np.zeros((1, self.d_hidden), dtype=np.float32))
        self.w2 = ag_parameter((rng.standard_normal((self.d_hidden, self.d_out)) * 0.02).astype(np.float32))
        self.b2 = ag_parameter(np.zeros((1, self.d_out), dtype=np.float32))

    def parameters(self) -> list[AutoTensor]:
        return [self.w1, self.b1, self.w2, self.b2]

    def forward(self, x: Any) -> AutoTensor:
        tx = _as_tensor(x)
        h = ag_relu(ag_add(ag_matmul(tx, self.w1), self.b1))
        return ag_add(ag_matmul(h, self.w2), self.b2)

    def train_step(self, x: Any, y: Any, *, lr: float = 1e-2) -> float:
        ag_zero_grad(self.parameters())
        pred = self.forward(x)
        loss = ag_mse_loss(pred, y)
        loss.backward()
        ag_sgd_step(self.parameters(), lr=float(lr))
        return float(loss.data.reshape(-1)[0])


class TinyAutogradConvAttention:
    """Tiny conv->attention training helper for autograd bootstrap v1 smoke."""

    def __init__(self, *, seed: int = 20260410):
        rng = np.random.default_rng(int(seed))
        self.w_conv = ag_parameter((rng.standard_normal((4, 3, 3, 3)) * 0.04).astype(np.float32))
        self.b_conv = ag_parameter(np.zeros((4,), dtype=np.float32))
        self.w_proj = ag_parameter((rng.standard_normal((4, 4)) * 0.05).astype(np.float32))
        self.b_proj = ag_parameter(np.zeros((1, 4), dtype=np.float32))

    def parameters(self) -> list[AutoTensor]:
        return [self.w_conv, self.b_conv, self.w_proj, self.b_proj]

    def forward(self, x: Any) -> AutoTensor:
        tx = _as_tensor(x)
        conv = ag_relu(ag_conv2d(tx, self.w_conv, self.b_conv, stride_h=1, stride_w=1, pad_h=1, pad_w=1))
        n, c, h, w = [int(v) for v in conv.data.shape]
        flat = ag_reshape(conv, (n * h * w, c))
        attn = ag_attention(flat, flat, flat, causal=False)
        proj = ag_add(ag_matmul(attn, self.w_proj), self.b_proj)
        return proj

    def train_step(self, x: Any, y: Any, *, lr: float = 1.0e-2) -> float:
        ag_zero_grad(self.parameters())
        pred = self.forward(x)
        target = _as_tensor(y)
        if pred.data.shape != target.data.shape:
            raise ValueError("TinyAutogradConvAttention target shape mismatch")
        loss = ag_mse_loss(pred, target)
        loss.backward()
        ag_sgd_step(self.parameters(), lr=float(lr))
        return float(loss.data.reshape(-1)[0])


def lightning_matmul(a: Any, b: Any, device: str = "metal") -> np.ndarray:
    arr_a = _as_f32_c(a)
    arr_b = _as_f32_c(b)
    if arr_a.ndim != 2 or arr_b.ndim != 2:
        raise ValueError("matmul expects 2D arrays")
    m, k = arr_a.shape
    kb, n = arr_b.shape
    if k != kb:
        raise ValueError("matmul shape mismatch")
    if _resolve_engine(device) == "torch":
        torch, _ = _import_torch()
        if torch is None:
            raise RuntimeError("torch backend selected but torch is unavailable")
        torch_device = _torch_device_for(device)
        ta = torch.as_tensor(arr_a, dtype=torch.float32, device=torch_device)
        tb = torch.as_tensor(arr_b, dtype=torch.float32, device=torch_device)
        tc = torch.matmul(ta, tb)
        return tc.detach().to("cpu").contiguous().numpy()

    out = np.empty((m * n,), dtype=np.float32)
    lc.matmul_np_into(arr_a, arr_b, out, m, k, n, device)
    return out.reshape(m, n)


def lightning_matmul_into(a: Any, b: Any, out: Any, device: str = "metal") -> np.ndarray:
    arr_a = _as_f32_c(a)
    arr_b = _as_f32_c(b)
    arr_out = _as_out_f32_c_no_copy(out)
    if arr_a.ndim != 2 or arr_b.ndim != 2:
        raise ValueError("matmul expects 2D arrays")
    m, k = arr_a.shape
    kb, n = arr_b.shape
    if k != kb:
        raise ValueError("matmul shape mismatch")
    if arr_out.size != m * n:
        raise ValueError("out shape mismatch")
    if _resolve_engine(device) == "torch":
        torch, _ = _import_torch()
        if torch is None:
            raise RuntimeError("torch backend selected but torch is unavailable")
        torch_device = _torch_device_for(device)
        ta = torch.as_tensor(arr_a, dtype=torch.float32, device=torch_device)
        tb = torch.as_tensor(arr_b, dtype=torch.float32, device=torch_device)
        tc = torch.matmul(ta, tb).detach().to("cpu").contiguous().numpy()
        np.copyto(arr_out.reshape(m, n), tc, casting="no")
    else:
        lc.matmul_np_into(arr_a, arr_b, arr_out.reshape(-1), m, k, n, device)
    return arr_out.reshape(m, n)


def _get_or_create_attention_session(seq: int, head_dim: int, causal: bool, device: str):
    key = (seq, head_dim, causal, device)
    with _CACHE_LOCK:
        sess = _ATTN_SESSION_CACHE.get(key)
        if sess is not None:
            _ATTN_SESSION_CACHE.move_to_end(key)
            return sess

    sess = lc.AttentionSession(seq, head_dim, causal, device)
    with _CACHE_LOCK:
        _ATTN_SESSION_CACHE[key] = sess
        _ATTN_SESSION_CACHE.move_to_end(key)
        while len(_ATTN_SESSION_CACHE) > _SESSION_CACHE_LIMIT:
            _ATTN_SESSION_CACHE.popitem(last=False)
    return sess


def clear_attention_session_cache() -> None:
    with _CACHE_LOCK:
        _ATTN_SESSION_CACHE.clear()


def _pack_qkv_repeat_from_conv(conv_out: Any, *, seq: int, head_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    need = int(seq) * int(head_dim)
    total = need * 3
    flat = np.asarray(conv_out, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise ValueError("conv output must be non-empty")
    if flat.size >= total:
        packed = np.ascontiguousarray(flat[:total], dtype=np.float32)
    else:
        packed = np.empty((total,), dtype=np.float32)
        copied = min(flat.size, total)
        np.copyto(packed[0:copied], flat[0:copied], casting="no")
        while copied < total:
            chunk = min(copied, total - copied)
            np.copyto(packed[copied : copied + chunk], packed[0:chunk], casting="no")
            copied += chunk
    q = packed[0:need].reshape(int(seq), int(head_dim))
    k = packed[need : 2 * need].reshape(int(seq), int(head_dim))
    v = packed[2 * need : 3 * need].reshape(int(seq), int(head_dim))
    return q, k, v


def _torch_conv_relu_direct(
    x_arr: np.ndarray,
    w_arr: np.ndarray,
    b_arr: np.ndarray | None,
    *,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    device: str,
) -> np.ndarray:
    torch, F = _import_torch()
    if torch is None or F is None:
        raise RuntimeError("torch backend selected but torch is unavailable")
    torch_device = _torch_device_for(device)
    tx = torch.as_tensor(x_arr, dtype=torch.float32, device=torch_device)
    tw = torch.as_tensor(w_arr, dtype=torch.float32, device=torch_device)
    tb = None if b_arr is None else torch.as_tensor(b_arr, dtype=torch.float32, device=torch_device)
    ty = F.conv2d(tx, tw, tb, stride=(int(stride_h), int(stride_w)), padding=(int(pad_h), int(pad_w)))
    ty = F.relu(ty)
    return ty.detach().to("cpu").contiguous().numpy()


def _lc_conv_relu_direct(
    x_arr: np.ndarray,
    w_arr: np.ndarray,
    b_arr: np.ndarray | None,
    *,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    device: str,
) -> np.ndarray:
    if hasattr(lc, "lightning_conv_relu_nchw"):
        return np.asarray(
            lc.lightning_conv_relu_nchw(
                x_arr, w_arr, b_arr, int(stride_h), int(stride_w), int(pad_h), int(pad_w), device
            ),
            dtype=np.float32,
        )
    y = np.asarray(
        lc.conv2d_nchw(
            x_arr, w_arr, b_arr, int(stride_h), int(stride_w), int(pad_h), int(pad_w), device
        ),
        dtype=np.float32,
    )
    np.maximum(y, 0.0, out=y)
    return y


def _torch_attention_direct(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    seq: int,
    head_dim: int,
    device: str,
    causal: bool = False,
) -> np.ndarray:
    torch, F = _import_torch()
    if torch is None or F is None:
        raise RuntimeError("torch backend selected but torch is unavailable")
    torch_device = _torch_device_for(device)
    tq = torch.as_tensor(np.asarray(q, dtype=np.float32), dtype=torch.float32, device=torch_device).reshape(
        1, 1, int(seq), int(head_dim)
    )
    tk = torch.as_tensor(np.asarray(k, dtype=np.float32), dtype=torch.float32, device=torch_device).reshape(
        1, 1, int(seq), int(head_dim)
    )
    tv = torch.as_tensor(np.asarray(v, dtype=np.float32), dtype=torch.float32, device=torch_device).reshape(
        1, 1, int(seq), int(head_dim)
    )
    if hasattr(F, "scaled_dot_product_attention"):
        out = F.scaled_dot_product_attention(tq, tk, tv, is_causal=bool(causal))
    else:
        scale = float(head_dim) ** -0.5
        scores = torch.matmul(tq, tk.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, tv)
    return out.reshape(int(seq), int(head_dim)).detach().to("cpu").contiguous().numpy()


def _lc_attention_direct(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    seq: int,
    head_dim: int,
    device: str,
    causal: bool = False,
) -> np.ndarray:
    q_flat = np.asarray(q, dtype=np.float32).reshape(-1)
    k_flat = np.asarray(k, dtype=np.float32).reshape(-1)
    v_flat = np.asarray(v, dtype=np.float32).reshape(-1)
    expected = int(seq) * int(head_dim)
    if q_flat.size != expected or k_flat.size != expected or v_flat.size != expected:
        raise ValueError("q/k/v shape mismatch")
    if hasattr(lc, "lightning_attention"):
        out_flat = np.asarray(
            lc.lightning_attention(q_flat, k_flat, v_flat, int(seq), int(head_dim), bool(causal), device),
            dtype=np.float32,
        ).reshape(-1)
        return out_flat.reshape(int(seq), int(head_dim))

    out_flat = np.empty((expected,), dtype=np.float32)
    sess = _get_or_create_attention_session(int(seq), int(head_dim), bool(causal), device)
    sess.forward_into(q_flat, k_flat, v_flat, out_flat)
    return out_flat.reshape(int(seq), int(head_dim))


def _conv_attention_route_context(
    *,
    x_arr: np.ndarray,
    w_arr: np.ndarray,
    seq: int,
    head_dim: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    requested_device: str,
    effective_device: str,
    execution_mode: str,
    route_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    policy = dict(validate_route_policy(route_policy, strict=True).get("normalized", {}))
    conv_engine, conv_note = _resolve_engine_preference(effective_device, policy["conv"])
    attention_engine, attention_note = _resolve_engine_preference(effective_device, policy["attention"])
    graph_engine, graph_note = _resolve_engine_preference(effective_device, policy["graph"])

    requested_mode = str(execution_mode).strip().lower()
    if requested_mode not in {"eager", "graph"}:
        raise ValueError("execution_mode must be 'eager' or 'graph'")

    resolved_mode = requested_mode
    fallback_reason = "none"
    graph_shape_supported = _graph_conv_attention_shape_supported(
        x_arr=x_arr,
        w_arr=w_arr,
        stride_h=int(stride_h),
        stride_w=int(stride_w),
        pad_h=int(pad_h),
        pad_w=int(pad_w),
    )
    if requested_mode == "graph":
        if graph_engine != "lightning":
            resolved_mode = "eager"
            fallback_reason = "graph_engine_not_lightning"
        elif not graph_shape_supported:
            resolved_mode = "eager"
            fallback_reason = "graph_shape_unsupported"

    out_h = (int(x_arr.shape[2]) + 2 * int(pad_h) - int(w_arr.shape[2])) // int(stride_h) + 1 if x_arr.ndim == 4 and w_arr.ndim == 4 else 0
    out_w = (int(x_arr.shape[3]) + 2 * int(pad_w) - int(w_arr.shape[3])) // int(stride_w) + 1 if x_arr.ndim == 4 and w_arr.ndim == 4 else 0
    conv_elements = max(0, int(x_arr.shape[0]) * int(w_arr.shape[0]) * max(out_h, 0) * max(out_w, 0)) if x_arr.ndim == 4 and w_arr.ndim == 4 else 0
    attn_pack_elements = max(0, 3 * int(seq) * int(head_dim))
    boundary_switches: list[str] = []
    if conv_engine != attention_engine:
        boundary_switches.append("conv_to_attention_engine_switch")
    if requested_mode == "graph" and resolved_mode != "graph":
        boundary_switches.append("graph_to_eager_switch")
    boundary_switch_count = len(boundary_switches)
    zero_copy_eligible = boundary_switch_count == 0
    if zero_copy_eligible:
        boundary_copy_mode = "zero_copy"
        boundary_reason = "none"
    elif conv_engine != attention_engine:
        boundary_copy_mode = "fallback_copy"
        boundary_reason = "interop_engine_boundary_copy"
    elif requested_mode == "graph" and resolved_mode != "graph":
        boundary_copy_mode = "fallback_copy"
        boundary_reason = "interop_graph_boundary_copy"
    else:
        boundary_copy_mode = "fallback_copy"
        boundary_reason = "interop_boundary_copy_unknown"
    copy_bytes_estimate = int(max(conv_elements, attn_pack_elements) * 4) if not zero_copy_eligible else 0
    boundary_overhead_est_ns = int((boundary_switch_count * 15000) + (copy_bytes_estimate * 0.02))

    return {
        "requested_mode": requested_mode,
        "resolved_mode": resolved_mode,
        "requested_device": str(requested_device),
        "effective_device": str(effective_device),
        "requested_conv_engine": policy["conv"],
        "requested_attention_engine": policy["attention"],
        "requested_graph_engine": policy["graph"],
        "resolved_conv_engine": conv_engine,
        "resolved_attention_engine": attention_engine,
        "resolved_graph_engine": graph_engine,
        "conv_engine_resolution": conv_note,
        "attention_engine_resolution": attention_note,
        "graph_engine_resolution": graph_note,
        "graph_shape_supported": bool(graph_shape_supported),
        "graph_fallback_reason_code": fallback_reason,
        "boundary_switch_count": int(boundary_switch_count),
        "boundary_switches": boundary_switches,
        "boundary_copy_mode": boundary_copy_mode,
        "boundary_reason_code": boundary_reason,
        "boundary_copy_bytes_estimate": int(copy_bytes_estimate),
        "boundary_overhead_est_ns": int(boundary_overhead_est_ns),
        "zero_copy_eligible": bool(zero_copy_eligible),
        "sync_boundary_required": bool(boundary_switch_count > 0),
        "seq": int(seq),
        "head_dim": int(head_dim),
        "kernel_h": int(w_arr.shape[2]) if w_arr.ndim == 4 else -1,
        "kernel_w": int(w_arr.shape[3]) if w_arr.ndim == 4 else -1,
        "stride_h": int(stride_h),
        "stride_w": int(stride_w),
        "pad_h": int(pad_h),
        "pad_w": int(pad_w),
    }


def _graph_conv_attention_python(
    *,
    x_arr: np.ndarray,
    w_arr: np.ndarray,
    b_arr: np.ndarray | None,
    seq: int,
    head_dim: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    device: str,
) -> np.ndarray | None:
    if not hasattr(lc, "GraphIR") or not hasattr(lc.GraphIR, "execute_f32"):
        return None
    if not _graph_conv_attention_shape_supported(
        x_arr=x_arr,
        w_arr=w_arr,
        stride_h=int(stride_h),
        stride_w=int(stride_w),
        pad_h=int(pad_h),
        pad_w=int(pad_w),
    ):
        return None

    out_h = (int(x_arr.shape[2]) + 2 * int(pad_h) - 3) // int(stride_h) + 1
    out_w = (int(x_arr.shape[3]) + 2 * int(pad_w) - 3) // int(stride_w) + 1
    g = lc.GraphIR()
    tx = g.add_tensor(list(x_arr.shape), dtype="float32", name="x", constant=True)
    tw = g.add_tensor(list(w_arr.shape), dtype="float32", name="w", constant=True)
    tconv = g.add_tensor([int(x_arr.shape[0]), int(w_arr.shape[0]), int(out_h), int(out_w)], dtype="float32", name="conv")
    tq = g.add_tensor([int(seq), int(head_dim)], dtype="float32", name="q")
    tk = g.add_tensor([int(seq), int(head_dim)], dtype="float32", name="k")
    tv = g.add_tensor([int(seq), int(head_dim)], dtype="float32", name="v")
    to = g.add_tensor([int(seq), int(head_dim)], dtype="float32", name="out")

    conv_inputs = [tx, tw]
    feeds = {tx: x_arr, tw: w_arr}
    if b_arr is not None:
        tb = g.add_tensor(list(b_arr.shape), dtype="float32", name="b", constant=True)
        conv_inputs.append(tb)
        feeds[tb] = b_arr

    g.add_node(
        "conv2d_nchw3x3",
        conv_inputs,
        [tconv],
        attributes={
            "stride_h": int(stride_h),
            "stride_w": int(stride_w),
            "pad_h": int(pad_h),
            "pad_w": int(pad_w),
            "apply_relu": 1,
        },
    )
    g.add_node("qkv_pack_repeat", [tconv], [tq, tk, tv])
    g.add_node("attention_forward", [tq, tk, tv], [to])
    try:
        out = g.execute_f32(feeds, preferred_device=device)
    except RuntimeError:
        return None
    values = dict(out.get("values", {}))
    if to not in values:
        return None
    return np.asarray(values[to], dtype=np.float32)


def lightning_conv_attention_torchstrong_nchw_route_report(
    x: Any,
    conv_weight: Any,
    conv_bias: Any | None,
    seq: int,
    head_dim: int,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0,
    device: str = "metal",
    execution_mode: str = "eager",
    route_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x_arr = _as_f32_c(x)
    w_arr = _as_f32_c(conv_weight)
    _ = None if conv_bias is None else np.asarray(conv_bias, dtype=np.float32)
    effective_device = (
        "cpu"
        if _should_prefer_tiny_cpu_conv_attn_chain(
            device=device,
            batch=int(x_arr.shape[0]),
            in_channels=int(x_arr.shape[1]),
            in_h=int(x_arr.shape[2]),
            in_w=int(x_arr.shape[3]),
            out_channels=int(w_arr.shape[0]),
            kernel_h=int(w_arr.shape[2]),
            kernel_w=int(w_arr.shape[3]),
            stride_h=int(stride_h),
            stride_w=int(stride_w),
            pad_h=int(pad_h),
            pad_w=int(pad_w),
            seq_len=int(seq),
            head_dim=int(head_dim),
        )
        else device
    )
    return _conv_attention_route_context(
        x_arr=x_arr,
        w_arr=w_arr,
        seq=int(seq),
        head_dim=int(head_dim),
        stride_h=int(stride_h),
        stride_w=int(stride_w),
        pad_h=int(pad_h),
        pad_w=int(pad_w),
        requested_device=str(device),
        effective_device=str(effective_device),
        execution_mode=str(execution_mode),
        route_policy=route_policy,
    )


def _torch_conv_attention_torchstrong(
    x_arr: np.ndarray,
    w_arr: np.ndarray,
    b_arr: np.ndarray | None,
    *,
    seq: int,
    head_dim: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    device: str,
) -> np.ndarray:
    torch, F = _import_torch()
    if torch is None or F is None:
        raise RuntimeError("torch backend selected but torch is unavailable")

    torch_device = _torch_device_for(device)
    tx = torch.as_tensor(x_arr, dtype=torch.float32, device=torch_device)
    tw = torch.as_tensor(w_arr, dtype=torch.float32, device=torch_device)
    tb = None if b_arr is None else torch.as_tensor(b_arr, dtype=torch.float32, device=torch_device)

    conv = F.conv2d(tx, tw, tb, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    conv = F.relu(conv)

    need = int(seq) * int(head_dim)
    total = need * 3
    flat = conv.reshape(-1)
    if int(flat.numel()) < total:
        reps = (total + int(flat.numel()) - 1) // int(flat.numel())
        flat = flat.repeat(reps)

    q = flat[0:need].reshape(1, 1, int(seq), int(head_dim))
    k = flat[need : 2 * need].reshape(1, 1, int(seq), int(head_dim))
    v = flat[2 * need : 3 * need].reshape(1, 1, int(seq), int(head_dim))

    if hasattr(F, "scaled_dot_product_attention"):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    else:
        scale = float(head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)

    return out.reshape(-1).detach().to("cpu").contiguous().numpy()


def lightning_attention(
    q: Any,
    k: Any,
    v: Any,
    seq: int,
    head_dim: int,
    device: str = "metal",
    causal: bool = False,
) -> np.ndarray:
    if _resolve_engine(device) == "torch":
        torch, F = _import_torch()
        if torch is None or F is None:
            raise RuntimeError("torch backend selected but torch is unavailable")
        seq_i = int(seq)
        head_i = int(head_dim)
        expected = seq_i * head_i

        q_arr = np.asarray(q, dtype=np.float32)
        k_arr = np.asarray(k, dtype=np.float32)
        v_arr = np.asarray(v, dtype=np.float32)
        if q_arr.shape != k_arr.shape or q_arr.shape != v_arr.shape:
            raise ValueError("q/k/v shape mismatch")

        if q_arr.ndim == 1:
            if q_arr.size != expected:
                raise ValueError("attention input shape mismatch")
            q2 = _as_f32_c(q_arr.reshape(1, seq_i, head_i))
            k2 = _as_f32_c(k_arr.reshape(1, seq_i, head_i))
            v2 = _as_f32_c(v_arr.reshape(1, seq_i, head_i))
            restore = "1d"
        elif q_arr.ndim == 2:
            if q_arr.shape != (seq_i, head_i):
                raise ValueError("attention 2D input must be [seq, head_dim]")
            q2 = _as_f32_c(q_arr.reshape(1, seq_i, head_i))
            k2 = _as_f32_c(k_arr.reshape(1, seq_i, head_i))
            v2 = _as_f32_c(v_arr.reshape(1, seq_i, head_i))
            restore = "2d"
        elif q_arr.ndim == 3:
            if q_arr.shape[1] != seq_i or q_arr.shape[2] != head_i:
                raise ValueError("attention 3D input must be [batch_heads, seq, head_dim]")
            q2 = _as_f32_c(q_arr)
            k2 = _as_f32_c(k_arr)
            v2 = _as_f32_c(v_arr)
            restore = "3d"
        else:
            raise ValueError("unsupported attention input rank")

        torch_device = _torch_device_for(device)
        tq = torch.as_tensor(q2, dtype=torch.float32, device=torch_device).unsqueeze(1)
        tk = torch.as_tensor(k2, dtype=torch.float32, device=torch_device).unsqueeze(1)
        tv = torch.as_tensor(v2, dtype=torch.float32, device=torch_device).unsqueeze(1)
        tout = F.scaled_dot_product_attention(tq, tk, tv, is_causal=bool(causal)).squeeze(1)
        out = tout.detach().to("cpu").contiguous().numpy()

        if restore == "1d":
            return out.reshape(-1)
        if restore == "2d":
            return out.reshape(seq_i, head_i)
        return out

    if hasattr(lc, "lightning_attention"):
        seq_i = int(seq)
        head_i = int(head_dim)
        expected = seq_i * head_i
        q_arr = np.asarray(q, dtype=np.float32)
        k_arr = np.asarray(k, dtype=np.float32)
        v_arr = np.asarray(v, dtype=np.float32)
        if q_arr.shape != k_arr.shape or q_arr.shape != v_arr.shape:
            raise ValueError("q/k/v shape mismatch")
        if q_arr.ndim == 1:
            if q_arr.size != expected:
                raise ValueError("attention input shape mismatch")
            restore = "1d"
        elif q_arr.ndim == 2:
            if q_arr.shape != (seq_i, head_i):
                raise ValueError("attention 2D input must be [seq, head_dim]")
            restore = "2d"
        else:
            raise ValueError("attention rank >2 requires session path; upgrade core or use non-fused fallback")
        q_flat = q_arr.reshape(-1)
        k_flat = k_arr.reshape(-1)
        v_flat = v_arr.reshape(-1)
        if q_flat.size != expected or k_flat.size != expected or v_flat.size != expected:
            raise ValueError("q/k/v shape mismatch")
        out_flat = np.asarray(
            lc.lightning_attention(q_flat, k_flat, v_flat, seq_i, head_i, bool(causal), device),
            dtype=np.float32,
        ).reshape(-1)
        if restore == "2d":
            return out_flat.reshape(seq_i, head_i)
        return out_flat

    seq_i = int(seq)
    head_i = int(head_dim)
    expected = seq_i * head_i

    q_arr = np.asarray(q, dtype=np.float32)
    k_arr = np.asarray(k, dtype=np.float32)
    v_arr = np.asarray(v, dtype=np.float32)
    if q_arr.shape != k_arr.shape or q_arr.shape != v_arr.shape:
        raise ValueError("q/k/v shape mismatch")

    if q_arr.ndim == 1:
        if q_arr.size != expected:
            raise ValueError("attention input shape mismatch")
        q2 = _as_f32_c(q_arr.reshape(1, expected))
        k2 = _as_f32_c(k_arr.reshape(1, expected))
        v2 = _as_f32_c(v_arr.reshape(1, expected))
        out2 = np.empty_like(q2)
    elif q_arr.ndim == 2:
        if q_arr.shape != (seq_i, head_i):
            raise ValueError("attention 2D input must be [seq, head_dim]")
        q2 = _as_f32_c(q_arr.reshape(1, expected))
        k2 = _as_f32_c(k_arr.reshape(1, expected))
        v2 = _as_f32_c(v_arr.reshape(1, expected))
        out2 = np.empty_like(q2)
    elif q_arr.ndim == 3:
        if q_arr.shape[1] != seq_i or q_arr.shape[2] != head_i:
            raise ValueError("attention 3D input must be [batch_heads, seq, head_dim]")
        bh = q_arr.shape[0]
        q2 = _as_f32_c(q_arr.reshape(bh, expected))
        k2 = _as_f32_c(k_arr.reshape(bh, expected))
        v2 = _as_f32_c(v_arr.reshape(bh, expected))
        out2 = np.empty_like(q2)
    else:
        raise ValueError("unsupported attention input rank")

    sess = _get_or_create_attention_session(seq_i, head_i, bool(causal), device)
    for i in range(q2.shape[0]):
        sess.forward_into(q2[i], k2[i], v2[i], out2[i])

    if q_arr.ndim == 1:
        return out2.reshape(-1)
    if q_arr.ndim == 2:
        return out2.reshape(seq_i, head_i)
    return out2.reshape(q_arr.shape[0], seq_i, head_i)


def lightning_conv_relu_nchw(
    x: Any,
    weight: Any,
    bias: Any | None,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0,
    device: str = "metal",
    out: Any | None = None,
) -> np.ndarray:
    x_arr = _as_f32_c(x)
    w_arr = _as_f32_c(weight)
    b_arr = None if bias is None else np.asarray(bias, dtype=np.float32)

    if _resolve_engine(device) == "torch":
        torch, F = _import_torch()
        if torch is None or F is None:
            raise RuntimeError("torch backend selected but torch is unavailable")
        torch_device = _torch_device_for(device)
        tx = torch.as_tensor(x_arr, dtype=torch.float32, device=torch_device)
        tw = torch.as_tensor(w_arr, dtype=torch.float32, device=torch_device)
        tb = None if b_arr is None else torch.as_tensor(b_arr, dtype=torch.float32, device=torch_device)
        ty = F.conv2d(tx, tw, tb, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
        ty = F.relu(ty)
        y_np = ty.detach().to("cpu").contiguous().numpy()
        if out is not None:
            out_arr = _as_out_f32_c_no_copy(out)
            if out_arr.shape != y_np.shape:
                raise ValueError("out shape mismatch for conv+relu torch path")
            np.copyto(out_arr, y_np, casting="no")
            return out_arr
        return y_np

    if out is not None and hasattr(lc, "lightning_conv_relu_nchw_into"):
        out_arr = _as_out_f32_c_no_copy(out)
        lc.lightning_conv_relu_nchw_into(x_arr, w_arr, b_arr, out_arr, stride_h, stride_w, pad_h, pad_w, device)
        return out_arr

    if hasattr(lc, "lightning_conv_relu_nchw"):
        y = lc.lightning_conv_relu_nchw(x_arr, w_arr, b_arr, stride_h, stride_w, pad_h, pad_w, device)
        if isinstance(y, np.ndarray) and y.dtype == np.float32:
            return y
        return np.asarray(y, dtype=np.float32)

    if out is not None and hasattr(lc, "conv2d_nchw_into"):
        out_arr = _as_out_f32_c_no_copy(out)
        lc.conv2d_nchw_into(x_arr, w_arr, b_arr, out_arr, stride_h, stride_w, pad_h, pad_w, device)
        np.maximum(out_arr, 0.0, out=out_arr)
        return out_arr

    y = lc.conv2d_nchw(x_arr, w_arr, b_arr, stride_h, stride_w, pad_h, pad_w, device)
    np.maximum(y, 0.0, out=y)
    return y


def lightning_conv_relu_torchstrong_nchw(
    x: Any,
    weight: Any,
    bias: Any | None,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0,
    device: str = "metal",
    out: Any | None = None,
    workspace_cache: dict | None = None,
    cache_key: str | None = None,
    force_into: bool | None = None,
) -> np.ndarray:
    _ = workspace_cache
    _ = cache_key
    use_into = bool(force_into) if force_into is not None else hasattr(lc, "conv2d_nchw_into")
    if use_into and out is not None and hasattr(lc, "conv2d_nchw_into"):
        return lightning_conv_relu_nchw(
            x,
            weight,
            bias,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            device,
            out=out,
        )
    return lightning_conv_relu_nchw(
        x,
        weight,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        device,
        out=out,
    )


class ConvReLUResidentBlock:
    def __init__(
        self,
        weight: Any,
        bias: Any | None,
        *,
        stride_h: int = 1,
        stride_w: int = 1,
        pad_h: int = 0,
        pad_w: int = 0,
        device: str = "metal",
        policy: str = "auto",
        cache_key_prefix: str = "conv_resident",
    ):
        self.w = _as_f32_c(weight)
        self.b = None if bias is None else np.asarray(bias, dtype=np.float32)
        self.stride_h = int(stride_h)
        self.stride_w = int(stride_w)
        self.pad_h = int(pad_h)
        self.pad_w = int(pad_w)
        self.device = str(device)
        self.policy = str(policy)
        self.cache_key_prefix = str(cache_key_prefix)
        self.workspace_cache: dict[str, np.ndarray] = {}
        self._resident_session = None
        self._resident_started = False
        self._resident_shape_key = None
        self._batch_repeat = 8

    def _ensure_resident_session(self, x_arr: np.ndarray, out_arr: np.ndarray) -> bool:
        if self.device != "metal" or not hasattr(lc, "Conv2dMetalResidentSession"):
            return False
        if x_arr.ndim != 4 or self.w.ndim != 4 or out_arr.ndim != 4:
            return False

        n, ic, ih, iw = x_arr.shape
        oc, w_ic, kh, kw = self.w.shape
        if ic != w_ic:
            return False
        shape_key = (n, ic, ih, iw, oc, kh, kw, self.stride_h, self.stride_w, self.pad_h, self.pad_w)
        if self._resident_shape_key != shape_key or self._resident_session is None:
            self._resident_session = lc.Conv2dMetalResidentSession(
                n,
                ic,
                ih,
                iw,
                oc,
                kh,
                kw,
                self.stride_h,
                self.stride_w,
                self.pad_h,
                self.pad_w,
                True,
            )
            self._resident_shape_key = shape_key
            self._resident_started = False
        return True

    def run(self, x: Any, out: Any | None = None) -> np.ndarray:
        x_arr = _as_f32_c(x)
        if out is None:
            n, _, h, w = x_arr.shape
            kh, kw = self.w.shape[2], self.w.shape[3]
            oh = (h + 2 * self.pad_h - kh) // self.stride_h + 1
            ow = (w + 2 * self.pad_w - kw) // self.stride_w + 1
            want = (n, self.w.shape[0], oh, ow)
            key = f"{self.cache_key_prefix}/out/{want}"
            out_arr = self.workspace_cache.get(key)
            if out_arr is None or out_arr.shape != want:
                out_arr = np.empty(want, dtype=np.float32)
                self.workspace_cache[key] = out_arr
        else:
            out_arr = _as_out_f32_c_no_copy(out)

        if self._ensure_resident_session(x_arr, out_arr):
            if not self._resident_started:
                self._resident_session.start_into(x_arr, self.w, self.b, out_arr)
                self._resident_started = True
            else:
                self._resident_session.run_batch_sync_into(x_arr, self.w, self.b, out_arr, self._batch_repeat)
            return out_arr

        return lightning_conv_relu_torchstrong_nchw(
            x_arr,
            self.w,
            self.b,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.device,
            out=out_arr,
            workspace_cache=self.workspace_cache,
            cache_key=self.cache_key_prefix,
            force_into=True,
        )

    def state_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        key = (str(prefix).rstrip(".") + ".") if prefix else ""
        out: dict[str, np.ndarray] = {
            f"{key}weight": np.ascontiguousarray(self.w),
            f"{key}stride_h": np.asarray([self.stride_h], dtype=np.int32),
            f"{key}stride_w": np.asarray([self.stride_w], dtype=np.int32),
            f"{key}pad_h": np.asarray([self.pad_h], dtype=np.int32),
            f"{key}pad_w": np.asarray([self.pad_w], dtype=np.int32),
            f"{key}has_bias": np.asarray([1 if self.b is not None else 0], dtype=np.int32),
        }
        if self.b is not None:
            out[f"{key}bias"] = np.ascontiguousarray(self.b)
        return out

    def load_state_dict(self, state: dict[str, Any], strict: bool = True, prefix: str = "") -> None:
        key = (str(prefix).rstrip(".") + ".") if prefix else ""
        w_key = f"{key}weight"
        b_key = f"{key}bias"
        hb_key = f"{key}has_bias"
        if w_key not in state:
            raise KeyError(f"missing key: {w_key}")
        w = np.asarray(state[w_key], dtype=np.float32)
        if w.ndim != 4:
            raise ValueError("conv weight must be 4D")
        self.w = np.ascontiguousarray(w)

        has_bias = self.b is not None
        if hb_key in state:
            hb = np.asarray(state[hb_key]).reshape(-1)
            if hb.size > 0:
                has_bias = bool(int(hb[0]))

        if has_bias:
            if b_key not in state:
                if strict:
                    raise KeyError(f"missing key: {b_key}")
                self.b = np.zeros((self.w.shape[0],), dtype=np.float32)
            else:
                b = np.asarray(state[b_key], dtype=np.float32).reshape(-1)
                if b.shape != (self.w.shape[0],):
                    raise ValueError(f"conv bias shape mismatch: expected {(self.w.shape[0],)} got {tuple(b.shape)}")
                self.b = np.ascontiguousarray(b)
        else:
            self.b = None

        for field in ("stride_h", "stride_w", "pad_h", "pad_w"):
            fk = f"{key}{field}"
            if fk in state:
                val = np.asarray(state[fk]).reshape(-1)
                if val.size > 0:
                    setattr(self, field, int(val[0]))
            elif strict:
                raise KeyError(f"missing key: {fk}")

        # Reset resident session cache after weight/config changes.
        self._resident_session = None
        self._resident_started = False
        self._resident_shape_key = None
        self.workspace_cache.clear()


class AttentionResidentBlock:
    def __init__(self, seq: int, head_dim: int, *, causal: bool = False, device: str = "metal"):
        self.seq = int(seq)
        self.head_dim = int(head_dim)
        self.causal = bool(causal)
        self.device = str(device)
        self.expected = self.seq * self.head_dim
        self._out_cache: dict[tuple[int, ...], np.ndarray] = {}

    def _as_2d(self, t: Any) -> tuple[np.ndarray, tuple[int, ...]]:
        arr = np.asarray(t, dtype=np.float32)
        original_shape = arr.shape
        if arr.ndim == 1:
            if arr.size != self.expected:
                raise ValueError("attention input shape mismatch")
            return _as_f32_c(arr.reshape(1, self.expected)), original_shape
        if arr.ndim == 2:
            if arr.shape != (self.seq, self.head_dim):
                raise ValueError("attention 2D input must be [seq, head_dim]")
            return _as_f32_c(arr.reshape(1, self.expected)), original_shape
        if arr.ndim == 3:
            if arr.shape[1] != self.seq or arr.shape[2] != self.head_dim:
                raise ValueError("attention 3D input must be [batch_heads, seq, head_dim]")
            return _as_f32_c(arr.reshape(arr.shape[0], self.expected)), original_shape
        raise ValueError("unsupported attention input rank")

    def _reshape_back(self, out2: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
        if len(original_shape) == 1:
            return out2.reshape(-1)
        if len(original_shape) == 2:
            return out2.reshape(self.seq, self.head_dim)
        return out2.reshape(original_shape[0], self.seq, self.head_dim)

    def run(self, q: Any, k: Any, v: Any) -> np.ndarray:
        q2, q_shape = self._as_2d(q)
        k2, k_shape = self._as_2d(k)
        v2, v_shape = self._as_2d(v)
        if q2.shape != k2.shape or q2.shape != v2.shape:
            raise ValueError("q/k/v shape mismatch")
        if q_shape != k_shape or q_shape != v_shape:
            raise ValueError("q/k/v shape mismatch")

        out2 = self._out_cache.get(q2.shape)
        if out2 is None:
            out2 = np.empty_like(q2)
            self._out_cache[q2.shape] = out2

        sess = _get_or_create_attention_session(self.seq, self.head_dim, self.causal, self.device)
        for i in range(q2.shape[0]):
            sess.forward_into(q2[i], k2[i], v2[i], out2[i])

        return self._reshape_back(out2, q_shape)

    def state_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        key = (str(prefix).rstrip(".") + ".") if prefix else ""
        return {
            f"{key}seq": np.asarray([self.seq], dtype=np.int32),
            f"{key}head_dim": np.asarray([self.head_dim], dtype=np.int32),
            f"{key}causal": np.asarray([1 if self.causal else 0], dtype=np.int32),
        }

    def load_state_dict(self, state: dict[str, Any], strict: bool = True, prefix: str = "") -> None:
        key = (str(prefix).rstrip(".") + ".") if prefix else ""
        required = ("seq", "head_dim", "causal")
        missing = [name for name in required if f"{key}{name}" not in state]
        if missing and strict:
            raise KeyError(f"missing keys: {missing}")

        if f"{key}seq" in state:
            self.seq = int(np.asarray(state[f"{key}seq"]).reshape(-1)[0])
        if f"{key}head_dim" in state:
            self.head_dim = int(np.asarray(state[f"{key}head_dim"]).reshape(-1)[0])
        if f"{key}causal" in state:
            self.causal = bool(int(np.asarray(state[f"{key}causal"]).reshape(-1)[0]))
        self.expected = self.seq * self.head_dim
        self._out_cache.clear()


class ConvAttentionResidentPipeline:
    def __init__(
        self,
        conv_weight: Any,
        conv_bias: Any | None,
        *,
        seq: int,
        head_dim: int,
        conv_stride_h: int = 1,
        conv_stride_w: int = 1,
        conv_pad_h: int = 0,
        conv_pad_w: int = 0,
        device: str = "metal",
        conv_policy: str = "auto",
        cache_key_prefix: str = "conv_attn_pipeline",
    ):
        self.seq = int(seq)
        self.head_dim = int(head_dim)
        self.need = self.seq * self.head_dim
        self._conv = ConvReLUResidentBlock(
            conv_weight,
            conv_bias,
            stride_h=conv_stride_h,
            stride_w=conv_stride_w,
            pad_h=conv_pad_h,
            pad_w=conv_pad_w,
            device=device,
            policy=conv_policy,
            cache_key_prefix=f"{cache_key_prefix}/conv",
        )
        self._attn = AttentionResidentBlock(self.seq, self.head_dim, causal=False, device=device)
        self._qkv_tmp = np.empty((self.need * 3,), dtype=np.float32)

    def _qkv_views(self, conv_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flat = np.asarray(conv_out, dtype=np.float32).reshape(-1)
        total = self.need * 3
        if flat.size >= total:
            return flat[0 : self.need], flat[self.need : 2 * self.need], flat[2 * self.need : 3 * self.need]
        if flat.size == 0:
            raise ValueError("conv output must be non-empty")
        copied = min(flat.size, total)
        np.copyto(self._qkv_tmp[0:copied], flat[0:copied], casting="no")
        while copied < total:
            chunk = min(copied, total - copied)
            np.copyto(self._qkv_tmp[copied : copied + chunk], self._qkv_tmp[0:chunk], casting="no")
            copied += chunk
        return (
            self._qkv_tmp[0 : self.need],
            self._qkv_tmp[self.need : 2 * self.need],
            self._qkv_tmp[2 * self.need : 3 * self.need],
        )

    def run(self, x: Any) -> np.ndarray:
        conv_out = self._conv.run(x)
        q, k, v = self._qkv_views(conv_out)
        return self._attn.run(q, k, v)

    def state_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        key = (str(prefix).rstrip(".") + ".") if prefix else ""
        out = {}
        out.update(self._conv.state_dict(prefix=f"{key}conv"))
        out.update(self._attn.state_dict(prefix=f"{key}attn"))
        out[f"{key}need"] = np.asarray([self.need], dtype=np.int32)
        return out

    def load_state_dict(self, state: dict[str, Any], strict: bool = True, prefix: str = "") -> None:
        key = (str(prefix).rstrip(".") + ".") if prefix else ""
        self._conv.load_state_dict(state, strict=strict, prefix=f"{key}conv")
        self._attn.load_state_dict(state, strict=strict, prefix=f"{key}attn")
        self.seq = self._attn.seq
        self.head_dim = self._attn.head_dim
        self.need = self.seq * self.head_dim
        self._qkv_tmp = np.empty((self.need * 3,), dtype=np.float32)


def lightning_conv_attention_torchstrong_nchw(
    x: Any,
    conv_weight: Any,
    conv_bias: Any | None,
    seq: int,
    head_dim: int,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0,
    device: str = "metal",
    conv_policy: str = "auto",
    execution_mode: str = "eager",
    *,
    pipeline_cache: dict | None = None,
    cache_key: str | None = None,
    route_policy: dict[str, Any] | None = None,
) -> np.ndarray:
    mode = str(execution_mode).strip().lower()
    if mode not in {"eager", "graph"}:
        raise ValueError("execution_mode must be 'eager' or 'graph'")

    x_arr = _as_f32_c(x)
    w_arr = _as_f32_c(conv_weight)
    b_arr = None if conv_bias is None else np.asarray(conv_bias, dtype=np.float32)
    effective_device = (
        "cpu"
        if _should_prefer_tiny_cpu_conv_attn_chain(
            device=device,
            batch=int(x_arr.shape[0]),
            in_channels=int(x_arr.shape[1]),
            in_h=int(x_arr.shape[2]),
            in_w=int(x_arr.shape[3]),
            out_channels=int(w_arr.shape[0]),
            kernel_h=int(w_arr.shape[2]),
            kernel_w=int(w_arr.shape[3]),
            stride_h=int(stride_h),
            stride_w=int(stride_w),
            pad_h=int(pad_h),
            pad_w=int(pad_w),
            seq_len=int(seq),
            head_dim=int(head_dim),
        )
        else device
    )

    route = _conv_attention_route_context(
        x_arr=x_arr,
        w_arr=w_arr,
        seq=int(seq),
        head_dim=int(head_dim),
        stride_h=int(stride_h),
        stride_w=int(stride_w),
        pad_h=int(pad_h),
        pad_w=int(pad_w),
        requested_device=str(device),
        effective_device=str(effective_device),
        execution_mode=mode,
        route_policy=route_policy,
    )

    key = cache_key or (
        f"conv_attn/{x_arr.shape[1]}_{w_arr.shape[0]}_{w_arr.shape[2]}_{w_arr.shape[3]}_"
        f"{seq}_{head_dim}_{stride_h}_{stride_w}_{pad_h}_{pad_w}_{conv_policy}_{effective_device}_"
        f"{route['resolved_mode']}_{route['resolved_conv_engine']}_{route['resolved_attention_engine']}"
    )

    def _run_eager_hybrid() -> np.ndarray:
        conv_engine = str(route["resolved_conv_engine"])
        attn_engine = str(route["resolved_attention_engine"])

        if conv_engine == "torch" and attn_engine == "torch":
            return _torch_conv_attention_torchstrong(
                x_arr,
                w_arr,
                b_arr,
                seq=int(seq),
                head_dim=int(head_dim),
                stride_h=int(stride_h),
                stride_w=int(stride_w),
                pad_h=int(pad_h),
                pad_w=int(pad_w),
                device=str(effective_device),
            )

        if conv_engine == "lightning" and attn_engine == "lightning":
            if hasattr(lc, "lightning_conv_attention_torchstrong_nchw"):
                try:
                    return np.asarray(
                        lc.lightning_conv_attention_torchstrong_nchw(
                            x_arr,
                            w_arr,
                            b_arr,
                            int(seq),
                            int(head_dim),
                            int(stride_h),
                            int(stride_w),
                            int(pad_h),
                            int(pad_w),
                            str(effective_device),
                            "eager",
                        ),
                        dtype=np.float32,
                    )
                except TypeError:
                    return np.asarray(
                        lc.lightning_conv_attention_torchstrong_nchw(
                            x_arr,
                            w_arr,
                            b_arr,
                            int(seq),
                            int(head_dim),
                            int(stride_h),
                            int(stride_w),
                            int(pad_h),
                            int(pad_w),
                            str(effective_device),
                        ),
                        dtype=np.float32,
                    )
                except RuntimeError:
                    pass

            if pipeline_cache is not None:
                pipe = pipeline_cache.get(key)
                if pipe is None:
                    pipe = ConvAttentionResidentPipeline(
                        w_arr,
                        b_arr,
                        seq=seq,
                        head_dim=head_dim,
                        conv_stride_h=stride_h,
                        conv_stride_w=stride_w,
                        conv_pad_h=pad_h,
                        conv_pad_w=pad_w,
                        device=str(effective_device),
                        conv_policy=conv_policy,
                        cache_key_prefix=key,
                    )
                    pipeline_cache[key] = pipe
                return pipe.run(x_arr)

            pipe = ConvAttentionResidentPipeline(
                w_arr,
                b_arr,
                seq=seq,
                head_dim=head_dim,
                conv_stride_h=stride_h,
                conv_stride_w=stride_w,
                conv_pad_h=pad_h,
                conv_pad_w=pad_w,
                device=str(effective_device),
                conv_policy=conv_policy,
                cache_key_prefix=key,
            )
            return pipe.run(x_arr)

        conv_out = (
            _torch_conv_relu_direct(
                x_arr,
                w_arr,
                b_arr,
                stride_h=int(stride_h),
                stride_w=int(stride_w),
                pad_h=int(pad_h),
                pad_w=int(pad_w),
                device=str(effective_device),
            )
            if conv_engine == "torch"
            else _lc_conv_relu_direct(
                x_arr,
                w_arr,
                b_arr,
                stride_h=int(stride_h),
                stride_w=int(stride_w),
                pad_h=int(pad_h),
                pad_w=int(pad_w),
                device=str(effective_device),
            )
        )
        q, k, v = _pack_qkv_repeat_from_conv(conv_out, seq=int(seq), head_dim=int(head_dim))
        if attn_engine == "torch":
            return _torch_attention_direct(
                q,
                k,
                v,
                seq=int(seq),
                head_dim=int(head_dim),
                device=str(effective_device),
                causal=False,
            )
        return _lc_attention_direct(
            q,
            k,
            v,
            seq=int(seq),
            head_dim=int(head_dim),
            device=str(effective_device),
            causal=False,
        )

    if str(route["resolved_mode"]) == "graph":
        if hasattr(lc, "lightning_conv_attention_torchstrong_nchw"):
            try:
                return np.asarray(
                    lc.lightning_conv_attention_torchstrong_nchw(
                        x_arr,
                        w_arr,
                        b_arr,
                        int(seq),
                        int(head_dim),
                        int(stride_h),
                        int(stride_w),
                        int(pad_h),
                        int(pad_w),
                        str(effective_device),
                        "graph",
                    ),
                    dtype=np.float32,
                )
            except TypeError:
                # Old build with eager-only signature: continue with Python graph path.
                pass
            except RuntimeError:
                # Runtime graph rejection: deterministic eager fallback.
                pass

        graph_out = _graph_conv_attention_python(
            x_arr=x_arr,
            w_arr=w_arr,
            b_arr=b_arr,
            seq=int(seq),
            head_dim=int(head_dim),
            stride_h=int(stride_h),
            stride_w=int(stride_w),
            pad_h=int(pad_h),
            pad_w=int(pad_w),
            device=str(effective_device),
        )
        if graph_out is not None:
            return np.asarray(graph_out, dtype=np.float32)

    return _run_eager_hybrid()


def lightning_conv_attention_torchstrong_nchw_ab_report(
    x: Any,
    conv_weight: Any,
    conv_bias: Any | None,
    seq: int,
    head_dim: int,
    stride_h: int = 1,
    stride_w: int = 1,
    pad_h: int = 0,
    pad_w: int = 0,
    device: str = "metal",
    conv_policy: str = "auto",
    route_policy: dict[str, Any] | None = None,
    warmup: int = 1,
    repeat: int = 5,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    warmup_i = int(warmup)
    repeat_i = int(repeat)
    atol_f = float(atol)
    rtol_f = float(rtol)
    if warmup_i < 0:
        raise ValueError("warmup must be >= 0")
    if repeat_i <= 0:
        raise ValueError("repeat must be > 0")
    if atol_f < 0.0 or rtol_f < 0.0:
        raise ValueError("atol/rtol must be >= 0")

    engine = _resolve_engine(device)
    if route_policy is None and engine == "lightning" and hasattr(lc, "lightning_conv_attention_torchstrong_nchw_ab_report"):
        out = lc.lightning_conv_attention_torchstrong_nchw_ab_report(
            x,
            conv_weight,
            conv_bias,
            int(seq),
            int(head_dim),
            int(stride_h),
            int(stride_w),
            int(pad_h),
            int(pad_w),
            device,
            warmup_i,
            repeat_i,
            atol_f,
            rtol_f,
        )
        return dict(out)

    x_arr = _as_f32_c(x)
    w_arr = _as_f32_c(conv_weight)
    b_arr = None if conv_bias is None else np.asarray(conv_bias, dtype=np.float32)

    def _run(mode: str) -> tuple[np.ndarray, float]:
        for _ in range(warmup_i):
            lightning_conv_attention_torchstrong_nchw(
                x_arr,
                w_arr,
                b_arr,
                seq=int(seq),
                head_dim=int(head_dim),
                stride_h=int(stride_h),
                stride_w=int(stride_w),
                pad_h=int(pad_h),
                pad_w=int(pad_w),
                device=device,
                conv_policy=conv_policy,
                execution_mode=mode,
                route_policy=route_policy,
            )
        t0 = time.perf_counter()
        out_arr = None
        for _ in range(repeat_i):
            out_arr = lightning_conv_attention_torchstrong_nchw(
                x_arr,
                w_arr,
                b_arr,
                seq=int(seq),
                head_dim=int(head_dim),
                stride_h=int(stride_h),
                stride_w=int(stride_w),
                pad_h=int(pad_h),
                pad_w=int(pad_w),
                device=device,
                conv_policy=conv_policy,
                execution_mode=mode,
                route_policy=route_policy,
            )
        t1 = time.perf_counter()
        assert out_arr is not None
        return np.asarray(out_arr, dtype=np.float32).reshape(-1), ((t1 - t0) * 1000.0) / float(repeat_i)

    eager_route = lightning_conv_attention_torchstrong_nchw_route_report(
        x_arr,
        w_arr,
        b_arr,
        seq=int(seq),
        head_dim=int(head_dim),
        stride_h=int(stride_h),
        stride_w=int(stride_w),
        pad_h=int(pad_h),
        pad_w=int(pad_w),
        device=device,
        execution_mode="eager",
        route_policy=route_policy,
    )
    graph_route = lightning_conv_attention_torchstrong_nchw_route_report(
        x_arr,
        w_arr,
        b_arr,
        seq=int(seq),
        head_dim=int(head_dim),
        stride_h=int(stride_h),
        stride_w=int(stride_w),
        pad_h=int(pad_h),
        pad_w=int(pad_w),
        device=device,
        execution_mode="graph",
        route_policy=route_policy,
    )

    eager_out, eager_ms = _run("eager")
    graph_out, graph_ms = _run("graph")

    abs_diff = np.abs(eager_out - graph_out)
    max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
    mean_abs = float(np.mean(abs_diff)) if abs_diff.size else 0.0
    rel_diff = abs_diff / (np.abs(eager_out) + 1e-12)
    max_rel = float(np.max(rel_diff)) if rel_diff.size else 0.0
    allclose = bool(np.all(abs_diff <= (atol_f + rtol_f * np.abs(eager_out))))

    winner = "tie"
    if graph_ms < eager_ms:
        winner = "graph"
    elif eager_ms < graph_ms:
        winner = "eager"

    return {
        "seq_len": int(seq),
        "head_dim": int(head_dim),
        "warmup": warmup_i,
        "repeat": repeat_i,
        "device": device,
        "allclose": allclose,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "max_rel_diff": max_rel,
        "atol": atol_f,
        "rtol": rtol_f,
        "eager_ms": float(eager_ms),
        "graph_ms": float(graph_ms),
        "graph_over_eager": float(graph_ms / eager_ms) if eager_ms > 0.0 else 0.0,
        "eager_over_graph": float(eager_ms / graph_ms) if graph_ms > 0.0 else 0.0,
        "winner": winner,
        "eager_route": eager_route,
        "graph_route": graph_route,
        "graph_fallback_reason_code": str(graph_route.get("graph_fallback_reason_code", "none")),
    }


def _install_lc_api_engine_bridge() -> bool:
    """Install engine-aware wrappers on lc.api while preserving direct LC callsites."""
    global _LC_API_BRIDGE_INSTALLED
    api = getattr(lc, "api", None)
    if api is None:
        return False

    with _LC_API_BRIDGE_LOCK:
        if _LC_API_BRIDGE_INSTALLED:
            return True

        direct_names = (
            "clear_attention_session_cache",
            "conv_relu_nchw",
            "conv_relu_nchw_into",
            "attention",
            "attention_into",
            "conv_attention_torchstrong_nchw",
            "conv_attention_torchstrong_nchw_into",
            "conv_attention_torchstrong_nchw_ab_report",
            "conv_attention_torchstrong_nchw_route_report",
        )
        for name in direct_names:
            fn = getattr(api, name, None)
            if callable(fn):
                _LC_API_DIRECT_EXPORTS[name] = fn
                direct_alias = f"{name}_lightning_direct"
                if not hasattr(api, direct_alias):
                    setattr(api, direct_alias, fn)

        def _api_set_engine(name: str) -> None:
            set_backend(name)

        def _api_get_engine() -> str:
            return get_backend()

        api.set_engine = _api_set_engine
        api.get_engine = _api_get_engine
        api.set_backend = _api_set_engine
        api.get_backend = _api_get_engine

        def _api_clear_attention_session_cache() -> None:
            clear_attention_session_cache()
            direct_clear = _LC_API_DIRECT_EXPORTS.get("clear_attention_session_cache")
            if callable(direct_clear):
                direct_clear()

        api.clear_attention_session_cache = _api_clear_attention_session_cache
        api.save_checkpoint = save_checkpoint
        api.load_checkpoint = load_checkpoint
        api.validate_checkpoint = validate_checkpoint
        api.checkpoint_conversion_diagnostics = checkpoint_conversion_diagnostics
        api.validate_checkpoint_conversion = validate_checkpoint_conversion
        api.save_model_checkpoint = save_model_checkpoint
        api.load_model_checkpoint = load_model_checkpoint
        api.validate_route_policy = validate_route_policy
        api.TinyTransformerRunner = TinyTransformerRunner
        api.create_torch_module_wrapper = create_torch_module_wrapper
        api.create_tf_keras_layer_wrapper = create_tf_keras_layer_wrapper

        def _api_conv_relu_nchw(
            x: Any,
            w: Any,
            bias: Any | None = None,
            stride_h: int = 1,
            stride_w: int = 1,
            pad_h: int = 0,
            pad_w: int = 0,
            device: str = "metal",
        ) -> np.ndarray:
            return lightning_conv_relu_nchw(
                x,
                w,
                bias,
                stride_h=stride_h,
                stride_w=stride_w,
                pad_h=pad_h,
                pad_w=pad_w,
                device=device,
            )

        def _api_conv_relu_nchw_into(
            x: Any,
            w: Any,
            bias: Any | None,
            out: Any,
            stride_h: int = 1,
            stride_w: int = 1,
            pad_h: int = 0,
            pad_w: int = 0,
            device: str = "metal",
        ) -> None:
            lightning_conv_relu_nchw(
                x,
                w,
                bias,
                stride_h=stride_h,
                stride_w=stride_w,
                pad_h=pad_h,
                pad_w=pad_w,
                device=device,
                out=out,
            )

        def _api_attention(
            q: Any,
            k: Any,
            v: Any,
            seq_len: int,
            head_dim: int,
            causal: bool = False,
            device: str = "metal",
        ) -> np.ndarray:
            return lightning_attention(
                q,
                k,
                v,
                seq=int(seq_len),
                head_dim=int(head_dim),
                device=device,
                causal=bool(causal),
            )

        def _api_attention_into(
            q: Any,
            k: Any,
            v: Any,
            out: Any,
            seq_len: int,
            head_dim: int,
            causal: bool = False,
            device: str = "metal",
        ) -> None:
            out_arr = _as_out_f32_c_no_copy(out).reshape(-1)
            result = np.asarray(
                lightning_attention(
                    q,
                    k,
                    v,
                    seq=int(seq_len),
                    head_dim=int(head_dim),
                    device=device,
                    causal=bool(causal),
                ),
                dtype=np.float32,
            ).reshape(-1)
            if out_arr.size != result.size:
                raise ValueError("out shape mismatch for attention_into")
            np.copyto(out_arr, result, casting="no")

        def _api_conv_attention_torchstrong_nchw(
            x: Any,
            w: Any,
            bias: Any | None,
            seq_len: int,
            head_dim: int,
            stride_h: int = 1,
            stride_w: int = 1,
            pad_h: int = 0,
            pad_w: int = 0,
            device: str = "metal",
            execution_mode: str = "eager",
            route_policy: dict[str, Any] | None = None,
        ) -> np.ndarray:
            return lightning_conv_attention_torchstrong_nchw(
                x,
                w,
                bias,
                seq=int(seq_len),
                head_dim=int(head_dim),
                stride_h=int(stride_h),
                stride_w=int(stride_w),
                pad_h=int(pad_h),
                pad_w=int(pad_w),
                device=device,
                execution_mode=execution_mode,
                route_policy=route_policy,
            )

        def _api_conv_attention_torchstrong_nchw_into(
            x: Any,
            w: Any,
            bias: Any | None,
            out: Any,
            seq_len: int,
            head_dim: int,
            stride_h: int = 1,
            stride_w: int = 1,
            pad_h: int = 0,
            pad_w: int = 0,
            device: str = "metal",
            execution_mode: str = "eager",
            route_policy: dict[str, Any] | None = None,
        ) -> None:
            out_arr = _as_out_f32_c_no_copy(out).reshape(-1)
            result = np.asarray(
                lightning_conv_attention_torchstrong_nchw(
                    x,
                    w,
                    bias,
                    seq=int(seq_len),
                    head_dim=int(head_dim),
                    stride_h=int(stride_h),
                    stride_w=int(stride_w),
                    pad_h=int(pad_h),
                    pad_w=int(pad_w),
                    device=device,
                    execution_mode=execution_mode,
                    route_policy=route_policy,
                ),
                dtype=np.float32,
            ).reshape(-1)
            if out_arr.size != result.size:
                raise ValueError("out shape mismatch for conv_attention_torchstrong_nchw_into")
            np.copyto(out_arr, result, casting="no")

        def _api_conv_attention_torchstrong_nchw_ab_report(
            x: Any,
            w: Any,
            bias: Any | None,
            seq_len: int,
            head_dim: int,
            stride_h: int = 1,
            stride_w: int = 1,
            pad_h: int = 0,
            pad_w: int = 0,
            device: str = "metal",
            route_policy: dict[str, Any] | None = None,
            warmup: int = 1,
            repeat: int = 5,
            atol: float = 1e-4,
            rtol: float = 1e-4,
        ) -> dict[str, Any]:
            return lightning_conv_attention_torchstrong_nchw_ab_report(
                x,
                w,
                bias,
                seq=int(seq_len),
                head_dim=int(head_dim),
                stride_h=int(stride_h),
                stride_w=int(stride_w),
                pad_h=int(pad_h),
                pad_w=int(pad_w),
                device=device,
                route_policy=route_policy,
                warmup=int(warmup),
                repeat=int(repeat),
                atol=float(atol),
                rtol=float(rtol),
            )

        def _api_conv_attention_torchstrong_nchw_route_report(
            x: Any,
            w: Any,
            bias: Any | None,
            seq_len: int,
            head_dim: int,
            stride_h: int = 1,
            stride_w: int = 1,
            pad_h: int = 0,
            pad_w: int = 0,
            device: str = "metal",
            execution_mode: str = "eager",
            route_policy: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            return lightning_conv_attention_torchstrong_nchw_route_report(
                x,
                w,
                bias,
                seq=int(seq_len),
                head_dim=int(head_dim),
                stride_h=int(stride_h),
                stride_w=int(stride_w),
                pad_h=int(pad_h),
                pad_w=int(pad_w),
                device=device,
                execution_mode=execution_mode,
                route_policy=route_policy,
            )

        api.conv_relu_nchw = _api_conv_relu_nchw
        api.conv_relu_nchw_into = _api_conv_relu_nchw_into
        api.attention = _api_attention
        api.attention_into = _api_attention_into
        api.conv_attention_torchstrong_nchw = _api_conv_attention_torchstrong_nchw
        api.conv_attention_torchstrong_nchw_into = _api_conv_attention_torchstrong_nchw_into
        api.conv_attention_torchstrong_nchw_ab_report = _api_conv_attention_torchstrong_nchw_ab_report
        api.conv_attention_torchstrong_nchw_route_report = _api_conv_attention_torchstrong_nchw_route_report

        _LC_API_BRIDGE_INSTALLED = True
    return True


_install_lc_api_engine_bridge()
