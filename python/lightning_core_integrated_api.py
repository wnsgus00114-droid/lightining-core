from __future__ import annotations

from collections import OrderedDict
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
_LC_API_BRIDGE_LOCK = Lock()
_LC_API_BRIDGE_INSTALLED = False
_LC_API_DIRECT_EXPORTS: dict[str, Any] = {}


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

        reps = (total + flat.size - 1) // flat.size
        tiled = np.tile(flat, reps)
        np.copyto(self._qkv_tmp, tiled[0:total])
        return (
            self._qkv_tmp[0 : self.need],
            self._qkv_tmp[self.need : 2 * self.need],
            self._qkv_tmp[2 * self.need : 3 * self.need],
        )

    def run(self, x: Any) -> np.ndarray:
        conv_out = self._conv.run(x)
        q, k, v = self._qkv_views(conv_out)
        return self._attn.run(q, k, v)


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
) -> np.ndarray:
    mode = str(execution_mode)
    if mode not in {"eager", "graph"}:
        raise ValueError("execution_mode must be 'eager' or 'graph'")

    x_arr = _as_f32_c(x)
    w_arr = _as_f32_c(conv_weight)
    b_arr = None if conv_bias is None else np.asarray(conv_bias, dtype=np.float32)

    # Hybrid execution policy:
    # - graph mode is currently lightning-core GraphIR only.
    # - torch backend always executes eager conv->attn path (graph request falls back deterministically).
    engine = _resolve_engine(device)
    if engine == "torch":
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
            device=device,
        )

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
                    device,
                    mode,
                ),
                dtype=np.float32,
            )
        except TypeError:
            if mode != "eager":
                raise RuntimeError("graph execution_mode requires a newer lightning_core build") from None
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
                    device,
                ),
                dtype=np.float32,
            )
        except RuntimeError:
            # Some builds expose the fused entrypoint but reject graph mode on specific
            # device/shape combinations at runtime. Fall through to Python graph/eager
            # implementation so execution policy remains deterministic.
            pass

    key = cache_key or (
        f"conv_attn/{x_arr.shape[1]}_{w_arr.shape[0]}_{w_arr.shape[2]}_{w_arr.shape[3]}_"
        f"{seq}_{head_dim}_{stride_h}_{stride_w}_{pad_h}_{pad_w}_{conv_policy}_{device}_{mode}"
    )

    if mode == "graph":
        if not hasattr(lc, "GraphIR") or not hasattr(lc.GraphIR, "execute_f32"):
            raise RuntimeError("graph execution_mode requires GraphIR.execute_f32 support")
        if stride_h != 1 or stride_w != 1 or pad_h != 1 or pad_w != 1:
            raise ValueError("graph execution_mode currently supports stride=1,pad=1")
        if w_arr.ndim != 4 or w_arr.shape[2] != 3 or w_arr.shape[3] != 3:
            raise ValueError("graph execution_mode currently supports 3x3 conv weights")

        conv_g = lc.GraphIR()
        tx = conv_g.add_tensor(list(x_arr.shape), dtype="float32", name="x", constant=True)
        tw = conv_g.add_tensor(list(w_arr.shape), dtype="float32", name="w", constant=True)
        out_h = (x_arr.shape[2] + 2 * int(pad_h) - 3) // int(stride_h) + 1
        out_w = (x_arr.shape[3] + 2 * int(pad_w) - 3) // int(stride_w) + 1
        tconv = conv_g.add_tensor([int(x_arr.shape[0]), int(w_arr.shape[0]), int(out_h), int(out_w)], dtype="float32", name="conv")
        if b_arr is not None:
            tb = conv_g.add_tensor(list(b_arr.shape), dtype="float32", name="b", constant=True)
            conv_g.add_node("conv2d_nchw3x3s1p1", [tx, tw, tb], [tconv])
            try:
                conv_res = conv_g.execute_f32({tx: x_arr, tw: w_arr, tb: b_arr}, preferred_device=device)
            except RuntimeError:
                conv_res = None
        else:
            conv_g.add_node("conv2d_nchw3x3s1p1", [tx, tw], [tconv])
            try:
                conv_res = conv_g.execute_f32({tx: x_arr, tw: w_arr}, preferred_device=device)
            except RuntimeError:
                conv_res = None

        if conv_res is None:
            mode = "eager"
        else:
            conv_flat = np.asarray(conv_res["values"][tconv], dtype=np.float32).reshape(-1)
            need = int(seq) * int(head_dim)
            total = need * 3
            if conv_flat.size >= total:
                packed = conv_flat[:total].copy()
            else:
                reps = (total + conv_flat.size - 1) // conv_flat.size
                packed = np.tile(conv_flat, reps)[:total].astype(np.float32, copy=False)
            q = packed[0:need]
            k = packed[need : 2 * need]
            v = packed[2 * need : 3 * need]

            attn_g = lc.GraphIR()
            tq = attn_g.add_tensor([int(seq), int(head_dim)], dtype="float32", name="q", constant=True)
            tk = attn_g.add_tensor([int(seq), int(head_dim)], dtype="float32", name="k", constant=True)
            tv = attn_g.add_tensor([int(seq), int(head_dim)], dtype="float32", name="v", constant=True)
            to = attn_g.add_tensor([int(seq), int(head_dim)], dtype="float32", name="out")
            attn_g.add_node("attention_forward", [tq, tk, tv], [to])
            try:
                out_dict = attn_g.execute_f32({tq: q, tk: k, tv: v}, preferred_device=device)
            except RuntimeError:
                out_dict = None
            if out_dict is not None:
                return np.asarray(out_dict["values"][to], dtype=np.float32)
            mode = "eager"

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
                device=device,
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
        device=device,
        conv_policy=conv_policy,
        cache_key_prefix=key,
    )
    return pipe.run(x_arr)


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
    if engine == "lightning" and hasattr(lc, "lightning_conv_attention_torchstrong_nchw_ab_report"):
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
            )
        t1 = time.perf_counter()
        assert out_arr is not None
        return np.asarray(out_arr, dtype=np.float32).reshape(-1), ((t1 - t0) * 1000.0) / float(repeat_i)

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
                warmup=int(warmup),
                repeat=int(repeat),
                atol=float(atol),
                rtol=float(rtol),
            )

        api.conv_relu_nchw = _api_conv_relu_nchw
        api.conv_relu_nchw_into = _api_conv_relu_nchw_into
        api.attention = _api_attention
        api.attention_into = _api_attention_into
        api.conv_attention_torchstrong_nchw = _api_conv_attention_torchstrong_nchw
        api.conv_attention_torchstrong_nchw_into = _api_conv_attention_torchstrong_nchw_into
        api.conv_attention_torchstrong_nchw_ab_report = _api_conv_attention_torchstrong_nchw_ab_report

        _LC_API_BRIDGE_INSTALLED = True
    return True


_install_lc_api_engine_bridge()
