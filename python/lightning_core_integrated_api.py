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
    # Integrated mode always targets lightning_core directly.
    _ = name


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


def lightning_attention(q: Any, k: Any, v: Any, seq: int, head_dim: int, device: str = "metal") -> np.ndarray:
    if hasattr(lc, "lightning_attention"):
        seq_i = int(seq)
        head_i = int(head_dim)
        expected = seq_i * head_i
        q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
        k_arr = np.asarray(k, dtype=np.float32).reshape(-1)
        v_arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if q_arr.size != expected or k_arr.size != expected or v_arr.size != expected:
            raise ValueError("q/k/v shape mismatch")
        return lc.lightning_attention(q_arr, k_arr, v_arr, seq_i, head_i, False, device)

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

    sess = _get_or_create_attention_session(seq_i, head_i, False, device)
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

    if hasattr(lc, "lightning_conv_attention_torchstrong_nchw"):
        try:
            return np.asarray(
                lc.lightning_conv_attention_torchstrong_nchw(
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
                    mode,
                ),
                dtype=np.float32,
            )
        except TypeError:
            if mode != "eager":
                raise RuntimeError("graph execution_mode requires a newer lightning_core build") from None
            return np.asarray(
                lc.lightning_conv_attention_torchstrong_nchw(
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
                ),
                dtype=np.float32,
            )

    x_arr = _as_f32_c(x)
    w_arr = _as_f32_c(conv_weight)
    b_arr = None if conv_bias is None else np.asarray(conv_bias, dtype=np.float32)
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
            conv_res = conv_g.execute_f32({tx: x_arr, tw: w_arr, tb: b_arr}, preferred_device=device)
        else:
            conv_g.add_node("conv2d_nchw3x3s1p1", [tx, tw], [tconv])
            conv_res = conv_g.execute_f32({tx: x_arr, tw: w_arr}, preferred_device=device)

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
        out_dict = attn_g.execute_f32({tq: q, tk: k, tv: v}, preferred_device=device)
        return np.asarray(out_dict["values"][to], dtype=np.float32)

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

    if hasattr(lc, "lightning_conv_attention_torchstrong_nchw_ab_report"):
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
