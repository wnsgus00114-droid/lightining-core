#!/usr/bin/env python3
"""Python stress smoke: graph fallback contract + numerical boundary checks."""

from __future__ import annotations

import numpy as np

import lightning_core as lc
import lightning_core_integrated_api as lc_api


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _device() -> str:
    name = str(lc.backend_name()).lower()
    if name in {"metal", "cpu", "cuda"}:
        return name
    return "cpu"


def _graph_mm_add_f32_case(
    *,
    m: int,
    k: int,
    n: int,
    sync_mode: str,
    device: str,
    rng: np.random.Generator,
) -> None:
    a = (rng.random((m, k), dtype=np.float32) * 2.0) - 1.0
    b = (rng.random((k, n), dtype=np.float32) * 2.0) - 1.0
    bias = (rng.random((m, n), dtype=np.float32) * 0.2) - 0.1

    g = lc.GraphIR()
    ta = g.add_tensor([m, k], dtype="float32", name="a", constant=True)
    tb = g.add_tensor([k, n], dtype="float32", name="b", constant=True)
    tbias = g.add_tensor([m, n], dtype="float32", name="bias", constant=True)
    tmm = g.add_tensor([m, n], dtype="float32", name="mm")
    tout = g.add_tensor([m, n], dtype="float32", name="out")
    g.add_node("matmul", [ta, tb], [tmm])
    g.add_node("vector_add", [tmm, tbias], [tout])

    result = dict(
        g.execute_f32(
            {ta: a, tb: b, tbias: bias},
            preferred_device=device,
            sync_mode=sync_mode,
            trace_sync_boundary=(sync_mode == "always"),
            enable_fusion_v1=False,
        )
    )
    graph_out = np.asarray(result["values"][tout], dtype=np.float32).reshape(m, n)

    mm = np.asarray(lc.matmul2d(a, b, device), dtype=np.float32).reshape(-1)
    eager_out = np.asarray(lc.vector_add(mm, bias.reshape(-1), device), dtype=np.float32).reshape(m, n)
    _require(np.allclose(graph_out, eager_out, atol=1.0e-4, rtol=1.0e-4), "f32 graph/eager mismatch")

    summary = dict(result.get("summary", {}))
    _require(int(summary.get("total_nodes", 0)) == 2, "plan summary total_nodes mismatch")
    _require(int(summary.get("planned_dispatch_groups", 0)) >= 1, "planned_dispatch_groups must be >=1")


def _graph_mm_add_f64_case(*, m: int, k: int, n: int, rng: np.random.Generator) -> None:
    a = (rng.random((m, k)) * 2.0 - 1.0).astype(np.float64)
    b = (rng.random((k, n)) * 2.0 - 1.0).astype(np.float64)
    bias = (rng.random((m, n)) * 0.2 - 0.1).astype(np.float64)

    g = lc.GraphIR()
    ta = g.add_tensor([m, k], dtype="float64", name="a", constant=True)
    tb = g.add_tensor([k, n], dtype="float64", name="b", constant=True)
    tbias = g.add_tensor([m, n], dtype="float64", name="bias", constant=True)
    tmm = g.add_tensor([m, n], dtype="float64", name="mm")
    tout = g.add_tensor([m, n], dtype="float64", name="out")
    g.add_node("matmul", [ta, tb], [tmm])
    g.add_node("vector_add", [tmm, tbias], [tout])

    out = dict(
        g.execute_f64(
            {ta: a, tb: b, tbias: bias},
            preferred_device="cpu",
            sync_mode="auto",
            enable_fusion_v1=False,
        )
    )
    graph_out = np.asarray(out["values"][tout], dtype=np.float64).reshape(m, n)
    ref = (a @ b) + bias
    _require(np.allclose(graph_out, ref, atol=1.0e-8, rtol=1.0e-8), "f64 graph/numpy mismatch")


def _graph_noncontiguous_layout_case(*, m: int, k: int, n: int, device: str, rng: np.random.Generator) -> None:
    a_full = (rng.random((m * 2, k * 2), dtype=np.float32) * 2.0) - 1.0
    b_full = (rng.random((k * 2, n * 2), dtype=np.float32) * 2.0) - 1.0
    bias_full = (rng.random((m * 2, n * 2), dtype=np.float32) * 0.2) - 0.1
    a = a_full[::2, ::2]
    b = b_full[::2, ::2]
    bias = bias_full[::2, ::2]
    _require(not a.flags.c_contiguous, "layout stress input should be non-contiguous")
    _require(not b.flags.c_contiguous, "layout stress input should be non-contiguous")

    g = lc.GraphIR()
    ta = g.add_tensor([m, k], dtype="float32", name="a", constant=True)
    tb = g.add_tensor([k, n], dtype="float32", name="b", constant=True)
    tbias = g.add_tensor([m, n], dtype="float32", name="bias", constant=True)
    tmm = g.add_tensor([m, n], dtype="float32", name="mm")
    tout = g.add_tensor([m, n], dtype="float32", name="out")
    g.add_node("matmul", [ta, tb], [tmm])
    g.add_node("vector_add", [tmm, tbias], [tout])

    out = dict(
        g.execute_f32(
            {ta: a, tb: b, tbias: bias},
            preferred_device=device,
            sync_mode="auto",
            enable_fusion_v1=False,
        )
    )
    graph_out = np.asarray(out["values"][tout], dtype=np.float32).reshape(m, n)
    ref = (np.ascontiguousarray(a) @ np.ascontiguousarray(b)) + np.ascontiguousarray(bias)
    _require(np.allclose(graph_out, ref, atol=1.0e-4, rtol=1.0e-4), "layout stress mismatch")


def _fallback_contract_case(*, device: str, rng: np.random.Generator) -> None:
    x = rng.random((1, 3, 8, 8), dtype=np.float32)
    w5 = rng.random((16, 3, 5, 5), dtype=np.float32)
    b = rng.random((16,), dtype=np.float32)
    eager = np.asarray(
        lc_api.lightning_conv_attention_torchstrong_nchw(
            x,
            w5,
            b,
            seq=48,
            head_dim=48,
            stride_h=1,
            stride_w=1,
            pad_h=2,
            pad_w=2,
            device=device,
            execution_mode="eager",
        ),
        dtype=np.float32,
    ).reshape(-1)
    graph_req = np.asarray(
        lc_api.lightning_conv_attention_torchstrong_nchw(
            x,
            w5,
            b,
            seq=48,
            head_dim=48,
            stride_h=1,
            stride_w=1,
            pad_h=2,
            pad_w=2,
            device=device,
            execution_mode="graph",
        ),
        dtype=np.float32,
    ).reshape(-1)
    _require(
        np.allclose(eager, graph_req, atol=1.0e-4, rtol=1.0e-4),
        "graph-request fallback output should match eager output",
    )


def _graph_supported_conv_path_case(*, device: str, rng: np.random.Generator) -> None:
    x = rng.random((1, 3, 8, 8), dtype=np.float32)
    w3 = rng.random((16, 3, 3, 3), dtype=np.float32)
    b = rng.random((16,), dtype=np.float32)
    for stride_h, stride_w, pad_h, pad_w, seq, head_dim in (
        (1, 1, 1, 1, 48, 48),
        (2, 2, 1, 1, 64, 32),
        (1, 1, 0, 0, 64, 32),
    ):
        eager = np.asarray(
            lc_api.lightning_conv_attention_torchstrong_nchw(
                x,
                w3,
                b,
                seq=seq,
                head_dim=head_dim,
                stride_h=stride_h,
                stride_w=stride_w,
                pad_h=pad_h,
                pad_w=pad_w,
                device=device,
                execution_mode="eager",
            ),
            dtype=np.float32,
        ).reshape(-1)
        graph_req = np.asarray(
            lc_api.lightning_conv_attention_torchstrong_nchw(
                x,
                w3,
                b,
                seq=seq,
                head_dim=head_dim,
                stride_h=stride_h,
                stride_w=stride_w,
                pad_h=pad_h,
                pad_w=pad_w,
                device=device,
                execution_mode="graph",
            ),
            dtype=np.float32,
        ).reshape(-1)
        _require(
            np.allclose(eager, graph_req, atol=1.0e-4, rtol=1.0e-4),
            "graph-request supported 3x3 shape should match eager output",
        )


def main() -> None:
    seed = 20260410
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    device = _device()

    for sync_mode in ("auto", "always", "never"):
        for m, k, n in ((8, 8, 8), (8, 16, 8), (16, 8, 16), (24, 16, 12)):
            _graph_mm_add_f32_case(m=m, k=k, n=n, sync_mode=sync_mode, device=device, rng=rng)

    for m, k, n in ((4, 4, 4), (6, 8, 5), (8, 6, 7)):
        _graph_mm_add_f64_case(m=m, k=k, n=n, rng=rng)

    for m, k, n in ((8, 8, 8), (12, 8, 10), (16, 16, 8)):
        _graph_noncontiguous_layout_case(m=m, k=k, n=n, device=device, rng=rng)

    for _ in range(6):
        _fallback_contract_case(device=device, rng=rng)
        _graph_supported_conv_path_case(device=device, rng=rng)

    print("python graph fallback/numerical stress smoke: ok")


if __name__ == "__main__":
    main()
