#!/usr/bin/env python3
"""Quick public benchmark for Lightning Core vs Torch MPS.

Usage:
  python benchmarks/python/quick_bench.py
  python benchmarks/python/quick_bench.py --iters 200 --warmup 40 --out benchmark_results/quick_bench.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from statistics import median

import numpy as np

import lightning_core as lc

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return float(median(samples))


def _torch_mps_available() -> bool:
    return torch is not None and torch.backends.mps.is_available()  # type: ignore[union-attr]


def bench_matmul(warmup: int, iters: int):
    cases = [(256, 256, 256), (1024, 1024, 1024), (2048, 2048, 2048)]
    out = []
    for m, k, n in cases:
        a = np.random.rand(m, k).astype(np.float32)
        b = np.random.rand(k, n).astype(np.float32)

        lc_ms = _time_ms(lambda: lc.matmul2d(a, b, "metal"), warmup, iters)

        torch_mps_ms = float("nan")
        if _torch_mps_available():
            ta = torch.from_numpy(a).to("mps")
            tb = torch.from_numpy(b).to("mps")

            def _torch_fn():
                torch.mps.synchronize()  # type: ignore[attr-defined]
                _ = ta @ tb
                torch.mps.synchronize()  # type: ignore[attr-defined]

            torch_mps_ms = _time_ms(_torch_fn, warmup, iters)

        out.append(
            {
                "suite": "quick_bench",
                "bench": "matmul2d",
                "shape": f"m={m},k={k},n={n}",
                "lightning_core_ms": lc_ms,
                "torch_mps_ms": torch_mps_ms,
                "speedup_torch_over_lc": (torch_mps_ms / lc_ms) if np.isfinite(torch_mps_ms) and lc_ms > 0 else float("nan"),
            }
        )
    return out


def bench_attention(warmup: int, iters: int):
    cases = [(8, 16), (96, 48), (256, 64)]
    out = []
    for seq, dim in cases:
        q = np.random.rand(seq, dim).astype(np.float32)
        k = np.random.rand(seq, dim).astype(np.float32)
        v = np.random.rand(seq, dim).astype(np.float32)

        lc_ms = _time_ms(lambda: lc.attention2d(q, k, v, False, "metal"), warmup, iters)

        torch_mps_ms = float("nan")
        if _torch_mps_available():
            tq = torch.from_numpy(q).to("mps").unsqueeze(0).unsqueeze(0)
            tk = torch.from_numpy(k).to("mps").unsqueeze(0).unsqueeze(0)
            tv = torch.from_numpy(v).to("mps").unsqueeze(0).unsqueeze(0)

            def _torch_fn():
                torch.mps.synchronize()  # type: ignore[attr-defined]
                _ = F.scaled_dot_product_attention(tq, tk, tv, is_causal=False)
                torch.mps.synchronize()  # type: ignore[attr-defined]

            torch_mps_ms = _time_ms(_torch_fn, warmup, iters)

        out.append(
            {
                "suite": "quick_bench",
                "bench": "attention2d",
                "shape": f"seq={seq},head_dim={dim}",
                "lightning_core_ms": lc_ms,
                "torch_mps_ms": torch_mps_ms,
                "speedup_torch_over_lc": (torch_mps_ms / lc_ms) if np.isfinite(torch_mps_ms) and lc_ms > 0 else float("nan"),
            }
        )
    return out


def bench_conv(warmup: int, iters: int):
    cases = [(1, 3, 16, 16, 16, 3), (1, 3, 32, 32, 16, 3), (2, 3, 32, 32, 16, 3)]
    out = []
    for n, c, h, w, oc, ksz in cases:
        x = np.random.rand(n, c, h, w).astype(np.float32)
        wgt = np.random.rand(oc, c, ksz, ksz).astype(np.float32)
        b = np.random.rand(oc).astype(np.float32)

        lc_ms = _time_ms(lambda: lc.conv2d_nchw(x, wgt, b, 1, 1, 1, 1, "metal"), warmup, iters)

        torch_mps_ms = float("nan")
        if _torch_mps_available():
            tx = torch.from_numpy(x).to("mps")
            tw = torch.from_numpy(wgt).to("mps")
            tb = torch.from_numpy(b).to("mps")

            def _torch_fn():
                torch.mps.synchronize()  # type: ignore[attr-defined]
                _ = F.relu(F.conv2d(tx, tw, tb, stride=1, padding=1))
                torch.mps.synchronize()  # type: ignore[attr-defined]

            torch_mps_ms = _time_ms(_torch_fn, warmup, iters)

        out.append(
            {
                "suite": "quick_bench",
                "bench": "conv2d_relu",
                "shape": f"batch={n},in_ch={c},h={h},w={w},out_ch={oc},k={ksz}",
                "lightning_core_ms": lc_ms,
                "torch_mps_ms": torch_mps_ms,
                "speedup_torch_over_lc": (torch_mps_ms / lc_ms) if np.isfinite(torch_mps_ms) and lc_ms > 0 else float("nan"),
            }
        )
    return out


def save_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["suite", "bench", "shape", "lightning_core_ms", "torch_mps_ms", "speedup_torch_over_lc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Lightning Core quick benchmark")
    p.add_argument("--warmup", type=int, default=40)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--out", type=Path, default=Path("benchmark_results/quick_bench.csv"))
    args = p.parse_args()

    rows = []
    rows.extend(bench_attention(args.warmup, args.iters))
    rows.extend(bench_conv(args.warmup, args.iters))
    rows.extend(bench_matmul(args.warmup, args.iters))

    save_csv(args.out, rows)

    print("saved:", args.out)
    print("\n=== Quick Bench (median ms) ===")
    for r in rows:
        sp = r["speedup_torch_over_lc"]
        sp_txt = f"{sp:.2f}x" if np.isfinite(sp) else "n/a"
        torch_txt = f"{r['torch_mps_ms']:.6f}ms" if np.isfinite(r["torch_mps_ms"]) else "n/a"
        print(
            f"[{r['bench']}] {r['shape']} | LC={r['lightning_core_ms']:.6f}ms "
            f"TorchMPS={torch_txt} torch/lc={sp_txt}"
        )


if __name__ == "__main__":
    main()
