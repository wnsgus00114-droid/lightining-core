from __future__ import annotations

import csv
import json
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

import lightning_core as lc


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "benchmarks" / "reports" / "2026-03-29"


def _sync_mps() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _time_ms(fn, warmup: int, iters: int) -> float:
    samples = []
    for i in range(warmup + iters):
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            samples.append(dt)
    return statistics.mean(samples)


@contextmanager
def _env(**kwargs: str):
    old = {}
    for k, v in kwargs.items():
        old[k] = os.environ.get(k)
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _torch_mps_mm_ms(ta_mps: torch.Tensor, tb_mps: torch.Tensor, warmup: int = 4, iters: int = 20) -> float:
    def _run() -> None:
        _ = torch.matmul(ta_mps, tb_mps)
        _sync_mps()

    return _time_ms(_run, warmup=warmup, iters=iters)


def _lc_one_shot_ms(a: np.ndarray, b: np.ndarray, out: np.ndarray, m: int, k: int, n: int, warmup: int = 3, iters: int = 16) -> float:
    return _time_ms(
        lambda: lc.matmul_np_into(a, b, out, m, k, n, "metal"),
        warmup=warmup,
        iters=iters,
    )


def _lc_resident_steady_ms(
    a: np.ndarray,
    b: np.ndarray,
    out: np.ndarray,
    m: int,
    k: int,
    n: int,
    loops_per_sample: int = 10,
    warmup: int = 2,
    iters: int = 12,
) -> float:
    session = lc.MatMulMetalResidentSession(m, k, n)
    session.start_into(a, b, out)

    def _run_batch() -> None:
        for _ in range(loops_per_sample):
            session.run_into(a, b, out)
        session.sync_into(a, b, out)

    batch_ms = _time_ms(_run_batch, warmup=warmup, iters=iters)
    return batch_ms / float(loops_per_sample)


def bench_large_gemm_sweep() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shapes = [
        (1024, 1024, 1024),
        (1536, 1536, 1536),
        (2048, 2048, 2048),
        (3072, 3072, 3072),
        (4096, 1024, 4096),
    ]

    tune_root = OUT_DIR / "matmul_tune_profiles"
    tune_root.mkdir(parents=True, exist_ok=True)

    modes = [
        {
            "name": "balanced_default",
            "env": {
                "CJ_MATMUL_PREFER_MPS_ON_LARGE": "1",
                "CJ_MATMUL_TRY_KERNEL_ON_LARGE": "1",
                "CJ_MATMUL_MPS_HYST_PCT": "2.0",
            },
        },
        {
            "name": "aggressive_mps",
            "env": {
                "CJ_MATMUL_PREFER_MPS_ON_LARGE": "1",
                "CJ_MATMUL_TRY_KERNEL_ON_LARGE": "0",
                "CJ_MATMUL_MPS_HYST_PCT": "5.0",
            },
        },
        {
            "name": "kernel_favor",
            "env": {
                "CJ_MATMUL_PREFER_MPS_ON_LARGE": "0",
                "CJ_MATMUL_TRY_KERNEL_ON_LARGE": "1",
                "CJ_MATMUL_MPS_HYST_PCT": "0.0",
            },
        },
    ]

    all_rows: list[dict] = []
    best_rows: list[dict] = []

    for m, k, n in shapes:
        a = np.random.rand(m, k).astype(np.float32)
        b = np.random.rand(k, n).astype(np.float32)
        out = np.empty((m * n,), dtype=np.float32)

        ta_cpu = torch.from_numpy(a)
        tb_cpu = torch.from_numpy(b)

        if torch.backends.mps.is_available():
            ta_mps = ta_cpu.to("mps")
            tb_mps = tb_cpu.to("mps")
            torch_mps_ms = _torch_mps_mm_ms(ta_mps, tb_mps)
        else:
            torch_mps_ms = float("nan")

        for mode in modes:
            mode_name = mode["name"]
            tune_file = str(tune_root / f"{mode_name}_{m}x{k}x{n}.csv")
            with _env(CJ_MATMUL_TUNE_CACHE_FILE=tune_file, **mode["env"]):
                lc_one_shot_ms = _lc_one_shot_ms(a, b, out, m, k, n)
                lc_resident_ms = _lc_resident_steady_ms(a, b, out, m, k, n)

            row = {
                "shape": f"m={m},k={k},n={n}",
                "mode": mode_name,
                "lc_one_shot_ms": lc_one_shot_ms,
                "lc_resident_steady_ms": lc_resident_ms,
                "lc_best_ms": min(lc_one_shot_ms, lc_resident_ms),
                "torch_mps_ms": torch_mps_ms,
                "speedup_torch_over_lc_best": (
                    torch_mps_ms / min(lc_one_shot_ms, lc_resident_ms)
                    if torch_mps_ms == torch_mps_ms and min(lc_one_shot_ms, lc_resident_ms) > 0
                    else float("nan")
                ),
            }
            all_rows.append(row)

        rows_for_shape = [r for r in all_rows if r["shape"] == f"m={m},k={k},n={n}"]
        best = min(rows_for_shape, key=lambda r: r["lc_best_ms"])
        best_rows.append(best)

    csv_path = OUT_DIR / "large_gemm_auto_sweep.csv"
    json_path = OUT_DIR / "large_gemm_auto_sweep.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shape",
                "mode",
                "lc_one_shot_ms",
                "lc_resident_steady_ms",
                "lc_best_ms",
                "torch_mps_ms",
                "speedup_torch_over_lc_best",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    payload = {
        "backend": lc.backend_name(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "all_rows": all_rows,
        "best_rows_per_shape": best_rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")
    return payload


def main() -> None:
    payload = bench_large_gemm_sweep()
    print("\n=== Best policy per shape ===")
    for row in payload["best_rows_per_shape"]:
        print(
            f"{row['shape']}: mode={row['mode']} "
            f"lc_best={row['lc_best_ms']:.4f}ms torch_mps={row['torch_mps_ms']:.4f}ms "
            f"speedup={row['speedup_torch_over_lc_best']:.2f}x"
        )


if __name__ == "__main__":
    main()
