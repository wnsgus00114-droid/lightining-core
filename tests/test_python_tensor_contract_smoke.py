#!/usr/bin/env python3
"""Python smoke test: tensor shape/layout/lifetime contract freeze guards."""

from __future__ import annotations

import numpy as np

import lightning_core as lc


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _expect_fail(fn, msg: str) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(msg)


def main() -> None:
    t = lc.Tensor([2, 3], "cpu")
    t.validate_contract()

    c = t.contract()
    _require(c["shape"] == [2, 3], "tensor contract shape mismatch")
    _require(c["layout"] == "contiguous", "tensor contract layout mismatch")
    _require(c["numel"] == 6 and c["storage_numel"] == 6, "tensor contract numel mismatch")

    src = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    t.from_numpy(src)
    src[0] = 777.0
    host = t.to_numpy()
    _require(float(host[0]) != 777.0, "tensor must not alias source numpy lifetime")

    host[1] = 888.0
    host2 = t.to_numpy()
    _require(float(host2[1]) != 888.0, "to_numpy result must be detached copy")

    flat = t.view([6])
    flat.validate_contract()
    t.validate_view_contract(flat)
    _require(flat.layout() == "contiguous", "flat view should be contiguous")

    sliced = t.slice(0, 1, 2)
    _require(sliced.layout() == "strided", "slice view should be strided")
    _require(sliced.offset_elements() == 3, "slice offset mismatch")
    sliced.validate_contract_for_storage(t.storage_numel())

    v = t.to_host_view_numpy(sliced)
    _require(np.allclose(v, np.array([4, 5, 6], dtype=np.float32)), "slice values mismatch")

    t.from_list([10, 11, 12, 13, 14, 15])
    v2 = t.to_host_view_numpy(sliced)
    _require(np.allclose(v2, np.array([13, 14, 15], dtype=np.float32)), "slice alias update mismatch")

    _expect_fail(lambda: t.view([5]), "view shape mismatch should fail")
    _expect_fail(lambda: t.read_strided([2, 3], [3], 0), "read_strided rank mismatch should fail")
    _expect_fail(lambda: t.read_strided([2, 3], [3, 0], 0), "read_strided zero stride should fail")
    _expect_fail(lambda: t.read_strided([2, 3], [3, -1], 0), "read_strided negative stride should fail")
    _expect_fail(lambda: sliced.validate_contract_for_storage(2), "view storage bound check should fail")

    if hasattr(lc, "metal_available") and lc.metal_available():
        tm = lc.Tensor([2, 3], "metal")
        _expect_fail(lambda: tm.validate_view_contract(flat), "cross-device view contract should fail")

    print("python tensor contract smoke: ok")


if __name__ == "__main__":
    main()
