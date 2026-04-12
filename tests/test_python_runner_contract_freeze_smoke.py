#!/usr/bin/env python3
"""Python smoke: v0.4.0-rc0 runner contract freeze hard-gate."""

from __future__ import annotations

import hashlib
import json

import lightning_core_integrated_api as lc_api


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    _require(hasattr(lc_api, "runner_config_schema"), "runner_config_schema must exist")
    _require(hasattr(lc_api, "runner_contract_manifest"), "runner_contract_manifest must exist")
    _require(hasattr(lc_api, "validate_runner_config"), "validate_runner_config must exist")

    schema = lc_api.runner_config_schema()
    manifest = lc_api.runner_contract_manifest()
    _require(str(schema.get("schema_version", "")) == "runner_contract_v2", "schema version mismatch")
    _require(str(schema.get("freeze_id", "")) == "v0.4.0-rc0", "freeze id mismatch")
    _require(str(manifest.get("freeze_id", "")) == "v0.4.0-rc0", "manifest freeze id mismatch")

    schema_core = dict(schema)
    schema_core.pop("schema_version", None)
    schema_core.pop("freeze_id", None)
    schema_core.pop("schema_hash_sha256", None)
    expected_hash = hashlib.sha256(
        json.dumps(schema_core, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _require(str(schema.get("schema_hash_sha256", "")) == expected_hash, "schema hash mismatch")
    _require(str(manifest.get("schema_hash_sha256", "")) == expected_hash, "manifest hash mismatch")

    ok_cfg = lc_api.validate_runner_config(
        {
            "mode": "graph",
            "device": "cpu",
            "seed": 20260411,
            "layout": "seq_dmodel_2d",
            "dtype": "float32",
            "route_policy": {"conv": "auto", "attention": "auto", "graph": "auto"},
        },
        strict=True,
    )
    _require(bool(ok_cfg.get("ok", False)), "valid runner config should pass")

    bad_missing = lc_api.validate_runner_config(
        {
            "mode": "graph",
            "device": "cpu",
            "seed": 20260411,
            "layout": "seq_dmodel_2d",
            "dtype": "float32",
        },
        strict=False,
    )
    _require(not bool(bad_missing.get("ok", True)), "missing route_policy must fail")
    _require(str(bad_missing.get("code", "")) == "runner_config_missing_required_fields", "missing-field code mismatch")

    bad_unknown = lc_api.validate_runner_config(
        {
            "mode": "graph",
            "device": "cpu",
            "seed": 20260411,
            "layout": "seq_dmodel_2d",
            "dtype": "float32",
            "route_policy": {"conv": "auto", "attention": "auto", "graph": "auto"},
            "unknown_field": "boom",
        },
        strict=False,
    )
    _require(not bool(bad_unknown.get("ok", True)), "unknown field must fail")
    _require(str(bad_unknown.get("code", "")) == "runner_config_unknown_fields", "unknown-field code mismatch")

    print("python runner contract freeze smoke: ok")


if __name__ == "__main__":
    main()

