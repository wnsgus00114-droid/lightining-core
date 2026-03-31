#!/usr/bin/env python3
"""Generate runtime capability and tested-environment matrices for docs.

This script updates marker blocks in:
- README.md
- docs/capability_matrix.md

It can also refresh a checked-in runtime capability snapshot JSON by querying
the installed `lightning_core` Python module.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

README_START = "<!-- AUTO-CAPABILITY-MATRIX:BEGIN -->"
README_END = "<!-- AUTO-CAPABILITY-MATRIX:END -->"
DOCS_START = "<!-- AUTO-CAPABILITY-MATRIX:BEGIN -->"
DOCS_END = "<!-- AUTO-CAPABILITY-MATRIX:END -->"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def bool_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return "-"


def norm_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value if value else "-"
    return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def try_capture_runtime_snapshot() -> dict[str, Any]:
    import importlib

    lc = importlib.import_module("lightning_core")
    out: dict[str, Any] = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "runtime_api",
        "backend_name": None,
        "memory_model_name": None,
        "timeline_api": {
            "runtime_trace_enable": hasattr(lc, "runtime_trace_enable"),
            "runtime_trace_events": hasattr(lc, "runtime_trace_events"),
            "runtime_trace_timeline": hasattr(lc, "runtime_trace_timeline"),
            "runtime_trace_clear": hasattr(lc, "runtime_trace_clear"),
        },
        "backends": [],
        "active_backend_capabilities": {},
    }

    try:
        out["backend_name"] = lc.backend_name()
    except Exception as exc:  # pragma: no cover - best effort metadata
        out["backend_name_error"] = str(exc)
    try:
        out["memory_model_name"] = lc.memory_model_name()
    except Exception as exc:  # pragma: no cover - best effort metadata
        out["memory_model_name_error"] = str(exc)

    for device in ("metal", "cpu", "cuda"):
        row: dict[str, Any] = {"device": device}
        try:
            caps = lc.runtime_backend_capabilities(device)
            row.update(dict(caps))
            row["query_status"] = "ok"
        except Exception as exc:
            row["query_status"] = "error"
            row["error"] = str(exc)
        out["backends"].append(row)

    try:
        out["active_backend_capabilities"] = dict(lc.runtime_active_backend_capabilities())
    except Exception as exc:
        out["active_backend_capabilities"] = {"query_status": "error", "error": str(exc)}

    return out


def render_runtime_capability_table(snapshot: dict[str, Any]) -> str:
    headers = [
        "Device",
        "Built",
        "Available",
        "Compute",
        "Memory",
        "Sync",
        "Profiling",
        "Trace",
        "Sync Policy",
        "Memory Model",
        "Query",
    ]
    rows: list[list[str]] = []
    for item in snapshot.get("backends", []):
        rows.append(
            [
                norm_cell(item.get("device")),
                bool_cell(item.get("built")),
                bool_cell(item.get("available")),
                bool_cell(item.get("compute_surface")),
                bool_cell(item.get("memory_surface")),
                bool_cell(item.get("sync_surface")),
                bool_cell(item.get("profiling_surface")),
                bool_cell(item.get("runtime_trace_surface")),
                bool_cell(item.get("sync_policy_surface")),
                norm_cell(item.get("memory_model")),
                norm_cell(item.get("query_status")),
            ]
        )
    if not rows:
        rows = [["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]]
    return md_table(headers, rows)


def render_runtime_api_surface_table(snapshot: dict[str, Any]) -> str:
    tl = snapshot.get("timeline_api", {})
    headers = ["Runtime API Surface", "Available", "Notes"]
    rows = [
        ["runtime_trace_enable", bool_cell(tl.get("runtime_trace_enable")), "Enable/disable trace capture"],
        ["runtime_trace_events", bool_cell(tl.get("runtime_trace_events")), "Raw runtime event list"],
        ["runtime_trace_timeline", bool_cell(tl.get("runtime_trace_timeline")), "Sorted/grouped timeline report"],
        ["runtime_trace_clear", bool_cell(tl.get("runtime_trace_clear")), "Clear ring buffer events"],
        [
            "runtime_backend_capabilities",
            "Yes",
            "Per-backend contract query (metal/cpu/cuda)",
        ],
        [
            "runtime_active_backend_capabilities",
            "Yes",
            "Capability contract of current active backend",
        ],
    ]
    return md_table(headers, rows)


def render_tested_env_table(entries: list[dict[str, Any]]) -> str:
    headers = ["Date", "Scope", "Hardware / OS", "Python", "Torch", "Status", "Notes"]
    rows: list[list[str]] = []
    for item in entries:
        rows.append(
            [
                norm_cell(item.get("date")),
                norm_cell(item.get("scope")),
                norm_cell(item.get("hardware")),
                norm_cell(item.get("python")),
                norm_cell(item.get("torch")),
                norm_cell(item.get("status")),
                norm_cell(item.get("notes")),
            ]
        )
    if not rows:
        rows = [["-", "-", "-", "-", "-", "-", "-"]]
    return md_table(headers, rows)


def render_block(snapshot: dict[str, Any], env_entries: list[dict[str, Any]]) -> str:
    generated = norm_cell(snapshot.get("generated_at_utc"))
    backend_name = norm_cell(snapshot.get("backend_name"))
    memory_model = norm_cell(snapshot.get("memory_model_name"))
    capability = render_runtime_capability_table(snapshot)
    runtime_api = render_runtime_api_surface_table(snapshot)
    environments = render_tested_env_table(env_entries)
    return (
        "### Runtime Capability Matrix (Auto-generated)\n\n"
        f"- Snapshot generated at (UTC): `{generated}`\n"
        f"- Active backend at snapshot time: `{backend_name}`\n"
        f"- Active memory model: `{memory_model}`\n"
        "- Note: `Available` is host-dependent. Regenerate the snapshot on your target machine for exact values.\n"
        "- Generated by: `python scripts/generate_capability_docs.py --refresh-runtime-snapshot`\n\n"
        f"{capability}\n\n"
        "### Runtime Trace / Capability API Surface (Auto-generated)\n\n"
        f"{runtime_api}\n\n"
        "### Tested Environment Matrix (Auto-generated)\n\n"
        f"{environments}\n"
    )


def replace_between_markers(text: str, start: str, end: str, payload: str) -> str:
    if start not in text or end not in text:
        raise ValueError(f"Markers not found: {start} .. {end}")
    left, rest = text.split(start, 1)
    _, right = rest.split(end, 1)
    return left + start + "\n\n" + payload.rstrip() + "\n\n" + end + right


def write_or_check(path: Path, content: str, check: bool) -> bool:
    old = path.read_text(encoding="utf-8")
    changed = old != content
    if check:
        return changed
    if changed:
        path.write_text(content, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--docs-page", type=Path, default=Path("docs/capability_matrix.md"))
    parser.add_argument("--env-json", type=Path, default=Path("docs/tested_environments.json"))
    parser.add_argument("--runtime-json", type=Path, default=Path("docs/runtime_capabilities.json"))
    parser.add_argument("--refresh-runtime-snapshot", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    readme_path = (root / args.readme).resolve()
    docs_path = (root / args.docs_page).resolve()
    env_json_path = (root / args.env_json).resolve()
    runtime_json_path = (root / args.runtime_json).resolve()

    if args.refresh_runtime_snapshot:
        runtime_snapshot = try_capture_runtime_snapshot()
        save_json(runtime_json_path, runtime_snapshot)
    else:
        if not runtime_json_path.exists():
            raise FileNotFoundError(
                f"Runtime snapshot not found: {runtime_json_path}. "
                "Run with --refresh-runtime-snapshot first."
            )
        runtime_snapshot = load_json(runtime_json_path)

    env_entries = load_json(env_json_path)
    payload = render_block(runtime_snapshot, env_entries)

    readme_text = readme_path.read_text(encoding="utf-8")
    readme_next = replace_between_markers(readme_text, README_START, README_END, payload)
    docs_text = docs_path.read_text(encoding="utf-8")
    docs_next = replace_between_markers(docs_text, DOCS_START, DOCS_END, payload)

    readme_changed = write_or_check(readme_path, readme_next, args.check)
    docs_changed = write_or_check(docs_path, docs_next, args.check)

    status = {
        "readme_changed": readme_changed,
        "docs_changed": docs_changed,
        "runtime_snapshot": str(runtime_json_path),
        "env_manifest": str(env_json_path),
    }
    print(json.dumps(status, ensure_ascii=False))

    if args.check and (readme_changed or docs_changed):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
