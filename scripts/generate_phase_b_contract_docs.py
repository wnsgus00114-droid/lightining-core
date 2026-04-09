#!/usr/bin/env python3
"""Generate Phase B graph contract docs from JSON source of truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

README_START = "<!-- AUTO-PHASE-B-CONTRACT:BEGIN -->"
README_END = "<!-- AUTO-PHASE-B-CONTRACT:END -->"
DOCS_START = README_START
DOCS_END = README_END


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _replace_between(text: str, start: str, end: str, payload: str) -> str:
    if start not in text or end not in text:
        raise ValueError(f"markers not found: {start} .. {end}")
    left, rest = text.split(start, 1)
    _, right = rest.split(end, 1)
    return left + start + "\n\n" + payload.rstrip() + "\n\n" + end + right


def _write_or_check(path: Path, content: str, check: bool) -> bool:
    old = path.read_text(encoding="utf-8")
    changed = old != content
    if not check and changed:
        path.write_text(content, encoding="utf-8")
    return changed


def _render_payload(contract: dict[str, Any]) -> str:
    graph = list(contract.get("graph_support", []))
    fallback = list(contract.get("fallback_rules", []))
    passes = list(contract.get("validation_passes", []))
    tol = dict(contract.get("numerical_tolerance", {}))
    ci = dict(contract.get("ci_constants", {}))

    graph_rows = [
        [
            str(x.get("op", "-")),
            str(x.get("rank", "-")),
            str(x.get("dtype", "-")),
            str(x.get("layout", "-")),
            str(x.get("shape_constraint", "-")),
        ]
        for x in graph
    ] or [["-", "-", "-", "-", "-"]]

    fallback_rows = [
        [str(x.get("code", "-")), str(x.get("description", "-"))]
        for x in fallback
    ] or [["-", "-"]]

    tol_rows = []
    for dtype, item in tol.items():
        item = dict(item)
        tol_rows.append([dtype, str(item.get("atol", "-")), str(item.get("rtol", "-"))])
    if not tol_rows:
        tol_rows = [["-", "-", "-"]]

    ci_rows = []
    for section, params in ci.items():
        for key, value in dict(params).items():
            ci_rows.append([section, key, str(value)])
    if not ci_rows:
        ci_rows = [["-", "-", "-"]]

    lines = []
    lines.append("### Phase B Graph Contract (Auto-generated)")
    lines.append("")
    lines.append(f"- Contract version: `{contract.get('contract_version', '-')}`")
    lines.append(f"- As-of date: `{contract.get('as_of_date', '-')}`")
    lines.append("- Source of truth: `docs/phase_b_graph_contract.json`")
    lines.append("- CI sync checker: `python scripts/check_phase_b_contract_sync.py`")
    lines.append("")
    lines.append("#### Graph Support Scope")
    lines.append("")
    lines.append(_md_table(["Op", "Rank", "DType", "Layout", "Shape Constraint"], graph_rows))
    lines.append("")
    lines.append("#### Fallback Reason Codes")
    lines.append("")
    lines.append(_md_table(["Reason Code", "Description"], fallback_rows))
    lines.append("")
    lines.append("#### Validation Passes")
    lines.append("")
    if passes:
        for p in passes:
            lines.append(f"- `{p}`")
    else:
        lines.append("- `-`")
    lines.append("")
    lines.append("#### Numerical Tolerances")
    lines.append("")
    lines.append(_md_table(["DType", "atol", "rtol"], tol_rows))
    lines.append("")
    lines.append("#### CI Constants (Phase B)")
    lines.append("")
    lines.append(_md_table(["Section", "Key", "Value"], ci_rows))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--contract-json", type=Path, default=Path("docs/phase_b_graph_contract.json"))
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--docs-page", type=Path, default=Path("docs/phase_b_contracts.md"))
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    root = args.repo_root.resolve()
    contract_path = (root / args.contract_json).resolve()
    readme_path = (root / args.readme).resolve()
    docs_path = (root / args.docs_page).resolve()

    contract = _load_json(contract_path)
    payload = _render_payload(contract)

    readme_next = _replace_between(readme_path.read_text(encoding="utf-8"), README_START, README_END, payload)
    docs_next = _replace_between(docs_path.read_text(encoding="utf-8"), DOCS_START, DOCS_END, payload)

    readme_changed = _write_or_check(readme_path, readme_next, args.check)
    docs_changed = _write_or_check(docs_path, docs_next, args.check)

    status = {
        "readme_changed": readme_changed,
        "docs_changed": docs_changed,
        "contract_manifest": str(contract_path),
        "docs_page": str(docs_path),
    }
    print(json.dumps(status, ensure_ascii=False))

    if args.check and (readme_changed or docs_changed):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
