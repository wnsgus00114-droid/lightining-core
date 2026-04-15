#!/usr/bin/env python3
"""Generate Phase F framework contract docs from JSON source of truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

README_START = "<!-- AUTO-PHASE-F-CONTRACT:BEGIN -->"
README_END = "<!-- AUTO-PHASE-F-CONTRACT:END -->"
DOCS_START = README_START
DOCS_END = README_END


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_between(text: str, start: str, end: str, payload: str) -> str:
    if start not in text or end not in text:
        raise ValueError(f"markers not found: {start} .. {end}")
    left, rest = text.split(start, 1)
    _, right = rest.split(end, 1)
    return left + start + "\n\n" + payload.rstrip() + "\n\n" + end + right


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _write_or_check(path: Path, content: str, *, check: bool) -> bool:
    old = path.read_text(encoding="utf-8")
    changed = old != content
    if changed and not check:
        path.write_text(content, encoding="utf-8")
    return changed


def _render(contract: dict[str, Any]) -> str:
    api = dict(contract.get("api_version", {}))
    const = dict(contract.get("ci_constants", {}).get("phase_f_rc_audit", {}))
    const_rows = [[str(k), str(v)] for k, v in sorted(const.items(), key=lambda kv: str(kv[0]))]
    if not const_rows:
        const_rows = [["-", "-"]]

    lines: list[str] = []
    lines.append("### Phase F RC Contract (Auto-generated)")
    lines.append("")
    lines.append(f"- Contract version: `{contract.get('contract_version', '-')}`")
    lines.append(f"- As-of date: `{contract.get('as_of_date', '-')}`")
    lines.append("- Source of truth: `docs/phase_f_framework_contract.json`")
    lines.append("- CI sync checker: `python scripts/check_phase_f_contract_sync.py`")
    lines.append("")
    lines.append("#### API/ABI Freeze Baseline")
    lines.append("")
    lines.append(
        _md_table(
            ["Field", "Value"],
            [
                ["major", str(api.get("major", "-"))],
                ["minor", str(api.get("minor", "-"))],
                ["patch", str(api.get("patch", "-"))],
                ["label", str(api.get("label", "-"))],
            ],
        )
    )
    lines.append("")
    lines.append("#### CI Constants (Phase F RC Audit)")
    lines.append("")
    lines.append(_md_table(["Key", "Value"], const_rows))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--contract-json", type=Path, default=Path("docs/phase_f_framework_contract.json"))
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--docs-page", type=Path, default=Path("docs/phase_f_contracts.md"))
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    root = args.repo_root.resolve()
    contract_path = (root / args.contract_json).resolve()
    readme_path = (root / args.readme).resolve()
    docs_path = (root / args.docs_page).resolve()

    payload = _render(_load_json(contract_path))
    readme_next = _replace_between(readme_path.read_text(encoding="utf-8"), README_START, README_END, payload)
    docs_next = _replace_between(docs_path.read_text(encoding="utf-8"), DOCS_START, DOCS_END, payload)

    readme_changed = _write_or_check(readme_path, readme_next, check=args.check)
    docs_changed = _write_or_check(docs_path, docs_next, check=args.check)

    print(
        json.dumps(
            {
                "readme_changed": readme_changed,
                "docs_changed": docs_changed,
                "contract": str(contract_path),
                "docs_page": str(docs_path),
            },
            ensure_ascii=False,
        )
    )
    if args.check and (readme_changed or docs_changed):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
