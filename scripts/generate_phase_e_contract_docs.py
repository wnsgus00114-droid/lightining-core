#!/usr/bin/env python3
"""Generate Phase E engine-federation contract docs from JSON source of truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

README_START = "<!-- AUTO-PHASE-E-CONTRACT:BEGIN -->"
README_END = "<!-- AUTO-PHASE-E-CONTRACT:END -->"
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


def _rows_from_list(items: list[Any]) -> list[list[str]]:
    return [[str(x)] for x in items] if items else [["-"]]


def _render_payload(contract: dict[str, Any]) -> str:
    engines = list(contract.get("engine_values", []))
    route_keys = list(contract.get("route_policy_keys", []))
    torch_codes = list(contract.get("torch_reason_codes", []))
    tf_codes = list(contract.get("tf_reason_codes", []))
    coreml_codes = list(contract.get("coreml_reason_codes", []))
    mlx_codes = list(contract.get("mlx_reason_codes", []))
    constants = dict(contract.get("ci_constants", {}).get("phase_e_exit_audit", {}))

    const_rows = [[str(k), str(v)] for k, v in sorted(constants.items(), key=lambda kv: str(kv[0]))]
    if not const_rows:
        const_rows = [["-", "-"]]

    lines: list[str] = []
    lines.append("### Phase E Engine Federation Contract (Auto-generated)")
    lines.append("")
    lines.append(f"- Contract version: `{contract.get('contract_version', '-')}`")
    lines.append(f"- As-of date: `{contract.get('as_of_date', '-')}`")
    lines.append("- Source of truth: `docs/engine_federation_contract.json`")
    lines.append("- CI sync checker: `python scripts/check_phase_e_contract_sync.py`")
    lines.append("")
    lines.append("#### Engine Values")
    lines.append("")
    lines.append(_md_table(["Engine"], _rows_from_list(engines)))
    lines.append("")
    lines.append("#### Route Policy Keys")
    lines.append("")
    lines.append(_md_table(["Key"], _rows_from_list(route_keys)))
    lines.append("")
    lines.append("#### Bridge Reason Codes")
    lines.append("")
    lines.append(_md_table(["Torch"], _rows_from_list(torch_codes)))
    lines.append("")
    lines.append(_md_table(["TensorFlow"], _rows_from_list(tf_codes)))
    lines.append("")
    lines.append(_md_table(["CoreML"], _rows_from_list(coreml_codes)))
    lines.append("")
    lines.append(_md_table(["MLX"], _rows_from_list(mlx_codes)))
    lines.append("")
    lines.append("#### CI Constants (Phase E Exit Audit)")
    lines.append("")
    lines.append(_md_table(["Key", "Value"], const_rows))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--contract-json", type=Path, default=Path("docs/engine_federation_contract.json"))
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--docs-page", type=Path, default=Path("docs/phase_e_contracts.md"))
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    root = args.repo_root.resolve()
    contract_path = (root / args.contract_json).resolve()
    readme_path = (root / args.readme).resolve()
    docs_path = (root / args.docs_page).resolve()

    payload = _render_payload(_load_json(contract_path))

    readme_next = _replace_between(readme_path.read_text(encoding="utf-8"), README_START, README_END, payload)
    docs_next = _replace_between(docs_path.read_text(encoding="utf-8"), DOCS_START, DOCS_END, payload)

    readme_changed = _write_or_check(readme_path, readme_next, args.check)
    docs_changed = _write_or_check(docs_path, docs_next, args.check)

    print(
        json.dumps(
            {
                "readme_changed": readme_changed,
                "docs_changed": docs_changed,
                "contract_manifest": str(contract_path),
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
