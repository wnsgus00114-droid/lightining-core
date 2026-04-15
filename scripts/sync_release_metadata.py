#!/usr/bin/env python3
"""Sync and validate release metadata from pyproject version."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


README_RELEASE_LINE_RE = re.compile(r"(?m)^Current public release: \*\*v[^\*]+\*\* \([^)]+\)\.$")
README_RELEASE_TRAIN_RE = re.compile(r"(?m)^Current release train: \*\*v[^\*]+\*\*\.$")
README_ROADMAP_BASELINE_RE = re.compile(
    r"(?m)^Roadmap baseline is now aligned to \*\*v[^\*]+\*\* and tracked in detail in \[ROADMAP\.md\]\(ROADMAP\.md\)\.$"
)
ROADMAP_VERSION_CONTEXT_RE = re.compile(r"(?m)^Version context: v[^ ]+ \([^)]+\)$")
ROADMAP_BASELINE_HEADING_RE = re.compile(r"(?m)^## 3\) Current Baseline \(v[^)]+\)$")
ROADMAP_RELEASE_TRAIN_RE = re.compile(r"(?m)^## 11\) Release-Train Detail \(v[^ ]+ -> v1\.0\)$")

RELEASE_NOTES_BEGIN = "<!-- AUTO-RELEASE-NOTES:BEGIN -->"
RELEASE_NOTES_END = "<!-- AUTO-RELEASE-NOTES:END -->"


@dataclass
class VersionEntry:
    file: str
    field: str
    observed: str
    expected: str

    @property
    def match(self) -> bool:
        return self.observed == self.expected


def _replace_required(text: str, rx: re.Pattern[str], replacement: str, *, label: str) -> str:
    out, n = rx.subn(replacement, text, count=1)
    if n != 1:
        raise SystemExit(f"[release_sync] expected exactly one match for {label}")
    return out


def _replace_between_markers(text: str, begin: str, end: str, payload: str) -> str:
    start = text.find(begin)
    stop = text.find(end)
    if start < 0 or stop < 0 or stop < start:
        raise SystemExit("[release_sync] release notes markers are missing or invalid")
    start += len(begin)
    return text[:start] + "\n\n" + payload.rstrip() + "\n\n" + text[stop:]


def _write_or_check(path: Path, new_text: str, *, check: bool) -> bool:
    old_text = path.read_text(encoding="utf-8") if path.exists() else ""
    changed = old_text != new_text
    if changed and not check:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_text, encoding="utf-8")
    return changed


def _load_pyproject_version(path: Path) -> str:
    obj = tomllib.loads(path.read_text(encoding="utf-8"))
    project = obj.get("project", {})
    version = str(project.get("version", "")).strip()
    if not version:
        raise SystemExit("[release_sync] project.version is missing in pyproject.toml")
    return version


def _normalized_tag(version_or_tag: str) -> str:
    raw = str(version_or_tag).strip()
    if not raw:
        return ""
    return raw if raw.startswith("v") else f"v{raw}"


def _release_notes_template() -> str:
    return (
        "# Release Notes\n\n"
        f"{RELEASE_NOTES_BEGIN}\n"
        "_auto-generated_\n"
        f"{RELEASE_NOTES_END}\n"
    )


def _extract_regex_group(text: str, rx: re.Pattern[str]) -> str:
    m = rx.search(text)
    return str(m.group(1)).strip() if m else ""


def _extract_release_date_from_readme(text: str) -> str:
    m = re.search(r"(?m)^Current public release: \*\*v[^\*]+\*\* \((\d{4}-\d{2}-\d{2})\)\.$", text)
    return str(m.group(1)).strip() if m else ""


def _extract_from_json(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get(key, "")).strip()


def _optional_path(path: Path) -> Path | None:
    raw = str(path).strip()
    if raw in {"", "."}:
        return None
    return path


def _report_md(entries: list[VersionEntry], *, expected_tag: str, expected_date: str, check: bool) -> str:
    lines = []
    lines.append("## Release Version Sync Report")
    lines.append("")
    lines.append(f"- expected_version: `{expected_tag}`")
    lines.append(f"- expected_date: `{expected_date}`")
    lines.append(f"- mode: `{'check' if check else 'sync'}`")
    lines.append("")
    lines.append("| File | Field | Observed | Expected | Match |")
    lines.append("| --- | --- | --- | --- | --- |")
    for e in entries:
        lines.append(f"| `{e.file}` | `{e.field}` | `{e.observed}` | `{e.expected}` | `{e.match}` |")
    return "\n".join(lines)


def _sync_contract_json(path: Path, *, tag: str, date_text: str, check: bool) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["release_baseline_version"] = tag
    payload["release_baseline_date"] = date_text
    new_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    return _write_or_check(path, new_text, check=check)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--roadmap", type=Path, default=Path("ROADMAP.md"))
    p.add_argument("--phase-b-contract", type=Path, default=Path("docs/phase_b_graph_contract.json"))
    p.add_argument("--phase-c-contract", type=Path, default=Path("docs/phase_c_engine_contract.json"))
    p.add_argument("--phase-d-contract", type=Path, default=Path("docs/phase_d_runner_contract.json"))
    p.add_argument("--phase-e-contract", type=Path, default=Path("docs/engine_federation_contract.json"))
    p.add_argument("--phase-f-contract", type=Path, default=Path("docs/phase_f_framework_contract.json"))
    p.add_argument("--test-matrix-contract", type=Path, default=Path("docs/test_matrix_contract.json"))
    p.add_argument("--release-notes", type=Path, default=Path("docs/release_notes.md"))
    p.add_argument("--release-date", type=str, default="")
    p.add_argument("--expected-version", type=str, default="")
    p.add_argument("--report-json", type=Path, default=Path(""))
    p.add_argument("--report-md", type=Path, default=Path(""))
    p.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()

    version = _load_pyproject_version(args.pyproject)
    tag = _normalized_tag(version)
    release_date = str(args.release_date).strip()
    if not release_date:
        if args.check and args.readme.exists():
            observed_readme = args.readme.read_text(encoding="utf-8")
            release_date = _extract_release_date_from_readme(observed_readme)
        if not release_date:
            release_date = datetime.now(timezone.utc).date().isoformat()

    expected_tag = _normalized_tag(args.expected_version)
    if expected_tag and expected_tag != tag:
        raise SystemExit(
            f"[release_sync] expected version/tag mismatch: expected={expected_tag} source={tag} (from {args.pyproject})"
        )

    readme_text = args.readme.read_text(encoding="utf-8")
    roadmap_text = args.roadmap.read_text(encoding="utf-8")
    release_notes_text = args.release_notes.read_text(encoding="utf-8") if args.release_notes.exists() else _release_notes_template()

    new_readme = readme_text
    new_readme = _replace_required(
        new_readme,
        README_RELEASE_LINE_RE,
        f"Current public release: **{tag}** ({release_date}).",
        label="README current public release",
    )
    new_readme = _replace_required(
        new_readme,
        README_RELEASE_TRAIN_RE,
        f"Current release train: **{tag}**.",
        label="README current release train",
    )
    new_readme = _replace_required(
        new_readme,
        README_ROADMAP_BASELINE_RE,
        f"Roadmap baseline is now aligned to **{tag}** and tracked in detail in [ROADMAP.md](ROADMAP.md).",
        label="README roadmap baseline line",
    )

    new_roadmap = roadmap_text
    new_roadmap = _replace_required(
        new_roadmap,
        ROADMAP_VERSION_CONTEXT_RE,
        f"Version context: {tag} ({release_date})",
        label="ROADMAP version context",
    )
    new_roadmap = _replace_required(
        new_roadmap,
        ROADMAP_BASELINE_HEADING_RE,
        f"## 3) Current Baseline ({tag})",
        label="ROADMAP current baseline heading",
    )
    new_roadmap = _replace_required(
        new_roadmap,
        ROADMAP_RELEASE_TRAIN_RE,
        f"## 11) Release-Train Detail ({tag} -> v1.0)",
        label="ROADMAP release-train heading",
    )

    notes_payload = "\n".join(
        [
            f"## {tag} ({release_date})",
            "",
            "- Source of truth: `pyproject.toml` `project.version`",
            "- Synced targets: `README.md`, `ROADMAP.md`, `docs/phase_b_graph_contract.json`, `docs/phase_c_engine_contract.json`, `docs/phase_d_runner_contract.json`, `docs/engine_federation_contract.json`, `docs/phase_f_framework_contract.json`, `docs/test_matrix_contract.json`",
            "- Generated by: `python scripts/sync_release_metadata.py`",
        ]
    )
    new_release_notes = _replace_between_markers(release_notes_text, RELEASE_NOTES_BEGIN, RELEASE_NOTES_END, notes_payload)

    changed = {
        "README.md": _write_or_check(args.readme, new_readme, check=args.check),
        "ROADMAP.md": _write_or_check(args.roadmap, new_roadmap, check=args.check),
        "docs/release_notes.md": _write_or_check(args.release_notes, new_release_notes, check=args.check),
        "docs/phase_b_graph_contract.json": _sync_contract_json(
            args.phase_b_contract, tag=tag, date_text=release_date, check=args.check
        ),
        "docs/phase_c_engine_contract.json": _sync_contract_json(
            args.phase_c_contract, tag=tag, date_text=release_date, check=args.check
        ),
        "docs/phase_d_runner_contract.json": _sync_contract_json(
            args.phase_d_contract, tag=tag, date_text=release_date, check=args.check
        ),
        "docs/engine_federation_contract.json": _sync_contract_json(
            args.phase_e_contract, tag=tag, date_text=release_date, check=args.check
        ),
        "docs/phase_f_framework_contract.json": _sync_contract_json(
            args.phase_f_contract, tag=tag, date_text=release_date, check=args.check
        ),
        "docs/test_matrix_contract.json": _sync_contract_json(
            args.test_matrix_contract, tag=tag, date_text=release_date, check=args.check
        ),
    }

    readme_cur = args.readme.read_text(encoding="utf-8") if not args.check else new_readme
    roadmap_cur = args.roadmap.read_text(encoding="utf-8") if not args.check else new_roadmap
    notes_cur = args.release_notes.read_text(encoding="utf-8") if not args.check else new_release_notes

    entries = [
        VersionEntry("pyproject.toml", "project.version", tag, tag),
        VersionEntry(
            "README.md",
            "current_public_release",
            _extract_regex_group(readme_cur, re.compile(r"(?m)^Current public release: \*\*(v[^\*]+)\*\* \([^)]+\)\.$")),
            tag,
        ),
        VersionEntry(
            "README.md",
            "current_release_train",
            _extract_regex_group(readme_cur, re.compile(r"(?m)^Current release train: \*\*(v[^\*]+)\*\*\.$")),
            tag,
        ),
        VersionEntry(
            "README.md",
            "roadmap_baseline_line",
            _extract_regex_group(
                readme_cur,
                re.compile(
                    r"(?m)^Roadmap baseline is now aligned to \*\*(v[^\*]+)\*\* and tracked in detail in \[ROADMAP\.md\]\(ROADMAP\.md\)\.$"
                ),
            ),
            tag,
        ),
        VersionEntry(
            "ROADMAP.md",
            "version_context",
            _extract_regex_group(roadmap_cur, re.compile(r"(?m)^Version context: (v[^ ]+) \([^)]+\)$")),
            tag,
        ),
        VersionEntry(
            "ROADMAP.md",
            "current_baseline_heading",
            _extract_regex_group(roadmap_cur, re.compile(r"(?m)^## 3\) Current Baseline \((v[^)]+)\)$")),
            tag,
        ),
        VersionEntry(
            "ROADMAP.md",
            "release_train_heading",
            _extract_regex_group(roadmap_cur, re.compile(r"(?m)^## 11\) Release-Train Detail \((v[^ ]+) -> v1\.0\)$")),
            tag,
        ),
        VersionEntry(
            "docs/phase_b_graph_contract.json",
            "release_baseline_version",
            _extract_from_json(args.phase_b_contract, "release_baseline_version"),
            tag,
        ),
        VersionEntry(
            "docs/phase_c_engine_contract.json",
            "release_baseline_version",
            _extract_from_json(args.phase_c_contract, "release_baseline_version"),
            tag,
        ),
        VersionEntry(
            "docs/phase_d_runner_contract.json",
            "release_baseline_version",
            _extract_from_json(args.phase_d_contract, "release_baseline_version"),
            tag,
        ),
        VersionEntry(
            "docs/engine_federation_contract.json",
            "release_baseline_version",
            _extract_from_json(args.phase_e_contract, "release_baseline_version"),
            tag,
        ),
        VersionEntry(
            "docs/phase_f_framework_contract.json",
            "release_baseline_version",
            _extract_from_json(args.phase_f_contract, "release_baseline_version"),
            tag,
        ),
        VersionEntry(
            "docs/test_matrix_contract.json",
            "release_baseline_version",
            _extract_from_json(args.test_matrix_contract, "release_baseline_version"),
            tag,
        ),
        VersionEntry(
            "docs/release_notes.md",
            "latest_heading",
            _extract_regex_group(notes_cur, re.compile(r"(?m)^## (v[^ ]+) \([^)]+\)$")),
            tag,
        ),
    ]

    report = {
        "status": "ok",
        "mode": "check" if args.check else "sync",
        "expected_version": tag,
        "expected_date": release_date,
        "expected_tag_match": (expected_tag == tag) if expected_tag else True,
        "changed": changed,
        "entries": [
            {"file": e.file, "field": e.field, "observed": e.observed, "expected": e.expected, "match": e.match}
            for e in entries
        ],
    }
    report["overall_match"] = bool(report["expected_tag_match"] and all(e.match for e in entries))
    if not report["overall_match"]:
        report["status"] = "drift_detected"

    report_json_path = _optional_path(args.report_json)
    report_md_path = _optional_path(args.report_md)
    if report_json_path is not None:
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if report_md_path is not None:
        report_md_path.parent.mkdir(parents=True, exist_ok=True)
        report_md_path.write_text(
            _report_md(entries, expected_tag=tag, expected_date=release_date, check=args.check), encoding="utf-8"
        )

    print(json.dumps(report, ensure_ascii=False))
    if args.check and not report["overall_match"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
