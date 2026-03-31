#!/usr/bin/env python3
"""Generate roadmap progress history blocks for README and ROADMAP.

Single source of truth:
- docs/roadmap_updates.json

Outputs:
- README.md marker block
- ROADMAP.md marker block
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

README_START = "<!-- AUTO-ROADMAP-HISTORY:BEGIN -->"
README_END = "<!-- AUTO-ROADMAP-HISTORY:END -->"
ROADMAP_START = "<!-- AUTO-ROADMAP-HISTORY:BEGIN -->"
ROADMAP_END = "<!-- AUTO-ROADMAP-HISTORY:END -->"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def norm(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def parse_date_yyyy_mm_dd(text: str) -> dt.date:
    return dt.datetime.strptime(text, "%Y-%m-%d").date()


def ensure_sorted_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(e: dict[str, Any]) -> tuple[dt.date, str, str]:
        date = parse_date_yyyy_mm_dd(norm(e.get("date"), "1970-01-01"))
        return (date, norm(e.get("milestone"), ""), norm(e.get("title"), ""))

    return sorted(entries, key=key, reverse=True)


def slugify(text: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return out or "update"


def grouped_by_date(entries: list[dict[str, Any]]) -> OrderedDict[str, list[dict[str, Any]]]:
    out: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for e in entries:
        date = norm(e.get("date"))
        if date not in out:
            out[date] = []
        out[date].append(e)
    return out


def build_summary_table(entries: list[dict[str, Any]]) -> str:
    groups = grouped_by_date(entries)
    rows: list[list[str]] = []
    for date, items in groups.items():
        milestones: list[str] = []
        for it in items:
            m = norm(it.get("milestone"), "")
            if m and m not in milestones:
                milestones.append(m)
        titles = [norm(it.get("title")) for it in items[:2]]
        if len(items) > 2:
            titles.append(f"... (+{len(items) - 2} more)")
        rows.append(
            [
                date,
                str(len(items)),
                ", ".join(milestones) if milestones else "-",
                " / ".join(titles),
            ]
        )
    if not rows:
        rows = [["-", "0", "-", "-"]]
    return md_table(["Date", "Updates", "Milestones", "Highlights"], rows)


def build_details(entries: list[dict[str, Any]]) -> str:
    groups = grouped_by_date(entries)
    chunks: list[str] = []
    for date, items in groups.items():
        chunks.append(f"#### {date} ({len(items)} updates)")
        chunks.append("")
        for it in items:
            status = norm(it.get("status"), "completed")
            area = norm(it.get("area"), "general")
            milestone = norm(it.get("milestone"), "-")
            title = norm(it.get("title"))
            source = norm(it.get("source"), "")
            line = f"- [{status}] [{milestone}] [{area}] {title}"
            if source and source != "-":
                line += f" (`{source}`)"
            chunks.append(line)
        chunks.append("")
    if not chunks:
        return "- (no updates yet)"
    return "\n".join(chunks).rstrip()


def render_block(entries: list[dict[str, Any]]) -> str:
    total = len(entries)
    summary = build_summary_table(entries)
    details = build_details(entries)
    return (
        "### Progress History (Auto-generated)\n\n"
        f"- Total tracked updates: `{total}`\n"
        "- Source of truth: `docs/roadmap_updates.json`\n"
        "- Quick add command:\n"
        "  `python scripts/generate_roadmap_history.py --add --date YYYY-MM-DD --milestone M-A --area runtime --title \"your update\"`\n\n"
        "**Date Summary**\n\n"
        f"{summary}\n\n"
        "**Detailed Timeline**\n\n"
        f"{details}\n"
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


def append_update(
    entries: list[dict[str, Any]],
    date: str,
    milestone: str,
    area: str,
    title: str,
    status: str,
    source: str,
) -> list[dict[str, Any]]:
    parse_date_yyyy_mm_dd(date)  # validate format
    item = {
        "id": f"{date}-{slugify(area)}-{slugify(title)}",
        "date": date,
        "milestone": milestone,
        "area": area,
        "title": title,
        "status": status,
        "source": source,
    }
    entries.append(item)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--roadmap", type=Path, default=Path("ROADMAP.md"))
    parser.add_argument("--updates-json", type=Path, default=Path("docs/roadmap_updates.json"))
    parser.add_argument("--check", action="store_true")

    parser.add_argument("--add", action="store_true")
    parser.add_argument("--date", type=str, default=dt.date.today().isoformat())
    parser.add_argument("--milestone", type=str, default="M-A")
    parser.add_argument("--area", type=str, default="general")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--status", type=str, default="completed")
    parser.add_argument("--source", type=str, default="")

    args = parser.parse_args()

    root = args.repo_root.resolve()
    readme_path = (root / args.readme).resolve()
    roadmap_path = (root / args.roadmap).resolve()
    updates_path = (root / args.updates_json).resolve()

    entries_raw = load_json(updates_path)
    if not isinstance(entries_raw, list):
        raise TypeError("docs/roadmap_updates.json must contain a JSON array")
    entries: list[dict[str, Any]] = [dict(x) for x in entries_raw]

    if args.add:
        if not args.title.strip():
            raise ValueError("--title is required when using --add")
        entries = append_update(
            entries=entries,
            date=args.date,
            milestone=args.milestone,
            area=args.area,
            title=args.title,
            status=args.status,
            source=args.source,
        )
        entries = ensure_sorted_entries(entries)
        if not args.check:
            save_json(updates_path, entries)
    else:
        entries = ensure_sorted_entries(entries)

    payload = render_block(entries)

    readme_text = readme_path.read_text(encoding="utf-8")
    readme_next = replace_between_markers(readme_text, README_START, README_END, payload)
    roadmap_text = roadmap_path.read_text(encoding="utf-8")
    roadmap_next = replace_between_markers(roadmap_text, ROADMAP_START, ROADMAP_END, payload)

    readme_changed = write_or_check(readme_path, readme_next, args.check)
    roadmap_changed = write_or_check(roadmap_path, roadmap_next, args.check)

    status = {
        "updates_total": len(entries),
        "readme_changed": readme_changed,
        "roadmap_changed": roadmap_changed,
        "updates_json": str(updates_path),
    }
    print(json.dumps(status, ensure_ascii=False))

    if args.check and (readme_changed or roadmap_changed):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
