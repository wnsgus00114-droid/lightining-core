#!/usr/bin/env python3
"""Check local markdown links in README/docs.

Rules:
- External links (http/https/mailto/...) are ignored.
- Build artifact links (build/, benchmark_results/) are ignored.
- Relative file links must exist.
- In-file/markdown anchors are validated for markdown targets.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

IGNORE_PREFIXES = (
    "build/",
    "../build/",
    "../../build/",
    "benchmark_results/",
    "../benchmark_results/",
    "../../benchmark_results/",
)


def iter_markdown_files(root: Path, inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        p = (root / item).resolve()
        if p.is_dir():
            files.extend(sorted(x for x in p.rglob("*.md") if x.is_file()))
        elif p.is_file() and p.suffix.lower() == ".md":
            files.append(p)
    # preserve deterministic order and uniqueness
    return sorted(set(files))


def slugify_heading(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("`", "")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text


def collect_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEADING_RE.match(line)
        if not m:
            continue
        slug = slugify_heading(m.group(1))
        if slug:
            anchors.add(slug)
    return anchors


def should_ignore_target(target: str) -> bool:
    if not target:
        return True
    if target.startswith("#"):
        return False
    if SCHEME_RE.match(target):
        return True
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if target.startswith("/"):
        return True
    for prefix in IGNORE_PREFIXES:
        if target.startswith(prefix):
            return True
    return False


def resolve_target(current: Path, target: str) -> tuple[Path | None, str | None]:
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if "#" in target:
        path_part, anchor = target.split("#", 1)
        return ((current.parent / path_part).resolve() if path_part else current.resolve(), anchor)
    return ((current.parent / target).resolve(), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "paths",
        nargs="*",
        default=["README.md", "docs"],
        help="Markdown files or directories to scan.",
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    md_files = iter_markdown_files(root, args.paths)
    if not md_files:
        print("No markdown files found for link check.", file=sys.stderr)
        return 1

    anchor_cache: dict[Path, set[str]] = {}
    errors: list[str] = []
    checked_links = 0

    for md in md_files:
        text = md.read_text(encoding="utf-8")
        for m in LINK_RE.finditer(text):
            raw_target = m.group(1).strip()
            if should_ignore_target(raw_target):
                continue

            resolved, anchor = resolve_target(md, raw_target)
            if resolved is None:
                continue
            checked_links += 1

            if not resolved.exists():
                rel_src = md.relative_to(root).as_posix()
                rel_tgt = resolved.relative_to(root).as_posix() if resolved.is_absolute() and root in resolved.parents else str(resolved)
                errors.append(f"{rel_src}: missing target `{raw_target}` -> `{rel_tgt}`")
                continue

            if anchor:
                if resolved.suffix.lower() != ".md":
                    continue
                anchors = anchor_cache.get(resolved)
                if anchors is None:
                    anchors = collect_anchors(resolved)
                    anchor_cache[resolved] = anchors
                if anchor not in anchors:
                    rel_src = md.relative_to(root).as_posix()
                    rel_tgt = resolved.relative_to(root).as_posix()
                    errors.append(f"{rel_src}: missing anchor `#{anchor}` in `{rel_tgt}`")

    if errors:
        print(f"checked_markdown_files={len(md_files)} checked_links={checked_links} errors={len(errors)}")
        for err in errors:
            print(err)
        return 1

    print(f"checked_markdown_files={len(md_files)} checked_links={checked_links} errors=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
