#!/usr/bin/env python3
"""Generate Python/C++ API reference pages for the docs site.

This script keeps generated API pages in sync with source files:
- docs/reference/python_api_generated.md
- docs/reference/cpp_api_generated.md
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REPO_BLOB_BASE = "https://github.com/wnsgus00114-droid/lightning-core/blob/main"


PY_BINDING_GLOBS = ("python/bindings/bind_*.cpp",)
CPP_HEADERS = (
    "include/lightning_core/core/runtime.hpp",
    "include/lightning_core/core/attention.hpp",
    "include/lightning_core/core/graph.hpp",
    "include/lightning_core/core/tensor.hpp",
    "include/lightning_core/core/ops/policy.hpp",
    "include/lightning_core/core/ops.hpp",
    "include/lightning_core/lightning_core.h",
)
CPP_FUNCTION_HEADERS = (
    "include/lightning_core/core/runtime.hpp",
    "include/lightning_core/core/attention.hpp",
)


MODULE_DEF_TOKEN_RE = re.compile(r"\b(?:m|mod)\.def\(\s*([A-Za-z_][A-Za-z0-9_]*|\"[^\"]+\")")
SUBMODULE_RE = re.compile(r"\b(?:m|mod)\.def_submodule\(\s*\"([^\"]+)\"")
CLASS_RE = re.compile(r"py::class_<[^>]+>\s*\([^,]+,\s*\"([^\"]+)\"")
METHOD_RE = re.compile(r"^\s*\.(?:def|def_static)\(\s*\"([^\"]+)\"")
CHAR_ALIAS_TERNARY_RE = re.compile(
    r"const\s+char\s*\*\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^;]*\?\s*\"([^\"]+)\"\s*:\s*\"([^\"]+)\"\s*;"
)
CHAR_ALIAS_DIRECT_RE = re.compile(r"const\s+char\s*\*\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\"([^\"]+)\"\s*;")


TYPE_RE = re.compile(r"^\s*(enum\s+class|struct|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
FUNC_RE = re.compile(
    r"^\s*(?:inline\s+)?(?:template\s*<[^>]+>\s*)?(?:[\w:<>~]+\s+|[\w:<>~]+\s*[*&]\s*)+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
C_API_FUNC_RE = re.compile(r"^\s*[\w\s\*]+?\s+(lc[A-Za-z0-9_]+)\s*\((.*)\)\s*;\s*$")


BANNED_FUNC_NAMES = {"if", "for", "while", "switch", "return", "sizeof", "constexpr", "static_assert"}


@dataclass(frozen=True)
class CppSymbol:
    kind: str
    name: str
    header: str
    line: int


def rel_blob_url(path: str) -> str:
    return f"{REPO_BLOB_BASE}/{path}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _collect_char_aliases(text: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for m in CHAR_ALIAS_TERNARY_RE.finditer(text):
        var = m.group(1)
        out.setdefault(var, set()).update([m.group(2), m.group(3)])
    for m in CHAR_ALIAS_DIRECT_RE.finditer(text):
        var = m.group(1)
        out.setdefault(var, set()).add(m.group(2))
    return out


def generate_python_reference(root: Path) -> str:
    files: list[Path] = []
    for glob_pat in PY_BINDING_GLOBS:
        files.extend(sorted((root / Path(glob_pat).parent).glob(Path(glob_pat).name)))
    files = sorted(set(files))

    lines: list[str] = []
    lines.append("# Python API Reference (Generated)")
    lines.append("")
    lines.append("Generated from pybind binding sources in `python/bindings/`.")
    lines.append("")
    lines.append("Regenerate:")
    lines.append("`python scripts/generate_api_reference_docs.py`")
    lines.append("")

    all_submodules: set[str] = set()
    all_module_functions: set[str] = set()
    all_classes: dict[str, set[str]] = {}

    for path in files:
        text = path.read_text(encoding="utf-8")
        aliases = _collect_char_aliases(text)

        submodules: set[str] = set()
        module_functions: set[str] = set()
        classes: dict[str, set[str]] = {}
        current_class: str | None = None

        for line in text.splitlines():
            for sm in SUBMODULE_RE.finditer(line):
                submodules.add(sm.group(1))

            class_match = CLASS_RE.search(line)
            if class_match:
                current_class = class_match.group(1)
                classes.setdefault(current_class, set())

            for tok in MODULE_DEF_TOKEN_RE.finditer(line):
                token = tok.group(1)
                if token.startswith('"') and token.endswith('"'):
                    module_functions.add(token[1:-1])
                else:
                    values = aliases.get(token)
                    if values:
                        module_functions.update(values)

            if current_class:
                for method in METHOD_RE.finditer(line):
                    classes.setdefault(current_class, set()).add(method.group(1))

        all_submodules.update(submodules)
        all_module_functions.update(module_functions)
        for cls, methods in classes.items():
            all_classes.setdefault(cls, set()).update(methods)

        src_rel = path.relative_to(root).as_posix()
        lines.append(f"## `{path.name}`")
        lines.append("")
        lines.append(f"Source: [{src_rel}]({rel_blob_url(src_rel)})")
        lines.append("")

        if submodules:
            lines.append("Submodules:")
            for name in sorted(submodules):
                lines.append(f"- `{name}`")
            lines.append("")

        if module_functions:
            lines.append("Module-level functions:")
            for fn in sorted(module_functions):
                lines.append(f"- `{fn}`")
            lines.append("")

        if classes:
            lines.append("Classes:")
            for cls in sorted(classes):
                methods = sorted(classes[cls])
                lines.append(f"- `{cls}`")
                if methods:
                    lines.append("  - methods: " + ", ".join(f"`{m}`" for m in methods))
            lines.append("")

    summary_rows = [
        ["Binding files", str(len(files))],
        ["Submodules", str(len(all_submodules))],
        ["Module functions", str(len(all_module_functions))],
        ["Classes", str(len(all_classes))],
    ]
    lines.insert(4, md_table(["Metric", "Count"], summary_rows))
    lines.insert(5, "")

    return "\n".join(lines).rstrip() + "\n"


def _iter_cpp_symbols(path: Path, root: Path, extract_functions: bool) -> tuple[list[CppSymbol], list[CppSymbol], list[CppSymbol]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rel = path.relative_to(root).as_posix()
    types: list[CppSymbol] = []
    functions: list[CppSymbol] = []
    c_api_functions: list[CppSymbol] = []

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        t = TYPE_RE.match(line)
        if t:
            kind = t.group(1).replace(" ", "_")
            types.append(CppSymbol(kind=kind, name=t.group(2), header=rel, line=idx))
            continue

        c_api_match = C_API_FUNC_RE.match(line)
        if c_api_match:
            c_api_functions.append(CppSymbol(kind="c_api_function", name=c_api_match.group(1), header=rel, line=idx))
            continue

        if extract_functions:
            if "{" in line:
                continue
            if "(" in line and "=" in line and line.index("=") < line.index("("):
                continue
            if not stripped.endswith(";"):
                continue
            f = FUNC_RE.match(line)
            if not f:
                continue
            fn_name = f.group(1)
            if fn_name in BANNED_FUNC_NAMES:
                continue
            functions.append(CppSymbol(kind="function", name=fn_name, header=rel, line=idx))

    return types, functions, c_api_functions


def generate_cpp_reference(root: Path) -> str:
    symbol_types: list[CppSymbol] = []
    symbol_functions: list[CppSymbol] = []
    symbol_c_api_functions: list[CppSymbol] = []

    for rel in CPP_HEADERS:
        path = root / rel
        if not path.exists():
            continue
        types, funcs, c_funcs = _iter_cpp_symbols(path, root, rel in CPP_FUNCTION_HEADERS)
        symbol_types.extend(types)
        symbol_functions.extend(funcs)
        symbol_c_api_functions.extend(c_funcs)

    symbol_types = sorted(set(symbol_types), key=lambda s: (s.header, s.name, s.line))
    symbol_functions = sorted(set(symbol_functions), key=lambda s: (s.header, s.name, s.line))
    symbol_c_api_functions = sorted(set(symbol_c_api_functions), key=lambda s: (s.header, s.name, s.line))

    lines: list[str] = []
    lines.append("# C/C++ API Reference (Generated)")
    lines.append("")
    lines.append("Generated from public headers under `include/lightning_core/`.")
    lines.append("")
    lines.append("Regenerate:")
    lines.append("`python scripts/generate_api_reference_docs.py`")
    lines.append("")

    summary_rows = [
        ["Headers scanned", str(len(CPP_HEADERS))],
        ["Types (enum/struct/class)", str(len(symbol_types))],
        ["C++ functions", str(len(symbol_functions))],
        ["C API functions", str(len(symbol_c_api_functions))],
    ]
    lines.append(md_table(["Metric", "Count"], summary_rows))
    lines.append("")

    lines.append("## Header Set")
    lines.append("")
    for rel in CPP_HEADERS:
        lines.append(f"- [{rel}]({rel_blob_url(rel)})")
    lines.append("")

    if symbol_c_api_functions:
        rows = [
            [
                f"`{s.name}`",
                f"[{s.header}:{s.line}]({rel_blob_url(s.header)}#L{s.line})",
            ]
            for s in symbol_c_api_functions
        ]
        lines.append("## C API Functions")
        lines.append("")
        lines.append(md_table(["Function", "Source"], rows))
        lines.append("")

    if symbol_types:
        rows = [
            [
                f"`{s.name}`",
                s.kind.replace("_", " "),
                f"[{s.header}:{s.line}]({rel_blob_url(s.header)}#L{s.line})",
            ]
            for s in symbol_types
        ]
        lines.append("## Types")
        lines.append("")
        lines.append(md_table(["Name", "Kind", "Source"], rows))
        lines.append("")

    if symbol_functions:
        rows = [
            [
                f"`{s.name}`",
                f"[{s.header}:{s.line}]({rel_blob_url(s.header)}#L{s.line})",
            ]
            for s in symbol_functions
        ]
        lines.append("## C++ Function Symbols")
        lines.append("")
        lines.append(md_table(["Function", "Source"], rows))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_or_check(path: Path, content: str, check: bool) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    changed = old != content
    if not check and changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--python-out",
        type=Path,
        default=Path("docs/reference/python_api_generated.md"),
    )
    parser.add_argument(
        "--cpp-out",
        type=Path,
        default=Path("docs/reference/cpp_api_generated.md"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    py_out = (root / args.python_out).resolve()
    cpp_out = (root / args.cpp_out).resolve()

    python_doc = generate_python_reference(root)
    cpp_doc = generate_cpp_reference(root)

    py_changed = write_or_check(py_out, python_doc, args.check)
    cpp_changed = write_or_check(cpp_out, cpp_doc, args.check)

    print(
        "{"
        f"\"python_doc_changed\": {str(py_changed).lower()}, "
        f"\"cpp_doc_changed\": {str(cpp_changed).lower()}, "
        f"\"python_out\": \"{py_out}\", "
        f"\"cpp_out\": \"{cpp_out}\""
        "}"
    )

    if args.check and (py_changed or cpp_changed):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
