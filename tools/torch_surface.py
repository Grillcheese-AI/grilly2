"""List the torch symbols a Python codebase uses, and which grilly provides.

    python tools/torch_surface.py PATH [PATH ...] [--grilly]

Walks every .py file under the given paths (skipping virtualenvs, caches and
build output), resolves names bound by `import torch...` / `from torch...
import ...`, and records every fully-qualified torch attribute reached from
them, e.g. `torch.cat`, `torch.nn.Linear`, `torch.nn.functional.cross_entropy`.

This counts module-level symbols only. Tensor methods (`x.view`, `x.sum`) are
reached through values, not names, and are not visible to a static scan.

With --grilly, each symbol is also checked against the importable `grilly`
package (with `torch` replaced by `grilly` in the name) and the report says
how many resolve.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import pathlib
import sys
from collections import defaultdict

SKIP_DIRS = {".git", "__pycache__", "build", "dist", "node_modules", "site-packages"}


def python_files(root: pathlib.Path):
    if root.is_file():
        yield root
        return
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if parts & SKIP_DIRS or any(p.startswith(".venv") or p == "venv" for p in path.parts):
            continue
        yield path


def dotted(node: ast.AST) -> list[str] | None:
    """`a.b.c` -> ["a", "b", "c"]; None if the chain does not start at a Name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return None


class Scan(ast.NodeVisitor):
    def __init__(self) -> None:
        self.bound: dict[str, str] = {}  # local name -> torch qualified name
        self.found: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "torch" or alias.name.startswith("torch."):
                if alias.asname:
                    self.bound[alias.asname] = alias.name
                else:  # `import torch.nn` binds `torch`
                    self.bound["torch"] = "torch"
                self.found.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if node.level == 0 and (mod == "torch" or mod.startswith("torch.")):
            for alias in node.names:
                if alias.name == "*":
                    continue
                qual = f"{mod}.{alias.name}"
                self.bound[alias.asname or alias.name] = qual
                self.found.add(qual)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = dotted(node)
        if chain and chain[0] in self.bound:
            self.found.add(".".join([self.bound[chain[0]], *chain[1:]]))
            return  # the full chain is recorded; skip its prefixes
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.bound:
            self.found.add(self.bound[node.id])


def scan(paths: list[pathlib.Path]) -> dict[str, set[str]]:
    """symbol -> files using it"""
    uses: dict[str, set[str]] = defaultdict(set)
    for root in paths:
        for path in python_files(root):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError):
                continue
            s = Scan()
            s.visit(tree)
            for sym in s.found:
                uses[sym].add(str(path))
    # A module like `torch.nn` that is only a prefix of other symbols is not
    # itself a symbol the code calls.
    return {
        sym: files
        for sym, files in uses.items()
        if not any(other.startswith(sym + ".") for other in uses) or sym.count(".") >= 3
    }


def resolves_in_grilly(sym: str) -> bool:
    parts = sym.split(".")
    parts[0] = "grilly"
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return False
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=pathlib.Path)
    ap.add_argument("--grilly", action="store_true", help="check each symbol against grilly")
    args = ap.parse_args()

    uses = scan(args.paths)
    have = {s for s in uses if args.grilly and resolves_in_grilly(s)}
    for sym in sorted(uses):
        mark = ("ok  " if sym in have else "--  ") if args.grilly else ""
        print(f"{mark}{sym}  ({len(uses[sym])} files)")
    print(f"\n{len(uses)} distinct torch symbols", end="")
    print(f"; {len(have)} resolve in grilly" if args.grilly else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
