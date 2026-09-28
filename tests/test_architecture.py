"""Executable license-boundary and package-independence checks."""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "src" / "fast_ofm_core"
FORBIDDEN_IMPORTS = (
    "openflexure_microscope_server",
    "openflexure_stitching",
    "labthings_fastapi",
)


def imported_names(path: Path) -> set[str]:
    """Return absolute import targets from one Python source file."""
    tree = ast.parse(path.read_text())
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            values.add(node.module)
    return values


def test_core_has_no_openflexure_or_labthings_imports() -> None:
    """Keep implementation out of the GPL/LGPL application process."""
    violations: list[str] = []
    for path in SOURCE.rglob("*.py"):
        for name in imported_names(path):
            if any(name == root or name.startswith(f"{root}.") for root in FORBIDDEN_IMPORTS):
                violations.append(f"{path.relative_to(SOURCE)}: {name}")
    assert violations == []


def test_core_tree_contains_no_openflexure_package_namespace() -> None:
    """Prevent accidental vendoring even when import checks would miss data files."""
    names = {path.name for path in SOURCE.rglob("*")}
    assert "openflexure_microscope_server" not in names
    assert "openflexure_stitching" not in names
