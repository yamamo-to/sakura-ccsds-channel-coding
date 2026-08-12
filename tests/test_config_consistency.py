"""Guard against drift between setup.py extras and pyproject.toml metadata.

``pyproject.toml`` is the authoritative source of package metadata for modern
setuptools builds, but ``setup.py`` still declares its own ``extras_require``.
These two declarations must stay in sync; this module parses both files (without
executing ``setup.py``) and enforces that the ``dev`` extra lists are identical.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_dev_extras() -> list[str]:
    """Return the ``dev`` optional-dependencies list from pyproject.toml."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["optional-dependencies"]["dev"]


def _setup_py_dev_extras() -> list[str]:
    """Extract the ``dev`` extras list from setup.py's ``setup()`` call."""
    tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            for kw in node.keywords:
                if kw.arg == "extras_require":
                    extras = ast.literal_eval(kw.value)
                    return extras["dev"]
    raise AssertionError("setup() call with extras_require not found in setup.py")


def test_dev_extras_match_pyproject() -> None:
    """The dev extras in setup.py must exactly match pyproject.toml."""
    assert _setup_py_dev_extras() == _pyproject_dev_extras()
