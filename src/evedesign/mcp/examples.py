"""
Listing and extraction of the example notebooks under examples/ for the MCP server's
list_examples/get_example tools.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _examples_root() -> Path | None:
    # src/evedesign/mcp/examples.py -> repo root is four parents up
    # won't work if pip-installed and not cloned (no examples in pypi package...)
    candidate = Path(__file__).resolve().parents[3] / "examples"
    return candidate if candidate.is_dir() else None


def _title_from_notebook(cells: list[dict[str, Any]]) -> str | None:
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            text = "".join(cell.get("source", [])).strip()
            if text:
                return text.lstrip("#").strip().splitlines()[0]
    return None


def _not_found_error() -> dict[str, Any]:
    return {
        "ok": False,
        "error": (
            "examples/ directory not found (expected when evedesign is installed from a "
            "package rather than run from the repo directly)"
        ),
    }


def list_examples() -> dict[str, Any]:
    """
    List the example notebooks under examples/, one subdirectory per topic

    Only available when running from a source checkout of the repository: examples/ is not
    part of the installed package.

    Returns
    -------
    Name, topic and title for each notebook found
    """
    root = _examples_root()
    if root is None:
        return _not_found_error()

    found = []
    for notebook_path in sorted(root.glob("*/*.ipynb")):
        try:
            data = json.loads(notebook_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        found.append(
            {
                "name": notebook_path.relative_to(root).as_posix(),
                "topic": notebook_path.parent.name,
                "title": _title_from_notebook(data.get("cells", [])),
            }
        )
    return {"ok": True, "examples": found, "n_total": len(found)}


def get_example(name: str) -> dict[str, Any]:
    """
    Read one example notebook's markdown and code cells as text

    Parameters
    ----------
    name
        Notebook name as returned by list_examples(), e.g.
        "scoring_and_transforming_instances/scoring_and_transforming_instances.ipynb"

    Returns
    -------
    Title and cells (each with its type and source) for the requested notebook
    """
    root = _examples_root()
    if root is None:
        return _not_found_error()

    notebook_path = (root / name).resolve()
    if root not in notebook_path.parents or notebook_path.suffix != ".ipynb":
        return {"ok": False, "error": f"not a valid example name: {name!r}"}
    if not notebook_path.is_file():
        return {"ok": False, "error": f"no example named {name!r}", "hint": "call list_examples() for valid names"}

    try:
        data = json.loads(notebook_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"could not read notebook: {exc}"}

    cells = []
    for cell in data.get("cells", []):
        cell_type = cell.get("cell_type")
        if cell_type not in ("markdown", "code"):
            continue
        cells.append({"type": cell_type, "source": "".join(cell.get("source", []))})

    return {
        "ok": True,
        "name": name,
        "title": _title_from_notebook(data.get("cells", [])),
        "cells": cells,
    }
