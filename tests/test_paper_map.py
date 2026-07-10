"""PAPER_MAP.md anchor resolver — PROGRAM.md §14.

Parses ``PAPER_MAP.md`` (a table ``paper ref | symbol | anchor``) and asserts every
``module:QualifiedSymbol`` anchor imports and resolves (importlib + getattr, dotted
attrs allowed). Written so it PASSES (skips) today and ACTIVATES the moment the docs
agent writes PAPER_MAP.md — a moved-but-not-updated entry then fails CI.

The anchor module part may be written either as a dotted module path
(``rina.simulation``) or as a file path (``rina/simulation.py``); both are normalized to
an importable module before resolution.
"""

from __future__ import annotations

import importlib
import os
import re

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PAPER_MAP = os.path.join(_REPO_ROOT, "PAPER_MAP.md")


def _module_to_import(mod: str) -> str:
    """Normalize a file-path-or-dotted module token to an importable module name."""
    mod = mod.strip()
    if mod.endswith(".py"):
        mod = mod[:-3]
    # rina/simulation.py -> rina.simulation ; rina/simulation -> rina.simulation
    mod = mod.replace("/", ".").replace("\\", ".")
    return mod


def _parse_anchors(text: str) -> list[tuple[str, str]]:
    """Extract (anchor_string, qualified_symbol_path) pairs from the markdown table.

    A row looks like::

        | F_S | ModelSim | rina/simulation.py:ModelSim |

    The anchor is the ``module:Symbol`` (or ``module:Sym.attr``) token; rows without a
    ``:``-bearing anchor (headers, separators, prose) are skipped.
    """
    anchors: list[tuple[str, str]] = []
    # Match `module[.py]:QualifiedSymbol` inside a table cell or backticks.
    pat = re.compile(r"`?([\w./\\]+\.py|[\w.]+):([\w.]+)`?")
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        # skip the markdown header-separator row (---|---|---)
        if set(line.strip()) <= set("|-: "):
            continue
        for m in pat.finditer(line):
            mod_token, sym = m.group(1), m.group(2)
            # Heuristic: a real anchor's module token is a python module/path; require it
            # to reference rina/cli/lib so we do not pick up incidental "Eq.(8)" style text.
            if not (mod_token.startswith(("rina", "cli", "lib")) ):
                continue
            anchors.append((mod_token, sym))
    return anchors


def _load_anchors():
    if not os.path.exists(_PAPER_MAP):
        return None
    with open(_PAPER_MAP, encoding="utf-8") as f:
        return _parse_anchors(f.read())


def test_paper_map_anchors_resolve():
    anchors = _load_anchors()
    if anchors is None:
        pytest.skip("PAPER_MAP.md not yet written")
    assert anchors, "PAPER_MAP.md exists but no module:Symbol anchors were parsed"

    failures = []
    for mod_token, sym in anchors:
        mod_name = _module_to_import(mod_token)
        try:
            obj = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{mod_token}:{sym} -> import {mod_name!r} failed: {exc}")
            continue
        try:
            for part in sym.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            failures.append(f"{mod_token}:{sym} -> {mod_name} has no attribute {sym!r}")
    assert not failures, "unresolved PAPER_MAP.md anchors:\n" + "\n".join(failures)
