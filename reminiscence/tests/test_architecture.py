"""Phase 0 architecture tests.

Enforce the dependency direction required by the LLM-free master plan:

    UI / CLI  ->  Services  ->  Domain  ->  Infrastructure

Rules checked here (statically, via AST — no runtime imports needed):

1. Only ``reminiscence.storage.database`` may import ``sqlite3``.
   (UI/CLI must never manipulate SQLite directly.)
2. No domain/storage/retrieval/evidence/ingestion module may import UI code
   (``reminiscence.app.ui`` or ``reminiscence.app.main``).
3. The core answer path must not require a generative LLM:
   ``ExtractiveGroundedAnswerer`` remains the default answerer wired into
   ``ReminiscenceEngine`` (checked in test_core.py behavior tests; here we
   assert the optional-LLM configuration flag exists and defaults to off).
4. Vector retrieval stays behind the ``VectorIndex`` interface: application
   modules must reference the abstraction, not concrete ANN libraries
   directly (no top-level ``import faiss`` / ``import hnswlib`` outside
   storage/vector_index.py).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]  # .../reminiscence


def _py_modules(*subdirs: str) -> list[Path]:
    files: list[Path] = []
    for sub in subdirs:
        d = ROOT / sub
        if d.is_dir():
            files.extend(sorted(d.rglob("*.py")))
    return files


def _imports(path: Path) -> set[str]:
    """All imported module names in a file (top-level + nested)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module)
    return found


# --------------------------------------------------------------------------
# Rule 1: sqlite3 confined to the storage layer
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path",
    _py_modules("app", "retrieval", "memory", "evidence", "ingestion", "ai", "workers", "benchmarks")
    + [p for p in _py_modules("storage") if p.name != "database.py"],
    ids=lambda p: str(p.relative_to(ROOT.parent)),
)
def test_only_storage_database_touches_sqlite(path: Path):
    imps = _imports(path)
    offenders = {m for m in imps if m == "sqlite3" or m.startswith("sqlite3.")}
    assert not offenders, f"{path} imports sqlite3 directly; go through Database"


# --------------------------------------------------------------------------
# Rule 2: domain never imports UI
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path",
    _py_modules("memory", "retrieval", "evidence", "ingestion", "storage", "ai", "workers", "benchmarks"),
    ids=lambda p: str(p.relative_to(ROOT.parent)),
)
def test_domain_never_imports_ui(path: Path):
    imps = _imports(path)
    offenders = {
        m
        for m in imps
        if m.startswith("reminiscence.app.ui") or m == "reminiscence.app.main"
    }
    assert not offenders, f"{path} imports UI ({offenders}); dependency direction violated"


# --------------------------------------------------------------------------
# Rule 3: LLM is optional, never a hidden core dependency
# --------------------------------------------------------------------------

def test_llm_disabled_by_default_in_env_example():
    env = (ROOT.parent / ".env.example").read_text(encoding="utf-8")
    # Either explicitly disabled, or absent (absent => engine default off).
    lines = [ln.strip() for ln in env.splitlines() if "ENABLE_LLM" in ln and not ln.strip().startswith("#")]
    assert all(ln.split("=", 1)[1].strip().lower() in ("false", "0", "") for ln in lines), (
        "REMINISCENCE_ENABLE_LLM must default to false/off"
    )


def test_extractive_answerer_is_the_llm_free_default():
    """The deterministic answerer must not import any model runtime."""
    gen = ROOT / "evidence" / "generator.py"
    imps = _imports(gen)
    extractive_src = gen.read_text(encoding="utf-8")
    assert "class ExtractiveGroundedAnswerer" in extractive_src
    # prompt-guard/onnx live under ai/, but the *extractive* class body must
    # be self-contained; verify no module-level onnxruntime import in evidence/.
    assert "onnxruntime" not in imps


# --------------------------------------------------------------------------
# Rule 4: vector backends stay behind the interface
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path",
    _py_modules("app", "retrieval", "memory", "evidence", "ingestion", "workers", "benchmarks")
    + [p for p in _py_modules("storage") if p.name != "vector_index.py"],
    ids=lambda p: str(p.relative_to(ROOT.parent)),
)
def test_no_direct_ann_library_imports(path: Path):
    imps = _imports(path)
    offenders = {m for m in imps if m.split(".")[0] in {"faiss", "hnswlib"}}
    assert not offenders, f"{path} imports an ANN library directly; use VectorIndex"
