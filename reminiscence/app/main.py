"""REMINISCENCE command-line interface.

The PySide6 GUI (app/ui) binds to the same ReminiscenceEngine; this CLI makes
the engine fully usable/testable headless — including offline operation.

Usage:
  python -m reminiscence.app.main import <file-or-dir>
  python -m reminiscence.app.main search "query text"
  python -m reminiscence.app.main ask "question"
  python -m reminiscence.app.main timeline --days 30
  python -m reminiscence.app.main sources
  python -m reminiscence.app.main delete <source_id>
  python -m reminiscence.app.main status
  python -m reminiscence.app.main benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC
from pathlib import Path

from ..benchmarks.runner import BenchmarkRunner, current_machine
from .services.engine import ReminiscenceEngine


def _setup_logging(verbose: bool) -> None:
    # Structured local logging; raw document content is NEVER logged unless
    # REMINISCENCE_DEBUG_CONTENT=1 is explicitly set by a developer.
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def cmd_import(engine: ReminiscenceEngine, args) -> int:
    target = Path(args.path)
    files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
    failures = 0
    for f in files:
        try:
            res = engine.ingest_now(f)
            print(f"[ok] {f.name}: {res.events_created} memory events ({res.modality.value})")
            for w in res.warnings:
                print(f"     warning: {w}")
        except Exception as e:
            failures += 1
            print(f"[skip] {f}: {e}")
    return 1 if failures and failures == len(files) else 0


def cmd_search(engine: ReminiscenceEngine, args) -> int:
    cq = engine.classifier.classify(args.query)
    print(
        f"classified: {[c.value for c in cq.categories]}"
        + (f" window={cq.time_range[0].date()}..{cq.time_range[1].date()}" if cq.time_range else "")
    )
    results = engine.search(args.query, top_k=args.top)
    if not results:
        print("No memories found.")
        return 0
    for i, c in enumerate(results, 1):
        ev = c.event
        loc = []
        if ev.page is not None:
            loc.append(f"p.{ev.page}")
        if ev.section:
            loc.append(ev.section)
        if ev.timestamp_start is not None:
            loc.append(f"{ev.timestamp_start:.1f}s-{(ev.timestamp_end or ev.timestamp_start):.1f}s")
        snippet = ev.content.replace("\n", " ")[:120]
        print(
            f"{i:2d}. [{ev.modality.value}] score={c.score:.3f} "
            f"(sem={c.semantic:.2f}/lex={c.lexical:.2f}) {' · '.join(loc)}\n    {snippet}"
        )
    return 0


def cmd_ask(engine: ReminiscenceEngine, args) -> int:
    ans = engine.ask(args.question)
    print(
        json.dumps(
            {
                "answer": ans.answer,
                "grounded": ans.grounded,
                "evidence": ans.evidence,
                "backend": ans.backend,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_timeline(engine: ReminiscenceEngine, args) -> int:
    from datetime import datetime, timedelta

    end = datetime.now(UTC)
    start = end - timedelta(days=args.days)
    events = engine.timeline(start.isoformat(), end.isoformat())
    for ev in events:
        ts = ev.created_at or ""
        print(f"{ts}  [{ev.modality.value}] {ev.content[:90]!r}")
    print(f"\n{len(events)} events between {start.date()} and {end.date()}")
    return 0


def cmd_sources(engine: ReminiscenceEngine, args) -> int:
    for s in engine.sources():
        print(f"{s['id']}  {s['modality']:8s} {s['name']:40s} imported={s['imported_at']}")
    return 0


def cmd_delete(engine: ReminiscenceEngine, args) -> int:
    n = engine.delete_source(args.source_id)
    print(f"deleted source {args.source_id} ({'removed' if n else 'not found'})")
    return 0


def cmd_status(engine: ReminiscenceEngine, args) -> int:
    st = engine.offline_status()
    perf = engine.performance_snapshot()
    print(
        json.dumps(
            {"offline_status": st, "index_size": perf["index_size"], "machine": current_machine()},
            indent=2,
        )
    )
    return 0


def cmd_benchmark(engine: ReminiscenceEngine, args) -> int:
    """Local developer benchmark on THIS machine (never reference numbers)."""
    runner = BenchmarkRunner(db=engine.db)

    texts = [
        f"benchmark sample sentence number {i} about transformers and memory" for i in range(64)
    ]

    r1 = runner.bench_fn(
        lambda: engine.embedder.embed(texts),
        model_id=engine.embedder.model_id,
        task="embedding_batch64",
        input_desc="64 texts x ~10 words",
        iterations=5,
    )
    q = engine.embedder.embed(["complexity of self attention"])[0]
    r2 = runner.bench_fn(
        lambda: engine.index.search(q, k=10),
        model_id="numpy-ann",
        task="vector_search",
        input_desc=f"k=10 over {len(engine.index)} vectors",
        iterations=20,
    )
    r3 = runner.bench_fn(
        lambda: engine.db.fts_search("transformers attention", limit=20),
        model_id="sqlite-fts5",
        task="lexical_search",
        input_desc="query 'transformers attention'",
        iterations=20,
    )
    print(runner.report())
    print(
        "\nNOTE: NPU utilization requires QNN EP + on-target profiling; "
        "values above are measured on the current machine."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reminiscence", description="REMINISCENCE — Your Private Multimodal AI Memory"
    )
    p.add_argument("--data-dir", default=None, help="local data directory")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("import", help="ingest a file or directory")
    s.add_argument("path")
    s.set_defaults(fn=cmd_import)

    s = sub.add_parser("search", help="hybrid memory search")
    s.add_argument("query")
    s.add_argument("--top", type=int, default=8)
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("ask", help="grounded answer with evidence JSON")
    s.add_argument("question")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("timeline", help="chronological memories")
    s.add_argument("--days", type=int, default=30)
    s.set_defaults(fn=cmd_timeline)

    s = sub.add_parser("sources", help="list ingested sources")
    s.set_defaults(fn=cmd_sources)

    s = sub.add_parser("delete", help="explicitly delete a source + memories")
    s.add_argument("source_id")
    s.set_defaults(fn=cmd_delete)

    s = sub.add_parser("status", help="offline/AI status")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("benchmark", help="run local benchmarks on this machine")
    s.set_defaults(fn=cmd_benchmark)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    from .services.engine import EnginePaths

    paths = EnginePaths.default()
    if args.data_dir:
        base = Path(args.data_dir)
        paths = EnginePaths(base, base / "reminiscence.db", base / "models")
    engine = ReminiscenceEngine(paths=paths)
    try:
        return args.fn(engine, args)
    finally:
        engine.close()


if __name__ == "__main__":
    sys.exit(main())
