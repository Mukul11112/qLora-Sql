"""
Execution accuracy for text-to-SQL.

Why not string match: `SELECT name FROM t WHERE age > 30` and
`SELECT t.name FROM t WHERE t.age >= 31` are the same answer and differ as
strings. Exact-match scoring systematically understates a fine-tuned model and
is the main reason reported text-to-SQL numbers are not comparable. We run both
queries and compare what comes back.

Safety: predictions are model output and may contain DROP/DELETE/UPDATE. Every
connection here is opened read-only via a SQLite URI, so a destructive
prediction fails harmlessly instead of destroying the eval database.
"""
from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

Row = Tuple[Any, ...]

# Statements that must never run during evaluation.
_FORBIDDEN = re.compile(
    r"^\s*(drop|delete|update|insert|alter|create|replace|truncate|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)


@dataclass
class ExecResult:
    ok: bool
    rows: Optional[List[Row]] = None
    error: Optional[str] = None


@dataclass
class ScoreDetail:
    correct: bool
    reason: str
    gold_error: Optional[str] = None
    pred_error: Optional[str] = None


def _connect_readonly(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def run_query(
    db_path: str, sql: str, timeout_s: float = 10.0, allow_writes: bool = False
) -> ExecResult:
    """Execute one SELECT read-only and return its rows."""
    if not allow_writes and _FORBIDDEN.match(sql or ""):
        return ExecResult(False, error="refused: non-SELECT statement")
    if not (sql or "").strip():
        return ExecResult(False, error="empty query")

    conn = None
    try:
        conn = _connect_readonly(db_path)

        # Hard stop for runaway cartesian joins: SQLite calls this handler
        # periodically and a non-zero return aborts the query.
        deadline = [0]
        budget = max(1, int(timeout_s * 1_000_000 / 1000))

        def _guard() -> int:
            deadline[0] += 1
            return 1 if deadline[0] > budget else 0

        conn.set_progress_handler(_guard, 1000)

        cur = conn.execute(sql)
        rows = cur.fetchall()
        return ExecResult(True, rows=[tuple(r) for r in rows])
    except Exception as exc:  # noqa: BLE001 - any failure is a failed prediction
        return ExecResult(False, error=f"{type(exc).__name__}: {exc}")
    finally:
        if conn is not None:
            conn.close()


def _norm_cell(v: Any) -> Any:
    """Normalise one cell so trivial type/format differences don't count."""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        # Collapse 3.0 -> 3 so INTEGER vs REAL doesn't fail a correct query.
        if abs(v - round(v)) < 1e-9:
            return int(round(v))
        return round(v, 6)
    if isinstance(v, int):
        return v
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace").strip()
    return str(v).strip()


def _norm_rows(rows: Sequence[Row]) -> List[Row]:
    return [tuple(_norm_cell(c) for c in r) for r in rows]


def _sort_rows(rows: Sequence[Row]) -> List[Row]:
    # Sort by string form: rows may mix types across columns, which would make
    # a natural tuple sort raise TypeError.
    return sorted(rows, key=lambda r: tuple(str(c) for c in r))


def has_order_by(sql: str) -> bool:
    """True if the gold query pins row order, so the comparison must too."""
    stripped = re.sub(r"'[^']*'", "''", sql or "")
    return re.search(r"\border\s+by\b", stripped, re.IGNORECASE) is not None


def compare_rows(
    gold: Sequence[Row], pred: Sequence[Row], order_matters: bool
) -> Tuple[bool, str]:
    g, p = _norm_rows(gold), _norm_rows(pred)

    if len(g) != len(p):
        return False, f"row count {len(p)} != gold {len(g)}"

    if not g:
        return True, "both empty"

    if len(g[0]) != len(p[0]):
        return False, f"column count {len(p[0])} != gold {len(g[0])}"

    if order_matters:
        return (g == p, "exact row order match" if g == p else "row order differs")

    gs, ps = _sort_rows(g), _sort_rows(p)
    if gs == ps:
        return True, "set match (order-insensitive)"

    # Same data, columns emitted in a different order, e.g. SELECT b, a.
    if len(g[0]) > 1:
        gcols = _sort_rows([tuple(sorted((str(c) for c in r))) for r in g])
        pcols = _sort_rows([tuple(sorted((str(c) for c in r))) for r in p])
        if gcols == pcols:
            return True, "match ignoring column order"

    return False, "different values"


def score_one(
    db_path: str,
    gold_sql: str,
    pred_sql: str,
    timeout_s: float = 10.0,
) -> ScoreDetail:
    """Execution accuracy for a single prediction."""
    gold_res = run_query(db_path, gold_sql, timeout_s, allow_writes=True)
    if not gold_res.ok:
        # A broken gold query is a dataset problem, not a model failure.
        return ScoreDetail(False, "gold query failed", gold_error=gold_res.error)

    pred_res = run_query(db_path, pred_sql, timeout_s)
    if not pred_res.ok:
        return ScoreDetail(False, "prediction failed to execute",
                           pred_error=pred_res.error)

    ok, reason = compare_rows(
        gold_res.rows or [], pred_res.rows or [], has_order_by(gold_sql)
    )
    return ScoreDetail(ok, reason)
