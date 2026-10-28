"""
Build MLX-format training data from Spider.

Emits train.jsonl / valid.jsonl / test.jsonl with {"prompt", "completion"},
which is the `completions` format mlx-lm detects automatically. Combined with
`--mask-prompt`, loss is computed on the SQL only, not on the schema text. That
matters: schemas are 10-50x longer than the answers, so without masking the
model spends almost all its gradient budget learning to echo DDL.

Also writes eval_manifest.json, pairing every eval question with the .sqlite
file its gold query runs against, which the benchmark needs for execution
accuracy.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt import SYSTEM_PROMPT, build_user_turn  # noqa: E402


def schema_from_sqlite(db_file: Path, max_chars: int = 6000) -> Optional[str]:
    """Read real CREATE TABLE statements straight from the database file.

    Preferred over Spider's tables.json: it is exactly what the DB enforces,
    including types and foreign keys, which measurably helps join accuracy.
    """
    if not db_file.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception:
        return None

    stmts: List[str] = []
    for (sql,) in rows:
        s = " ".join(str(sql).split())          # collapse whitespace
        if s.lower().startswith("create table sqlite_"):
            continue
        stmts.append(s + ";")

    if not stmts:
        return None
    schema = "\n".join(stmts)
    return schema[:max_chars]


def load_spider_split(path: Path) -> List[dict]:
    with open(path) as f:
        return json.load(f)


def build_records(
    items: List[dict], db_root: Path, schema_cache: Dict[str, str]
) -> tuple[List[dict], List[dict]]:
    """Return (jsonl_records, manifest_records)."""
    jsonl, manifest = [], []
    skipped = 0

    for ex in items:
        db_id = ex.get("db_id")
        question = (ex.get("question") or "").strip()
        gold = (ex.get("query") or ex.get("SQL") or "").strip()
        if not (db_id and question and gold):
            skipped += 1
            continue

        if db_id not in schema_cache:
            db_file = db_root / db_id / f"{db_id}.sqlite"
            s = schema_from_sqlite(db_file)
            if s is None:
                skipped += 1
                continue
            schema_cache[db_id] = s
        schema = schema_cache[db_id]

        jsonl.append({
            "prompt": f"{SYSTEM_PROMPT}\n\n{build_user_turn(question, schema)}",
            "completion": " " + gold.rstrip(";"),
        })
        manifest.append({
            "db_id": db_id,
            "db_path": str((db_root / db_id / f"{db_id}.sqlite").resolve()),
            "question": question,
            "schema": schema,
            "gold_sql": gold,
        })

    if skipped:
        print(f"  skipped {skipped} items (missing fields or database file)")
    return jsonl, manifest


def write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(records):>6} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Spider -> MLX training data")
    ap.add_argument("--spider-root", required=True,
                    help="unzipped Spider dir (contains train_spider.json, dev.json, database/)")
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--valid-size", type=int, default=400)
    ap.add_argument("--eval-size", type=int, default=400,
                    help="how many dev questions to benchmark on (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.spider_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    db_root = root / "database"
    if not db_root.exists():
        raise SystemExit(f"No database/ directory under {root}")

    random.seed(args.seed)
    cache: Dict[str, str] = {}

    print("Training split:")
    train_items = load_spider_split(root / "train.json")
    train_recs, _ = build_records(train_items, db_root, cache)
    random.shuffle(train_recs)
    valid_recs = train_recs[: args.valid_size]
    train_recs = train_recs[args.valid_size:]
    write_jsonl(out / "train.jsonl", train_recs)
    write_jsonl(out / "valid.jsonl", valid_recs)

    print("Dev split (held out, used for execution accuracy):")
    dev_items = load_spider_split(root / "validation.json")
    dev_recs, dev_manifest = build_records(dev_items, db_root, cache)
    write_jsonl(out / "test.jsonl", dev_recs)

    if args.eval_size and args.eval_size < len(dev_manifest):
        random.shuffle(dev_manifest)
        dev_manifest = dev_manifest[: args.eval_size]
    with open(out / "eval_manifest.json", "w") as f:
        json.dump(dev_manifest, f, indent=2)
    print(f"  wrote {len(dev_manifest):>6} -> {out / 'eval_manifest.json'}")
    print(f"\nDatabases covered: {len(cache)}")


if __name__ == "__main__":
    main()
