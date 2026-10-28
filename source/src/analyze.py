#!/usr/bin/env python3
"""Compare SQL prediction files and diagnose fine-tuning regressions."""

import argparse
import json
import math
from pathlib import Path


def load_predictions(path):
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("records", data.get("predictions", data))

    if isinstance(data, dict):
        data = list(data.values())

    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of predictions")

    result = {}
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            continue

        question = row.get("question")
        if question is None:
            continue

        # Include database identity when available.
        key = (row.get("db_id", ""), question.strip())
        if key in result:
            raise ValueError(
                f"{path}: duplicate question/database key: {key!r}"
            )

        correct = row.get(
            "correct",
            row.get("exec_correct", row.get("execution_correct"))
        )

        if correct is None:
            raise ValueError(
                f"{path}: prediction {index} has no correctness field. "
                f"Available fields: {list(row.keys())}"
            )

        result[key] = row

    if not result:
        raise ValueError(
            f"{path}: no usable predictions. "
            "Check whether records contain a 'question' field."
        )

    return result


def sql_text(row, *fields):
    for field in fields:
        value = row.get(field)
        if value is not None:
            return str(value).strip()
    return "(not available)"


def exact_mcnemar_p(b, c):
    """Two-sided exact binomial test for discordant paired outcomes."""
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def compare(reference, candidate, ref_name, cand_name, examples=10):
    common = sorted(set(reference) & set(candidate))

    if not common:
        print(f"\nERROR: No matched questions between "
              f"{ref_name} and {cand_name}.")
        print("Check question text and database IDs in both files.")
        return

    a = {k: bool(reference[k].get(
        "correct",
        reference[k].get("exec_correct",
                         reference[k].get("execution_correct"))
    )) for k in common}

    b = {k: bool(candidate[k].get(
        "correct",
        candidate[k].get("exec_correct",
                         candidate[k].get("execution_correct"))
    )) for k in common}

    gained = [k for k in common if not a[k] and b[k]]
    lost = [k for k in common if a[k] and not b[k]]
    both_correct = sum(a[k] and b[k] for k in common)
    both_wrong = sum(not a[k] and not b[k] for k in common)

    ref_acc = sum(a.values()) / len(common)
    cand_acc = sum(b.values()) / len(common)
    p = exact_mcnemar_p(len(gained), len(lost))

    print("\n" + "=" * 72)
    print(f"REFERENCE: {ref_name}")
    print(f"CANDIDATE: {cand_name}")
    print("=" * 72)
    print(f"Matched questions : {len(common)}")
    print(f"Reference accuracy: {ref_acc:.2%}")
    print(f"Candidate accuracy: {cand_acc:.2%}")
    print(f"Accuracy change   : {(cand_acc-ref_acc)*100:+.2f} percentage points")
    print(f"Both correct      : {both_correct}")
    print(f"Both incorrect    : {both_wrong}")
    print(f"Improved          : {len(gained)}")
    print(f"Regressed         : {len(lost)}")
    print(f"McNemar exact p   : {p:.6f}")

    if p < 0.05:
        print("Paired difference: statistically significant at 5%.")
    else:
        print("Paired difference: not statistically significant at 5%.")

    def show(title, keys):
        print(f"\n--- {title} (up to {examples}) ---")
        if not keys:
            print("None.")
            return

        for i, key in enumerate(keys[:examples], 1):
            old = reference[key]
            new = candidate[key]
            print(f"\n{i}. Question: {key[1]}")
            print(f"   Database: {key[0] or '(not recorded)'}")
            print(f"   Gold SQL: {sql_text(new, 'gold_sql', 'gold')}")
            print(f"   Reference prediction: "
                  f"{sql_text(old, 'pred_sql', 'prediction')}")
            print(f"   Candidate prediction: "
                  f"{sql_text(new, 'pred_sql', 'prediction')}")
            error = new.get("pred_error")
            if error:
                print(f"   Candidate error: {str(error)[:300]}")

    show(f"REGRESSIONS: correct in {ref_name}, wrong in {cand_name}", lost)
    show(f"IMPROVEMENTS: wrong in {ref_name}, correct in {cand_name}", gained)


def main():
    parser = argparse.ArgumentParser(
        description="Compare base and fine-tuned text-to-SQL predictions."
    )
    parser.add_argument("reference", help="Reference prediction JSON")
    parser.add_argument("candidate", help="Candidate prediction JSON")
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    reference = load_predictions(args.reference)
    candidate = load_predictions(args.candidate)

    print(f"Loaded {len(reference)} reference predictions.")
    print(f"Loaded {len(candidate)} candidate predictions.")
    print(f"Reference-only questions: {len(set(reference) - set(candidate))}")
    print(f"Candidate-only questions: {len(set(candidate) - set(reference))}")

    compare(
        reference,
        candidate,
        Path(args.reference).stem,
        Path(args.candidate).stem,
        max(0, args.examples),
    )


if __name__ == "__main__":
    main()
