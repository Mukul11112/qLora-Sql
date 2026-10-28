"""
Benchmark text-to-SQL variants on Apple Silicon: accuracy, latency, memory.

Measurement notes that decide whether the numbers mean anything:

* MLX is lazily evaluated. Timing a call that only built a graph measures
  nothing. Every timed region here is forced with mx.eval/synchronize.
* Decoding is greedy (temp=0). Sampling makes accuracy irreproducible and
  makes two variants incomparable.
* A warmup pass runs first, so weight paging and kernel compilation don't land
  inside the measured window.
* Time to first token (prefill) and tokens/sec (decode) are reported
  separately. 4-bit mainly speeds up decode, which is memory-bandwidth bound;
  reporting one blended latency hides that.
* Peak memory is read from MLX's own allocator, reset per variant.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt import extract_sql, render_prompt  # noqa: E402
from sqleval import score_one  # noqa: E402

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


# ---------------------------------------------------------------- memory ----

def reset_peak_memory() -> None:
    for fn in ("reset_peak_memory", "metal.reset_peak_memory"):
        obj: Any = mx
        try:
            for part in fn.split("."):
                obj = getattr(obj, part)
            obj()
            return
        except AttributeError:
            continue


def peak_memory_gb() -> float:
    for fn in ("get_peak_memory", "metal.get_peak_memory"):
        obj: Any = mx
        try:
            for part in fn.split("."):
                obj = getattr(obj, part)
            return obj() / 1e9
        except AttributeError:
            continue
    return float("nan")


def sync() -> None:
    """Force all queued MLX work to finish, so timings are real."""
    try:
        mx.synchronize()
    except AttributeError:
        mx.eval(mx.array(0.0))


# ------------------------------------------------------------ generation ----

def generate_one(model, tokenizer, prompt_text: str, max_tokens: int) -> Dict[str, Any]:
    """Greedy-decode one prompt, timing prefill and decode separately."""
    sampler = make_sampler(temp=0.0)

    sync()
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    pieces: List[str] = []
    n_tokens = 0
    prompt_tokens = 0

    for resp in stream_generate(
        model, tokenizer, prompt=prompt_text,
        max_tokens=max_tokens, sampler=sampler,
    ):
        if t_first is None:
            t_first = time.perf_counter()
        pieces.append(resp.text)
        n_tokens += 1
        prompt_tokens = getattr(resp, "prompt_tokens", prompt_tokens) or prompt_tokens

    sync()
    t_end = time.perf_counter()

    ttft = (t_first - t_start) if t_first else float("nan")
    decode_s = (t_end - t_first) if t_first else float("nan")
    tps = (n_tokens - 1) / decode_s if t_first and decode_s > 0 and n_tokens > 1 else 0.0

    return {
        "text": "".join(pieces),
        "ttft_s": ttft,
        "total_s": t_end - t_start,
        "gen_tokens": n_tokens,
        "prompt_tokens": prompt_tokens,
        "decode_tps": tps,
    }


# ------------------------------------------------------------- one variant --

def evaluate_variant(
    name: str,
    model_path: str,
    manifest: List[dict],
    max_tokens: int,
    warmup: int,
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    print(f"\n=== {name} ===")
    print(f"  loading {model_path}" + (f" + adapter {adapter_path}" if adapter_path else ""))

    reset_peak_memory()
    t0 = time.perf_counter()
    kwargs = {"adapter_path": adapter_path} if adapter_path else {}
    model, tokenizer = load(model_path, **kwargs)
    sync()
    load_s = time.perf_counter() - t0
    weights_gb = peak_memory_gb()
    print(f"  loaded in {load_s:.1f}s, weights ~{weights_gb:.2f} GB")

    # Warmup: excluded from all reported timings.
    for ex in manifest[:warmup]:
        generate_one(model, tokenizer,
                     render_prompt(tokenizer, ex["question"], ex["schema"]),
                     max_tokens)

    reset_peak_memory()
    records, ttfts, tpss = [], [], []
    n_correct = n_exec = n_scored = 0

    for i, ex in enumerate(manifest, 1):
        prompt_text = render_prompt(tokenizer, ex["question"], ex["schema"])
        out = generate_one(model, tokenizer, prompt_text, max_tokens)
        pred_sql = extract_sql(out["text"])

        detail = score_one(ex["db_path"], ex["gold_sql"], pred_sql)
        if detail.gold_error:
            # Unusable item: don't let a broken gold query count against any model.
            continue

        n_scored += 1
        n_correct += int(detail.correct)
        n_exec += int(detail.pred_error is None)
        ttfts.append(out["ttft_s"])
        if out["decode_tps"] > 0:
            tpss.append(out["decode_tps"])

        records.append({
            "question": ex["question"], "db_id": ex["db_id"],
            "gold_sql": ex["gold_sql"], "pred_sql": pred_sql,
            "correct": detail.correct, "reason": detail.reason,
            "pred_error": detail.pred_error,
            "ttft_s": out["ttft_s"], "decode_tps": out["decode_tps"],
            "gen_tokens": out["gen_tokens"],
        })

        if i % 25 == 0 or i == len(manifest):
            acc = n_correct / max(1, n_scored)
            print(f"  [{i}/{len(manifest)}] exec-acc {acc:6.1%}  "
                  f"tps {statistics.median(tpss) if tpss else 0:.1f}")

    peak_gb = peak_memory_gb()

    def pct(values: List[float], q: float) -> float:
        if not values:
            return float("nan")
        s = sorted(values)
        return s[min(len(s) - 1, int(q * len(s)))]

    summary = {
        "variant": name,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "n_scored": n_scored,
        "execution_accuracy": n_correct / max(1, n_scored),
        "valid_sql_rate": n_exec / max(1, n_scored),
        "ttft_p50_s": statistics.median(ttfts) if ttfts else float("nan"),
        "ttft_p95_s": pct(ttfts, 0.95),
        "decode_tps_p50": statistics.median(tpss) if tpss else float("nan"),
        "load_s": load_s,
        "weights_gb": weights_gb,
        "peak_gb": peak_gb,
    }
    print(f"  -> exec-acc {summary['execution_accuracy']:.1%} | "
          f"valid-sql {summary['valid_sql_rate']:.1%} | "
          f"ttft p50 {summary['ttft_p50_s']:.2f}s | "
          f"{summary['decode_tps_p50']:.1f} tok/s | peak {peak_gb:.2f} GB")

    del model, tokenizer
    return {"summary": summary, "records": records}


# -------------------------------------------------------------------- main --

def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark SQL model variants on MLX")
    ap.add_argument("--manifest", required=True, help="eval_manifest.json")
    ap.add_argument("--variant", action="append", required=True, metavar="NAME=PATH[,ADAPTER]",
                    help="repeatable, e.g. --variant base-4bit=models/qwen-4bit")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    if args.limit:
        manifest = manifest[: args.limit]
    print(f"Evaluating {len(manifest)} questions across "
          f"{len({m['db_id'] for m in manifest})} databases")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for spec in args.variant:
        name, _, rest = spec.partition("=")
        path, _, adapter = rest.partition(",")
        res = evaluate_variant(name, path, manifest, args.max_tokens,
                               args.warmup, adapter or None)
        summaries.append(res["summary"])
        (out_dir / f"predictions_{name}.json").write_text(
            json.dumps(res["records"], indent=2))

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))

    # Comparison table
    print("\n" + "=" * 104)
    hdr = (f"{'variant':<22}{'exec-acc':>10}{'valid-sql':>11}"
           f"{'ttft p50':>11}{'tok/s':>9}{'weights GB':>12}{'peak GB':>10}")
    print(hdr)
    print("-" * 104)
    for s in summaries:
        print(f"{s['variant']:<22}{s['execution_accuracy']:>9.1%}"
              f"{s['valid_sql_rate']:>11.1%}{s['ttft_p50_s']:>10.2f}s"
              f"{s['decode_tps_p50']:>9.1f}{s['weights_gb']:>12.2f}{s['peak_gb']:>10.2f}")
    print("=" * 104)
    print(f"\nWrote {out_dir / 'summary.json'} and per-variant predictions.")


if __name__ == "__main__":
    main()
