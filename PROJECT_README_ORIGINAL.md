# QLoRA text-to-SQL on an M3 Pro

Fine-tune Qwen2.5-Coder-7B-Instruct on Spider with QLoRA, quantise to 4-bit, and
benchmark accuracy, latency and memory against the base model — entirely on-device.

## Why MLX and not bitsandbytes

The standard QLoRA stack (`bitsandbytes` NF4 + `peft` + CUDA kernels) does not run
on Apple Silicon. `bitsandbytes` 4-bit is CUDA-only; there is no usable MPS path.

MLX does the same thing natively: point `mlx_lm.lora` at a **4-bit quantised base**
and it freezes those weights and trains bf16 LoRA adapters over them. That is QLoRA.
The quantisation format differs (MLX uses group-wise affine 4-bit, not NF4), but the
method, memory profile and results are equivalent.

## Requirements

- Apple Silicon (M3 Pro), macOS 14+
- **18 GB unified memory minimum**, 36 GB comfortable
- ~35 GB free disk (base fp16 download ~15 GB, plus three derived checkpoints)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Step 0 — Get Spider

Execution accuracy needs the real `.sqlite` databases, not just question/SQL pairs.

```bash
pip install -U huggingface_hub
hf download xlangai/spider --repo-type dataset --local-dir spider_raw
unzip -q spider_raw/spider_data.zip -d .      # -> spider_data/
ls spider_data/              # train_spider.json  dev.json  database/  tables.json
```

If the zip layout differs, point `SPIDER_ROOT` at whatever directory contains
`train_spider.json` and `database/`.

## Step 1 — Quantise the base model to 4-bit

```bash
./run_all.sh quantize-base
```

Downloads ~15 GB of bf16 weights and writes a ~4.3 GB 4-bit checkpoint to
`models/qwen2.5-coder-7b-4bit`. **~15–25 min**, mostly download.

> Check `mlx_lm.convert --help` once — the flag for the source repo is `--hf-path`
> on older releases and `--model` on newer ones. Fix `run_all.sh` if yours differs.

## Step 2 — Build the training data

```bash
./run_all.sh prep
```

Produces `data/{train,valid,test}.jsonl` plus `data/eval_manifest.json`.

Two decisions here that drive most of the final accuracy:

- **Schemas come from `sqlite_master`**, i.e. real `CREATE TABLE` statements with
  types and foreign keys, rather than Spider's flattened column lists. Joins improve
  noticeably when the model can see the FK relationships.
- **Prompt masking.** Records are `{"prompt", "completion"}` and the config sets
  `mask_prompt: true`, so loss is computed on the SQL only. Schemas are 10–50× longer
  than the answers; without masking, nearly all the gradient signal goes into learning
  to reproduce DDL.

## Step 3 — QLoRA fine-tune

```bash
./run_all.sh train            # uses configs/qlora_m3pro_18gb.yaml
CONFIG=configs/qlora_m3pro_36gb.yaml ./run_all.sh train   # if you have 36 GB
```

**Expect roughly 2.5–5 hours for 1200 iterations on an M3 Pro.** Watch `train.log`:
validation loss should fall for the first few hundred iterations and then flatten.
If it turns upward, stop and use an earlier checkpoint from `adapters/`.

While this runs, keep other memory-hungry apps closed. If you hit swap, the run does
not crash — it gets several times slower, which is worse. Drop `num_layers` to 4 and
`max_seq_length` to 1024 if that happens.

## Step 4 — Fuse the adapter

```bash
./run_all.sh fuse
```

Merges the LoRA weights into the base to produce a standalone bf16 model at
`models/qwen-sql-fused`. **~2–4 min.**

## Step 5 — Quantise the fine-tuned model to 4-bit

```bash
./run_all.sh quantize-ft
```

`models/qwen-sql-4bit`, ~4.3 GB. **~3–5 min.** This is the deployable artifact.

## Step 6 — Benchmark

```bash
./run_all.sh bench            # or: EVAL_N=500 ./run_all.sh bench
```

Compares three variants on held-out Spider dev questions:

| variant | what it tells you |
|---|---|
| `base-4bit` | the baseline, quantised identically so the comparison isolates fine-tuning |
| `finetuned-4bit` | fused then re-quantised — the thing you would actually ship |
| `finetuned-adapter` | 4-bit base + live adapter — isolates any loss from re-quantisation |

**~25–50 min for 200 questions per variant.** Results land in
`results/summary.json` and per-variant prediction dumps, plus a printed table.

---

## How accuracy is measured

`src/sqleval.py` scores by **execution**: it runs the gold query and the predicted
query against the real database and compares the result sets. String matching would
mark `SELECT s.name FROM singer AS s WHERE s.age >= 31` wrong against
`SELECT name FROM singer WHERE age > 30` — same answer, different text — and would
systematically understate the fine-tuned model.

The comparison handles the cases that otherwise produce bogus scores:

- row order ignored **unless** the gold query has `ORDER BY`
- duplicate rows preserved, so `DISTINCT` errors are caught
- `INTEGER` vs `REAL` (`3` vs `3.0`) not penalised
- column order differences tolerated
- a prediction that fails to parse scores 0 rather than crashing the run
- a **broken gold query is excluded**, not charged to the model
- every connection is opened **read-only**, so a predicted `DROP TABLE` fails
  harmlessly instead of destroying your eval database

This scorer and the output parser are covered by 20 tests that run anywhere:

```bash
python3 tests/test_sqleval.py
```

## How latency is measured

MLX is lazily evaluated, so timing a call that only built a computation graph
measures nothing. `src/benchmark.py` forces `mx.synchronize()` around every timed
region, runs warmup passes first, decodes greedily (`temp=0`) for reproducibility,
and reports **time-to-first-token and decode tok/s separately**. 4-bit mainly speeds
up decode, which is memory-bandwidth bound; a single blended latency figure hides
the effect you are trying to see.

Peak memory comes from MLX's own allocator, reset per variant.

## What to expect from the results

I have not run this — no GPU or Apple Silicon in the environment where it was
written — so these are expectations to sanity-check against, not measurements:

- **Fine-tuning should help most on format compliance and schema grounding.**
  Watch `valid_sql_rate` as much as accuracy; base models lose a lot of points to
  markdown fences and hallucinated column names rather than bad reasoning.
- **Qwen2.5-Coder already has a strong SQL prior,** so the accuracy delta will be
  smaller than it would be on, say, Mistral-7B. If you want a dramatic before/after,
  swap the base. If you want the best final model, keep it.
- **4-bit re-quantisation typically costs a little accuracy.** The
  `finetuned-adapter` vs `finetuned-4bit` gap is exactly that cost. If it is large
  (more than a couple of points), try `--q-group-size 32`.
- If `finetuned-4bit` scores *below* `base-4bit`, suspect a prompt mismatch between
  training and eval before suspecting the training itself.

## Layout

```
configs/      QLoRA hyperparameters, 18 GB and 36 GB variants
src/prompt.py          prompt construction + SQL extraction, shared by train and eval
src/prepare_data.py    Spider -> MLX jsonl + eval manifest
src/sqleval.py         execution-accuracy scorer (tested)
src/benchmark.py       accuracy + latency + memory across variants
tests/                 20 tests, no GPU required
run_all.sh             step driver
```

`src/prompt.py` is deliberately the only place a prompt is built. Train/eval prompt
drift is the single most common cause of a fine-tune that looks fine in training and
fails at evaluation.
