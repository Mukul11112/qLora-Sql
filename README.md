# Qwen2.5-Coder-7B Text-to-SQL Fine-Tuning

## Project
Fine-tuning a 7B open model for text-to-SQL generation and benchmarking
the base and adapted models for execution accuracy, SQL validity,
inference latency, and memory usage.

## Reported benchmark
Evaluation set: 500 questions.

| Metric | Base model | Fine-tuned checkpoint |
|---|---:|---:|
| Execution accuracy | 76.6% | 78.6% |
| Valid SQL rate | 97.6% | 96.0% |
| Median generation speed | 40.5 tokens/s | 27.2 tokens/s |
| Peak memory | 5.03 GB | 5.54 GB |
| Median time to first token | 1.37 s | 1.44 s |

The accuracy difference is +2 percentage points. The paired exact
McNemar test returned p = 0.143307, so the difference was not
statistically significant at the 0.05 level.

## Method
The project uses a 4-bit MLX-format Qwen2.5-Coder-7B base model
and trains LoRA adapters with MLX-LM. Describe this accurately as
LoRA fine-tuning on a quantized base model; do not claim that a
merged fine-tuned model was separately quantized unless that step
was actually performed.

## Contents
- source/: code and available project scripts
- configs/: training configuration
- benchmark_results/: benchmark summaries and predictions
- data/: final training data and evaluation manifest, if available
- model_adapter/: final LoRA adapter files, if available
- REPORT.md: project report

## Reproduction
Use a compatible macOS environment, install the dependencies listed
in source/requirements.txt if provided, and review the training
configuration before running scripts. The base model must be available
locally or downloaded separately. Run the benchmark using the
provided evaluation manifest and the project's benchmark script,
following its supported command-line arguments.

## Limitations
The measured accuracy gain was not statistically significant.
The fine-tuned model generated tokens more slowly and used more peak
memory in this benchmark. Results apply to the evaluated 500 questions
and do not establish production readiness.
