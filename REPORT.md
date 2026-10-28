# QLoRA-Style Fine-Tuning of Qwen2.5-Coder-7B for Text-to-SQL

## 1. Objective
Fine-tune a 7B open model for SQL generation and compare the base and
fine-tuned models on execution accuracy, valid SQL rate, inference
latency, and peak memory.

## 2. Model and methodology
The project uses Qwen2.5-Coder-7B in 4-bit MLX format. LoRA adapters
were trained using MLX-LM. The final reported adapter corresponds to
the 400-iteration checkpoint. The workflow should be described as
LoRA fine-tuning with a quantized base model. A separate post-training
quantization of a merged model is not claimed here.

## 3. Evaluation
The reported comparison uses 500 questions. Execution accuracy measures
whether generated SQL produces the expected result under the benchmark's
execution evaluation. Valid SQL rate measures the proportion classified
as valid by the benchmark. Latency and memory figures are the recorded
benchmark measurements on the project's test hardware.

## 4. Results

| Metric | Base | Fine-tuned |
|---|---:|---:|
| Execution accuracy | 76.6% | 78.6% |
| Valid SQL rate | 97.6% | 96.0% |
| Median generation speed | 40.5 tokens/s | 27.2 tokens/s |
| Peak memory | 5.03 GB | 5.54 GB |
| Median time to first token | 1.37 s | 1.44 s |

Execution accuracy increased by 2 percentage points. Valid SQL rate
decreased by 1.6 percentage points. Median generation speed decreased
from 40.5 to 27.2 tokens/s, while peak memory increased by 0.51 GB.

## 5. Statistical comparison
For the paired 500-question comparison, 383 questions were correct for
the base model and 393 for the fine-tuned checkpoint. Both were correct
on 369 questions and both were wrong on 93. The base was correct while
the tuned model was wrong on 14 questions; the reverse occurred on 24.
The exact two-sided McNemar p-value was 0.143307. Therefore, the
observed accuracy difference was not statistically significant at
the 0.05 threshold.

## 6. Conclusion
The fine-tuned model showed a small measured improvement in execution
accuracy on this evaluation, with trade-offs in valid SQL rate, speed,
and memory. More held-out evaluation would be needed to establish whether
the improvement generalizes. These results do not establish production
readiness.

## 7. Reproducibility
The submission includes available source files, training configuration,
benchmark artifacts, evaluation manifest, and final adapter files.
Check README.md for the project structure. The base model may need to
be supplied separately. Verify the included files before submission.
