"""
Prompt construction, shared by training AND evaluation.

The single most common cause of a fine-tune that "works in training but not at
eval" is a prompt mismatch between the two. Everything that builds a prompt in
this repo goes through this module so that can't drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

SYSTEM_PROMPT = (
    "You are a precise text-to-SQL engine. Given a database schema and a "
    "question, reply with exactly one SQLite query that answers it. "
    "Output only the SQL, with no explanation and no markdown fences."
)


@dataclass
class SQLExample:
    """One text-to-SQL training/eval item."""
    question: str
    schema: str           # serialized CREATE TABLE statements
    gold_sql: str
    db_id: Optional[str] = None
    db_path: Optional[str] = None


def build_user_turn(question: str, schema: str) -> str:
    return f"""Database schema:
{schema.strip()}

Question: {question.strip()}

SQL:"""


def build_messages(question: str, schema: str) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_turn(question, schema)},
    ]


def render_prompt(tokenizer, question: str, schema: str) -> str:
    """Render the inference-time prompt (ends ready for the model to speak).

    Uses the tokenizer's own chat template so this stays correct across
    Qwen/ChatML, Mistral and Llama formats rather than hardcoding one.
    """
    msgs = build_messages(question, schema)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    # Fallback for base (non-instruct) checkpoints with no chat template.
    return f"{SYSTEM_PROMPT}\n\n{build_user_turn(question, schema)}\n"


def render_full(tokenizer, question: str, schema: str, sql: str) -> str:
    """Render prompt + target answer, for supervised training."""
    return render_prompt(tokenizer, question, schema) + sql.strip() + (
        tokenizer.eos_token or ""
    )


def extract_sql(raw: str) -> str:
    """Pull a single SQL statement out of raw model output.

    Models leak markdown fences, prose preambles and trailing commentary even
    after fine-tuning. Eval must be robust to that or you measure formatting
    compliance instead of SQL correctness.
    """
    text = (raw or "").strip()

    # Strip markdown fences, keeping the fenced body if present.
    if "```" in text:
        parts = text.split("```")
        for part in parts[1:]:
            body = part
            for lang in ("sqlite", "sql", "postgresql", "mysql"):
                if body.lower().startswith(lang):
                    body = body[len(lang):]
                    break
            if body.strip():
                text = body.strip()
                break

    # Drop a leading prose line if the SQL clearly starts further down.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    starts = ("select", "with", "insert", "update", "delete", "create")
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith(starts):
            text = "\n".join(lines[i:])
            break

    # Keep only the first statement.
    if ";" in text:
        text = text.split(";")[0]

    return text.strip()
