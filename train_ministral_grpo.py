# GRPO Training for NL2SQL with Ministral-3-14B-Reasoning-2512
#
# Uses TRL GRPOTrainer + PEFT LoRA (no Unsloth).
# Designed for a single NVIDIA B200 (192 GB) — no model sharding.
# Launch with DDP across multiple GPUs on the B200 node:
#
#   torchrun --nproc_per_node=1 train_ministral_grpo.py          # single-GPU
#   accelerate launch --multi_gpu --num_processes=N train_ministral_grpo.py  # DDP
#
# Monitor with TensorBoard:
#   tensorboard --logdir outputs/grpo_ministral --bind_all --port 6006

import os

os.environ["COMET_MODE"] = "DISABLED"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"

import logging

logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch._dynamo
import re
import json
import sqlite3
import random
import gc
import sys
import time
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig, GRPOTrainer
from transformers import (
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
    TrainerCallback,
)
from torch.utils.tensorboard import SummaryWriter

torch._dynamo.config.suppress_errors = True

# ============================================================================
#  CONFIGURATION
# ============================================================================
CONFIG = {
    # ── Model ──────────────────────────────────────────────────────────────
    "model_id": "mistralai/Ministral-3-14B-Reasoning-2512",
    "train_dataset_path": "data/reasoning_data_gen/train_val_split/training_data_train.jsonl",
    "val_dataset_path": "data/reasoning_data_gen/train_val_split/training_data_val.jsonl",
    "database_dir": "train_databases",
    "output_dir": "outputs/grpo_ministral",
    "seed": 42,

    # ── LoRA ───────────────────────────────────────────────────────────────
    "lora_rank": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # ── Sequence lengths ───────────────────────────────────────────────────
    "max_seq_length": 2100,
    "max_prompt_length": 900,
    "max_completion_length": 1050,

    # ── Training ───────────────────────────────────────────────────────────
    "num_generations": 4,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 3e-6,
    "num_train_epochs": 1,
    "max_steps": 500,
    "save_steps": 25,
    "eval_steps": 25,
    "logging_steps": 1,
    "warmup_ratio": 0.05,
    "temperature": 1.0,               # Mistral recommends 1.0 for reasoning
    "weight_decay": 0.001,

    # ── GRPO ───────────────────────────────────────────────────────────────
    "beta": 0.04,                      # KL penalty
    "max_grad_norm": 0.5,

    # ── Data limits (None = full) ──────────────────────────────────────────
    "train_limit": None,
    "val_limit": 50,

    # ── Resume ─────────────────────────────────────────────────────────────
    "resume_from_checkpoint": None,

    # ── Logging ────────────────────────────────────────────────────────────
    "reward_log_every": 10,

    # ── Thinking scaffold (appended to system prompt so reasoning activates) ─
    "thinking_scaffold": (
        "\n\n# HOW YOU SHOULD THINK AND ANSWER\n\n"
        "First draft your thinking process (inner monologue) until you arrive at a response.\n\n"
        "Your thinking process must follow the template below:"
        "[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. "
        "Be as casual and as long as you want until you are confident to generate the response "
        "to the user.[/THINK]Here, provide a self-contained response."
    ),

    # ── Reward configuration ───────────────────────────────────────────────
    "reward_config": {
        "syntax_execution_reward": True,
        "result_reward": True,
        "schema_linking_reward": True,
        "ngram_similarity_reward": True,
    },
    "use_composite_reward": True,
    "reward_weights": {
        "syntax_execution_reward": 0.20,
        "result_reward": 1.00,
        "schema_linking_reward": 0.30,
        "ngram_similarity_reward": 0.30,
    },
}


# ============================================================================
#  SQL EXTRACTION (adapted from unsloth_rl_grpo.py)
# ============================================================================

SQL_SELECT_REGEX = re.compile(
    r"(SELECT\s+.*?)(?:;|\n\n|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_ministral_response(text: str) -> Tuple[str, str]:
    """
    Parse Ministral reasoning model output.

    Returns (thinking, answer).

    Handles:
      1. [THINK]...[/THINK]answer  (special tokens present — skip_special_tokens=False)
      2. <think>...</think>answer   (fallback)
      3. No delimiters — entire text is answer
    """
    # Primary: [THINK]...[/THINK]
    m = re.search(r"\[THINK\](.*?)\[/THINK\](.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Fallback: <think>...</think>
    m = re.search(r"<think>(.*?)</think>(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    return "", text.strip()


def extract_sql(text: str) -> str:
    """Extract SQL from model response (handles Ministral format)."""
    _, content = parse_ministral_response(text)
    search_text = content if content else text

    # Fenced ```sql ... ```
    m = re.search(r"```sql\s*(.*?)```", search_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # Plain ``` with SELECT/WITH
    m = re.search(r"```\s*((?:SELECT|WITH).*?)```", search_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # CTE: WITH ... ; or WITH ... \n\n
    m = re.search(r"(WITH\s+.+?)(?:;(?:\s*$|\s*\n))", search_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        sql = m.group(1).strip()
        if _is_valid_cte(sql):
            return sql

    m = re.search(r"(WITH\s+.+?SELECT\s+.+?)(?:\n\n|\Z)", search_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        sql = m.group(1).strip()
        if _is_valid_cte(sql):
            return sql

    cte_sql = _extract_cte_balanced(search_text)
    if cte_sql:
        return cte_sql

    # If starts with WITH but CTE extraction failed, bail out
    if re.match(r"\s*WITH\b", search_text, re.IGNORECASE):
        return ""

    # Plain SELECT
    m = SQL_SELECT_REGEX.search(search_text)
    if m:
        return m.group(1).strip()

    for line in search_text.splitlines():
        stripped = line.strip().upper()
        if stripped.startswith("SELECT") or stripped.startswith("WITH"):
            return line.strip()

    return ""


def _is_valid_cte(sql: str) -> bool:
    if not sql or not sql.strip().upper().startswith("WITH"):
        return False
    if sql.count("(") != sql.count(")"):
        return False
    depth = 0
    sql_upper = sql.upper()
    last_select_outside = -1
    for i in range(len(sql)):
        if sql[i] == "(":
            depth += 1
        elif sql[i] == ")":
            depth -= 1
        elif depth == 0 and sql_upper[i : i + 6] == "SELECT":
            last_select_outside = i
    return last_select_outside > 10


def _extract_cte_balanced(text: str) -> Optional[str]:
    text_upper = text.upper()
    with_match = re.search(r"\bWITH\s+", text_upper)
    if not with_match:
        return None

    start_idx = with_match.start()
    depth = 0
    final_select_start = -1
    i = with_match.end()

    while i < len(text):
        char = text[i]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char in ("'", '"'):
            quote = char
            i += 1
            while i < len(text) and text[i] != quote:
                if text[i] == "\\":
                    i += 1
                i += 1
        elif depth == 0 and text_upper[i : i + 6] == "SELECT":
            final_select_start = i
        i += 1

    if final_select_start == -1:
        return None

    end_idx = len(text)
    for pattern, offset in [(";", 1), ("\n\n", 0), ("\n--", 0)]:
        pos = text.find(pattern, final_select_start)
        if pos != -1 and pos < end_idx:
            end_idx = pos + offset

    sql = text[start_idx:end_idx].strip()
    if sql and _is_valid_cte(sql):
        return sql
    if sql and sql.upper().startswith("WITH") and "SELECT" in sql.upper():
        if sql.count("(") == sql.count(")"):
            return sql
    return None


def extract_sql_from_ground_truth(gt_content: str) -> str:
    if not gt_content or not gt_content.strip():
        return ""
    stripped = gt_content.strip()
    upper = stripped.upper()
    if upper.startswith("SELECT") or upper.startswith("WITH"):
        return stripped.rstrip(";").strip()
    sql = extract_sql(gt_content)
    return sql if sql else stripped


def extract_thinking(text: str) -> str:
    thinking, _ = parse_ministral_response(text)
    return thinking


def extract_database_name(prompt: str) -> str:
    patterns = [
        r"Database:\s*(\w+)",
        r"database[:\s]+['\"]?(\w+)['\"]?",
        r"DB:\s*(\w+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


# ============================================================================
#  EXECUTION INFRASTRUCTURE (from unsloth_rl_grpo.py)
# ============================================================================

_execution_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()
_DB_CONNECTIONS: Dict[str, sqlite3.Connection] = {}
_DB_CONN_LOCK = threading.Lock()


def _get_db_connection(db_path: str) -> Optional[sqlite3.Connection]:
    with _DB_CONN_LOCK:
        if db_path in _DB_CONNECTIONS:
            return _DB_CONNECTIONS[db_path]
        try:
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=30)
            conn.text_factory = str
            conn.execute("PRAGMA query_only = ON")
            conn.execute("PRAGMA cache_size = -2000")
            _DB_CONNECTIONS[db_path] = conn
            return conn
        except Exception:
            _DB_CONNECTIONS[db_path] = None
            return None


def _close_all_db_connections():
    with _DB_CONN_LOCK:
        count = len(_DB_CONNECTIONS)
        for _, conn in _DB_CONNECTIONS.items():
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
        _DB_CONNECTIONS.clear()
    print(f"  Closed {count} pooled DB connections")


def _cache_key(text: str) -> str:
    return str(hash(text))


def _clear_cache():
    with _cache_lock:
        _execution_cache.clear()


def _get_cache(key: str) -> Optional[Dict[str, Any]]:
    with _cache_lock:
        return _execution_cache.get(key)


def _set_cache(key: str, value: Dict[str, Any]):
    with _cache_lock:
        _execution_cache[key] = value


def get_database_path(database_name: str, database_dir: str) -> str:
    if not database_name:
        return ""
    db_path = os.path.join(database_dir, database_name, f"{database_name}.sqlite")
    if os.path.exists(db_path):
        return db_path
    alt_path = os.path.join(database_dir, database_name, f"{database_name}.db")
    if os.path.exists(alt_path):
        return alt_path
    return ""


def execute_sql(sql: str, db_path: str, timeout: int = 60) -> Tuple[bool, Any]:
    if not db_path or not os.path.exists(db_path):
        return False, "Database not found"
    if not sql or not sql.strip():
        return False, "Empty SQL"
    try:
        conn = _get_db_connection(db_path)
        if conn is None:
            return False, "Could not open database"

        deadline = time.monotonic() + timeout

        def _timeout_handler():
            return 1 if time.monotonic() > deadline else 0

        conn.set_progress_handler(_timeout_handler, 1000)
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            cursor.close()
            return True, result
        finally:
            conn.set_progress_handler(None, 0)
    except sqlite3.OperationalError as e:
        err = str(e)
        if "interrupted" in err:
            return False, f"Query timed out after {timeout}s"
        if any(k in err for k in ("disk I/O error", "database disk image is malformed", "unable to open")):
            with _DB_CONN_LOCK:
                old = _DB_CONNECTIONS.pop(db_path, None)
                if old:
                    try:
                        old.close()
                    except Exception:
                        pass
        return False, err
    except Exception as e:
        return False, str(e)


def normalize_result(result: Any) -> Any:
    if result is None:
        return None
    if not isinstance(result, list):
        return result
    normalized = []
    for row in result:
        if isinstance(row, (list, tuple)):
            norm_row = tuple(
                cell.lower().strip() if isinstance(cell, str) else cell for cell in row
            )
            normalized.append(norm_row)
        else:
            normalized.append((row,))
    try:
        return sorted(normalized)
    except TypeError:
        return normalized


# ============================================================================
#  HELPER FUNCTIONS (from unsloth_rl_grpo.py)
# ============================================================================

def _get_completion_text(completion) -> str:
    if isinstance(completion, list):
        return completion[0]["content"] if completion else ""
    return str(completion)


def _get_prompt_text(prompt) -> str:
    if isinstance(prompt, list):
        for m in reversed(prompt):
            if m.get("role") == "user":
                return m["content"]
        return prompt[-1]["content"] if prompt else ""
    return str(prompt)


def _ensure_list(val, length: int) -> list:
    if val is None:
        return [None] * length
    if not isinstance(val, list):
        return [val] * length
    return val


def _tokenize_sql(sql: str) -> List[str]:
    if not sql:
        return []
    sql = sql.lower().strip()
    sql = re.sub(r"\s+", " ", sql)
    tokens = re.findall(
        r">=|<=|<>|!=|[a-z_][a-z0-9_]*\.?[a-z0-9_]*|'[^']*'|\"[^\"]*\"|\d+\.?\d*|[^\s]",
        sql,
    )
    return [t for t in tokens if t.strip()]


def _extract_schema_items(sql: str) -> Tuple[set, Dict[str, str]]:
    if not sql:
        return set(), {}
    sql_lower = sql.lower().strip()
    schema_items: set = set()
    alias_map: Dict[str, str] = {}

    sql_keywords = {
        "select", "from", "where", "join", "inner", "left", "right",
        "outer", "cross", "on", "and", "or", "not", "in", "between",
        "like", "is", "null", "group", "order", "by", "having", "limit",
        "union", "except", "intersect", "exists", "case", "when", "then",
        "else", "end", "as", "distinct", "all", "asc", "desc", "set",
        "into", "values", "update", "delete", "insert", "create", "drop",
        "alter", "table", "index", "view", "with", "recursive", "natural",
    }

    for match in re.finditer(
        r"(?:from|join)\s+([a-z_][a-z0-9_]*)(?:\s+(?:as\s+)?([a-z_][a-z0-9_]*))?",
        sql_lower,
    ):
        table = match.group(1)
        alias = match.group(2)
        if table not in sql_keywords:
            schema_items.add(table)
            if alias and alias not in sql_keywords:
                alias_map[alias] = table

    for match in re.finditer(r"([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)", sql_lower):
        qualifier = match.group(1)
        column = match.group(2)
        actual_table = alias_map.get(qualifier, qualifier)
        schema_items.add(f"{actual_table}.{column}")
        schema_items.add(actual_table)

    sql_keywords_ext = sql_keywords | {
        "count", "sum", "avg", "max", "min", "coalesce", "ifnull",
        "cast", "substr", "length", "upper", "lower", "trim",
        "strftime", "date", "time", "datetime", "julianday",
        "group_concat", "total", "abs", "round", "replace", "instr",
        "true", "false",
    }

    select_match = re.search(r"select\s+(.*?)\s+from\s", sql_lower, re.DOTALL)
    if select_match:
        for col in re.findall(r"(?<!\.)(?<![a-z0-9_])([a-z_][a-z0-9_]*)(?!\s*\()", select_match.group(1)):
            if col not in sql_keywords_ext and not col.isdigit():
                schema_items.add(col)

    for clause_kw in ["where", r"group\s+by", r"order\s+by", "having"]:
        clause_match = re.search(
            rf"{clause_kw}\s+(.*?)(?:group\s+by|order\s+by|having|limit|union|except|intersect|$)",
            sql_lower,
            re.DOTALL,
        )
        if clause_match:
            for col in re.findall(r"(?<!\.)(?<![a-z0-9_])([a-z_][a-z0-9_]*)(?!\s*\()", clause_match.group(1)):
                if col not in sql_keywords_ext and not col.isdigit():
                    schema_items.add(col)

    return schema_items, alias_map


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _get_ngrams(tokens: List[str], n: int) -> set:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


# ============================================================================
#  TENSORBOARD WRITER
# ============================================================================

_TB_WRITER: Optional[SummaryWriter] = None
_TB_STEP: int = 0


def _get_tb_writer() -> Optional[SummaryWriter]:
    return _TB_WRITER


def _log_tb_scalar(tag: str, value: float, step: Optional[int] = None):
    writer = _get_tb_writer()
    if writer is not None:
        writer.add_scalar(tag, value, step if step is not None else _TB_STEP)


# ============================================================================
#  REWARD FUNCTIONS
# ============================================================================

def syntax_execution_reward(
    completions, prompts, ground_truth_sql=None, database_name=None, **kwargs
) -> List[float]:
    """
    Syntax + Execution Reward.

    +2.0  SQL executes successfully
    -2.0  SQL fails to execute
     0.0  No SQL extracted / no DB
    """
    scores = []
    database_dir = CONFIG["database_dir"]
    ground_truth_sql = _ensure_list(ground_truth_sql, len(completions))
    database_name = _ensure_list(database_name, len(completions))

    for i, completion in enumerate(completions):
        response = _get_completion_text(completion)
        sql = extract_sql(response)
        key = _cache_key(response)

        cache_entry = {
            "sql": sql, "executed": False, "success": False,
            "result": None, "gt_result": None, "result_match": None, "db_path": "",
        }

        if not sql or not sql.strip():
            _set_cache(key, cache_entry)
            scores.append(0.0)
            continue

        db_name = database_name[i] if i < len(database_name) else None
        if not db_name:
            prompt_text = _get_prompt_text(prompts[i]) if i < len(prompts) else ""
            db_name = extract_database_name(prompt_text)

        db_path = get_database_path(db_name, database_dir)
        if not db_path:
            _set_cache(key, cache_entry)
            scores.append(0.0)
            continue

        success, result = execute_sql(sql, db_path)
        cache_entry.update({"executed": True, "success": success, "result": result, "db_path": db_path})
        _set_cache(key, cache_entry)
        scores.append(2.0 if success else -2.0)

    return scores


def result_reward(
    completions, prompts, ground_truth_sql=None, database_name=None, **kwargs
) -> List[float]:
    """
    Execution Accuracy Reward (RLEF) with partial credit.

    +3.0  Exact match
    +1.5  Partial match (correct columns, overlapping rows)
    -3.0  No match
     0.0  Cannot evaluate
    """
    scores = []
    ground_truth_sql = _ensure_list(ground_truth_sql, len(completions))
    database_name = _ensure_list(database_name, len(completions))

    for i, completion in enumerate(completions):
        text = _get_completion_text(completion)
        c_key = _cache_key(text)
        cached = _get_cache(c_key)

        db_name = database_name[i] if database_name[i] else extract_database_name(
            _get_prompt_text(prompts[i])
        )
        gt_sql = ground_truth_sql[i] if ground_truth_sql[i] else ""
        gt_sql = extract_sql_from_ground_truth(gt_sql)

        if not gt_sql or not db_name:
            scores.append(0.0)
            continue

        db_path = get_database_path(db_name, CONFIG["database_dir"])
        if not db_path:
            scores.append(0.0)
            continue

        # Predicted result
        pred_result = None
        if cached and cached.get("executed"):
            pred_result = cached.get("result")
        else:
            pred_sql = extract_sql(text)
            if pred_sql:
                success, result = execute_sql(pred_sql, db_path)
                if success:
                    pred_result = result

        # Ground truth result
        gt_success, gt_result = execute_sql(gt_sql, db_path)
        if not gt_success:
            scores.append(0.0)
            continue

        if pred_result is None:
            scores.append(-3.0)
            continue

        pred_norm = normalize_result(pred_result)
        gt_norm = normalize_result(gt_result)

        if pred_norm == gt_norm:
            scores.append(3.0)
            continue

        # Partial credit
        pred_cols = len(pred_norm[0]) if pred_norm and isinstance(pred_norm[0], tuple) else 0
        gt_cols = len(gt_norm[0]) if gt_norm and isinstance(gt_norm[0], tuple) else 0

        if pred_cols == gt_cols and pred_cols > 0:
            pred_counter = Counter(pred_norm)
            gt_counter = Counter(gt_norm)
            if pred_counter and gt_counter:
                intersection_count = sum((pred_counter & gt_counter).values())
                precision = intersection_count / sum(pred_counter.values()) if pred_counter else 0
                recall = intersection_count / sum(gt_counter.values()) if gt_counter else 0
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    if f1 > 0.5:
                        score = min(-3.0 + (f1 * 4.5), 1.5)
                        scores.append(score)
                        continue

        scores.append(-3.0)

    return scores


def schema_linking_reward(completions, prompts, ground_truth_sql=None, **kwargs) -> List[float]:
    """
    Schema Linking Reward: Jaccard similarity of referenced schema items.
    Range: 0.0 to 2.0
    """
    scores = []
    ground_truth_sql = _ensure_list(ground_truth_sql, len(completions))

    for i, completion in enumerate(completions):
        response = _get_completion_text(completion)
        pred_sql = extract_sql(response)
        gt_sql_raw = ground_truth_sql[i] if i < len(ground_truth_sql) else None
        gt_sql = extract_sql_from_ground_truth(gt_sql_raw) if gt_sql_raw else ""

        if not pred_sql or not gt_sql:
            scores.append(0.0)
            continue

        pred_items, _ = _extract_schema_items(pred_sql)
        gt_items, _ = _extract_schema_items(gt_sql)
        scores.append(_jaccard_similarity(pred_items, gt_items) * 2.0)

    return scores


def ngram_similarity_reward(completions, prompts, ground_truth_sql=None, **kwargs) -> List[float]:
    """
    N-gram Similarity Reward: Jaccard similarity of SQL token n-grams.
    Range: 0.0 to 2.0
    """
    scores = []
    ground_truth_sql = _ensure_list(ground_truth_sql, len(completions))

    for i, completion in enumerate(completions):
        response = _get_completion_text(completion)
        pred_sql = extract_sql(response)
        gt_sql_raw = ground_truth_sql[i] if i < len(ground_truth_sql) else None
        gt_sql = extract_sql_from_ground_truth(gt_sql_raw) if gt_sql_raw else ""

        if not pred_sql or not gt_sql:
            scores.append(0.0)
            continue

        pred_tokens = _tokenize_sql(pred_sql)
        gt_tokens = _tokenize_sql(gt_sql)
        if not pred_tokens or not gt_tokens:
            scores.append(0.0)
            continue

        similarities = []
        for n in [1, 2, 3]:
            sim = _jaccard_similarity(_get_ngrams(pred_tokens, n), _get_ngrams(gt_tokens, n))
            similarities.append(sim)

        scores.append(sum(similarities) / len(similarities) * 2.0)

    return scores


# ============================================================================
#  COMPOSITE REWARD
# ============================================================================

def composite_reward(
    completions, prompts,
    ground_truth_sql=None, database_name=None, **kwargs,
) -> List[float]:
    """
    Composite weighted reward — computes all sub-rewards and returns weighted sum.
    """
    global _TB_STEP
    _clear_cache()

    reward_config = CONFIG["reward_config"]
    reward_weights = CONFIG["reward_weights"]
    batch_size = len(completions)

    if not hasattr(composite_reward, "_call_count"):
        composite_reward._call_count = 0
    composite_reward._call_count += 1

    component_scores: Dict[str, List[float]] = {
        name: [0.0] * batch_size for name in reward_weights
    }

    # Order matters: syntax_execution_reward populates cache for result_reward
    if reward_config.get("syntax_execution_reward", True):
        component_scores["syntax_execution_reward"] = syntax_execution_reward(
            completions, prompts,
            ground_truth_sql=ground_truth_sql, database_name=database_name, **kwargs,
        )

    if reward_config.get("result_reward", True):
        component_scores["result_reward"] = result_reward(
            completions, prompts,
            ground_truth_sql=ground_truth_sql, database_name=database_name, **kwargs,
        )

    if reward_config.get("schema_linking_reward", False):
        component_scores["schema_linking_reward"] = schema_linking_reward(
            completions, prompts, ground_truth_sql=ground_truth_sql, **kwargs,
        )

    if reward_config.get("ngram_similarity_reward", False):
        component_scores["ngram_similarity_reward"] = ngram_similarity_reward(
            completions, prompts, ground_truth_sql=ground_truth_sql, **kwargs,
        )

    # Weighted sum
    final_scores = []
    for i in range(batch_size):
        total = sum(reward_weights.get(name, 0.0) * scores[i] for name, scores in component_scores.items())
        final_scores.append(total)

    # ── TensorBoard logging ──
    _TB_STEP = composite_reward._call_count
    for name, sub_scores in component_scores.items():
        if not reward_config.get(name, False):
            continue
        _log_tb_scalar(f"rewards/{name}_mean", np.mean(sub_scores))

    mean_final = np.mean(final_scores) if final_scores else 0.0
    _log_tb_scalar("rewards/composite_mean", mean_final)

    result_scores = component_scores["result_reward"]
    n_correct = sum(1 for s in result_scores if s > 0)
    n_evaluated = sum(1 for s in result_scores if s != 0)
    exec_accuracy = n_correct / max(n_evaluated, 1)
    _log_tb_scalar("rewards/execution_accuracy", exec_accuracy)

    # Periodic cleanup
    if composite_reward._call_count % 20 == 0:
        gc.collect()
        torch.cuda.empty_cache()

    # Console logging
    log_every = CONFIG.get("reward_log_every", 10)
    if composite_reward._call_count % log_every == 0:
        print(f"\n{'='*70}")
        print(f"REWARD BREAKDOWN (call #{composite_reward._call_count}, batch={batch_size})")
        print(f"{'='*70}")
        print(f"{'Component':<30} {'Weight':>6} {'Mean':>8} {'Min':>8} {'Max':>8}")
        print(f"{'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
        for name, sub_scores in component_scores.items():
            if not reward_config.get(name, False):
                continue
            w = reward_weights.get(name, 0.0)
            print(f"{name:<30} {w:>6.2f} {np.mean(sub_scores):>8.3f} {min(sub_scores):>8.3f} {max(sub_scores):>8.3f}")
        print(f"{'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
        print(f"{'WEIGHTED TOTAL':<30} {'':>6} {mean_final:>8.3f} {min(final_scores):>8.3f} {max(final_scores):>8.3f}")
        print(f"\nExecution accuracy: {n_correct}/{batch_size} ({exec_accuracy*100:.1f}%)")
        print(f"{'='*70}\n")

    return final_scores


# ============================================================================
#  DATASET PREPARATION
# ============================================================================

def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prepare_dataset_for_grpo(
    data: List[Dict[str, Any]],
    tokenizer,
    max_prompt_tokens: int,
    max_completion_tokens: int,
) -> Tuple[Dataset, Dict[str, int]]:
    """
    Build GRPO dataset from harmony-format JSONL records.

    For Ministral, the thinking scaffold is appended to the system prompt
    so the model produces [THINK]...[/THINK] reasoning.
    """
    processed = []
    stats = {"total": len(data), "no_user": 0, "prompt_too_long": 0,
             "total_too_long": 0, "completion_too_long": 0, "no_gt_sql": 0}
    max_seq = CONFIG["max_seq_length"]
    thinking_scaffold = CONFIG["thinking_scaffold"]

    for item in data:
        messages = item["messages"]

        # Build prompt messages (system + user only, no assistant)
        prompt_messages = []
        for m in messages:
            if m["role"] == "assistant":
                continue
            if m["role"] == "system":
                # Append thinking scaffold to system prompt
                prompt_messages.append({
                    "role": "system",
                    "content": m["content"] + thinking_scaffold,
                })
            else:
                prompt_messages.append({"role": m["role"], "content": m["content"]})

        if not any(m["role"] == "user" for m in prompt_messages):
            stats["no_user"] += 1
            continue

        # Tokenize prompt to check length
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_tokens)

        if prompt_len > max_prompt_tokens:
            stats["prompt_too_long"] += 1
            continue
        if prompt_len + max_completion_tokens > max_seq:
            stats["total_too_long"] += 1
            continue

        # Ground truth
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        gt_sql = ""
        gt_thinking = ""
        if assistant_msg:
            gt_sql_raw = assistant_msg.get("content", "")
            gt_thinking = assistant_msg.get("thinking", "")
            gt_sql = extract_sql_from_ground_truth(gt_sql_raw)

            # Filter if GT completion is too long
            gt_text = (gt_thinking + "\n" + gt_sql_raw) if gt_thinking else gt_sql_raw
            gt_tokens = len(tokenizer.encode(gt_text, add_special_tokens=False)) if gt_text else 0
            if gt_tokens > max_completion_tokens:
                stats["completion_too_long"] += 1
                continue

        if not gt_sql:
            stats["no_gt_sql"] += 1

        user_msg = next((m for m in messages if m["role"] == "user"), None)
        db_name = extract_database_name(user_msg["content"]) if user_msg else ""
        if not db_name:
            db_name = item.get("db_id", item.get("database", ""))

        processed.append({
            "prompt": prompt_messages,
            "ground_truth_sql": gt_sql,
            "ground_truth_thinking": gt_thinking,
            "database_name": db_name,
        })

    kept = len(processed)
    total_filtered = stats["total"] - kept

    print(f"\nDataset filtering results:")
    print(f"  Total input: {stats['total']}")
    print(f"  Kept: {kept} ({100*kept/max(stats['total'],1):.1f}%)")
    print(f"  Filtered (no user msg):            {stats['no_user']}")
    print(f"  Filtered (prompt > {max_prompt_tokens} tok):      {stats['prompt_too_long']}")
    print(f"  Filtered (total > {max_seq} tok):     {stats['total_too_long']}")
    print(f"  Filtered (GT > {max_completion_tokens} tok):    {stats['completion_too_long']}")
    print(f"  No ground truth SQL:               {stats['no_gt_sql']}")
    if total_filtered > 0:
        print(f"  Total filtered: {total_filtered} ({100*total_filtered/max(stats['total'],1):.1f}%)")

    return Dataset.from_list(processed), stats


# ============================================================================
#  MODEL LOADING (Ministral — no Unsloth, uses PEFT directly)
# ============================================================================

def load_model_and_tokenizer(config: Dict[str, Any]):
    """
    Load Ministral-3-14B-Reasoning-2512 in BF16 on a single GPU (no sharding).
    Apply LoRA via PEFT.

    B200 (192 GB) fits the full 14B model (~28 GB BF16) + LoRA + KV cache easily.
    For the current A100 40GB dev box, device_map="auto" is used as fallback.
    """
    model_id = config["model_id"]

    print(f"\n{'='*60}")
    print(f"  LOADING MODEL: {model_id}")
    print(f"{'='*60}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_mem_gb = props.total_memory / 1e9
        print(f"  GPU: {props.name}")
        print(f"  VRAM: {gpu_mem_gb:.1f} GB")
    else:
        gpu_mem_gb = 0

    print(f"  Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"  Loading model (BF16, ~28 GB) ...")
    load_start = datetime.now()

    # Ministral-3 is a vision-language model (Mistral3Config).
    # Must use Mistral3ForConditionalGeneration — AutoModelForCausalLM won't work.
    #
    # DDP mode (torchrun): each process loads the FULL model on its LOCAL GPU.
    #   - B200 (192 GB): 28 GB model fits trivially on one GPU.
    #   - A100 40 GB:    28 GB model + LoRA + optimizer ≈ 35-38 GB — tight but OK
    #                    with gradient checkpointing + batch_size=1.
    #
    # Non-DDP mode (plain python): use device_map="auto" to let HF shard across
    #   all available GPUs. This is only useful for dev/testing.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank >= 0

    if is_ddp:
        # DDP: load on the assigned GPU — do NOT use device_map="auto" which
        # would shard across all GPUs and collide with other ranks.
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
        print(f"  DDP mode: LOCAL_RANK={local_rank}, loading model on cuda:{local_rank}")
    elif gpu_mem_gb >= 100:
        # Single-process on a large GPU (B200 / H100 80GB)
        device_map = None
        print(f"  device_map: None (single-device)")
    else:
        # Single-process on small dev GPUs — shard across all
        device_map = "auto"
        print(f"  device_map: auto (sharding across GPUs for dev)")

    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
    )

    load_time = (datetime.now() - load_start).total_seconds()
    print(f"  Model loaded in {load_time:.1f}s")

    if torch.cuda.is_available():
        dev_idx = local_rank if is_ddp else 0
        print(f"  VRAM after load (cuda:{dev_idx}): "
              f"{torch.cuda.memory_allocated(dev_idx)/1e9:.2f} GB allocated")

    # ── Freeze vision encoder (NL2SQL — no images, no need to train vision) ─
    # Mistral3ForConditionalGeneration has a vision_model (ViT) that we never
    # use for text-only SQL generation.  Freezing it saves VRAM by eliminating
    # optimizer states for ~300 M vision params.
    # Note: LoRA adapters may still be added to vision attention layers by PEFT
    # (they share projection names with LM layers), but since the vision encoder
    # never runs (no image inputs), those adapters never receive gradients.
    vision_encoder = None
    for attr in ("vision_model", "vision_tower", "visual_encoder", "image_encoder"):
        if hasattr(model, attr):
            vision_encoder = getattr(model, attr)
            break
    if vision_encoder is None and hasattr(model, "model"):
        for attr in ("vision_model", "vision_tower"):
            if hasattr(model.model, attr):
                vision_encoder = getattr(model.model, attr)
                break

    if vision_encoder is not None:
        for param in vision_encoder.parameters():
            param.requires_grad = False
        vision_params = sum(p.numel() for p in vision_encoder.parameters()) / 1e6
        print(f"  Frozen vision encoder ({vision_params:.0f} M params)")
    else:
        print("  WARNING: vision encoder attribute not found — skipping explicit freeze")

    # ── Apply LoRA via PEFT ────────────────────────────────────────────────
    print(f"\n  Applying LoRA (rank={config['lora_rank']}, alpha={config['lora_alpha']}) ...")

    # The text model layers live under:
    #   model.language_model.model.layers.*.self_attn.{q,k,v,o}_proj
    #   model.language_model.model.layers.*.mlp.{gate,up,down}_proj
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print(f"{'='*60}\n")
    return model, tokenizer


# ============================================================================
#  CALLBACKS
# ============================================================================

class TensorBoardRewardCallback(TrainerCallback):
    """Log GRPOTrainer metrics to TensorBoard."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        writer = _get_tb_writer()
        if writer is None:
            return
        step = state.global_step
        for key, value in logs.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                writer.add_scalar(f"trainer/{key}", value, step)
        writer.flush()


class VerboseTrainingCallback(TrainerCallback):
    """Verbose progress logging."""

    def __init__(self):
        self.train_start_time = None
        self.step_times = []
        self.last_step_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"  TRAINING STARTED — {self.train_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Max steps: {args.max_steps}")
        print(f"  Batch: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = "
              f"{args.per_device_train_batch_size * args.gradient_accumulation_steps} effective")
        print(f"  Num generations: {CONFIG['num_generations']}")
        print(f"{'='*60}\n")

    def on_step_begin(self, args, state, control, **kwargs):
        self.last_step_time = datetime.now()

    def on_step_end(self, args, state, control, **kwargs):
        if self.last_step_time:
            dt = (datetime.now() - self.last_step_time).total_seconds()
            self.step_times.append(dt)
            if len(self.step_times) > 100:
                self.step_times = self.step_times[-100:]

        if state.global_step % max(args.logging_steps, 1) == 0 and state.global_step > 0:
            elapsed = (datetime.now() - self.train_start_time).total_seconds()
            avg_step = np.mean(self.step_times[-10:]) if self.step_times else 0
            remaining = avg_step * (args.max_steps - state.global_step)
            pct = 100 * state.global_step / max(args.max_steps, 1)
            filled = int(30 * state.global_step / max(args.max_steps, 1))
            bar = "█" * filled + "░" * (30 - filled)

            print(f"\n  [{bar}] {pct:.1f}% | Step {state.global_step}/{args.max_steps} | "
                  f"Elapsed: {elapsed/60:.1f}m | Step: {avg_step:.1f}s | ETA: {remaining/60:.1f}m")

            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(0) / 1e9
                peak = torch.cuda.max_memory_allocated(0) / 1e9
                print(f"    GPU: {alloc:.2f} GB alloc / {peak:.2f} GB peak")

    def on_train_end(self, args, state, control, **kwargs):
        total = (datetime.now() - self.train_start_time).total_seconds()
        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETED in {total/60:.1f} min ({total/3600:.2f} h)")
        print(f"  Total steps: {state.global_step}")
        if self.step_times:
            print(f"  Avg step: {np.mean(self.step_times):.2f}s")
        print(f"{'='*60}\n")


class GRPOMetricsCallback(TrainerCallback):
    """Track rolling reward average and alert on stalls."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.reward_history = []
        self.best_avg = float("-inf")
        self.steps_no_improve = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        reward = logs.get("reward", logs.get("train/reward"))
        if reward is not None:
            self.reward_history.append(reward)

        step = state.global_step
        if step > 0 and step % 25 == 0 and self.reward_history:
            window = self.reward_history[-self.window_size:]
            avg = np.mean(window)
            if avg > self.best_avg:
                self.best_avg = avg
                self.steps_no_improve = 0
            else:
                self.steps_no_improve += 25

            print(f"\n  Reward tracking (step {step}): rolling avg={avg:.4f}, best={self.best_avg:.4f}, "
                  f"no-improve={self.steps_no_improve} steps")

            if self.steps_no_improve >= 200:
                print(f"  WARNING: No improvement for {self.steps_no_improve} steps!")

            _log_tb_scalar("tracking/best_avg_reward", self.best_avg, step)


class SaveLoRAOnlyCallback(TrainerCallback):
    """Ensure checkpoints only contain LoRA adapter weights."""

    def on_save(self, args, state, control, model=None, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.exists(ckpt_dir):
            return

        all_files = os.listdir(ckpt_dir)
        full_model_files = [
            f for f in all_files
            if (f.startswith("model-") and f.endswith(".safetensors"))
            or f.startswith("pytorch_model")
        ]

        adapter_file = os.path.join(ckpt_dir, "adapter_model.safetensors")

        if os.path.exists(adapter_file):
            # Clean up any full-model shards
            for f in full_model_files:
                fpath = os.path.join(ckpt_dir, f)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                os.remove(fpath)
                print(f"  Deleted stray shard {f} ({size_mb:.1f} MB)")
        elif model is not None:
            try:
                model.save_pretrained(ckpt_dir)
                print(f"  Saved LoRA adapter to checkpoint-{state.global_step}")
                # Clean up full shards after explicit save
                for f in full_model_files:
                    fpath = os.path.join(ckpt_dir, f)
                    if os.path.exists(fpath):
                        os.remove(fpath)
            except Exception as e:
                print(f"  WARNING: Could not save LoRA adapter: {e}")


# ============================================================================
#  MAIN
# ============================================================================

def main():
    global _TB_WRITER

    config = CONFIG

    print(f"\n{'█'*60}", flush=True)
    print("  GRPO TRAINING — Ministral-3-14B-Reasoning NL2SQL", flush=True)
    print(f"{'█'*60}\n", flush=True)

    # Seed
    seed = config["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    tb_log_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    _TB_WRITER = SummaryWriter(log_dir=tb_log_dir)

    print(f"  Output:      {output_dir}")
    print(f"  TensorBoard: tensorboard --logdir {tb_log_dir} --bind_all --port 6006\n")

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Print config
    print("  Configuration:")
    for k, v in config.items():
        if isinstance(v, dict):
            print(f"    {k}:")
            for kk, vv in v.items():
                print(f"      {kk}: {vv}")
        else:
            print(f"    {k}: {v}")
    print()

    # System info
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  GPU VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print()

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)

    # Load datasets
    print("  Loading datasets ...")
    train_data = load_jsonl(config["train_dataset_path"], limit=config.get("train_limit"))
    val_data = load_jsonl(config["val_dataset_path"], limit=config.get("val_limit"))
    print(f"    Train: {len(train_data)} samples")
    print(f"    Val:   {len(val_data)} samples")

    train_dataset, train_stats = prepare_dataset_for_grpo(
        train_data, tokenizer,
        max_prompt_tokens=config["max_prompt_length"],
        max_completion_tokens=config["max_completion_length"],
    )
    val_dataset, val_stats = prepare_dataset_for_grpo(
        val_data, tokenizer,
        max_prompt_tokens=config["max_prompt_length"],
        max_completion_tokens=config["max_completion_length"],
    )
    print(f"\n  Train after filtering: {len(train_dataset)}")
    print(f"  Val after filtering:   {len(val_dataset)}")

    # Training args
    training_args = GRPOConfig(
        output_dir=output_dir,

        # Generation
        num_generations=config["num_generations"],
        max_completion_length=config["max_completion_length"],
        temperature=config["temperature"],

        # Optimisation
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="linear",
        optim="adamw_torch_fused",

        # Batch
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["num_generations"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"],
        bf16=True,

        # Gradient checkpointing handled by PEFT model already
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # GRPO
        beta=config["beta"],
        max_grad_norm=config["max_grad_norm"],

        # Logging
        logging_steps=config["logging_steps"],
        log_completions=True,
        report_to="tensorboard",
        logging_dir=tb_log_dir,

        # Saving
        save_steps=config["save_steps"],
        save_total_limit=5,
        save_only_model=True,

        # Eval
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        eval_on_start=True,

        # DDP
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
        seed=seed,

        # Disable DeepSpeed / vLLM
        use_vllm=False,
    )

    # Reward function
    reward_funcs = [composite_reward]

    print(f"\n  Reward functions: composite_reward (weighted)")
    print(f"  Enabled sub-rewards:")
    for name, enabled in config["reward_config"].items():
        w = config["reward_weights"].get(name, 0.0)
        print(f"    {'✓' if enabled else '✗'} {name}: {w:.2f}" if enabled else f"    ✗ {name}")

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[
            TensorBoardRewardCallback(),
            VerboseTrainingCallback(),
            GRPOMetricsCallback(window_size=50),
            SaveLoRAOnlyCallback(),
        ],
    )

    # Train
    resume_ckpt = config.get("resume_from_checkpoint")
    if resume_ckpt:
        print(f"\n  Resuming from: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save final model
    final_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n  Final LoRA adapter saved to: {final_path}")

    saved = os.listdir(final_path)
    total_mb = sum(
        os.path.getsize(os.path.join(final_path, f))
        for f in saved if os.path.isfile(os.path.join(final_path, f))
    ) / (1024 * 1024)
    print(f"  Files: {saved}")
    print(f"  Size:  {total_mb:.1f} MB")

    # Cleanup
    if _TB_WRITER is not None:
        _TB_WRITER.close()
    _close_all_db_connections()

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Output:      {output_dir}")
    print(f"  TensorBoard: tensorboard --logdir {tb_log_dir} --bind_all --port 6006")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
