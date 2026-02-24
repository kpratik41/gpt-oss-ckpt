# GRPO Training for NL2SQL with Unsloth
# Usage: python unsloth_rl_grpo.py
#
# Monitor with TensorBoard:
#   tensorboard --logdir /home/ec2-user/rl/outputs/grpo_training --bind_all --port 6006
#   Then SSH tunnel: ssh -L 6006:localhost:6006 <ec2-host>

import os
os.environ["COMET_MODE"] = "DISABLED"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"

import logging

# Only show WARNING and above for noisy libraries
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Show INFO for your own code
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["TQDM_DISABLE"] = "0"  # Ensure progress bars show

from unsloth import FastLanguageModel
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
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
from collections import defaultdict, Counter
import numpy as np
from torch.utils.tensorboard import SummaryWriter

torch._dynamo.config.suppress_errors = True

# ============== Configuration ==============
INLINE_CONFIG = {
    "base_model_path": "unsloth/gpt-oss-20b",
    "train_dataset_path": "data/reasoning_data_gen/train_val_split/training_data_train.jsonl",
    "val_dataset_path": "data/reasoning_data_gen/train_val_split/training_data_val.jsonl",
    "database_dir": "train_databases",
    "output_dir": "/home/ec2-user/rl/outputs/grpo_training",
    "seed": 42,

    # LoRA config — slightly larger rank since we have VRAM headroom
    "lora_rank": 32,
    "lora_alpha": 32,

    # Training config — tuned for single 80GB A100 with BF16
    "max_seq_length": 2100,
    "max_prompt_length": 900,
    "max_completion_length": 1050,
    "num_generations": 4,                    # ← More generations = better GRPO signal
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 3e-6,
    "num_train_epochs": 1,
    "max_steps": 500,
    "save_steps": 25,
    "eval_steps": 25,
    "logging_steps": 1,
    "warmup_ratio": 0.05,
    "temperature": 0.7,
    "load_in_4bit": False,                   # ← BF16 instead of 4-bit
    "offload_embedding": False,              # ← No need with 80GB
    "save_merged_model": False,

    # GRPO-specific
    "beta": 0.06,                            # ← KL penalty coefficient
    "max_grad_norm": 0.5,                    # ← Tighter gradient clipping for RL stability

    # Data limits (None = use full dataset)
    "train_limit": None,
    "val_limit": 50,          # eval generates 4 completions per sample → 50×35s ≈ 30min per eval

    # Resume training from a checkpoint (set to path like "outputs/grpo_training/grpo_run_.../checkpoint-25", or None)
    "resume_from_checkpoint": None,

    # Debug
    "debug_every_n_steps": 10,
    "reward_log_every": 10,
    "save_raw_completions": False,

    # ================== REWARD CONFIGURATION ==================
    # Enable/disable individual reward functions per training run
    "reward_config": {
        "format_reward": False,            # Check gpt-oss format + SQL extraction [-1.5, +2.0]
        "syntax_execution_reward": True,  # Check if SQL executes (syntax check) [-2.0, +2.0] — SATURATED, no signal
        "result_reward": True,            # Check if result matches ground truth (RLEF) [-3.0, +3.0]
        "schema_linking_reward": True,    # Jaccard similarity of schema items [0.0, +2.0] — low within-group variance
        "ngram_similarity_reward": True,  # Jaccard similarity of SQL n-grams [0.0, +2.0]
        "thinking_quality_reward": False,  # Thinking length + schema reference quality [0.0, +2.0]
        "llm_judge_reward": False,        # LLM-as-a-Judge — disabled (no external API) [0.0, +2.0]
    },

    # ================== REWARD WEIGHTS ==================
    # Weighted sum ensures correctness dominance:
    # No incorrect query can achieve higher total reward than a correct query.
    #
    # Constraint: w_result × 6.0 > Σ(w_i × range_i) for all other rewards
    # With these defaults: 1.0 × 6.0 = 6.0 > 0.30*3.5 + 0.30*4.0 + 0.20*2.0 + 0.20*2.0 + 0.15*2.0 = 3.35 ✓
    "use_composite_reward": True,
    "reward_weights": {
        "format_reward": 0.30,
        "syntax_execution_reward": 0.20,
        "result_reward": 1.00,
        "schema_linking_reward": 0.30,
        "ngram_similarity_reward": 0.30,
        "thinking_quality_reward": 0.15,
        "llm_judge_reward": 0.20,
    },

    # ================== LLM JUDGE CONFIGURATION ==================
    # Disabled by default — self-judging with the training model is unreliable.
    # Enable only if you have an external judge API.
    "llm_judge_config": {
        "enabled": False,
    },
}

# SQL extraction regex
SQL_SELECT_REGEX = re.compile(
    r"(SELECT\s+.*?)(?:;|\n\n|$)",
    flags=re.IGNORECASE | re.DOTALL
)


# ================= Execution Cache (thread-safe) =================
_execution_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()

# ================= SQLite Connection Pool (read-only) =================
_DB_CONNECTIONS: Dict[str, sqlite3.Connection] = {}
_DB_CONN_LOCK = threading.Lock()


def _get_db_connection(db_path: str) -> Optional[sqlite3.Connection]:
    """
    Get or create a cached read-only SQLite connection.
    """
    with _DB_CONN_LOCK:
        # Check if we already tried and failed (cached as None)
        if db_path in _DB_CONNECTIONS:
            return _DB_CONNECTIONS[db_path]  # May be None if previously failed
        
        try:
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(
                uri,
                uri=True,
                check_same_thread=False,
                timeout=30,
            )
            conn.text_factory = str
            conn.execute("PRAGMA query_only = ON")
            conn.execute("PRAGMA cache_size = -2000")
            _DB_CONNECTIONS[db_path] = conn
            return conn
        except Exception as e:
            _DB_CONNECTIONS[db_path] = None  # Cache the failure
            return None


def _close_all_db_connections():
    """Close all pooled connections — call at end of training."""
    with _DB_CONN_LOCK:
        count = len(_DB_CONNECTIONS)  # ← Capture count BEFORE clearing
        for path, conn in _DB_CONNECTIONS.items():
            try:
                conn.close()
            except Exception:
                pass
        _DB_CONNECTIONS.clear()
    print(f"  Closed {count} pooled DB connections")


def _cache_key(completion_text: str) -> str:
    """Generate a cache key from the full completion text."""
    return str(hash(completion_text))


def _clear_cache():
    """Clear the execution cache at the start of a new batch."""
    with _cache_lock:
        _execution_cache.clear()


def _get_cache(key: str) -> Optional[Dict[str, Any]]:
    with _cache_lock:
        return _execution_cache.get(key)


def _set_cache(key: str, value: Dict[str, Any]):
    with _cache_lock:
        _execution_cache[key] = value


# ================= Global TensorBoard Writer =================
_TB_WRITER: Optional[SummaryWriter] = None
_TB_STEP: int = 0


def _get_tb_writer() -> Optional[SummaryWriter]:
    return _TB_WRITER


def _log_tb_scalar(tag: str, value: float, step: Optional[int] = None):
    """Log a scalar to TensorBoard if writer is available."""
    writer = _get_tb_writer()
    if writer is not None:
        writer.add_scalar(tag, value, step if step is not None else _TB_STEP)


def _log_tb_scalars(main_tag: str, tag_scalar_dict: Dict[str, float], step: Optional[int] = None):
    """Log multiple scalars under one main tag to TensorBoard."""
    writer = _get_tb_writer()
    if writer is not None:
        writer.add_scalars(main_tag, tag_scalar_dict, step if step is not None else _TB_STEP)


# ================= Response Parsing for gpt-oss Format =================

def parse_gpt_oss_response(text: str) -> Tuple[str, str]:
    """
    Parse gpt-oss model response to extract thinking and final content.

    Returns: (thinking, content)

    Handles three formats:
    1. Full special tokens intact (e.g. offline inference):
       <|start|>assistant<|channel|>analysis<|message|>THINKING<|end|>
       <|start|>assistant<|channel|>final<|message|>CONTENT<|return|>

    2. Partial tokens (TRL strips the leading <|start|>assistant prefix):
       <|channel|>analysis<|message|>THINKING<|end|>
       <|start|>assistant<|channel|>final<|message|>CONTENT<|return|>

    3. No tokens — TRL decoded with skip_special_tokens=True (default in GRPOTrainer):
       Special tokens <|channel|>, <|message|>, <|end|>, <|start|> are ALL stripped.
       The remaining literal words "analysis" and "final" are kept, giving:
           "analysisTHINKINGassistantfinalCONTENT"
       This format is what reward functions actually see during GRPO training.
    """
    thinking = ""
    content = ""

    # --- Format 1 & 2: special tokens present ---
    analysis_match = re.search(
        r"<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|start\|>|$)",
        text, flags=re.DOTALL,
    )
    if analysis_match:
        thinking = analysis_match.group(1).strip()
    else:
        analysis_match = re.search(
            r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|channel\|>|$)",
            text, flags=re.DOTALL,
        )
        if analysis_match:
            thinking = analysis_match.group(1).strip()

    final_match = re.search(
        r"<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
        text, flags=re.DOTALL,
    )
    if final_match:
        content = final_match.group(1).strip()
    else:
        final_match = re.search(
            r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
            text, flags=re.DOTALL,
        )
        if final_match:
            content = final_match.group(1).strip()

    # --- Format 3: TRL stripped all special tokens (skip_special_tokens=True) ---
    # <|channel|>analysis<|message|> → "analysis"  (the literal word remains)
    # <|end|><|start|>assistant<|channel|>final<|message|> → "assistantfinal"
    # So the full completion looks like: "analysisTHINKINGassistantfinalCONTENT"
    if not thinking and not content:
        if text.startswith("analysis"):
            inner = text[len("analysis"):]
            af_idx = inner.find("assistantfinal")
            if af_idx != -1:
                thinking = inner[:af_idx].strip()
                content = inner[af_idx + len("assistantfinal"):].strip()
            else:
                # Truncated completion — no final channel reached; whole body is thinking
                thinking = inner.strip()

    # --- Last resort: treat entire text as content ---
    if not thinking and not content:
        content = text.strip()

    return thinking, content


def extract_sql(text: str) -> str:
    """Extract SQL from model response (handles gpt-oss format)."""
    thinking, content = parse_gpt_oss_response(text)
    search_text = content if content else text

    # Try fenced code block: ```sql ... ```
    m = re.search(r"```sql\s*(.*?)```", search_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try plain code block with SELECT
    m = re.search(r"```\s*(SELECT.*?)```", search_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # Try plain code block with WITH (CTE)
    m = re.search(r"```\s*(WITH.*?)```", search_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # ===== IMPROVED CTE HANDLING =====
    # For CTEs, we need to capture everything from WITH to the final semicolon
    # or end of statement. Use GREEDY match for the body but bounded by terminator.
    
    # Method 1: WITH ... ending with semicolon
    m = re.search(
        r"(WITH\s+.+?)(?:;(?:\s*$|\s*\n))",  # Greedy .+ captures full CTE
        search_text, 
        flags=re.IGNORECASE | re.DOTALL
    )
    if m:
        sql = m.group(1).strip()
        if _is_valid_cte(sql):
            return sql

    # Method 2: WITH that ends at double newline or EOF (no semicolon)
    m = re.search(
        r"(WITH\s+.+?SELECT\s+.+?)(?:\n\n|\Z)",  # Must have final SELECT
        search_text, 
        flags=re.IGNORECASE | re.DOTALL
    )
    if m:
        sql = m.group(1).strip()
        if _is_valid_cte(sql):
            return sql
    
    # Method 3: Bracket-balanced CTE extraction (most robust)
    cte_sql = _extract_cte_balanced(search_text)
    if cte_sql:
        return cte_sql
    # ===== END IMPROVED CTE HANDLING =====

    # If the text starts with WITH but all CTE extraction methods failed
    # (malformed/truncated CTE), bail out here rather than falling through to
    # SQL_SELECT_REGEX, which would return an inner-SELECT fragment from the
    # CTE body — incorrect SQL that may happen to pass syntax checks.
    if re.match(r'\s*WITH\b', search_text, re.IGNORECASE):
        return ""

    # Try SELECT pattern (non-CTE)
    m = SQL_SELECT_REGEX.search(search_text)
    if m:
        return m.group(1).strip()

    # Try line by line
    for line in search_text.splitlines():
        stripped = line.strip().upper()
        if stripped.startswith("SELECT") or stripped.startswith("WITH"):
            return line.strip()

    return ""


def _is_valid_cte(sql: str) -> bool:
    """
    Check if a CTE has balanced parentheses and ends with a SELECT.
    """
    if not sql:
        return False
    
    # Must start with WITH
    if not sql.strip().upper().startswith("WITH"):
        return False
    
    # Count parentheses - should be balanced
    open_count = sql.count('(')
    close_count = sql.count(')')
    if open_count != close_count:
        return False
    
    # Should have a final SELECT after the CTE definitions
    # Look for SELECT that's not inside parentheses
    depth = 0
    sql_upper = sql.upper()
    last_select_outside_parens = -1
    
    i = 0
    while i < len(sql):
        if sql[i] == '(':
            depth += 1
        elif sql[i] == ')':
            depth -= 1
        elif depth == 0 and sql_upper[i:i+6] == 'SELECT':
            last_select_outside_parens = i
        i += 1
    
    # The last SELECT at depth 0 should exist and not be at the very start
    return last_select_outside_parens > 10  # After "WITH x AS ("


def _extract_cte_balanced(text: str) -> Optional[str]:
    """
    Extract CTE using parenthesis balancing.
    
    This handles complex nested CTEs by tracking parenthesis depth
    and finding the final SELECT statement.
    """
    text_upper = text.upper()
    
    # Find WITH keyword
    with_match = re.search(r'\bWITH\s+', text_upper)
    if not with_match:
        return None
    
    start_idx = with_match.start()
    
    # Track parenthesis depth to find where CTE definitions end
    depth = 0
    in_cte_defs = True
    final_select_start = -1
    i = with_match.end()
    
    while i < len(text):
        char = text[i]
        
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            # When we return to depth 0 after being >0, we might be done with a CTE block
        elif char == "'" or char == '"':
            # Skip string literals
            quote_char = char
            i += 1
            while i < len(text) and text[i] != quote_char:
                if text[i] == '\\':
                    i += 1  # Skip escaped char
                i += 1
        elif depth == 0 and text_upper[i:i+6] == 'SELECT':
            # Found SELECT at top level - this could be the final one
            final_select_start = i
        
        i += 1
    
    if final_select_start == -1:
        return None
    
    # Now extract from WITH to end of final SELECT
    # Find the end: semicolon, double newline, or EOF
    end_patterns = [
        (';', 1),           # Semicolon (don't include it, or include - your choice)
        ('\n\n', 0),        # Double newline
        ('\n--', 0),        # Comment starts
    ]
    
    end_idx = len(text)
    for pattern, offset in end_patterns:
        pos = text.find(pattern, final_select_start)
        if pos != -1 and pos < end_idx:
            end_idx = pos + offset
    
    sql = text[start_idx:end_idx].strip()
    
    # Final validation
    if sql and _is_valid_cte(sql):
        return sql
    
    # Fallback: just return what we found if it looks reasonable
    if sql and sql.upper().startswith('WITH') and 'SELECT' in sql.upper():
        # Check balanced parens
        if sql.count('(') == sql.count(')'):
            return sql
    
    return None


def extract_sql_from_ground_truth(gt_content: str) -> str:
    """Extract SQL from ground truth content."""
    if not gt_content or not gt_content.strip():
        return ""

    stripped = gt_content.strip()
    upper = stripped.upper()
    if upper.startswith("SELECT") or upper.startswith("WITH"):
        return stripped.rstrip(";").strip()

    sql = extract_sql(gt_content)
    return sql if sql else stripped


def extract_thinking(text: str) -> str:
    """Extract thinking/reasoning from gpt-oss response."""
    thinking, _ = parse_gpt_oss_response(text)
    return thinking


def extract_database_name(prompt: str) -> str:
    """Extract database name from the prompt."""
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


def extract_question_and_hint(prompt: str) -> Tuple[str, str]:
    """Extract the natural language question and hint from the prompt."""
    question = ""
    hint = ""

    q_patterns = [
        r"Question:\s*(.+?)(?:Hint:|Schema:|$)",
        r"question[:\s]+(.+?)(?:hint|schema|$)",
    ]
    for pattern in q_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
        if match:
            question = match.group(1).strip()
            break

    h_patterns = [
        r"Hint:\s*(.+?)(?:Schema:|$)",
        r"hint[:\s]+(.+?)(?:schema|$)",
    ]
    for pattern in h_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE | re.DOTALL)
        if match:
            hint = match.group(1).strip()
            break

    return question, hint


def extract_schema_from_prompt(prompt: str) -> Tuple[set, set]:
    """
    Extract table and column names mentioned in the schema section of the prompt.
    """
    table_names = set()
    column_names = set()

    schema_match = re.search(
        r"(?:Schema|CREATE TABLE|Tables?:)(.*?)(?:Question:|Hint:|$)",
        prompt, re.IGNORECASE | re.DOTALL
    )

    search_text = schema_match.group(1) if schema_match else prompt

    # Always scan the full prompt for "Table: name" headers first —
    # schema_match consumes the first "Table:" token, so searching only
    # search_text would miss the first table name.
    for m in re.finditer(r'^Table:\s+(\w+)', prompt, re.IGNORECASE | re.MULTILINE):
        table_names.add(m.group(1).lower())

    # Extended SQL keywords to filter out of column name extraction
    sql_keywords = {
        'primary', 'foreign', 'unique', 'not', 'default', 'check',
        'references', 'constraint', 'key', 'null', 'autoincrement',
        'auto_increment', 'index', 'create', 'table', 'text', 'integer',
        'real', 'numeric', 'blob', 'varchar', 'char', 'int', 'float',
        'double', 'boolean', 'date', 'datetime', 'timestamp', 'if',
        'exists', 'temp', 'temporary', 'as', 'select', 'from', 'where',
    }

    # ── Format 1: CREATE TABLE DDL ─────────────────────────────────────
    for m in re.finditer(r'CREATE\s+TABLE\s+[`"\']?(\w+)[`"\']?', search_text, re.IGNORECASE):
        table_names.add(m.group(1).lower())

    # DDL-style column lines: "   col_name  TEXT NOT NULL ..."
    for m in re.finditer(
        r'^\s+[`"\']?([a-zA-Z_]\w*)[`"\']?\s+'
        r'(?:TEXT|INTEGER|REAL|NUMERIC|BLOB|VARCHAR|CHAR|INT|FLOAT|DOUBLE|BOOLEAN|DATE|DATETIME|TIMESTAMP)',
        search_text, re.IGNORECASE | re.MULTILINE
    ):
        col = m.group(1).lower()
        if col not in sql_keywords:
            column_names.add(col)

    # ── Format 2: Pipe-separated prompt format ─────────────────────────
    # Matches "Table: table_name" section headers
    for m in re.finditer(r'^Table:\s+(\w+)', search_text, re.IGNORECASE | re.MULTILINE):
        table_names.add(m.group(1).lower())

    # Matches "col_name (type ...)" entries at line start or after a pipe separator
    # e.g. "user_id (integer, FK->...) | list_id (integer, PK) | list_title (text)"
    for m in re.finditer(r'(?:^|\|)\s*([a-zA-Z_]\w*)\s+\(', search_text, re.MULTILINE):
        col = m.group(1).lower()
        if col not in sql_keywords:
            column_names.add(col)

    # ── Format 3: dot-notation references (table.column) ──────────────
    for m in re.finditer(r'(\w+)\.(\w+)', search_text):
        table_names.add(m.group(1).lower())
        column_names.add(m.group(2).lower())

    return table_names, column_names


def get_database_path(database_name: str, database_dir: str) -> str:
    """Get the full path to the SQLite database file."""
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
    """Execute SQL using pooled read-only connection with a hard timeout."""
    if not db_path or not os.path.exists(db_path):
        return False, "Database not found"

    if not sql or not sql.strip():
        return False, "Empty SQL"

    try:
        conn = _get_db_connection(db_path)
        if conn is None:
            return False, "Could not open database"

        # Enforce timeout via SQLite progress handler.
        # progress_handler is called every N SQLite VM opcodes; returning non-zero
        # raises OperationalError: interrupted — the only reliable way to stop a
        # runaway query (e.g. cartesian join, recursive CTE) on the same thread.
        deadline = time.monotonic() + timeout
        check_interval = 1000  # opcodes between checks (~1ms each at typical speed)
        def _timeout_handler():
            if time.monotonic() > deadline:
                return 1  # non-zero → interrupt
            return 0
        conn.set_progress_handler(_timeout_handler, check_interval)

        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            cursor.close()  # Close cursor (not connection) after each query
            return True, result
        finally:
            conn.set_progress_handler(None, 0)  # always clear handler after query

    except sqlite3.OperationalError as e:
        err = str(e)
        if "interrupted" in err:
            return False, f"Query timed out after {timeout}s"
        # If connection went stale, remove from pool so it gets recreated next call
        if "disk I/O error" in err or "database disk image is malformed" in err or "unable to open" in err:
            with _DB_CONN_LOCK:
                old_conn = _DB_CONNECTIONS.pop(db_path, None)
                if old_conn:
                    try:
                        old_conn.close()
                    except Exception:
                        pass
        return False, err
    except Exception as e:
        return False, str(e)

def normalize_result(result: Any) -> Any:
    """
    Normalize SQL query result for comparison.
    Handles: sorting, type conversion, None/NULL handling.
    """
    if result is None:
        return None
    if not isinstance(result, list):
        return result
    
    # Convert to list of tuples, normalize types
    normalized = []
    for row in result:
        if isinstance(row, (list, tuple)):
            # Convert each cell: None -> None, numbers stay, strings lowercase strip
            norm_row = tuple(
                cell.lower().strip() if isinstance(cell, str) else cell
                for cell in row
            )
            normalized.append(norm_row)
        else:
            normalized.append((row,))
    
    # Sort for order-independent comparison
    try:
        return sorted(normalized)
    except TypeError:
        # If rows aren't comparable (mixed types), return as-is
        return normalized

# ================== Helper: Get completion/prompt text ==================

def _get_completion_text(completion) -> str:
    """Extract raw text from a completion (handles list-of-dicts or string)."""
    if isinstance(completion, list):
        return completion[0]["content"] if completion else ""
    return str(completion)


def _get_prompt_text(prompt) -> str:
    """Extract raw text from a prompt (handles list-of-dicts or string)."""
    if isinstance(prompt, list):
        for m in reversed(prompt):
            if m.get("role") == "user":
                return m["content"]
        return prompt[-1]["content"] if prompt else ""
    return str(prompt)


def _ensure_list(val, length: int) -> list:
    """Ensure val is a list of the given length."""
    if val is None:
        return [None] * length
    if not isinstance(val, list):
        return [val] * length
    return val


# ================== N-gram / Schema Helpers ==================

def _tokenize_sql(sql: str) -> List[str]:
    """Tokenize SQL for n-gram and schema analysis."""
    if not sql:
        return []

    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)

    tokens = re.findall(
        r">=|<=|<>|!=|[a-z_][a-z0-9_]*\.?[a-z0-9_]*|'[^']*'|\"[^\"]*\"|\d+\.?\d*|[^\s]",
        sql
    )
    return [t for t in tokens if t.strip()]


def _extract_schema_items(sql: str) -> Tuple[set, Dict[str, str]]:
    """
    Extract referenced tables and columns from a SQL query.

    Returns:
        - schema_items: set of normalized schema references
        - alias_map: dict mapping aliases to table names
    """
    if not sql:
        return set(), {}

    sql_lower = sql.lower().strip()
    schema_items = set()
    alias_map = {}

    sql_keywords = {
        'select', 'from', 'where', 'join', 'inner', 'left', 'right',
        'outer', 'cross', 'on', 'and', 'or', 'not', 'in', 'between',
        'like', 'is', 'null', 'group', 'order', 'by', 'having', 'limit',
        'union', 'except', 'intersect', 'exists', 'case', 'when', 'then',
        'else', 'end', 'as', 'distinct', 'all', 'asc', 'desc', 'set',
        'into', 'values', 'update', 'delete', 'insert', 'create', 'drop',
        'alter', 'table', 'index', 'view', 'with', 'recursive', 'natural',
    }

    table_patterns = [
        r'(?:from|join)\s+([a-z_][a-z0-9_]*)(?:\s+(?:as\s+)?([a-z_][a-z0-9_]*))?'
    ]
    for pattern in table_patterns:
        for match in re.finditer(pattern, sql_lower):
            table = match.group(1)
            alias = match.group(2) if match.group(2) else None
            if table not in sql_keywords:
                schema_items.add(table)
                if alias and alias not in sql_keywords:
                    alias_map[alias] = table

    col_pattern = r'([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)'
    for match in re.finditer(col_pattern, sql_lower):
        qualifier = match.group(1)
        column = match.group(2)
        actual_table = alias_map.get(qualifier, qualifier)
        schema_items.add(f"{actual_table}.{column}")
        schema_items.add(actual_table)

    sql_keywords_extended = sql_keywords | {
        'count', 'sum', 'avg', 'max', 'min', 'coalesce', 'ifnull',
        'cast', 'substr', 'length', 'upper', 'lower', 'trim',
        'strftime', 'date', 'time', 'datetime', 'julianday',
        'group_concat', 'total', 'abs', 'round', 'replace', 'instr',
        'true', 'false',
    }

    select_match = re.search(r'select\s+(.*?)\s+from\s', sql_lower, re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        bare_cols = re.findall(r'(?<!\.)(?<![a-z0-9_])([a-z_][a-z0-9_]*)(?!\s*\()', select_clause)
        for col in bare_cols:
            if col not in sql_keywords_extended and not col.isdigit():
                schema_items.add(col)

    for clause_kw in ['where', r'group\s+by', r'order\s+by', 'having']:
        clause_match = re.search(
            rf'{clause_kw}\s+(.*?)(?:group\s+by|order\s+by|having|limit|union|except|intersect|$)',
            sql_lower, re.DOTALL
        )
        if clause_match:
            clause_text = clause_match.group(1)
            bare_cols = re.findall(r'(?<!\.)(?<![a-z0-9_])([a-z_][a-z0-9_]*)(?!\s*\()', clause_text)
            for col in bare_cols:
                if col not in sql_keywords_extended and not col.isdigit():
                    schema_items.add(col)

    return schema_items, alias_map


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def _get_ngrams(tokens: List[str], n: int) -> set:
    """Generate n-grams from a token list."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


# ================== Reward Functions ==================

def format_reward(completions, prompts, **kwargs) -> List[float]:
    """
    Format Reward: Enforce adherence to gpt-oss output structure.
    ...existing docstring...
    """
    scores = []

    ANALYSIS_OPEN = "<|start|>assistant<|channel|>analysis<|message|>"
    FINAL_OPEN = "<|start|>assistant<|channel|>final<|message|>"
    ANALYSIS_CLOSE = "<|end|>"
    FINAL_CLOSE = "<|return|>"

    # Partial markers (what reward functions actually receive after TRL strips special tokens)
    PARTIAL_ANALYSIS_OPEN = "analysis"   # <|channel|> and <|message|> stripped
    PARTIAL_FINAL_OPEN = "final"
    # Separator between analysis and final sections (stripped <|end|><|start|>assistant)
    ANALYSIS_TO_FINAL_SEP = "assistant"  # what remains between the two blocks

    batch_size = len(completions)

    # Track scoring breakdown for debugging
    if not hasattr(format_reward, "_call_count"):
        format_reward._call_count = 0
    format_reward._call_count += 1
    should_print = (format_reward._call_count <= 3) or (format_reward._call_count % 50 == 0)

    scoring_details = []  # Collect details for all completions in this batch

    for comp_idx, completion in enumerate(completions):
        response = _get_completion_text(completion)
        reward = 0.0
        details = []  # Track each scoring decision

        has_analysis_open = ANALYSIS_OPEN in response
        has_final_open = FINAL_OPEN in response

        has_partial_analysis = "<|channel|>analysis<|message|>" in response or response.lstrip().startswith("analysis")
        has_partial_final = "<|channel|>final<|message|>" in response

        # ── Use partial markers when full markers absent (TRL stripped special tokens) ──
        use_partial = not has_analysis_open and not has_final_open

        if use_partial:
            # Detect pattern: "analysis...assistant...final...SELECT..."
            # This is what reward functions actually receive
            has_analysis_section = response.lstrip().startswith("analysis") or "analysis" in response[:50]
            has_final_section = "final" in response and "assistant" in response

            if has_analysis_section:
                reward += 0.2
            if has_final_section:
                reward += 0.2

            # Check ordering: analysis text comes before "assistant" separator before final+SQL
            assistant_pos = response.find("assistant")
            final_pos = response.find("final", assistant_pos) if assistant_pos != -1 else -1
            if has_analysis_section and assistant_pos > 10 and final_pos > assistant_pos:
                reward += 0.2  # correct ordering
            
            # No duplicates check on partial
            if response.count("assistant") == 1:
                reward += 0.2

        # During GRPO training TRL strips the <|start|>assistant prefix from
        # the first completion token, so the analysis block always starts with
        # <|channel|>analysis<|message|> (partial form) rather than the full
        # <|start|>assistant<|channel|>analysis<|message|>.  Treat partial as
        # equivalent to full so the real runtime format is not penalised.
        PARTIAL_ANALYSIS_MARKER = "<|channel|>analysis<|message|>"
        effective_analysis_open = has_analysis_open or has_partial_analysis

        if effective_analysis_open:
            reward += 0.2
            if has_analysis_open:
                details.append("+0.2 analysis_open (full)")
                analysis_start_idx = response.index(ANALYSIS_OPEN) + len(ANALYSIS_OPEN)
            else:
                details.append("+0.2 analysis_open (partial — TRL stripped <|start|>assistant prefix)")
                p_idx = response.find(PARTIAL_ANALYSIS_MARKER)
                analysis_start_idx = (p_idx + len(PARTIAL_ANALYSIS_MARKER)) if p_idx != -1 else 0
            end_after_analysis = response.find(ANALYSIS_CLOSE, analysis_start_idx)
            if end_after_analysis != -1:
                final_after_analysis = response.find(FINAL_OPEN, analysis_start_idx)
                if final_after_analysis == -1 or end_after_analysis < final_after_analysis:
                    reward += 0.1
                    details.append("+0.1 analysis_closed_properly")
                else:
                    details.append(f"+0.0 analysis_close <|end|> found at {end_after_analysis} but FINAL_OPEN at {final_after_analysis}")
            else:
                details.append(f"+0.0 no <|end|> after analysis (searched from pos {analysis_start_idx})")
        else:
            details.append("+0.0 no analysis marker (full or partial)")

        if has_final_open:
            reward += 0.2
            details.append("+0.2 final_open")
            final_start_idx = response.index(FINAL_OPEN) + len(FINAL_OPEN)
            return_after_final = response.find(FINAL_CLOSE, final_start_idx)
            if return_after_final != -1:
                reward += 0.1
                details.append("+0.1 final_closed_with_return")
            else:
                details.append(f"+0.0 no <|return|> after final (searched from pos {final_start_idx})")
        else:
            details.append("+0.0 no FULL final marker")
            if has_partial_final:
                details.append("  ⚠️ BUT partial <|channel|>final<|message|> IS present!")

        # Check ordering — works for both full and partial analysis marker
        if effective_analysis_open and has_final_open:
            if has_analysis_open:
                analysis_pos = response.index(ANALYSIS_OPEN)
            else:
                analysis_pos = response.find(PARTIAL_ANALYSIS_MARKER)
                if analysis_pos == -1:
                    analysis_pos = 0
            final_pos = response.index(FINAL_OPEN)
            if analysis_pos < final_pos:
                reward += 0.2
                details.append(f"+0.2 correct_ordering (analysis@{analysis_pos} < final@{final_pos})")
            else:
                details.append(f"+0.0 wrong_ordering (analysis@{analysis_pos} >= final@{final_pos})")

        # No duplicates — count whichever marker form is present
        if has_analysis_open:
            analysis_count = response.count(ANALYSIS_OPEN)
        else:
            analysis_count = response.count(PARTIAL_ANALYSIS_MARKER)
        final_count = response.count(FINAL_OPEN)
        if analysis_count == 1 and final_count == 1:
            reward += 0.2
            details.append(f"+0.2 no_duplicates (analysis={analysis_count}, final={final_count})")
        else:
            details.append(f"+0.0 duplicates? (analysis_count={analysis_count}, final_count={final_count})")

        # Content quality checks
        thinking = extract_thinking(response)
        final_content = parse_gpt_oss_response(response)[1]
        sql = extract_sql(response)

        if thinking:
            thinking_words = thinking.split()
            word_count = len(thinking_words)
            thinking_lower = thinking.lower()

            if word_count > 20:
                reward += 0.25
                details.append(f"+0.25 thinking_length ({word_count} words)")
            else:
                details.append(f"+0.0 thinking_too_short ({word_count} words)")

            schema_indicators = [
                "table", "columns", "join", "select", "where", "from",
                "schema", "foreign key", "primary key", "the question",
                "need to", "should", "looking at",
            ]
            indicator_matches = sum(1 for s in schema_indicators if s in thinking_lower)
            if indicator_matches >= 2:
                reward += 0.25
                details.append(f"+0.25 schema_indicators ({indicator_matches} matches)")
            else:
                details.append(f"+0.0 few_schema_indicators ({indicator_matches} matches)")
        else:
            details.append(f"+0.0 no_thinking_extracted")

        if sql and sql.strip():
            reward += 0.25
            details.append(f"+0.25 has_sql")
            sql_upper = sql.strip().upper()
            if sql_upper.startswith("SELECT") or sql_upper.startswith("WITH"):
                reward += 0.25
                details.append(f"+0.25 sql_starts_correctly")
            else:
                details.append(f"+0.0 sql_starts_with: '{sql_upper[:20]}'")
        else:
            details.append(f"+0.0 no_sql_extracted")

        # Penalties
        if final_content and not sql:
            if len(final_content.split()) > 20:
                reward -= 0.5
                details.append(f"-0.5 final_has_text_but_no_sql ({len(final_content.split())} words)")

        if not has_final_open:
            if has_analysis_open:
                reward -= 1.0
                details.append(f"-1.0 has_analysis_but_no_final")
            else:
                if not sql:
                    reward -= 1.5
                    details.append(f"-1.5 no_format_no_sql")
                else:
                    reward -= 1.0
                    details.append(f"-1.0 no_final_but_has_sql")

        if has_final_open and final_content and sql and thinking:
            final_words = set(final_content.lower().split())
            thinking_words_set = set(thinking.lower().split())
            sql_words_set = set(sql.lower().split())

            non_sql_overlap = (final_words & thinking_words_set) - sql_words_set
            if len(non_sql_overlap) > len(sql_words_set):
                reward -= 0.5
                details.append(f"-0.5 final_contains_reasoning (overlap={len(non_sql_overlap)} > sql_words={len(sql_words_set)})")

        final_score = max(-1.5, min(2.0, reward))
        details.append(f"= FINAL: {final_score:.3f} (raw: {reward:.3f})")

        scoring_details.append((comp_idx, final_score, details))
        scores.append(final_score)

    # Print scoring breakdown
    if should_print:
        print(f"\n{'🏷️ '*20}")
        print(f"🏷️  FORMAT REWARD SCORING BREAKDOWN (call #{format_reward._call_count})")
        print(f"{'🏷️ '*20}")

        # Print score distribution for this batch
        print(f"\n  Batch scores: {[f'{s:.2f}' for s in scores]}")
        print(f"  Mean: {np.mean(scores):.3f}, Min: {min(scores):.3f}, Max: {max(scores):.3f}")

        # Count patterns
        n_full_analysis = sum(1 for c in completions if ANALYSIS_OPEN in _get_completion_text(c))
        n_partial_analysis = sum(1 for c in completions if "<|channel|>analysis<|message|>" in _get_completion_text(c))
        n_full_final = sum(1 for c in completions if FINAL_OPEN in _get_completion_text(c))
        n_partial_final = sum(1 for c in completions if "<|channel|>final<|message|>" in _get_completion_text(c))
        n_has_sql = sum(1 for c in completions if extract_sql(_get_completion_text(c)))

        print(f"\n  Batch pattern counts ({batch_size} completions):")
        print(f"    FULL  analysis marker: {n_full_analysis}/{batch_size}")
        print(f"    PARTIAL analysis marker: {n_partial_analysis}/{batch_size}")
        print(f"    FULL  final marker:    {n_full_final}/{batch_size}")
        print(f"    PARTIAL final marker:  {n_partial_final}/{batch_size}")
        print(f"    Has extractable SQL:   {n_has_sql}/{batch_size}")

        if n_partial_analysis > n_full_analysis or n_partial_final > n_full_final:
            print(f"\n    ⚠️  MORE partial markers than full markers!")
            print(f"    This means TRL is stripping the <|start|>assistant prefix from completions.")
            print(f"    → format_reward should check for PARTIAL markers too!")

        # Print detailed scoring for first 2 completions
        num_detail = min(2, len(scoring_details))
        for idx, score, details in scoring_details[:num_detail]:
            print(f"\n  ── Completion {idx} (score={score:.3f}) ──")
            for d in details:
                print(f"    {d}")

        print(f"\n{'🏷️ '*20}\n")

    return scores

def syntax_execution_reward(
    completions, prompts, ground_truth_sql=None, database_name=None, **kwargs
) -> List[float]:
    """
    Syntax Check + Execution Reward: Check if SQL is syntactically valid and executable.

    Also populates the execution cache for downstream rewards.

    Returns:
    +2.0: SQL executes successfully
    -2.0: SQL fails to execute
     0.0: No SQL extracted or database not found
    """

    scores = []
    database_dir = INLINE_CONFIG["database_dir"]

    ground_truth_sql = _ensure_list(ground_truth_sql, len(completions))
    database_name = _ensure_list(database_name, len(completions))

    for i, completion in enumerate(completions):
        response = _get_completion_text(completion)
        sql = extract_sql(response)
        key = _cache_key(response)

        cache_entry = {
            "sql": sql,
            "executed": False,
            "success": False,
            "result": None,
            "gt_result": None,
            "result_match": None,
            "db_path": "",
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
            cache_entry["db_path"] = ""
            _set_cache(key, cache_entry)
            scores.append(0.0)
            continue

        success, result = execute_sql(sql, db_path)

        cache_entry["executed"] = True
        cache_entry["success"] = success
        cache_entry["result"] = result
        cache_entry["db_path"] = db_path

        _set_cache(key, cache_entry)
        scores.append(2.0 if success else -2.0)

    return scores


def result_reward(
    completions, prompts, ground_truth_sql=None, database_name=None, **kwargs
) -> List[float]:
    """
    Execution Accuracy Reward (RLEF) with partial credit.

    MUST run after syntax_execution_reward (uses execution cache).

    Returns:
    +3.0: Results match exactly
    +1.5: Partial match — correct columns, subset of rows (or superset)
    -3.0: Results don't match at all
     0.0: Can't evaluate (no SQL / no DB)
    """
    scores = []

    ground_truth_sql = _ensure_list(ground_truth_sql, len(completions))
    database_name = _ensure_list(database_name, len(completions))

    for i, completion in enumerate(completions):
        text = _get_completion_text(completion)
        c_key = _cache_key(text)
        cached = _get_cache(c_key)

        score = 0.0

        db_name = database_name[i] if database_name[i] else extract_database_name(
            _get_prompt_text(prompts[i])
        )
        gt_sql = ground_truth_sql[i] if ground_truth_sql[i] else ""
        gt_sql = extract_sql_from_ground_truth(gt_sql)

        if not gt_sql or not db_name:
            scores.append(0.0)
            continue

        db_path = get_database_path(db_name, INLINE_CONFIG["database_dir"])
        if not db_path:
            scores.append(0.0)
            continue

        # Get predicted result from cache or execute
        pred_result = None
        if cached and cached.get("executed"):
            pred_result = cached.get("result")
        else:
            pred_sql = extract_sql(text)
            if pred_sql:
                success, result = execute_sql(pred_sql, db_path)
                if success:
                    pred_result = result

        # Execute ground truth
        gt_success, gt_result = execute_sql(gt_sql, db_path)
        if not gt_success:
            scores.append(0.0)
            continue

        if pred_result is None:
            scores.append(-3.0)
            continue

        # Normalize and compare
        pred_norm = normalize_result(pred_result)
        gt_norm = normalize_result(gt_result)

        if pred_norm == gt_norm:
            score = 3.0  # Exact match
        else:
            # ── Partial credit: reward structural similarity ──
            # Check column count match
            pred_cols = len(pred_norm[0]) if pred_norm and isinstance(pred_norm[0], tuple) else 0
            gt_cols = len(gt_norm[0]) if gt_norm and isinstance(gt_norm[0], tuple) else 0

            if pred_cols == gt_cols and pred_cols > 0:
                # Same number of columns — check row overlap
                pred_counter = Counter(pred_norm) if pred_norm else Counter()
                gt_counter = Counter(gt_norm) if gt_norm else Counter()
                if pred_counter and gt_counter:
                    intersection_count = sum((pred_counter & gt_counter).values())
                    precision = intersection_count / sum(pred_counter.values()) if pred_counter else 0
                    recall    = intersection_count / sum(gt_counter.values())   if gt_counter  else 0
                    

                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                        if f1 > 0.5:
                            # Significant overlap — partial credit
                            score = -3.0 + (f1 * 4.5)  # Maps f1=0.5→-0.75, f1=0.9→+1.05
                            score = min(score, 1.5)      # Cap partial credit at 1.5
                        else:
                            score = -3.0
                    else:
                        score = -3.0
                else:
                    score = -3.0
            else:
                score = -3.0  # Wrong structure entirely

        scores.append(score)

    return scores


def schema_linking_reward(completions, prompts, ground_truth_sql=None, **kwargs) -> List[float]:
    """
    Schema Linking Reward: Jaccard similarity of referenced schema items.

    Extracts tables and columns from both predicted and gold SQL,
    resolves aliases, and computes Jaccard similarity.

    Independent — does NOT depend on execution cache.

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

        similarity = _jaccard_similarity(pred_items, gt_items)
        scores.append(similarity * 2.0)

    return scores


def ngram_similarity_reward(completions, prompts, ground_truth_sql=None, **kwargs) -> List[float]:
    """
    N-gram Similarity Reward: Jaccard similarity of SQL token n-grams.

    Computes average Jaccard similarity across unigrams, bigrams, and trigrams.

    Independent — does NOT depend on execution cache.

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
            pred_ngrams = _get_ngrams(pred_tokens, n)
            gt_ngrams = _get_ngrams(gt_tokens, n)
            sim = _jaccard_similarity(pred_ngrams, gt_ngrams)
            similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities)
        scores.append(avg_similarity * 2.0)

    return scores


def thinking_quality_reward(completions, prompts, ground_truth_sql=None, **kwargs) -> List[float]:
    """
    Thinking Quality Reward: Evaluate chain-of-thought quality based on
    schema grounding and structural coherence.

    Unlike the old reasoning_reward which used generic keyword matching
    (easily gamed), this reward checks:

    1. Schema grounding: Does the thinking reference actual table/column names
       from the prompt's schema? (not generic SQL keywords)
    2. Length guardrail: Is the thinking a reasonable length (not too short
       = no reasoning, not too long = padding/repetition)?
    3. SQL coherence: Are schema items mentioned in thinking also present
       in the generated SQL? (thinking should be relevant to the answer)

    Scoring rubric (up to 2.0):

    Schema grounding (up to +0.8):
        +0.4: Thinking mentions ≥2 actual table names from the schema
        +0.4: Thinking mentions ≥3 actual column names from the schema

    Length guardrail (up to +0.4):
        +0.4: Thinking is 30-400 words (sweet spot)
        +0.2: Thinking is 15-30 or 400-600 words (acceptable)
        +0.0: Too short (<15) or too long (>600, likely repetitive)

    SQL coherence (up to +0.8):
        +0.4: ≥50% of tables mentioned in thinking appear in the SQL
        +0.4: ≥30% of columns mentioned in thinking appear in the SQL

    Range: 0.0 to 2.0
    """
    scores = []

    ground_truth_sql = _ensure_list(ground_truth_sql, len(completions))

    for i, completion in enumerate(completions):
        response = _get_completion_text(completion)
        thinking = extract_thinking(response)
        reward = 0.0

        if not thinking or len(thinking.strip()) < 10:
            scores.append(0.0)
            continue

        thinking_lower = thinking.lower()
        thinking_words = thinking_lower.split()
        word_count = len(thinking_words)

        # Get the prompt to extract actual schema items
        prompt_text = _get_prompt_text(prompts[i]) if i < len(prompts) else ""
        schema_tables, schema_columns = extract_schema_from_prompt(prompt_text)

        # ---- Schema grounding (up to +0.8) ----
        # Check if thinking references actual table names from the schema
        tables_mentioned = 0
        for table in schema_tables:
            if table in thinking_lower:
                tables_mentioned += 1

        if tables_mentioned >= 2:
            reward += 0.4
        elif tables_mentioned >= 1:
            reward += 0.2

        # Check if thinking references actual column names from the schema
        columns_mentioned = 0
        for col in schema_columns:
            if col in thinking_lower:
                columns_mentioned += 1

        if columns_mentioned >= 3:
            reward += 0.4
        elif columns_mentioned >= 1:
            reward += 0.2

        # ---- Length guardrail (up to +0.4) ----
        if 30 <= word_count <= 400:
            reward += 0.4
        elif 15 <= word_count < 30 or 400 < word_count <= 600:
            reward += 0.2
        # else: 0.0 (too short or too long)

        # ---- SQL coherence (up to +0.8) ----
        # Check that schema items discussed in thinking actually appear in the SQL
        pred_sql = extract_sql(response)
        if pred_sql:
            pred_sql_lower = pred_sql.lower()
            pred_items, _ = _extract_schema_items(pred_sql)

            # Tables discussed in thinking that also appear in SQL
            if schema_tables:
                tables_in_thinking = {t for t in schema_tables if t in thinking_lower}
                tables_in_sql = {t for t in schema_tables if t in pred_sql_lower}

                if tables_in_thinking:
                    overlap_ratio = len(tables_in_thinking & tables_in_sql) / len(tables_in_thinking)
                    if overlap_ratio >= 0.5:
                        reward += 0.4
                    elif overlap_ratio >= 0.25:
                        reward += 0.2

            # Columns discussed in thinking that also appear in SQL
            if schema_columns:
                cols_in_thinking = {c for c in schema_columns if c in thinking_lower}
                cols_in_sql = {c for c in schema_columns if c in pred_sql_lower}

                if cols_in_thinking:
                    col_overlap = len(cols_in_thinking & cols_in_sql) / len(cols_in_thinking)
                    if col_overlap >= 0.3:
                        reward += 0.4
                    elif col_overlap >= 0.15:
                        reward += 0.2

        scores.append(max(0.0, min(2.0, reward)))

    return scores


# ================== Composite Reward & Weights ==================

DEFAULT_REWARD_WEIGHTS = {
    "format_reward": 0.30,
    "syntax_execution_reward": 0.30,
    "result_reward": 1.00,
    "schema_linking_reward": 0.20,
    "ngram_similarity_reward": 0.20,
    "thinking_quality_reward": 0.15,
    "llm_judge_reward": 0.20,
}


def _verify_weight_constraint(weights: Dict[str, float], reward_config: Dict[str, bool]):
    """Verify that correct queries always outscore incorrect ones."""
    reward_ranges = {
        "format_reward": (-1.5, 2.0),
        "syntax_execution_reward": (-2.0, 2.0),
        "result_reward": (-3.0, 3.0),
        "schema_linking_reward": (0.0, 2.0),
        "ngram_similarity_reward": (0.0, 2.0),
        "thinking_quality_reward": (0.0, 2.0),
        "llm_judge_reward": (0.0, 2.0),
    }

    w_result = weights.get("result_reward", 1.0)
    result_gap = w_result * (
        reward_ranges["result_reward"][1] - reward_ranges["result_reward"][0]
    )

    max_other_advantage = 0.0
    for name, (rmin, rmax) in reward_ranges.items():
        if name == "result_reward":
            continue
        if not reward_config.get(name, False):
            continue
        w = weights.get(name, 0.0)
        max_other_advantage += w * (rmax - rmin)

    is_valid = result_gap > max_other_advantage

    if not is_valid:
        min_w = max_other_advantage / 6.0
        print("⚠ WARNING: Incorrect queries CAN outscore correct queries!")
        print(f"  result_gap = {result_gap:.3f}, max_other_advantage = {max_other_advantage:.3f}")
        print(f"  Minimum safe result_reward weight: {min_w:.3f}")
    else:
        print(f"✅ Weight constraint satisfied: result_gap={result_gap:.3f} > other_advantage={max_other_advantage:.3f}")

    return is_valid


def composite_reward(
    completions,
    prompts,
    ground_truth_sql=None,
    database_name=None,
    ground_truth_thinking=None,
    **kwargs
) -> List[float]:
    """
    Composite Weighted Reward: Single reward function that computes all
    sub-rewards and returns their weighted sum.

    Enables:
    1. Proper weighting with correctness dominance guarantee
    2. Per-component logging to TensorBoard
    3. Consistent cache lifecycle (clear once, use everywhere)
    """
    global _TB_STEP
    _clear_cache()

    reward_config = INLINE_CONFIG.get("reward_config", {})
    reward_weights = INLINE_CONFIG.get("reward_weights", DEFAULT_REWARD_WEIGHTS)
    batch_size = len(completions)

    # ===== FORMAT INVESTIGATION: Print raw completion structure =====
    if not hasattr(composite_reward, "_call_count"):
        composite_reward._call_count = 0
    composite_reward._call_count += 1  # ← Increment FIRST
    
    should_print_debug = (composite_reward._call_count == 1 and INLINE_CONFIG.get("save_raw_completions", False))
    
    if should_print_debug and INLINE_CONFIG.get("save_raw_completions", True):
        print("\n" + "🔬" * 35)
        print(f"🔬 FORMAT INVESTIGATION — composite_reward call #{composite_reward._call_count}")
        print("🔬" * 35)
        
        # Print completion structure info
        print(f"\n  batch_size: {batch_size}")
        print(f"  type(completions): {type(completions)}")
        if batch_size > 0:
            print(f"  type(completions[0]): {type(completions[0])}")
            if isinstance(completions[0], dict):
                print(f"  len(completions[0]): {len(completions[0])}")
                if len(completions[0]) > 0:
                    print(f"  type(completions[0][0]): {type(completions[0][0])}")
                    if isinstance(completions[0][0], dict):
                        print(f"  completions[0][0].keys(): {completions[0][0].keys()}")
        
        # Print first 2 raw completions
        num_to_inspect = min(2, batch_size)
        for idx in range(num_to_inspect):
            completion = completions[idx]
            raw_text = _get_completion_text(completion)
            
            print(f"\n  {'─'*60}")
            print(f"  📝 Completion {idx + 1}/{batch_size}")
            print(f"  {'─'*60}")
            print(f"  Raw text length: {len(raw_text)} chars, {len(raw_text.split())} words")
            
            # Print first 800 chars with visible special tokens
            print(f"\n  FULL RAW TEXT (first 800 chars):")
            print(f"  ┌{'─'*58}┐")
            for line in raw_text[:800].split('\n'):
                print(f"  │ {line[:56]:<56} │")
            if len(raw_text) > 800:
                print(f"  │ {'... [TRUNCATED]':<56} │")
            print(f"  └{'─'*58}┘")
            
            # Check ALL possible format markers (full and partial)
            markers_to_check = {
                # Full markers (with <|start|>assistant prefix)
                "<|start|>assistant<|channel|>analysis<|message|>": "FULL analysis open",
                "<|start|>assistant<|channel|>final<|message|>": "FULL final open",
                # Partial markers (without prefix — what model might generate)
                "<|channel|>analysis<|message|>": "PARTIAL analysis open",
                "<|channel|>final<|message|>": "PARTIAL final open",
                # Individual tokens
                "<|start|>": "start token",
                "<|channel|>": "channel token",
                "<|message|>": "message token",
                "<|end|>": "end token",
                "<|return|>": "return token",
                # Common alternatives the model might generate instead
                "assistant": "word 'assistant'",
                "analysis": "word 'analysis'",
                "final": "word 'final'",
                "```sql": "SQL code fence",
                "```": "code fence",
                "SELECT": "SELECT keyword",
                "select": "select keyword",
            }
            
            print(f"\n  FORMAT MARKER SCAN:")
            for marker, description in markers_to_check.items():
                count = raw_text.count(marker)
                if count > 0:
                    # Find position of first occurrence
                    pos = raw_text.index(marker)
                    context_start = max(0, pos - 20)
                    context_end = min(len(raw_text), pos + len(marker) + 20)
                    context = raw_text[context_start:context_end].replace('\n', '\\n')
                    print(f"    ✅ {description:<30} count={count}  pos={pos}  context: ...{context}...")
                else:
                    print(f"    ❌ {description:<30} NOT FOUND")
            
            # Show what extract_thinking and extract_sql get
            thinking = extract_thinking(raw_text)
            sql = extract_sql(raw_text)
            thinking_parsed, content_parsed = parse_gpt_oss_response(raw_text)
            
            print(f"\n  PARSING RESULTS:")
            print(f"    parse_gpt_oss_response() →")
            print(f"      thinking: {len(thinking_parsed)} chars → '{thinking_parsed[:150]}{'...' if len(thinking_parsed) > 150 else ''}'")
            print(f"      content:  {len(content_parsed)} chars → '{content_parsed[:150]}{'...' if len(content_parsed) > 150 else ''}'")
            print(f"    extract_thinking() → {len(thinking)} chars")
            print(f"    extract_sql() → '{sql[:200]}{'...' if len(sql) > 200 else ''}'" if sql else "    extract_sql() → EMPTY")
        
        # Also print the prompt structure for context
        if batch_size > 0:
            print(f"\n  {'─'*60}")
            print(f"  📋 Prompt structure (first prompt):")
            print(f"  {'─'*60}")
            prompt = prompts[0]
            print(f"  type(prompt): {type(prompt)}")
            if isinstance(prompt, list):
                for m_idx, m in enumerate(prompt):
                    if isinstance(m, dict):
                        print(f"    msg[{m_idx}]: role={m.get('role', '?')}, content_len={len(m.get('content', ''))}")
                    else:
                        print(f"    msg[{m_idx}]: {type(m)} = {str(m)[:100]}")
            else:
                print(f"  prompt (first 200 chars): {str(prompt)[:200]}")
        
        print("\n" + "🔬" * 35 + "\n")
    # ===== END FORMAT INVESTIGATION =====

    component_scores = {
        "format_reward": [0.0] * batch_size,
        "syntax_execution_reward": [0.0] * batch_size,
        "result_reward": [0.0] * batch_size,
        "schema_linking_reward": [0.0] * batch_size,
        "ngram_similarity_reward": [0.0] * batch_size,
        "thinking_quality_reward": [0.0] * batch_size,
        "llm_judge_reward": [0.0] * batch_size,
    }

    # Compute each enabled sub-reward (order matters for cache!)
    if reward_config.get("format_reward", True):
        component_scores["format_reward"] = format_reward(
            completions, prompts, **kwargs
        )

    if reward_config.get("syntax_execution_reward", True) or reward_config.get(
        "execution_reward", True
    ):
        component_scores["syntax_execution_reward"] = syntax_execution_reward(
            completions,
            prompts,
            ground_truth_sql=ground_truth_sql,
            database_name=database_name,
            **kwargs,
        )

    if reward_config.get("result_reward", True):
        component_scores["result_reward"] = result_reward(
            completions,
            prompts,
            ground_truth_sql=ground_truth_sql,
            database_name=database_name,
            **kwargs,
        )

    if reward_config.get("schema_linking_reward", False):
        component_scores["schema_linking_reward"] = schema_linking_reward(
            completions,
            prompts,
            ground_truth_sql=ground_truth_sql,
            **kwargs,
        )

    if reward_config.get("ngram_similarity_reward", False):
        component_scores["ngram_similarity_reward"] = ngram_similarity_reward(
            completions,
            prompts,
            ground_truth_sql=ground_truth_sql,
            **kwargs,
        )

    if reward_config.get("thinking_quality_reward", True):
        component_scores["thinking_quality_reward"] = thinking_quality_reward(
            completions,
            prompts,
            ground_truth_sql=ground_truth_sql,
            **kwargs,
        )

    if reward_config.get("llm_judge_reward", False):
        # LLM judge disabled — no self-judging
        pass

    # Compute weighted sum
    final_scores = []
    for i in range(batch_size):
        weighted_sum = 0.0
        for name, sub_scores in component_scores.items():
            w = reward_weights.get(name, 0.0)
            weighted_sum += w * sub_scores[i]
        final_scores.append(weighted_sum)

    # ---- TensorBoard logging ----
    _TB_STEP = composite_reward._call_count

    # Log to TensorBoard every call
    for name, sub_scores in component_scores.items():
        if not reward_config.get(name, name in ("format_reward",)):
            continue
        mean_s = sum(sub_scores) / len(sub_scores) if sub_scores else 0.0
        _log_tb_scalar(f"rewards/{name}_mean", mean_s)

    mean_final = sum(final_scores) / len(final_scores) if final_scores else 0.0
    _log_tb_scalar("rewards/composite_mean", mean_final)

    # Execution accuracy
    result_scores = component_scores["result_reward"]
    n_correct = sum(1 for s in result_scores if s > 0)
    n_evaluated = sum(1 for s in result_scores if s != 0)
    exec_accuracy = n_correct / max(n_evaluated, 1)
    _log_tb_scalar("rewards/execution_accuracy", exec_accuracy)
    _log_tb_scalar("rewards/n_correct", float(n_correct))
    _log_tb_scalar("rewards/n_evaluated", float(n_evaluated))

    # SQL extraction rate
    syntax_scores = component_scores["syntax_execution_reward"]
    n_extracted = sum(1 for s in syntax_scores if s != 0)
    n_executable = sum(1 for s in syntax_scores if s > 0)
    _log_tb_scalar("rewards/sql_extraction_rate", n_extracted / max(batch_size, 1))
    _log_tb_scalar("rewards/sql_execution_rate", n_executable / max(batch_size, 1))

    # ---- Generation quality diagnostics ----
    # Track per-prompt generation quality (important for GRPO signal quality)
    num_generations = INLINE_CONFIG.get("num_generations", 4)
    if batch_size >= num_generations and batch_size % num_generations == 0:
        num_prompts = batch_size // num_generations
        all_gens_no_sql_count = 0
        some_gens_no_sql_count = 0
        all_gens_failed_exec_count = 0
        no_variance_count = 0

        for p_idx in range(num_prompts):
            start = p_idx * num_generations
            end = start + num_generations
            prompt_syntax_scores = syntax_scores[start:end]
            prompt_result_scores = result_scores[start:end]
            prompt_final_scores = final_scores[start:end]

            n_with_sql = sum(1 for s in prompt_syntax_scores if s != 0)
            n_no_sql = num_generations - n_with_sql

            if n_no_sql == num_generations:
                all_gens_no_sql_count += 1
            elif n_no_sql > 0:
                some_gens_no_sql_count += 1

            # Count prompts where all generations failed execution
            n_exec_fail = sum(1 for s in prompt_syntax_scores if s < 0)
            if n_exec_fail == num_generations:
                all_gens_failed_exec_count += 1

            # Check if all final_scores are identical (no GRPO signal)
            if len(set(round(s, 4) for s in prompt_final_scores)) <= 1:
                no_variance_count += 1

        _log_tb_scalar("diagnostics/prompts_all_gens_no_sql", float(all_gens_no_sql_count))
        _log_tb_scalar("diagnostics/prompts_some_gens_no_sql", float(some_gens_no_sql_count))
        _log_tb_scalar("diagnostics/prompts_all_gens_failed_exec", float(all_gens_failed_exec_count))
        _log_tb_scalar("diagnostics/prompts_no_reward_variance", float(no_variance_count))
        _log_tb_scalar("diagnostics/pct_all_gens_no_sql", 100.0 * all_gens_no_sql_count / max(num_prompts, 1))
        _log_tb_scalar("diagnostics/pct_no_reward_variance", 100.0 * no_variance_count / max(num_prompts, 1))

    # Memory stats
    if torch.cuda.is_available():
        _log_tb_scalar("system/gpu_memory_allocated_gb", torch.cuda.memory_allocated(0) / 1e9)
        _log_tb_scalar("system/gpu_memory_reserved_gb", torch.cuda.memory_reserved(0) / 1e9)

    # Periodic cleanup
    if composite_reward._call_count % 20 == 0:
        gc.collect()
        torch.cuda.empty_cache()

    # Console logging (periodic)
    log_every = INLINE_CONFIG.get("reward_log_every", 10)

    if composite_reward._call_count % log_every == 0:
        print("\n" + "=" * 70)
        print(
            f"REWARD BREAKDOWN (call #{composite_reward._call_count}, batch_size={batch_size})"
        )
        print("=" * 70)
        print(
            f"{'Component':<30} {'Weight':>6} {'Mean':>8} {'Min':>8} {'Max':>8} {'W×Mean':>8}"
        )
        print(f"{'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for name, sub_scores in component_scores.items():
            if not reward_config.get(name, name == "format_reward"):
                continue

            w = reward_weights.get(name, 0.0)
            mean_s = sum(sub_scores) / len(sub_scores) if sub_scores else 0.0
            min_s = min(sub_scores) if sub_scores else 0.0
            max_s = max(sub_scores) if sub_scores else 0.0

            print(
                f"{name:<30} {w:>6.2f} {mean_s:>8.3f} {min_s:>8.3f} {max_s:>8.3f} {w*mean_s:>8.3f}"
            )

        min_final = min(final_scores) if final_scores else 0.0
        max_final = max(final_scores) if final_scores else 0.0

        print(f"{'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        print(
            f"{'WEIGHTED TOTAL':<30} {'':>6} {mean_final:>8.3f} {min_final:>8.3f} {max_final:>8.3f}"
        )

        n_incorrect = sum(1 for s in result_scores if s < 0)
        n_unknown = sum(1 for s in result_scores if s == 0)

        print(
            f"\nExecution accuracy: {n_correct}/{batch_size} correct, "
            f"{n_incorrect} wrong, {n_unknown} unknown "
            f"({exec_accuracy*100:.1f}%)"
        )

        # Print generation quality diagnostics
        if batch_size >= num_generations and batch_size % num_generations == 0:
            num_prompts = batch_size // num_generations
            print(f"\n📊 Per-prompt generation diagnostics ({num_prompts} prompts × {num_generations} gens):")
            print(f"   All {num_generations} gens no SQL:      {all_gens_no_sql_count}/{num_prompts} — ⚠️ zero useful signal" if all_gens_no_sql_count > 0 else f"   All {num_generations} gens no SQL:      0/{num_prompts} ✅")
            print(f"   Some gens no SQL:        {some_gens_no_sql_count}/{num_prompts} — ok, negative contrast helps learning")
            print(f"   All gens failed exec:    {all_gens_failed_exec_count}/{num_prompts}" + (" — ⚠️ model struggling" if all_gens_failed_exec_count > num_prompts * 0.5 else ""))
            print(f"   No reward variance:      {no_variance_count}/{num_prompts}" + (" — ⚠️ degenerate, no GRPO signal" if no_variance_count > num_prompts * 0.3 else ""))

        print("=" * 70 + "\n")

    return final_scores


# ================== Build Reward Functions ==================

def build_reward_functions(config: Dict[str, Any]) -> Tuple[List, List[str]]:
    """
    Build reward function(s) based on config.

    Returns a SINGLE composite reward function that computes all sub-rewards
    internally with proper weighting.
    """
    reward_config = config.get("reward_config", {})  # composite_reward reads from INLINE_CONFIG directly

    reward_weights = config.get("reward_weights", DEFAULT_REWARD_WEIGHTS)
    use_composite = config.get("use_composite_reward", True)

    if use_composite:
        _verify_weight_constraint(reward_weights, reward_config)

        reward_funcs = [composite_reward]
        reward_names = ["composite_reward (weighted)"]

        print("\nUsing COMPOSITE weighted reward function")
        print("Enabled sub-rewards and weights:")
        for name, w in reward_weights.items():
            enabled = reward_config.get(name, False)
            status = f"{w:.2f}" if enabled else "DISABLED"
            print(f"  {'✓' if enabled else '✗'} {name}: {status}")

        # Print theoretical score ranges
        min_possible = 0.0
        max_possible = 0.0
        reward_ranges = {
            "format_reward": (-1.5, 2.0),
            "syntax_execution_reward": (-2.0, 2.0),
            "result_reward": (-3.0, 3.0),
            "schema_linking_reward": (0.0, 2.0),
            "ngram_similarity_reward": (0.0, 2.0),
            "thinking_quality_reward": (0.0, 2.0),
            "llm_judge_reward": (0.0, 2.0),
        }

        for name, (rmin, rmax) in reward_ranges.items():
            if reward_config.get(name, False):
                w = reward_weights.get(name, 0.0)
                min_possible += w * rmin
                max_possible += w * rmax

        print(
            f"\nTheoretical composite score range: "
            f"[{min_possible:.2f}, {max_possible:.2f}]"
        )

        return reward_funcs, reward_names

    else:
        # Unweighted list fallback
        reward_funcs = []
        reward_names = []

        if reward_config.get("format_reward", True):
            reward_funcs.append(format_reward)
            reward_names.append("format_reward [-1.5, +2.0]")

        if reward_config.get("syntax_execution_reward", True) \
           or reward_config.get("execution_reward", True):
            reward_funcs.append(syntax_execution_reward)
            reward_names.append("syntax_execution_reward [-2.0, +2.0]")

        if reward_config.get("result_reward", True):
            reward_funcs.append(result_reward)
            reward_names.append("result_reward [-3.0, +3.0]")

        if reward_config.get("schema_linking_reward", False):
            reward_funcs.append(schema_linking_reward)
            reward_names.append("schema_linking_reward [0.0, +2.0]")

        if reward_config.get("ngram_similarity_reward", False):
            reward_funcs.append(ngram_similarity_reward)
            reward_names.append("ngram_similarity_reward [0.0, +2.0]")

        if reward_config.get("thinking_quality_reward", True):
            reward_funcs.append(thinking_quality_reward)
            reward_names.append("thinking_quality_reward [0.0, +2.0]")

        print("\nUsing UNWEIGHTED reward list (TRL sums equally)")
        print("⚠️ WARNING: No correctness dominance guarantee!")
        print("Enabled reward functions (in order):")
        for name in reward_names:
            print(f"  • {name}")

        return reward_funcs, reward_names


# ============== Dataset Analysis ==============

def analyze_dataset_tokens(data: List[Dict[str, Any]], tokenizer, name: str = "Dataset"):
    """Analyze token distribution in the dataset."""
    print(f"\n{'='*60}")
    print(f"📊 TOKEN ANALYSIS: {name}")
    print(f"{'='*60}")

    prompt_lengths = []
    completion_lengths = []
    total_lengths = []
    thinking_lengths = []
    sql_lengths = []

    for item in data:
        messages = item["messages"]

        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
                    )
        prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        prompt_lengths.append(prompt_tokens)

        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        if assistant_msg:
            content = assistant_msg.get("content", "")
            thinking = assistant_msg.get("thinking", "")

            content_tokens = len(tokenizer.encode(content, add_special_tokens=False))
            thinking_tokens = len(tokenizer.encode(thinking, add_special_tokens=False)) if thinking else 0

            completion_lengths.append(content_tokens + thinking_tokens)
            thinking_lengths.append(thinking_tokens)
            sql_lengths.append(content_tokens)

        total_lengths.append(prompt_lengths[-1] + (completion_lengths[-1] if completion_lengths else 0))

    def print_stats(stat_name: str, values: List[int]):
        if not values:
            return
        arr = np.array(values)
        print(f"\n  {stat_name}:")
        print(f"    Min:    {arr.min():,}")
        print(f"    Max:    {arr.max():,}")
        print(f"    Mean:   {arr.mean():,.1f}")
        print(f"    Median: {np.median(arr):,.1f}")
        print(f"    Std:    {arr.std():,.1f}")
        print(f"    P90:    {np.percentile(arr, 90):,.1f}")
        print(f"    P95:    {np.percentile(arr, 95):,.1f}")
        print(f"    P99:    {np.percentile(arr, 99):,.1f}")

    print(f"\n  Total samples: {len(data)}")
    print_stats("Prompt tokens", prompt_lengths)
    print_stats("Completion tokens (thinking + SQL)", completion_lengths)
    print_stats("Thinking tokens", thinking_lengths)
    print_stats("SQL/Content tokens", sql_lengths)
    print_stats("Total tokens (prompt + completion)", total_lengths)

    max_prompt = INLINE_CONFIG["max_prompt_length"]
    max_completion = INLINE_CONFIG["max_completion_length"]

    prompt_truncated = sum(1 for p in prompt_lengths if p > max_prompt)
    completion_truncated = sum(1 for c in completion_lengths if c > max_completion)

    print(f"\n  ⚠️  Truncation warnings:")
    print(f"    Prompts > {max_prompt} tokens: {prompt_truncated} ({100*prompt_truncated/len(data):.1f}%)")
    print(f"    Completions > {max_completion} tokens: {completion_truncated} ({100*completion_truncated/len(data):.1f}%)")

    print(f"\n  📈 Prompt length distribution:")
    bins = [0, 256, 512, 1024, 1536, 2048, 3072, 4096, float('inf')]
    for i in range(len(bins) - 1):
        count = sum(1 for p in prompt_lengths if bins[i] <= p < bins[i+1])
        bar = "█" * (count * 40 // max(len(data), 1))
        label = f"{bins[i]}-{bins[i+1] if bins[i+1] != float('inf') else '∞'}"
        print(f"    {label:>12}: {bar} ({count})")

    print(f"\n{'='*60}\n")

    return {
        "prompt_lengths": prompt_lengths,
        "completion_lengths": completion_lengths,
        "thinking_lengths": thinking_lengths,
        "total_lengths": total_lengths,
    }


# ============== Dataset Preparation ==============

def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load JSONL file."""
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
    max_prompt_tokens: int = 2048,
    max_completion_tokens: int = 1536,
) -> Tuple[Dataset, Dict[str, int]]:
    """
    Prepare dataset for GRPO training with gpt-oss reasoning model.
    Includes filtering by token length (never truncates — only filters).
    
    Returns:
        (dataset, filter_stats) — the dataset and a dict of filtering statistics
    """
    processed_data = []
    filtered_prompt_too_long = 0
    filtered_total_too_long = 0
    filtered_no_user_msg = 0
    filtered_no_gt_sql = 0
    filtered_completion_too_long = 0
    max_seq_length = INLINE_CONFIG.get("max_seq_length", 4096)

    prompt_token_lengths = []  # Track all prompt lengths for reporting
    completion_token_lengths = []  # Track all completion lengths for reporting

    for item in data:
        messages = item["messages"]

        prompt_messages = []
        for m in messages:
            if m["role"] != "assistant":
                prompt_messages.append({"role": m["role"], "content": m["content"]})

        if not any(m["role"] == "user" for m in prompt_messages):
            filtered_no_user_msg += 1
            continue

        # Tokenize the prompt to get exact length
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_tokens)
        prompt_token_lengths.append(prompt_len)

        if prompt_len > max_prompt_tokens:
            filtered_prompt_too_long += 1
            continue

        if prompt_len + max_completion_tokens > max_seq_length:
            filtered_total_too_long += 1
            continue

        # Extract ground truth completion and measure its length
        assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
        if assistant_msg:
            gt_sql_raw = assistant_msg.get("content", "")
            gt_thinking = assistant_msg.get("thinking", "")
            gt_sql = extract_sql_from_ground_truth(gt_sql_raw)

            # Measure ground truth completion length (thinking + content)
            # This approximates what the model would need to generate
            gt_completion_text = ""
            if gt_thinking:
                gt_completion_text += gt_thinking + "\n"
            if gt_sql_raw:
                gt_completion_text += gt_sql_raw
            
            gt_completion_tokens = len(tokenizer.encode(gt_completion_text, add_special_tokens=False)) if gt_completion_text else 0
            completion_token_lengths.append(gt_completion_tokens)

            # Filter if ground truth completion exceeds max_completion_length
            # The model cannot generate the correct answer if it's longer than the hard cap
            if gt_completion_tokens > max_completion_tokens:
                filtered_completion_too_long += 1
                continue
        else:
            gt_sql = ""
            gt_thinking = ""
            gt_sql_raw = ""
            completion_token_lengths.append(0)

        # Warn if no ground truth SQL (can't compute result_reward)
        if not gt_sql:
            filtered_no_gt_sql += 1
            # Still include — format/syntax rewards still work

        user_msg = next((m for m in messages if m["role"] == "user"), None)
        db_name = extract_database_name(user_msg["content"]) if user_msg else ""

        if not db_name:
            db_name = item.get("db_id", item.get("database", ""))

        processed_data.append({
            "prompt": prompt_messages,
            "ground_truth_sql": gt_sql,
            "ground_truth_thinking": gt_thinking,
            "database_name": db_name,
        })

    total_input = len(data)
    total_kept = len(processed_data)

    filter_stats = {
        "total_input": total_input,
        "total_kept": total_kept,
        "filtered_no_user_msg": filtered_no_user_msg,
        "filtered_prompt_too_long": filtered_prompt_too_long,
        "filtered_total_too_long": filtered_total_too_long,
        "filtered_completion_too_long": filtered_completion_too_long,
        "filtered_no_gt_sql": filtered_no_gt_sql,
        "max_prompt_tokens": max_prompt_tokens,
        "max_completion_tokens": max_completion_tokens,
        "max_seq_length": max_seq_length,
        "prompt_token_lengths": prompt_token_lengths,
        "completion_token_lengths": completion_token_lengths,
    }

    total_filtered = filtered_prompt_too_long + filtered_total_too_long + filtered_no_user_msg + filtered_completion_too_long

    print("\nDataset filtering results:")
    print(f"  Total input examples: {total_input}")
    print(f"  Kept: {total_kept} ({100 * total_kept / max(total_input, 1):.1f}%)")
    print(f"  Filtered (no user message):                    {filtered_no_user_msg}")
    print(f"  Filtered (prompt > {max_prompt_tokens} tokens):               {filtered_prompt_too_long}")
    print(f"  Filtered (prompt + completion > {max_seq_length} tokens):  {filtered_total_too_long}")
    print(f"  Filtered (GT completion > {max_completion_tokens} tokens):     {filtered_completion_too_long}")
    print(f"  Samples with no ground truth SQL:              {filtered_no_gt_sql}")
    if total_filtered > 0:
        print(f"  ⚠️ Total filtered out: {total_filtered} ({100 * total_filtered / max(total_input, 1):.1f}%)")
    if total_kept == 0:
        print("  ❌ WARNING: No examples survived filtering!")

    if prompt_token_lengths:
        arr = np.array(prompt_token_lengths)
        print(f"\n  📊 Prompt token length distribution (BEFORE filtering):")
        print(f"    Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.0f}, Median: {np.median(arr):.0f}")
        print(f"    P90: {np.percentile(arr, 90):.0f}, P95: {np.percentile(arr, 95):.0f}, P99: {np.percentile(arr, 99):.0f}")
        n_over = sum(1 for p in prompt_token_lengths if p > max_prompt_tokens)
        print(f"    Samples > {max_prompt_tokens} tokens: {n_over}/{len(prompt_token_lengths)} ({100*n_over/len(prompt_token_lengths):.1f}%)")

    if completion_token_lengths:
        arr = np.array(completion_token_lengths)
        print(f"\n  📊 GT Completion token length distribution (BEFORE filtering):")
        print(f"    Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.0f}, Median: {np.median(arr):.0f}")
        print(f"    P90: {np.percentile(arr, 90):.0f}, P95: {np.percentile(arr, 95):.0f}, P99: {np.percentile(arr, 99):.0f}")
        n_over = sum(1 for c in completion_token_lengths if c > max_completion_tokens)
        print(f"    Samples > {max_completion_tokens} tokens: {n_over}/{len(completion_token_lengths)} ({100*n_over/len(completion_token_lengths):.1f}%)")

    if total_kept > 0:
        ex = processed_data[0]
        print(f"\n  Sample entry:")
        print(f"    database_name: {ex['database_name']}")
        print(f"    ground_truth_sql: {ex['ground_truth_sql'][:100]}...")
        print(f"    prompt roles: {[m['role'] for m in ex['prompt']]}")

    return Dataset.from_list(processed_data), filter_stats


def _verify_unsloth_kernels(model):
    """
    Verify Unsloth 2026.x kernel activation.
    
    In Unsloth 2026.x the architecture is:
      - Base weights (q_proj etc): Unsloth 4-bit patched with fused forward
      - LoRA adapters (lora_A/B):  Standard torch.nn.Linear (tiny, negligible)
      - Combined forward:          Unsloth fused kernel handles both together
    
    So checking for 'unsloth' in lora_A/lora_B module is WRONG.
    Instead check: are the BASE projection layers using Unsloth's forward?
    """
    print("\n🔍 VERIFYING UNSLOTH KERNEL ACTIVATION (2026.x):")
    print("─" * 55)

    issues = []

    # ── Check 1: Base projection layers use Unsloth forward ──
    target_layers = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                     'gate_proj', 'up_proj', 'down_proj']
    unsloth_forward_count = 0
    std_forward_count = 0
    checked = set()

    for name, module in model.named_modules():
        layer_match = any(name.endswith(t) for t in target_layers)
        if not layer_match or name in checked:
            continue
        checked.add(name)

        fwd = type(module).forward
        fwd_module = getattr(fwd, '__module__', '') or ''
        fwd_file = getattr(getattr(fwd, '__code__', None), 'co_filename', '') or ''

        is_unsloth = (
            'unsloth' in fwd_module.lower() or
            'unsloth' in fwd_file.lower() or
            'compiled_cache' in fwd_file.lower()
        )

        if is_unsloth:
            unsloth_forward_count += 1
        else:
            std_forward_count += 1
            if std_forward_count <= 3:
                print(f"  ⚠️  Standard forward on: {name}")
                print(f"       → {fwd_module} | {fwd_file[-60:]}")

    if unsloth_forward_count > 0:
        print(f"  ✅ Unsloth-patched projection layers: {unsloth_forward_count}")
    else:
        print(f"  ⚠️  No Unsloth-patched projection layers found")
        issues.append("No Unsloth-patched projection layers")

    if std_forward_count > 0:
        print(f"  ⚠️  Standard PyTorch projection layers: {std_forward_count}")
        issues.append(f"{std_forward_count} standard projection layers")

    # ── Check 2: LoRA adapters exist (torch.nn.Linear is CORRECT here) ──
    lora_a_count = sum(
        1 for name, _ in model.named_modules() if 'lora_A' in name
    )
    lora_b_count = sum(
        1 for name, _ in model.named_modules() if 'lora_B' in name
    )
    if lora_a_count > 0:
        print(f"  ✅ LoRA adapters: {lora_a_count} lora_A + {lora_b_count} lora_B")
        print(f"     (torch.nn.Linear for adapters is correct — tiny matrices, fused at forward)")
    else:
        print(f"  ❌ No LoRA adapters found")
        issues.append("No LoRA adapters found")

    # ── Check 3: Triton ──
    try:
        import triton
        print(f"  ✅ Triton v{triton.__version__} (required for Unsloth kernels)")
    except ImportError:
        print(f"  ❌ Triton NOT available")
        issues.append("Triton not available")

    # ── Check 4: xformers (your output showed 0.0.34) ──
    try:
        import xformers
        print(f"  ✅ xformers v{xformers.__version__} (memory-efficient attention)")
    except ImportError:
        print(f"  ℹ️  xformers not available (Flash Attention fallback)")

    # ── Check 5: lora_dropout=0 ──
    dropout_violations = [
        f"{name}={d.p}"
        for name, module in model.named_modules()
        if hasattr(module, "lora_dropout")
        for key, d in module.lora_dropout.items()
        if hasattr(d, "p") and d.p != 0
    ]
    if dropout_violations:
        print(f"  ❌ lora_dropout violations: {dropout_violations[:3]}")
        issues.append(f"lora_dropout != 0: {dropout_violations[:3]}")
    else:
        print(f"  ✅ lora_dropout=0 confirmed")

    # ── Check 6: Compiled cache ──
    cache_path = "/home/ec2-user/rl/unsloth_compiled_cache"
    if os.path.exists(cache_path):
        n = len(os.listdir(cache_path))
        print(f"  ✅ Compiled cache: {n} files at {cache_path}")
    else:
        print(f"  ℹ️  Compiled cache not yet built (created on first training step)")

    # ── Summary ──
    print("─" * 55)
    if not issues:
        print("  ✅ Unsloth 2026.x fully active — fused kernels confirmed")
    else:
        print(f"  ⚠️  Issues found:")
        for issue in issues:
            print(f"     • {issue}")
    print("─" * 55 + "\n")


# ============== Model Loading ==============

def load_model_and_tokenizer(config: Dict[str, Any]):
    """Load model with Unsloth optimizations and LoRA."""

    print("\n" + "="*60)
    print("🚀 LOADING MODEL")
    print("="*60)
    print(f"  Base model:     {config['base_model_path']}")
    print(f"  Load in 4bit:   {config['load_in_4bit']}")
    print(f"  Max seq length: {config['max_seq_length']}")
    print(f"  Offload embed:  {config.get('offload_embedding', True)}")

    if torch.cuda.is_available():
        print(f"\n  📱 GPU Info:")
        print(f"    Device:       {torch.cuda.get_device_name(0)}")
        print(f"    Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"    Free Memory:  {torch.cuda.memory_reserved(0) / 1e9:.1f} GB reserved")

    print(f"\n  ⏳ Loading model (this may take a few minutes)...")
    load_start = datetime.now()

    # Determine dtype — BF16 for A100 80GB, 4-bit only if explicitly requested
    load_in_4bit = config.get("load_in_4bit", False)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model_path"],
        max_seq_length=config["max_seq_length"],
        load_in_4bit=load_in_4bit,
        dtype=torch.bfloat16 if not load_in_4bit else None,  # ← Explicit BF16
        offload_embedding=config.get("offload_embedding", False),
    )

    load_time = (datetime.now() - load_start).total_seconds()
    print(f"  ✅ Model loaded in {load_time:.1f}s")
    print(f"  📐 Precision: {'4-bit quantized' if load_in_4bit else 'BF16 (full precision)'}")

    if torch.cuda.is_available():
        print(f"\n  📊 Memory after model load:")
        print(f"    Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"    Reserved:  {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    print(f"\n  Adding LoRA adapters...")
    print(f"    Rank:  {config['lora_rank']}")
    print(f"    Alpha: {config['lora_alpha']}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_rank"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=config["lora_alpha"],
        lora_dropout=0,           # ← Required for Unsloth optimized kernels
        bias="none",              # ← Required for Unsloth
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
    )

    # ── Verify Unsloth kernels are actually active ──────────────────
    _verify_unsloth_kernels(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n  📈 Parameter Summary:")
    print(f"    Total params:     {total_params:,}")
    print(f"    Trainable params: {trainable_params:,}")
    print(f"    Trainable %:      {100 * trainable_params / total_params:.4f}%")

    if torch.cuda.is_available():
        print(f"\n  📊 Memory after LoRA:")
        print(f"    Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"    Reserved:  {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

    has_thinking = 'thinking' in str(tokenizer.chat_template) if tokenizer.chat_template else False
    print(f"\n  💬 Chat template supports 'thinking': {has_thinking}")

    print("="*60 + "\n")

    return model, tokenizer


# ============== Callbacks ==============

class TensorBoardRewardCallback(TrainerCallback):
    """
    Log GRPOTrainer's built-in metrics to TensorBoard.

    GRPOTrainer logs: loss, reward, reward_std, kl, lr, etc.
    This callback ensures they all go to TensorBoard.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        writer = _get_tb_writer()
        if writer is None:
            return

        step = state.global_step

        # Log all numeric metrics from GRPOTrainer
        for key, value in logs.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                # Clean up key name for TensorBoard
                tb_key = key.replace("/", "_")
                writer.add_scalar(f"trainer/{tb_key}", value, step)

        # Specifically log key GRPO metrics under their own group
        grpo_keys = {
            "loss": "grpo/loss",
            "reward": "grpo/reward_mean",
            "reward_std": "grpo/reward_std",
            "kl": "grpo/kl_divergence",
            "learning_rate": "grpo/learning_rate",
            "completion_length": "grpo/completion_length",
            "clip_ratio": "grpo/clip_ratio",
        }

        for src_key, dst_key in grpo_keys.items():
            # Try exact key and train/ prefixed key
            value = logs.get(src_key, logs.get(f"train/{src_key}"))
            if value is not None and isinstance(value, (int, float)):
                writer.add_scalar(dst_key, value, step)

        writer.flush()


class VerboseTrainingCallback(TrainerCallback):
    """Callback for verbose training progress logging."""

    def __init__(self):
        self.train_start_time = None
        self.step_times = []
        self.last_step_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = datetime.now()
        print("\n" + "🎯"*30)
        print("🚀 TRAINING STARTED")
        print("🎯"*30)
        print(f"  Start time: {self.train_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Max steps:  {args.max_steps}")
        print(f"  Batch size: {args.per_device_train_batch_size}")
        print(f"  Grad accum: {args.gradient_accumulation_steps}")
        print(f"  Effective batch: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print(f"  Num generations: {INLINE_CONFIG['num_generations']}")
        print("🎯"*30 + "\n")

    def on_step_begin(self, args, state, control, **kwargs):
        self.last_step_time = datetime.now()

    def on_step_end(self, args, state, control, **kwargs):
        if self.last_step_time:
            step_duration = (datetime.now() - self.last_step_time).total_seconds()
            self.step_times.append(step_duration)
            # Keep only last 100 steps to avoid memory growth
            if len(self.step_times) > 100:
                self.step_times = self.step_times[-100:]

        if state.global_step % max(args.logging_steps, 1) == 0 and state.global_step > 0:
            elapsed = (datetime.now() - self.train_start_time).total_seconds()
            avg_step_time = np.mean(self.step_times[-10:]) if self.step_times else 0
            remaining_steps = args.max_steps - state.global_step
            eta_seconds = avg_step_time * remaining_steps

            progress_pct = 100 * state.global_step / max(args.max_steps, 1)
            bar_width = 30
            filled = int(bar_width * state.global_step / max(args.max_steps, 1))
            bar = "█" * filled + "░" * (bar_width - filled)

            print(f"\n📊 [{bar}] {progress_pct:.1f}% | "
                  f"Step {state.global_step}/{args.max_steps} | "
                  f"Elapsed: {elapsed/60:.1f}m | "
                  f"Step: {avg_step_time:.1f}s | "
                  f"ETA: {eta_seconds/60:.1f}m")

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                peak = torch.cuda.max_memory_allocated(0) / 1e9
                print(f"   💾 GPU: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved / {peak:.2f}GB peak")

                # Log to TensorBoard
                _log_tb_scalar("system/gpu_allocated_gb", allocated, state.global_step)
                _log_tb_scalar("system/gpu_reserved_gb", reserved, state.global_step)
                _log_tb_scalar("system/gpu_peak_gb", peak, state.global_step)
                _log_tb_scalar("system/step_time_seconds", avg_step_time, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        total_time = (datetime.now() - self.train_start_time).total_seconds()
        print("\n" + "✅"*30)
        print("🏁 TRAINING COMPLETED")
        print("✅"*30)
        print(f"  Total time:    {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"  Total steps:   {state.global_step}")
        if self.step_times:
            print(f"  Avg step time: {np.mean(self.step_times):.2f}s")
            print(f"  Min step time: {np.min(self.step_times):.2f}s")
            print(f"  Max step time: {np.max(self.step_times):.2f}s")
        print("✅"*30 + "\n")


class GRPOMetricsCallback(TrainerCallback):
    """Track and report key metrics during GRPO training."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.reward_history = []
        self.best_avg_reward = float("-inf")
        self.steps_since_improvement = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step = state.global_step
        reward = logs.get("reward", logs.get("train/reward", None))
        if reward is not None:
            self.reward_history.append(reward)

        if step > 0 and step % 25 == 0 and self.reward_history:
            window = self.reward_history[-self.window_size:]
            avg_reward = sum(window) / len(window)

            print(f"\n{'─'*60}")
            print(f"📈 REWARD TRACKING — Step {step}")
            print(f"{'─'*60}")
            print(f"  Rolling avg reward (last {len(window)}): {avg_reward:.4f}")

            if len(self.reward_history) > self.window_size:
                prev_window = self.reward_history[
                    -2*self.window_size:-self.window_size
                ]
                if prev_window:
                    prev_avg = sum(prev_window) / len(prev_window)
                    delta = avg_reward - prev_avg
                    trend = (
                        "📈 improving" if delta > 0.01 else
                        "📉 degrading" if delta < -0.01 else
                        "➡️  flat"
                    )
                    print(f"  Trend: {trend} (Δ={delta:+.4f})")

                    _log_tb_scalar("tracking/reward_delta", delta, step)

            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.steps_since_improvement = 0
                print(f"  🎉 New best avg reward!")
            else:
                self.steps_since_improvement += 25

            print(f"  Best avg: {self.best_avg_reward:.4f}")
            print(f"  Steps since improvement: {self.steps_since_improvement}")

            _log_tb_scalar("tracking/best_avg_reward", self.best_avg_reward, step)
            _log_tb_scalar("tracking/steps_since_improvement", float(self.steps_since_improvement), step)

            # Early stopping warning
            if self.steps_since_improvement >= 200:
                print(f"  ⚠️  WARNING: No improvement for {self.steps_since_improvement} steps!")
                print(f"  Consider stopping training or adjusting hyperparameters.")

            print(f"{'─'*60}\n")


class SaveLoRAOnlyCallback(TrainerCallback):
    """
    Ensure intermediate checkpoints only save LoRA adapters (not full model shards).

    On save, this callback:
      1. Checks what the trainer wrote to the checkpoint directory.
      2. If adapter_model.safetensors already exists → deletes any stray full-model
         shards (model-*.safetensors, pytorch_model*.bin) and returns.
      3. If NO adapter file was produced → explicitly calls model.save_pretrained()
         to write the LoRA adapter, THEN deletes full-model shards.
      4. If that explicit save also fails → logs an error loudly so the user knows.
    """

    def on_save(self, args, state, control, model=None, **kwargs):
        checkpoint_dir = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}"
        )

        if not os.path.exists(checkpoint_dir):
            print(f"  ⚠️ SaveLoRAOnlyCallback: checkpoint dir does not exist: {checkpoint_dir}")
            return

        all_files = os.listdir(checkpoint_dir)

        # Full-model shard files written by a non-PEFT save path
        full_model_files = [
            f for f in all_files
            if (f.startswith("model-") and f.endswith(".safetensors"))
            or f.startswith("pytorch_model")
        ]

        adapter_file = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        adapter_config = os.path.join(checkpoint_dir, "adapter_config.json")

        total_size_mb = sum(
            os.path.getsize(os.path.join(checkpoint_dir, f))
            for f in all_files
            if os.path.isfile(os.path.join(checkpoint_dir, f))
        ) / (1024 * 1024)

        print(f"💾 Checkpoint step {state.global_step}: {total_size_mb:.1f} MB")
        print(f"  Files: {sorted(all_files)}")

        adapter_present = os.path.exists(adapter_file)
        if adapter_present or os.path.exists(adapter_config):
            print("  ✓ LoRA adapter files found in checkpoint")

        # ── Case 1: adapter already present → just clean up any stray shards ──
        if adapter_present:
            if full_model_files:
                print(
                    f"  ⚠️ Removing {len(full_model_files)} stray full-model files "
                    "(LoRA adapter already saved)"
                )
                for f in full_model_files:
                    fpath = os.path.join(checkpoint_dir, f)
                    size_mb = os.path.getsize(fpath) / (1024 * 1024)
                    os.remove(fpath)
                    print(f"    Deleted {f} ({size_mb:.1f} MB)")
            return

        # ── Case 2: no adapter file — explicitly write LoRA adapter now ──
        print(
            f"  ⚠️ adapter_model.safetensors missing from checkpoint-{state.global_step}!"
            " Attempting explicit model.save_pretrained()..."
        )
        if model is None:
            print("  ❌ 'model' not available in callback kwargs — cannot save LoRA adapter!")
            if full_model_files:
                print("  Keeping full-model shards as the only fallback.")
            return

        try:
            model.save_pretrained(checkpoint_dir)
            print(f"  ✅ Explicit save_pretrained() succeeded for checkpoint-{state.global_step}")
        except Exception as exc:
            print(f"  ❌ model.save_pretrained() failed: {exc}")
            if full_model_files:
                print("  Keeping full-model shards as fallback (adapter save failed).")
            return

        # Verify the adapter was actually written
        if os.path.exists(adapter_file):
            print(f"  ✓ adapter_model.safetensors confirmed in checkpoint-{state.global_step}")
            for f in full_model_files:
                fpath = os.path.join(checkpoint_dir, f)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                os.remove(fpath)
                print(f"    Deleted full-model shard {f} ({size_mb:.1f} MB)")
        else:
            print(
                f"  ❌ save_pretrained() ran but adapter_model.safetensors still missing!"
                " Keeping full-model shards as fallback."
            )


# ============== Enhanced Trainer ==============

class GRPOTrainerWithDebug(GRPOTrainer):
    """Extended GRPO Trainer that logs sample completions for debugging."""

    def __init__(self, *args, debug_every_n_steps: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_every_n_steps = debug_every_n_steps
        self._step_count = 0
        self._raw_completion_log_path = os.path.join(
            self.args.output_dir, "raw_completions.txt"
        )
        # Required by Unsloth's compiled GRPOTrainer for evaluation loop
        if not hasattr(self, 'current_gradient_accumulation_steps'):
            self.current_gradient_accumulation_steps = self.args.gradient_accumulation_steps

    def _generate_and_score_completions(self, inputs):
        """Override to capture completions for debugging and handle OOM."""
        gc.collect()
        torch.cuda.empty_cache()

        try:
            result = super()._generate_and_score_completions(inputs)
        except torch.cuda.OutOfMemoryError:
            print("⚠️ OOM during generation! Clearing cache and retrying...")
            gc.collect()
            torch.cuda.empty_cache()
            result = super()._generate_and_score_completions(inputs)
        except RuntimeError as e:
            if "invalid for input of size" in str(e):
                print(f"⚠️ Reshape error during generation: {e}")
                raise
            raise

        self._step_count += 1

        # Always save raw completions for the FIRST step, then every N steps
        if INLINE_CONFIG.get("save_raw_completions", True):
            if self._step_count == 1 or self._step_count % self.debug_every_n_steps == 0:
                try:
                    self._save_raw_completions(result, inputs)
                except Exception as e:
                    print(f"[DEBUG Step {self._step_count}] Raw save failed: {e}")

        return result

    def _save_raw_completions(self, result, inputs):
        """
        Save raw completions exactly as returned by the model — before any
        reward processing, special-token stripping, or parsing.

        Writes to: <output_dir>/raw_completions.txt
        Appends each call so you can track evolution across steps.
        """
        os.makedirs(os.path.dirname(self._raw_completion_log_path), exist_ok=True)

        # Extract completions from result (GRPOTrainer returns a dict or dataclass)
        completions = None
        prompts = None

        if isinstance(result, dict):
            completions = result.get("completions", result.get("completion_ids"))
            prompts = result.get("prompts", result.get("prompt_ids"))
        elif hasattr(result, "completions"):
            completions = result.completions
            prompts = getattr(result, "prompts", None)
        elif hasattr(result, "completion_ids"):
            completions = result.completion_ids
            prompts = getattr(result, "prompt_ids", None)

        # Also try inputs dict for prompts
        if prompts is None and isinstance(inputs, dict):
            prompts = inputs.get("prompts", inputs.get("input_ids"))

        with open(self._raw_completion_log_path, "a", encoding="utf-8") as f:
            separator = "=" * 80
            f.write(f"\n{separator}\n")
            f.write(f"STEP {self._step_count}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{separator}\n")

            if completions is None:
                f.write(f"[WARNING] Could not extract completions from result.\n")
                f.write(f"result type: {type(result)}\n")
                if isinstance(result, dict):
                    f.write(f"result keys: {list(result.keys())}\n")
                elif hasattr(result, '__dict__'):
                    f.write(f"result attrs: {list(result.__dict__.keys())}\n")
                return

            # Determine if we have token IDs or already-decoded text
            num_to_save = min(4, len(completions))

            for i in range(num_to_save):
                comp = completions[i]
                f.write(f"\n{'─'*60}\n")
                f.write(f"COMPLETION {i+1}/{len(completions)}\n")
                f.write(f"{'─'*60}\n")

                # --- Decode token IDs if needed ---
                if isinstance(comp, torch.Tensor):
                    # Raw token IDs — decode with ALL special tokens preserved
                    raw_text = self.processing_class.decode(
                        comp,
                        skip_special_tokens=False,   # ← Keep ALL special tokens
                        clean_up_tokenization_spaces=False,
                    )
                    f.write(f"[SOURCE: token IDs tensor, shape={comp.shape}]\n\n")

                    # Also decode WITH skip_special_tokens for comparison
                    clean_text = self.processing_class.decode(
                        comp,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    f.write(f"--- RAW (skip_special_tokens=False) ---\n")
                    f.write(raw_text)
                    f.write(f"\n\n--- CLEAN (skip_special_tokens=True) ---\n")
                    f.write(clean_text)

                elif isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], int):
                    # List of token IDs
                    raw_text = self.processing_class.decode(
                        comp,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    f.write(f"[SOURCE: token ID list, len={len(comp)}]\n\n")
                    f.write(f"--- RAW (skip_special_tokens=False) ---\n")
                    f.write(raw_text)
                    clean_text = self.processing_class.decode(
                        comp,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    f.write(f"\n\n--- CLEAN (skip_special_tokens=True) ---\n")
                    f.write(clean_text)

                elif isinstance(comp, list) and len(comp) > 0 and isinstance(comp[0], dict):
                    # Already decoded list-of-dicts (TRL format: [{"role": ..., "content": ...}])
                    f.write(f"[SOURCE: list-of-dicts (already decoded by TRL)]\n")
                    f.write(f"[⚠️  Special tokens may have been stripped by TRL before this point!]\n\n")
                    for msg in comp:
                        f.write(f"  role={msg.get('role', '?')}\n")
                        f.write(f"  content:\n")
                        f.write(msg.get("content", ""))
                        f.write("\n")

                elif isinstance(comp, str):
                    f.write(f"[SOURCE: plain string (already decoded)]\n")
                    f.write(f"[⚠️  Special tokens may have been stripped!]\n\n")
                    f.write(comp)

                else:
                    f.write(f"[SOURCE: unknown type {type(comp)}]\n")
                    f.write(repr(comp)[:500])

                f.write(f"\n")

                # Token ID dump for the first completion only (useful for debugging special tokens)
                if i == 0 and isinstance(comp, (torch.Tensor, list)):
                    token_ids = comp.tolist() if isinstance(comp, torch.Tensor) else comp
                    if isinstance(token_ids[0], int):
                        f.write(f"\n--- TOKEN ID DUMP (first 50 tokens) ---\n")
                        for j, tid in enumerate(token_ids[:50]):
                            tok_str = self.processing_class.convert_ids_to_tokens(tid)
                            f.write(f"  [{j:3d}] id={tid:6d}  tok={repr(tok_str)}\n")
                        if len(token_ids) > 50:
                            f.write(f"  ... ({len(token_ids) - 50} more tokens)\n")

            f.write(f"\n{separator}\n")

        print(f"[DEBUG Step {self._step_count}] Raw completions saved → {self._raw_completion_log_path}")

    def _print_debug_completions(self, completions, prompts):
        """Print sample completions for debugging gpt-oss format."""
        print(f"\n{'*'*70}")
        print(f"🔍 DEBUG COMPLETIONS at step {self._step_count}")
        print(f"{'*'*70}")

        num_to_print = min(2, len(completions))
        for i in range(num_to_print):
            completion = completions[i]
            response = _get_completion_text(completion)

            print(f"\n--- Completion {i+1}/{len(completions)} ---")
            print("Raw response (first 500 chars):")
            print(response[:500])
            print(f"\n(total length: {len(response)} chars)")

            # Format markers
            markers = {
                "<|start|>assistant": "<|start|>assistant" in response,
                "<|channel|>analysis": "<|channel|>analysis" in response,
                "<|channel|>final": "<|channel|>final" in response,
                "<|end|>": "<|end|>" in response,
                "<|return|>": "<|return|>" in response,
            }

            print("\ngpt-oss format markers:")
            for marker, present in markers.items():
                print(f"  {'✓' if present else '✗'} {marker}")

            thinking = extract_thinking(response)
            sql = extract_sql(response)

            print(f"\n  Thinking ({len(thinking)} chars): {thinking[:150] if thinking else 'NONE'}...")
            print(f"  SQL: {sql[:200] if sql else 'NONE'}")

        print(f"{'*'*70}\n")


# ============== Main Training Function ==============

def main(config: Dict[str, Any]):
    """Main training function for gpt-oss reasoning model."""
    global _TB_WRITER

    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  GRPO TRAINING FOR NL2SQL REASONING MODEL".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60 + "\n")

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"grpo_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize TensorBoard writer
    tb_log_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    _TB_WRITER = SummaryWriter(log_dir=tb_log_dir)
    print(f"📊 TensorBoard logs: {tb_log_dir}")
    print(f"   To monitor: tensorboard --logdir {tb_log_dir} --bind_all --port 6006")
    print(f"   Then SSH:   ssh -L 6006:localhost:6006 <ec2-host>\n")

    # Log config to TensorBoard
    config_text = json.dumps(config, indent=2, default=str)
    _TB_WRITER.add_text("config", f"```json\n{config_text}\n```", 0)

    # Print full configuration
    print("📋 CONFIGURATION:")
    print("─"*40)
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    print("─"*40)
    print(f"  Output dir: {output_dir}")
    print()

    # System info
    print("💻 SYSTEM INFO:")
    print("─"*40)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {gpu_mem_gb:.1f} GB")
        print(f"  GPU count: {torch.cuda.device_count()}")

        _TB_WRITER.add_text("system/gpu", torch.cuda.get_device_name(0), 0)
        _TB_WRITER.add_text("system/gpu_memory", f"{gpu_mem_gb:.1f} GB", 0)
    print("─"*40 + "\n")

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)

    # LLM judge is disabled — no self-judging
    print("ℹ️  LLM Judge disabled (self-judging with training model is unreliable)")

    # Load datasets
    print("📂 LOADING DATASETS...")
    print("─"*40)

    train_limit = config.get("train_limit", None)
    val_limit = config.get("val_limit", None)

    train_data = load_jsonl(config["train_dataset_path"], limit=train_limit)
    val_data = load_jsonl(config["val_dataset_path"], limit=val_limit)
    print(f"  Train samples loaded: {len(train_data)} (limit: {train_limit})")
    print(f"  Val samples loaded:   {len(val_data)} (limit: {val_limit})")
    print("─"*40 + "\n")

    # Log dataset sizes to TensorBoard
    _TB_WRITER.add_scalar("dataset/train_raw_size", len(train_data), 0)
    _TB_WRITER.add_scalar("dataset/val_raw_size", len(val_data), 0)

    # Analyze token distribution
    analyze_dataset_tokens(train_data, tokenizer, "Training Data")

    # Show example
    if train_data:
        print("📝 EXAMPLE DATA SAMPLE:")
        print("─"*40)
        example = train_data[0]
        example_messages = example["messages"]
        prompt_msgs = [m for m in example_messages if m["role"] != "assistant"]
        assistant_msg = next((m for m in example_messages if m["role"] == "assistant"), None)

        print(f"  Prompt messages: {len(prompt_msgs)}")
        for m in prompt_msgs:
            preview = m["content"][:100].replace('\n', ' ')
            print(f"    [{m['role']}]: {preview}...")

        if assistant_msg:
            print(f"\n  Assistant response:")
            print(f"    Has thinking: {'thinking' in assistant_msg}")
            if 'thinking' in assistant_msg:
                print(f"    Thinking preview: {assistant_msg['thinking'][:100]}...")
            print(f"    Content preview: {assistant_msg.get('content', '')[:100]}...")
        print("─"*40 + "\n")

    # Prepare datasets with filtering
    max_prompt_tokens = config.get("max_prompt_length", 2048)
    max_completion_tokens = config.get("max_completion_length", 1536)

    train_dataset, train_filter_stats = prepare_dataset_for_grpo(
        train_data, tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_completion_tokens=max_completion_tokens,
    )
    val_dataset, val_filter_stats = prepare_dataset_for_grpo(
        val_data, tokenizer,
        max_prompt_tokens=max_prompt_tokens,
        max_completion_tokens=max_completion_tokens,
    )

    print(f"\n✅ Train samples after filtering: {len(train_dataset)}")
    print(f"✅ Val samples after filtering: {len(val_dataset)}")

    _TB_WRITER.add_scalar("dataset/train_filtered_size", len(train_dataset), 0)
    _TB_WRITER.add_scalar("dataset/val_filtered_size", len(val_dataset), 0)

    # Log filter stats to TensorBoard
    for key in ["filtered_prompt_too_long", "filtered_total_too_long", "filtered_no_user_msg"]:
        _TB_WRITER.add_scalar(f"dataset/train_{key}", train_filter_stats[key], 0)
        _TB_WRITER.add_scalar(f"dataset/val_{key}", val_filter_stats[key], 0)

    # ========== PROMINENT TRAINING CONFIGURATION SUMMARY ==========
    print("\n" + "🔷" * 35)
    print("🔷  TRAINING CONFIGURATION SUMMARY")
    print("🔷" * 35)

    is_4bit = config.get("load_in_4bit", False)
    model_vram_est = "~13 GB" if is_4bit else "~40 GB"
    kv_cache_est = "~6-10 GB" if config["num_generations"] <= 4 else "~8-14 GB"
    activation_est = "~8-12 GB"
    optimizer_est = "~1 GB"
    total_est = "~24-32 GB" if is_4bit else "~57-77 GB"
    total_available = "80 GB"

    print(f"""
  📦 DATA:
    Train: {len(train_dataset)} samples (filtered from {train_filter_stats['total_input']}, dropped {train_filter_stats['total_input'] - len(train_dataset)})
    Val:   {len(val_dataset)} samples (filtered from {val_filter_stats['total_input']}, dropped {val_filter_stats['total_input'] - len(val_dataset)})

  📏 SEQUENCE LENGTHS (controls what gets FILTERED vs what gets GENERATED):
    max_prompt_length:     {config['max_prompt_length']} tokens — prompts longer than this are FILTERED OUT (not truncated)
    max_completion_length: {config['max_completion_length']} tokens — model generation HARD STOPS at this length
    max_seq_length:        {config['max_seq_length']} tokens — model positional encoding / KV-cache limit
    Effective max total:   {config['max_prompt_length']} + {config['max_completion_length']} = {config['max_prompt_length'] + config['max_completion_length']} tokens

  🎲 GENERATION:
    num_generations:       {config['num_generations']} completions per prompt (GRPO needs multiple for advantage estimation)
    temperature:           {config['temperature']}
    Generation cannot exceed {config['max_completion_length']} tokens (enforced by GRPOConfig.max_completion_length)

  📊 BATCH:
    per_device_train_batch_size: {config['per_device_train_batch_size']}
    gradient_accumulation_steps: {config['gradient_accumulation_steps']}
    effective_batch_size:        {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']}
    
  📐 PRECISION: {'4-bit quantized (MXFP4)' if is_4bit else 'BF16 (full precision)'}
    
  🧠 VRAM ESTIMATE (approximate):
    Model ({'4-bit' if is_4bit else 'BF16'}):    {model_vram_est}
    KV-cache ({config['num_generations']} gen × {config['max_seq_length']} tokens): {kv_cache_est}
    Activations:      {activation_est} (gradient checkpointing ON)
    Optimizer (8-bit): {optimizer_est}
    Estimated total:  {total_est} / {total_available} available
""")
    print("🔷" * 35 + "\n")

    # Training config
    training_args = GRPOConfig(
        temperature=config["temperature"],
        num_generations=config["num_generations"],
        max_prompt_length=config["max_prompt_length"],
        max_completion_length=config["max_completion_length"],

        learning_rate=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.001),
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="linear",
        optim=config.get("optim", "adamw_torch"),

        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["num_generations"],  # Must be divisible by num_generations
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"],
        bf16=True,

        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=5,
        output_dir=output_dir,
        report_to="tensorboard",        # ← Enable TensorBoard!
        logging_dir=tb_log_dir,          # ← Point to our TB dir
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
        seed=seed,

        beta=config.get("beta", 0.04),
        max_grad_norm=config.get("max_grad_norm", 0.5),

        # Evaluation
        eval_strategy="steps",
        eval_steps=config.get("eval_steps", 50),
        eval_on_start=True,  # Get baseline metrics before any training        

        # Save only LoRA adapters
        save_only_model=True,
    )

    # Debug: verify gradient clipping is enabled
    print(f"  max_grad_norm: {training_args.max_grad_norm}")    

    # Build reward functions
    reward_funcs, reward_names = build_reward_functions(config)

    # Print reward structure
    print("\n🏆 REWARD STRUCTURE:")
    print("─"*60)
    print(f"  Mode: {'COMPOSITE (weighted)' if config.get('use_composite_reward', True) else 'UNWEIGHTED (list)'}")
    print(f"  Active reward functions ({len(reward_names)}):")
    for name in reward_names:
        print(f"    ✓ {name}")

    reward_config = config.get("reward_config", {})
    all_rewards = [
        "format_reward", "syntax_execution_reward", "result_reward",
        "schema_linking_reward", "ngram_similarity_reward",
        "thinking_quality_reward", "llm_judge_reward",
    ]
    disabled = [r for r in all_rewards if not reward_config.get(r, False)]
    if disabled:
        print(f"  Disabled: {disabled}")
    print("─"*60 + "\n")

    # Initialize trainer
    trainer = GRPOTrainerWithDebug(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        debug_every_n_steps=config.get("debug_every_n_steps", 10),
        callbacks=[
            TensorBoardRewardCallback(),
            VerboseTrainingCallback(),
            GRPOMetricsCallback(window_size=50),
            SaveLoRAOnlyCallback(),
        ],
    )

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"💾 Saved config to: {config_path}\n")

    # Start training
    resume_checkpoint = config.get("resume_from_checkpoint", None)
    if resume_checkpoint:
        print(f"▶️  Resuming from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    print("\n" + "*" * 80)
    print("💾 SAVING FINAL MODEL")
    print("*" * 80)

    final_model_path = os.path.join(output_dir, "final_model")

    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    saved_files = os.listdir(final_model_path)
    total_size_mb = sum(
        os.path.getsize(os.path.join(final_model_path, f))
        for f in saved_files
        if os.path.isfile(os.path.join(final_model_path, f))
    ) / (1024 * 1024)

    print(f"  Saved LoRA adapters to: {final_model_path}")
    print(f"  Files: {saved_files}")
    print(f"  Total size: {total_size_mb:.1f} MB")

    if config.get("save_merged_model", False):
        merged_path = os.path.join(output_dir, "merged_model")
        model.save_pretrained_merged(
            merged_path,
            tokenizer,
            save_method="merged_16bit"
        )
        print(f"  Saved merged model to: {merged_path}")

    # Close TensorBoard writer
    if _TB_WRITER is not None:
        _TB_WRITER.close()

    # Close pooled DB connections
    _close_all_db_connections()

    print("\n" + "🎉"*20)
    print("  TRAINING COMPLETE!")
    print(f"  Output:      {output_dir}")
    print(f"  TensorBoard: tensorboard --logdir {tb_log_dir} --bind_all --port 6006")
    print("🎉"*20 + "\n")


if __name__ == "__main__":
    main(INLINE_CONFIG)