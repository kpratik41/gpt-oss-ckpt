# recipe/retool/generated_tools.py
# Async tool functions for VERL router.
# Exposed via your GeneratedToolsRouter (e.g., tool name "finance_data" or "router_tools")
# so the LLM can call (example):
# {"name":"finance_data","arguments":{"name":"sqlite_peek","args":{"db_id":"...", "sql":"SELECT ..."}}}

import os
import re
import sqlite3
import time
from typing import Dict, List, Tuple, Any, Optional
import os
import sqlite3
from typing import List, Dict
# add to the top-level __all__

# keep existing imports (os, re, sqlite3 already present)
from typing import Any  # if not already imported
import re

__all__ = ["sqlite_peek", "sqlite_query", "bm25_search_sqlite", "consensus_at_1"]
# __all__ = ["sqlite_peek", "sqlite_query", "bm25_search_sqlite"]


# -------------------------
# Internals / guardrails
# -------------------------

_READONLY_SELECT_RX = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)

def _roots_from_env() -> List[str]:
    """Read colon/OS-pathsep separated roots from BIRD_DB_ROOTS."""
    roots = os.environ.get("BIRD_DB_ROOTS", "")
    parts = [p for p in roots.split(os.pathsep) if p]
    return parts

def _candidate_names(db_id: str) -> List[str]:
    """Candidate filenames/paths tried under each root."""
    return [
        f"{db_id}.sqlite", f"{db_id}.sqlite3", f"{db_id}.db",
        os.path.join(db_id, f"{db_id}.sqlite"),
        os.path.join(db_id, f"{db_id}.sqlite3"),
        os.path.join(db_id, f"{db_id}.db"),
    ]

def _resolve_db_path(db_id: str) -> Optional[str]:
    """Resolve a DB path from env roots + db_id."""
    for r in _roots_from_env():
        for c in _candidate_names(db_id):
            p = os.path.join(r, c)
            if os.path.isfile(p):
                return p
    return None

def _ensure_select_only(sql: str) -> None:
    """Ensure SQL starts with SELECT and does not contain obvious mutating patterns."""
    if not isinstance(sql, str) or not _READONLY_SELECT_RX.match(sql):
        raise ValueError("Only read-only SELECT statements are allowed.")
    # cheap defense-in-depth against multiple statements or PRAGMA etc.
    lower = sql.lower()
    forbidden = [" pragma ", " attach ", " detach ", " insert ", " update ", " delete ", " drop ", " create ", " alter "]
    if any(tok in lower for tok in forbidden):
        raise ValueError("Only read-only SELECT statements are allowed (mutating/DDL tokens detected).")

def _sanitize_sql(sql: str) -> str:
    """Fix common model-generated SQL issues before execution."""
    import re as _re
    # Replace \" with ' (backslash-escaped double quotes to single quotes)
    sql = sql.replace('\\"', "'")
    # Replace standalone backslashes that aren't part of valid SQL
    sql = _re.sub(r"\\(?![%_'])", "", sql)
    # Replace double-quoted identifiers with backtick-quoted (SQLite compat)
    # e.g. "T-CHO" -> `T-CHO`, but skip string literals inside single quotes
    sql = _re.sub(r'"([A-Za-z][A-Za-z0-9_ \-]*)"', r'`\1`', sql)
    # Replace square-bracket identifiers with backticks: [U-PRO] -> `U-PRO`
    sql = _re.sub(r'\[([A-Za-z][A-Za-z0-9_ \-]*)\]', r'`\1`', sql)
    # Strip trailing non-SQL characters (model sometimes appends ? or other junk)
    sql = _re.sub(r'[?!]+\s*$', '', sql.rstrip())
    # Fix double-single-quotes around values: ''#'' -> '#', ''value'' -> 'value'
    sql = _re.sub(r"''([^']+)''", r"'\1'", sql)
    return sql


def _get_column_names(db_path: str) -> set:
    """Get all column names from all tables in the database (cached)."""
    if not hasattr(_get_column_names, "_cache"):
        _get_column_names._cache = {}
    if db_path in _get_column_names._cache:
        return _get_column_names._cache[db_path]
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        cols = set()
        for t in tables:
            cur.execute(f"PRAGMA table_info(`{t}`);")
            for row in cur.fetchall():
                cols.add(row[1])  # column name
        conn.close()
        _get_column_names._cache[db_path] = cols
        return cols
    except Exception:
        return set()


def _fix_column_names(sql: str, db_path: str) -> str:
    """Fix underscore-for-hyphen column name substitutions using actual schema.
    
    Models often generate T_BIL instead of T-BIL, U_PRO instead of U-PRO, etc.
    This checks the actual database columns and corrects mismatches.
    """
    import re as _re
    real_cols = _get_column_names(db_path)
    if not real_cols:
        return sql
    
    # Build a lookup: lowercase normalized name -> real name
    # For columns with hyphens: T-BIL -> t-bil, T-CHO -> t-cho
    hyphen_cols = {}
    for col in real_cols:
        if '-' in col or ' ' in col:
            # Map underscore version to real name
            underscore_ver = col.replace('-', '_').replace(' ', '_').lower()
            hyphen_cols[underscore_ver] = col
    
    if not hyphen_cols:
        return sql
    
    # Find identifiers in SQL that might be underscore versions of hyphenated columns
    # Match: table.COLUMN or standalone COLUMN (not inside quotes/backticks)
    def replace_col(match):
        prefix = match.group(1) or ""  # table alias + dot, or empty
        col_name = match.group(2)
        col_lower = col_name.lower()
        if col_lower in hyphen_cols:
            real_name = hyphen_cols[col_lower]
            return f"{prefix}`{real_name}`"
        return match.group(0)
    
    # Match: optional "alias." prefix followed by an identifier containing underscore
    # but NOT already inside backticks
    fixed = _re.sub(
        r'(?<!`)(\b\w+\.)?(\b[A-Za-z]\w*_\w*\b)(?!`)',
        replace_col,
        sql
    )
    return fixed

def _fuzzy_match_column(requested: str, valid_columns: list) -> Optional[str]:
    """Find the closest matching column name using multiple strategies.

    Handles: CamelCase typos (CreationDate->CreaionDate), underscore/hyphen/space
    swaps (T_BIL->T-BIL, NSLP_Provision_Status->NSLP Provision Status),
    and transposed chars (HBG->HGB, cardKingdomFoildId->cardKingdomFoilId).
    Returns the best match or None.
    """
    if not requested or not valid_columns:
        return None
    req_clean = requested.strip().strip('`"[]')
    req_lower = req_clean.lower()

    # 1. Exact match (case-insensitive)
    for col in valid_columns:
        if col.lower() == req_lower:
            return col

    # 2. Normalize: replace _/- with space, compare
    def normalize(s):
        return s.lower().replace('_', ' ').replace('-', ' ').strip()
    req_norm = normalize(req_clean)
    for col in valid_columns:
        if normalize(col) == req_norm:
            return col

    # 3. Strip all separators and compare
    def strip_seps(s):
        return s.lower().replace('_', '').replace('-', '').replace(' ', '')
    req_stripped = strip_seps(req_clean)
    for col in valid_columns:
        if strip_seps(col) == req_stripped:
            return col

    # 4. difflib fuzzy match (catches typos like CreaionDate, HBG->HGB)
    from difflib import get_close_matches
    cutoff = 0.6 if len(req_lower) <= 4 else 0.75
    matches = get_close_matches(req_lower, [c.lower() for c in valid_columns], n=1, cutoff=cutoff)
    if matches:
        for col in valid_columns:
            if col.lower() == matches[0]:
                return col

    return None


def _connect_ro(db_path: str, busy_ms: int = 5000) -> sqlite3.Connection:
    """Open a read-only SQLite connection with safe PRAGMAs."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=busy_ms / 1000.0)
    conn.execute("PRAGMA query_only=ON;")
    conn.execute("PRAGMA foreign_keys=OFF;")
    return conn

def _set_progress_guard(conn: sqlite3.Connection, hard_timeout_s: float, max_steps: int) -> None:
    """Abort long-running queries via VM progress handler and soft deadline."""
    soft_deadline = time.perf_counter() + max(0.75 * hard_timeout_s, min(0.5, hard_timeout_s))
    steps = 0
    def _progress() -> int:
        nonlocal steps
        steps += 1
        if (max_steps and steps > max_steps) or (time.perf_counter() > soft_deadline):
            return 1
        return 0
    conn.set_progress_handler(_progress, 1000)

def _run_select(
    db_path: str,
    sql: str,
    *,
    hard_timeout_s: float = 45.0,
    busy_ms: int = 5000,
    max_steps: int = 15_000_000,
    cap_rows: Optional[int] = None,
) -> Tuple[List[List[Any]], List[str]]:
    """Execute read-only SELECT with guardrails and optional fetch cap."""
    sql = _sanitize_sql(sql)
    sql = _fix_column_names(sql, db_path)
    _ensure_select_only(sql)
    start = time.perf_counter()
    conn = _connect_ro(db_path, busy_ms=busy_ms)
    _set_progress_guard(conn, hard_timeout_s, max_steps)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        if cap_rows is not None and cap_rows >= 0:
            rows_t = cur.fetchmany(cap_rows)
        else:
            rows_t = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [list(r) for r in rows_t]
        return rows, cols
    finally:
        try:
            conn.close()
        finally:
            _ = (time.perf_counter() - start)

def _force_limit(sql: str, default_limit: int) -> str:
    """Append LIMIT if missing (naive but effective)."""
    if " limit " not in sql.lower():
        return sql.rstrip().rstrip(";") + f" LIMIT {int(default_limit)}"
    return sql

# -------------------------
# Public async API
# -------------------------

import asyncio
import math
import statistics
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

# expects your existing helpers:
# _resolve_db_path(db_id) -> str
# _force_limit(sql, limit) -> str
# _run_select(db_path, sql, hard_timeout_s, busy_ms, max_steps, cap_rows=None) -> (rows, cols)

def _is_number_like(x) -> bool:
    if x is None: 
        return False
    if isinstance(x, (int, float)): 
        return True
    try:
        float(str(x))
        return True
    except Exception:
        return False

def _maybe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _strish(x) -> str:
    if x is None:
        return ""
    s = str(x)
    # guard insanely long blobs
    return s if len(s) <= 10_000 else s[:10_000]

def _char_class_profile(s: str) -> Dict[str, Any]:
    # simple character-class summary for a string
    has_upper = any(c.isupper() for c in s)
    has_lower = any(c.islower() for c in s)
    has_digit = any(c.isdigit() for c in s)
    has_space = any(c.isspace() for c in s)
    has_punct = any(not (c.isalnum() or c.isspace()) for c in s)
    all_digits = s.isdigit() and len(s) > 0
    # crude date-ish / iso-ish detector
    looks_date = False
    if len(s) in (8, 10, 19):
        if "-" in s or "/" in s or ":" in s:
            looks_date = True
    looks_json = (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))
    return {
        "has_upper": has_upper,
        "has_lower": has_lower,
        "has_digit": has_digit,
        "has_space": has_space,
        "has_punct": has_punct,
        "all_digits": all_digits,
        "looks_date_like": looks_date,
        "looks_json_like": looks_json,
    }

def _prefixes(s: str, lens=(2, 3, 4)) -> Dict[str, str]:
    out = {}
    for L in lens:
        out[f"prefix_{L}"] = s[:L] if len(s) >= L else s
    return out



def _profile_columns_from_sample(rows: List[Tuple[Any, ...]], cols: List[str], *, topk: int = 10) -> Dict[str, Any]:
    n = len(rows)
    col_profiles: Dict[str, Any] = {}
    # transpose columns cheaply
    col_data = defaultdict(list)
    for r in rows:
        for i, v in enumerate(r):
            col_data[cols[i]].append(v)

    for name in cols:
        data = col_data[name]
        nulls = sum(1 for v in data if v is None)
        non_null = n - nulls
        distinct_vals = set(v for v in data if v is not None)
        numeric_vals = []
        str_vals = []
        for v in data:
            if v is None:
                continue
            if _is_number_like(v):
                f = _maybe_float(v)
                if f is not None and not math.isnan(f) and not math.isinf(f):
                    numeric_vals.append(f)
            s = _strish(v)
            str_vals.append(s)

        # shape for numbers
        num_profile = None
        if numeric_vals:
            try:
                num_profile = {
                    "min": min(numeric_vals),
                    "max": max(numeric_vals),
                    "mean": statistics.fmean(numeric_vals) if len(numeric_vals) > 0 else None,
                    "median": statistics.median(numeric_vals) if len(numeric_vals) > 0 else None,
                }
            except Exception:
                num_profile = {
                    "min": min(numeric_vals) if numeric_vals else None,
                    "max": max(numeric_vals) if numeric_vals else None,
                }

        # shape for strings
        len_stats = None
        cls_hints = None
        if str_vals:
            lens = [len(s) for s in str_vals]
            try:
                len_stats = {
                    "min_len": min(lens),
                    "max_len": max(lens),
                    "mean_len": statistics.fmean(lens),
                    "median_len": statistics.median(lens),
                }
            except Exception:
                len_stats = {
                    "min_len": min(lens),
                    "max_len": max(lens),
                }
            # aggregate simple char-class hints on a small subset
            sample_for_cls = str_vals[: min(256, len(str_vals))]
            agg = Counter()
            for s in sample_for_cls:
                cc = _char_class_profile(s)
                for k, v in cc.items():
                    agg[(k, bool(v))] += 1
            # convert to rates
            cls_hints = {}
            denom = max(1, len(sample_for_cls))
            for k in ("has_upper", "has_lower", "has_digit", "has_space", "has_punct", "all_digits", "looks_date_like", "looks_json_like"):
                cls_hints[k] = agg.get((k, True), 0) / denom

        # top-k values (from sample)
        # stringify to avoid Counter treating floats/ints differently from strings in later display
        value_counter = Counter([_strish(v) if v is not None else "<NULL>" for v in data])
        topk_values = value_counter.most_common(topk)

        # common prefixes (just from the top few non-null)
        prefixes = {}
        for s in [sv for sv in str_vals[: min(10, len(str_vals))]]:
            for k, v in _prefixes(s).items():
                prefixes.setdefault(k, Counter())
                prefixes[k][v] += 1
        top_prefixes = {k: cnt.most_common(5) for k, cnt in prefixes.items()} if prefixes else {}

        # simple min/max over raw (string) ordering to give a hint even for non-numerics
        raw_min = None
        raw_max = None
        try:
            non_null_raw = [v for v in data if v is not None]
            if non_null_raw:
                raw_min = min(non_null_raw)
                raw_max = max(non_null_raw)
        except Exception:
            pass

        # MinHash sketch (over stringified non-nulls)

        col_profiles[name] = {
            "sampled_rows": n,
            "nulls": nulls,
            "non_nulls": non_null,
            "distinct_in_sample": len(distinct_vals),
            "numeric_shape": num_profile,
            "string_len_shape": len_stats,
            # "string_charclass_hints": cls_hints,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "topk_values": topk_values,
            # "top_prefixes": top_prefixes,
        }

    return col_profiles



async def sqlite_peek(
    db_id: str,
    table: str,
    columns: List[str],
    *,
    limit: int = 10,
    timeout_s: float = 120.0,
    vm_step_limit: int = 20_000_000,
    busy_timeout_ms: int = 5_000,
    profile: bool = True,
    profile_scan_rows: int = 5000,
    profile_topk: int = 10,
    where: Optional[str] = None,
    **_extra,
) -> Dict[str, Any]:
    """
    Preview column samples and small profiles from a SQLite table (read-only).
    Returns representative rows and basic stats for each column to help the agent
    understand data ranges, types, and distinctness.
    """
    import time
    t0 = time.time()
    try:
        db_path = _resolve_db_path(db_id)
        if not db_path:
            return {"error": "no_db_found", "message": f"Could not resolve db_id='{db_id}'"}

        if not table or not isinstance(table, str):
            return {"error": "invalid_table", "message": "You must provide a valid table name"}
        if not columns or not isinstance(columns, list):
            return {"error": "invalid_columns", "message": "You must provide a non-empty list of columns"}

        # Pre-process: model sometimes puts multiple columns in one string
        # e.g. ["\"Col1\", \"Col2\""] instead of ["Col1", "Col2"]
        expanded = []
        for col in columns:
            if isinstance(col, str) and ',' in col and ('"' in col or '`' in col):
                # Split comma-separated columns and clean each
                parts = [p.strip().strip('"').strip('`').strip('[').strip(']').strip() for p in col.split(',')]
                expanded.extend(p for p in parts if p)
            else:
                expanded.append(col)
        columns = expanded

        # Validate table and columns exist
        try:
            conn_info = _connect_ro(db_path, busy_ms=int(busy_timeout_ms))
            cur_info = conn_info.cursor()
            
            # Check table exists
            cur_info.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
            if not cur_info.fetchone():
                conn_info.close()
                return {"error": "table_not_found", "message": f"Table '{table}' does not exist in database"}
            
            # Get valid columns and their types
            cur_info.execute(f"PRAGMA table_info(`{table}`);")
            table_info = cur_info.fetchall()
            conn_info.close()
            
            valid_columns = {col_info[1].lower(): {"name": col_info[1], "type": col_info[2] or "UNKNOWN"} for col_info in table_info}
            
            # Validate requested columns
            invalid_cols = []
            column_types = {}
            cleaned_columns = []  # map input col names to real schema names
            all_valid_names = [ci[1] for ci in table_info]
            corrected_cols = {}  # original -> corrected
            for col in columns:
                # Strip quotes the model may have added: "Col Name" or `Col Name` or [Col Name]
                clean = col.strip().strip('"').strip('`').strip('[').strip(']').strip()
                match_key = clean.lower()
                if match_key in valid_columns:
                    real_name = valid_columns[match_key]["name"]
                    column_types[real_name] = valid_columns[match_key]["type"]
                    cleaned_columns.append(real_name)
                else:
                    # Try fuzzy matching
                    fuzzy = _fuzzy_match_column(clean, all_valid_names)
                    if fuzzy:
                        column_types[fuzzy] = valid_columns.get(fuzzy.lower(), {}).get("type", "UNKNOWN")
                        cleaned_columns.append(fuzzy)
                        corrected_cols[col] = fuzzy
                    else:
                        invalid_cols.append(col)
            
            if invalid_cols:
                return {
                    "error": "columns_not_found", 
                    "message": f"Column(s) not found: {invalid_cols}",
                    "valid_columns": list(valid_columns.keys()),
                    "hint": "Check column name spelling"
                }
            # Use cleaned column names for queries
            columns = cleaned_columns
        except sqlite3.OperationalError as e:
            # Continue without validation if PRAGMA fails
            column_types = {}

        where_sql = ""
        if isinstance(where, str) and where.strip():
            where_sql = f" WHERE ({where.strip()})"

        col_results: Dict[str, Any] = {}
        tbl_q = f"`{table}`"

        # ---- Strategy: fetch ALL requested columns in ONE query using ROWID sampling ----
        # This avoids N separate full-table scans. We use ROWID-based random sampling
        # which is O(1) per row on rowid lookups instead of sequential scan.
        col_qs = ", ".join(f"`{c}`" for c in columns)

        # --- 1) Small preview: per-column DISTINCT values (avoids repetitive samples) ---
        preview_data: Dict[str, list] = {c: [] for c in columns}
        for col in columns:
            try:
                col_q = f"`{col}`"
                sql_preview = _force_limit(f"SELECT DISTINCT {col_q} FROM {tbl_q}{where_sql}", limit)
                rows, _ = _run_select(
                    db_path,
                    sql_preview,
                    hard_timeout_s=float(timeout_s),
                    busy_ms=int(busy_timeout_ms),
                    max_steps=int(vm_step_limit),
                    cap_rows=None,
                )
                preview_data[col] = [r[0] for r in rows] if rows else []
            except Exception:
                pass

        # --- 2) Profile sample: use ROWID-based random sampling for speed ---
        profile_data: Dict[str, list] = {c: [] for c in columns}
        if profile:
            # Cap profile_scan_rows to avoid excessive scanning
            effective_scan = min(profile_scan_rows, 2000)
            try:
                # First try fast ROWID-based sampling: pick random rowids from the range
                # This is O(sample_size) instead of O(table_size)
                conn_tmp = _connect_ro(db_path, busy_ms=int(busy_timeout_ms))
                cur_tmp = conn_tmp.cursor()
                cur_tmp.execute(f"SELECT MIN(rowid), MAX(rowid) FROM {tbl_q}")
                row_range = cur_tmp.fetchone()
                conn_tmp.close()

                if row_range and row_range[0] is not None and row_range[1] is not None:
                    min_rid, max_rid = int(row_range[0]), int(row_range[1])
                    rid_span = max_rid - min_rid + 1

                    if rid_span <= effective_scan * 3:
                        # Table is small enough, just scan sequentially
                        sql_profile = _force_limit(f"SELECT {col_qs} FROM {tbl_q}{where_sql}", effective_scan)
                    else:
                        # Sample random rowids — much faster than sequential scan on large tables
                        import random
                        sample_rids = sorted(random.sample(range(min_rid, max_rid + 1), min(effective_scan * 2, rid_span)))
                        # Use rowid IN (...) with a reasonable batch; SQLite handles this efficiently
                        # Chunk to avoid overly long IN clauses
                        rid_str = ",".join(str(r) for r in sample_rids[:effective_scan * 2])
                        sql_profile = f"SELECT {col_qs} FROM {tbl_q} WHERE rowid IN ({rid_str}){' AND ' + where.strip() if where_sql else ''} LIMIT {effective_scan}"
                else:
                    sql_profile = _force_limit(f"SELECT {col_qs} FROM {tbl_q}{where_sql}", effective_scan)

                prof_rows, _ = _run_select(
                    db_path,
                    sql_profile,
                    hard_timeout_s=float(timeout_s),
                    busy_ms=int(busy_timeout_ms),
                    max_steps=int(vm_step_limit),
                    cap_rows=None,
                )
                for row in (prof_rows or []):
                    for i, col in enumerate(columns):
                        profile_data[col].append(row[i] if i < len(row) else None)
            except sqlite3.OperationalError:
                pass  # profile will be empty; samples still available
            except Exception:
                pass

        # --- 3) Build per-column results from the batched data ---
        for col in columns:
            try:
                preview_values = preview_data.get(col, [])
                col_profile = None

                if profile and profile_data.get(col):
                    flat_rows = profile_data[col]
                    full_profile = _profile_columns_from_sample(
                        [(v,) for v in flat_rows], [col], topk=int(profile_topk)
                    )[col]

                    col_profile = {
                        "distinct": full_profile.get("distinct_in_sample", 0),
                        "nulls": full_profile.get("nulls", 0),
                    }
                    if full_profile.get("numeric_shape"):
                        col_profile["min"] = full_profile["numeric_shape"].get("min")
                        col_profile["max"] = full_profile["numeric_shape"].get("max")
                    elif full_profile.get("raw_min") is not None:
                        col_profile["min"] = full_profile["raw_min"]
                        col_profile["max"] = full_profile["raw_max"]
                    if full_profile.get("topk_values"):
                        col_profile["top_values"] = [v[0] for v in full_profile["topk_values"][:5]]

                col_results[col] = {
                    "samples": preview_values,
                    "type": column_types.get(col, "UNKNOWN"),
                }
                if col_profile:
                    col_results[col]["stats"] = col_profile
            except Exception as e:
                col_results[col] = {"error": "profile_error", "message": str(e)}

        result = {"columns": col_results}
        # If any columns were fuzzy-corrected, tell the model the correct names
        if corrected_cols:
            corrections = [f"'{k}' → '{v}'" for k, v in corrected_cols.items()]
            result["warning"] = (f"Column name(s) auto-corrected: {', '.join(corrections)}. "
                                f"Use the corrected name(s) with backticks in your SQL (e.g. `{list(corrected_cols.values())[0]}`).")
        return result

    except sqlite3.OperationalError as e:
        err_str = str(e).lower()
        if "no such table" in err_str:
            return {"error": "table_not_found", "message": f"Table '{table}' does not exist"}
        elif "interrupted" in err_str:
            return {"error": "timeout", "message": "Query exceeded time or step limit", "limit_hit": "vm_steps_or_timeout"}
        return {"error": "sqlite_error", "message": str(e)}
    except Exception as e:
        return {"error": "unexpected_error", "message": str(e)}


async def bm25_search_sqlite(
    db_id: str,
    table: str,
    column: str,
    query: str,
    top_k: int = 10,
    *,
    timeout_s: Optional[float] = None,
    busy_timeout_ms: Optional[int] = None,
    vm_step_limit: Optional[int] = None,
    where: Optional[str] = None,
    distinct: bool = True,
    case_sensitive: bool = False,
    min_chars: int = 0,
    max_chars: int = 200,
) -> List[Dict[str, Any]]:
    """
    Perform hybrid BM25 + regex/substring text search over a SQLite column (read-only).
    Returns 2X results: top_k from BM25 ranking + top_k from regex/substring matching.
    Helps locate text values semantically or literally matching a keyword query within a table.
    """
    def _tok(s: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", (s or "").lower())

    def _regex_score(text: str, query_terms: List[str], case_sensitive: bool = False) -> float:
        """Score based on substring/regex matches - higher = more matches."""
        if not text or not query_terms:
            return 0.0
        score = 0.0
        check_text = text if case_sensitive else text.lower()
        for term in query_terms:
            check_term = term if case_sensitive else term.lower()
            # Exact substring match
            if check_term in check_text:
                score += 2.0
            # Word boundary match (more precise)
            pattern = r'\b' + re.escape(check_term) + r'\b'
            flags = 0 if case_sensitive else re.IGNORECASE
            if re.search(pattern, text, flags):
                score += 3.0
            # Prefix match
            if check_text.startswith(check_term):
                score += 1.5
        return score

    try:
        if not isinstance(query, str) or not query.strip():
            return [{"error": "invalid_query", "message": "Query must be a non-empty string"}]
        if not table or not column:
            return [{"error": "missing_table_or_column", "message": "Both table and column are required"}]
        k = max(1, int(top_k))
    except Exception as e:
        return [{"error": "validation_error", "message": str(e)}]

    try:
        db_path = _resolve_db_path(db_id)
        if not db_path:
            return [{"error": "no_db_found", "message": f"Could not resolve db_id='{db_id}'"}]
    except Exception as e:
        return [{"error": "resolve_error", "message": str(e)}]

    try:
        from rank_bm25 import BM25Okapi
        _has_bm25 = True
    except Exception:
        _has_bm25 = False

    # Get column type info first
    column_type = None
    try:
        conn_info = _connect_ro(db_path, busy_ms=int(busy_timeout_ms) if busy_timeout_ms is not None else 2000)
        cur_info = conn_info.cursor()
        cur_info.execute(f"PRAGMA table_info(`{table}`);")
        table_info = cur_info.fetchall()
        conn_info.close()
        
        # Find column type
        for col_info in table_info:
            if col_info[1].lower() == column.lower():
                column_type = col_info[2].upper() if col_info[2] else "UNKNOWN"
                break
        
        if column_type is None:
            # Try fuzzy matching before giving up
            valid_cols = [ci[1] for ci in table_info]
            fuzzy = _fuzzy_match_column(column, valid_cols)
            if fuzzy:
                _original_column = column
                column = fuzzy  # use the corrected column name
                for col_info in table_info:
                    if col_info[1] == fuzzy:
                        column_type = col_info[2].upper() if col_info[2] else "UNKNOWN"
                        break
            else:
                return [{"error": "column_not_found", "message": f"Column '{column}' not found in table '{table}'", "hint": "Check column name spelling", "valid_columns": valid_cols[:20]}]
    except Exception as e:
        # Continue without type info if PRAGMA fails
        pass

    rows = []
    try:
        conn = _connect_ro(db_path, busy_ms=int(busy_timeout_ms) if busy_timeout_ms is not None else 5000)
        if vm_step_limit or timeout_s:
            _set_progress_guard(
                conn,
                hard_timeout_s=float(timeout_s) if timeout_s is not None else 45.0,
                max_steps=int(vm_step_limit) if vm_step_limit is not None else 0,
            )

        cur = conn.cursor()
        col_q = f"`{column}`"
        tbl_q = f"`{table}`"
        distinct_sql = "DISTINCT " if distinct else ""
        if where and isinstance(where, str) and where.strip():
            where_sql = f" WHERE ({where.strip()}) AND {col_q} IS NOT NULL"
        else:
            where_sql = f" WHERE {col_q} IS NOT NULL"
        collation = "" if case_sensitive else " COLLATE NOCASE"

        sql = f"SELECT {distinct_sql}{col_q} FROM {tbl_q}{where_sql}{collation};"
        cur.execute(sql)
        rows = cur.fetchall()
    except sqlite3.OperationalError as e:
        err_str = str(e).lower()
        if "no such table" in err_str:
            return [{"error": "table_not_found", "message": f"Table '{table}' does not exist"}]
        elif "no such column" in err_str:
            return [{"error": "column_not_found", "message": f"Column '{column}' not found in table '{table}'"}]
        elif "interrupted" in err_str:
            return [{"error": "timeout", "message": "Query exceeded time or step limit", "limit_hit": "vm_steps_or_timeout"}]
        return [{"error": "sqlite_error", "message": str(e)}]
    except Exception as e:
        return [{"error": "sqlite_error", "message": str(e)}]
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not rows:
        return [{"error": "no_data", "message": f"No non-null values found in {table}.{column}", "column_type": column_type}]

    # Convert values to strings, handling non-string types
    values: List[str] = []
    non_string_count = 0
    for r in rows:
        v = r[0]
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
        else:
            # Convert non-string values (INTEGER, REAL, etc.) to string
            s = str(v).strip()
            non_string_count += 1
        if s and len(s) >= int(min_chars):
            values.append(s[: int(max_chars)])

    if not values:
        hint = "Use sqlite_peek or direct SQL WHERE clause for numeric lookups" if column_type in ("INTEGER", "REAL", "NUMERIC") else "Column may be empty or all values filtered"
        return [{"error": "no_searchable_values", "message": f"No values to search in {table}.{column}", "column_type": column_type, "hint": hint}]

    # Helper: direct SQL fallback when BM25 tokenization fails
    def _direct_search_fallback(query_str, values_list, k_limit):
        """Fall back to exact/LIKE matching + DISTINCT when BM25 can't tokenize."""
        matched = []
        seen = set()
        q_lower = query_str.lower().strip()
        # 1. Exact matches first
        for v in values_list:
            if v.lower() == q_lower and v not in seen:
                matched.append(v)
                seen.add(v)
        # 2. Substring/LIKE matches
        for v in values_list:
            if v in seen:
                continue
            if q_lower in v.lower():
                matched.append(v)
                seen.add(v)
                if len(matched) >= k_limit:
                    break
        # 3. If still nothing, return all distinct values (capped)
        if not matched:
            for v in values_list:
                if v not in seen:
                    matched.append(v)
                    seen.add(v)
                    if len(seen) >= k_limit:
                        break
        return matched

    corpus: List[List[str]] = []
    kept: List[str] = []
    for v in values:
        toks = _tok(v)
        if toks:
            corpus.append(toks)
            kept.append(v)

    if not corpus:
        # BM25 can't tokenize any values — fall back to direct search
        return _direct_search_fallback(query, values, k)

    q_tokens = _tok(query)
    if not q_tokens:
        # Query itself can't be tokenized (e.g. '-', '=', '#') — fall back to direct search
        return _direct_search_fallback(query, values, k)

    results = []
    seen_contents = set()

    # --- BM25 Search (top_k results) ---
    try:
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:k]
        
        for idx, score in bm25_ranked:
            content = kept[idx]
            if content not in seen_contents:
                seen_contents.add(content)
                results.append(content)
    except Exception as e:
        # If BM25 fails, continue with regex only
        pass

    # --- Regex/Substring Search (top_k results) ---
    try:
        regex_scored = []
        for idx, text in enumerate(kept):
            score = _regex_score(text, q_tokens, case_sensitive)
            if score > 0:
                regex_scored.append((idx, score, text))
        
        # Sort by score descending, then by length (prefer shorter matches)
        regex_ranked = sorted(regex_scored, key=lambda x: (-x[1], len(x[2])))[:k]
        
        for idx, score, content in regex_ranked:
            if content not in seen_contents:
                seen_contents.add(content)
                results.append(content)
    except Exception as e:
        # If regex fails, return what we have from BM25
        pass

    # If no results found, provide helpful feedback
    if not results:
        return [{"error": "no_matches", "message": f"No matches found for '{query}' in {table}.{column}", "hint": "Try different search terms or use sqlite_peek to explore values"}]

    # Return combined results as simple list of matching values
    # If column was fuzzy-corrected, prepend a warning so model uses correct name in SQL
    if '_original_column' in dir() and _original_column != column:
        return [{"warning": f"Column '{_original_column}' not found. Auto-corrected to '{column}'. Use `{column}` (with backticks if special chars) in your SQL."}] + results
    return results

def _quote_variants(sql: str) -> list:
    """Generate SQL variants with different identifier quoting to handle column name issues.
    
    SQLite treats "X" as a string literal if X isn't a column, but `X` as an identifier.
    Models often use "col-name" when they should use `col-name` for columns with special chars.
    """
    import re as _re
    variants = [sql]
    # Variant: replace double-quoted identifiers with backtick-quoted
    # Match "word-with-special-chars" that look like column names (contain hyphen, space, etc.)
    dq_to_bt = _re.sub(r'"([A-Za-z][A-Za-z0-9_ -]*[A-Za-z0-9])"', r'`\1`', sql)
    if dq_to_bt != sql:
        variants.append(dq_to_bt)
    return variants


async def sqlite_query(
    db_id: str,
    sql: str,
    *,
    timeout_s: float = 45.0,
    vm_step_limit: int = 15_000_000,
    busy_timeout_ms: int = 5000,
    max_return_rows: Optional[int] = 100,
    limit: Optional[int] = None,
    **_extra,
) -> Dict[str, Any]:
    """
    Execute the agent's **actual** read-only SQL and return results, with guardrails.

    This function is intended as the **ultimate verification step** before producing
    a `<final_answer>` with `<sql_code>...</sql_code>`.The query may need rewriting if the results are different from expectations. In a text-to-SQL workflow,
    the LLM may use lighter probes (`sqlite_peek`, `bm25_search_sqlite`) during
    reasoning. But before committing to an answer, it should always call
    `sqlite_query` once with the **exact final SQL** it intends to surface.

    Why:
      • Ensures the SQL is executable on the target DB (valid syntax, schema alignment).
      • Confirms that results are returned within resource and safety limits.
      • Catches runtime errors (bad joins, invalid column names, etc.) before vending.
      • Allows truncation detection if the result set is too large.

    Args:
      db_id: Logical database id (e.g., "financial", "formula_1").
      sql: Final SELECT-only SQL. No automatic LIMIT is injected here;
           the agent must include LIMIT if result cardinality may be large.
      timeout_s: Wallclock guard (default 45s).
      vm_step_limit: SQLite VM step cap (default 15,000,000).
      busy_timeout_ms: SQLite busy timeout for readonly connection (default 5000).
      max_return_rows: If set, cap returned rows and set "truncated": true when reached.

    Returns:
      {"rows":[...], "columns":[...], "db_path": "...", "truncated": bool}
      or {"error":"..."}.

    Notes:
      - Enforces **SELECT-only**.
      - Executes with SQLite query_only=ON (read-only, no writes).
      - Always run this once per query before vending `<final_answer>`.
      - If a result set is massive, only the first `max_return_rows` are returned
        and `"truncated": true` is set, signaling the agent to revise with LIMIT.
    """
    try:
        db_path = _resolve_db_path(db_id)
        if not db_path:
            return {"error": "no_db_found", "message": f"Could not resolve db_id='{db_id}'"}

        # Try original SQL first, then retry with quote normalization on column errors
        last_err = None
        for attempt_sql in _quote_variants(sql):
            try:
                rows, cols = _run_select(
                    db_path,
                    attempt_sql,
                    hard_timeout_s=float(timeout_s),
                    busy_ms=int(busy_timeout_ms),
                    max_steps=int(vm_step_limit),
                    cap_rows=int(max_return_rows) if max_return_rows is not None else None,
                )
                truncated = False
                if max_return_rows is not None and len(rows) >= max_return_rows:
                    truncated = True
                result = {"rows": rows, "columns": cols, "truncated": truncated}
                
                # If SQL was auto-corrected, tell the model so it uses the fixed version
                corrected = _sanitize_sql(attempt_sql)
                corrected = _fix_column_names(corrected, db_path)
                if corrected != sql:
                    result["corrected_sql"] = corrected
                
                # Add nudges for suspicious results
                if len(rows) == 0:
                    result["warning"] = ("Query returned 0 rows. This likely means your WHERE/JOIN "
                                        "conditions are too restrictive or reference wrong values. "
                                        "Double-check filter values, table names, and join keys.")
                elif (len(rows) == 1 and len(rows[0]) == 1 and 
                      isinstance(rows[0][0], (int, float)) and rows[0][0] == 0):
                    result["warning"] = ("Query returned a single zero value. If this is a COUNT or SUM, "
                                        "it likely means no rows matched your conditions. Verify your "
                                        "WHERE clause filters and JOIN keys are correct.")
                elif rows and all(v is None for row in rows for v in row):
                    result["warning"] = ("All result values are NULL. This usually means a division or "
                                        "arithmetic operation has NULL inputs. Check if your columns "
                                        "contain NULLs and use CAST, COALESCE, or NULLIF to handle them.")
                
                return result
            except sqlite3.OperationalError as e:
                err_str = str(e).lower()
                if "no such column" in err_str:
                    last_err = e
                    continue  # try next quote variant
                raise  # re-raise non-column errors
        # All variants failed
        raise last_err
    except ValueError as e:
        # From _ensure_select_only
        return {"error": "invalid_sql", "message": str(e), "hint": "Only SELECT statements are allowed"}
    except sqlite3.OperationalError as e:
        err_str = str(e).lower()
        if "no such table" in err_str:
            return {"error": "table_not_found", "message": str(e), "hint": "Check table name spelling"}
        elif "no such column" in err_str:
            # Try to suggest valid columns from the tables used in the query
            suggestions = []
            try:
                conn = _connect_ro(db_path)
                # Extract table names from the SQL
                import re
                table_names = re.findall(r'(?:FROM|JOIN)\s+[`"\[]?(\w+)[`"\]]?', sql, re.IGNORECASE)
                for tbl in set(table_names):
                    try:
                        cur = conn.cursor()
                        cur.execute(f"PRAGMA table_info(`{tbl}`)")
                        cols = [row[1] for row in cur.fetchall()]
                        suggestions.append(f"{tbl}: [{', '.join(cols)}]")
                    except:
                        pass
                conn.close()
            except:
                pass
            result = {"error": "column_not_found", "message": str(e), "hint": "Check column name spelling"}
            if suggestions:
                result["valid_columns"] = "; ".join(suggestions)
            return result
        elif "syntax error" in err_str:
            return {"error": "syntax_error", "message": str(e), "hint": "Check SQL syntax"}
        elif "interrupted" in err_str:
            return {"error": "timeout", "message": "Query exceeded time or step limit", "limit_hit": "vm_steps_or_timeout", "vm_step_limit": vm_step_limit, "timeout_s": timeout_s}
        elif "ambiguous column" in err_str:
            return {"error": "ambiguous_column", "message": str(e), "hint": "Use table.column notation to disambiguate"}
        return {"error": "sqlite_error", "message": str(e)}
    except Exception as e:
        return {"error": "unexpected_error", "message": str(e)}



async def consensus_at_1(
    db_id: str,
    sqls: List[str],
    *,
    timeout_s: float = 45.0,
    vm_step_limit: int = 15_000_000,
    busy_timeout_ms: int = 5000,
    max_return_rows: Optional[int] = 100,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute N SELECT SQL candidates, cluster by (columns, rows) equivalence, and return majority.

    Returns on success:
      {
        "consensus_sql": str,
        "columns": [str,...],
        "rows": [[...], ...],
        "largest_cluster_size": int,
        "num_clusters": int,
        "memberships": [cluster_index|-1,...],
        "chosen_index": int,
      }

    Returns on error:
      {
        "error": "<code>",
        "message": "<human-friendly text>",
      }
    """
    try:
        if not isinstance(sqls, list) or len(sqls) == 0:
            return {"error": "no_sql_candidates", "message": "No SQL candidates were provided.", "per_sql": [], "normalized_sqls": []}
        try:
            db_path = _resolve_db_path(db_id)
            if not db_path:
                return {"error": "db_not_found", "message": f"Could not resolve db_id='{db_id}'", "per_sql": [], "normalized_sqls": []}
        except Exception as e:
            return {"error": "db_not_found", "message": f"Could not resolve db_id='{db_id}': {e}", "per_sql": [], "normalized_sqls": []}
    except Exception as e:
        return {"error": "init_error", "message": f"Initialization failed: {e}", "per_sql": [], "normalized_sqls": []}

    def _strip_sql_comments(s: str) -> str:
        """Robustly strip SQL comments (line and block) from the beginning of SQL.

        Handles edge cases:
        - "-- comment\\nSELECT ..." (normal multiline)
        - "-- commentSELECT ..." (no newline: comment glued to SQL)
        - "/* block */ SELECT ..."
        - "/* block */SELECT ..."
        - Multiple stacked comments
        - BOM and leading whitespace
        """
        t = s.lstrip("\ufeff").strip()
        # Iteratively strip leading comments
        changed = True
        while changed and t:
            changed = False
            # Block comments: /* ... */
            if t.startswith("/*"):
                end = t.find("*/", 2)
                if end != -1:
                    t = t[end + 2:].lstrip()
                    changed = True
                    continue
                else:
                    # Unclosed block comment — treat entire thing as comment
                    return ""
            # Line comments: -- ...
            if t.startswith("--"):
                line_end = t.find("\n")
                if line_end != -1:
                    # Normal case: newline exists, skip the comment line
                    t = t[line_end + 1:].lstrip()
                    changed = True
                    continue
                else:
                    # No newline — entire string is one line starting with --
                    # Model often generates: "-- Strategy 1: blahSELECT ..."
                    # or "-- Multistep with CTEWITH cte AS ..."
                    #
                    # Strategy: find the earliest SQL statement keyword.
                    # Prefer SELECT over WITH to avoid false positives
                    # (e.g. "with" appearing in English comment text).
                    rest = t[2:]  # skip the -- prefix
                    upper_rest = rest.upper()

                    def _find_kw(kw):
                        """Find all valid keyword positions, checking what follows."""
                        positions = []
                        start = 0
                        while True:
                            pos = upper_rest.find(kw, start)
                            if pos == -1:
                                break
                            after = pos + len(kw)
                            # Keyword must be followed by space/tab/paren/newline/end
                            # to avoid matching substrings like "SELECTED" or "WITHDRAW"
                            if after >= len(rest) or rest[after] in (" ", "\t", "(", "\n"):
                                positions.append(pos)
                            start = after  # keep searching
                        return positions

                    select_positions = _find_kw("SELECT")
                    with_positions = _find_kw("WITH")

                    # Filter WITH positions to only those that look like real CTEs
                    # (followed by: identifier AS (...))
                    valid_with_positions = []
                    for wp in with_positions:
                        cte_check = rest[wp + 4:].lstrip()
                        if re.match(r'(?i)\w+\s+AS\s*\(', cte_check):
                            valid_with_positions.append(wp)

                    # Combine all candidate positions and pick the earliest
                    all_positions = select_positions + valid_with_positions

                    if all_positions:
                        best_pos = min(all_positions)
                        t = rest[best_pos:]
                        changed = True
                        continue
                    else:
                        # Entire string is just a comment with no SQL
                        return ""
        return t

    def _normalize_sql(s: str) -> Tuple[Optional[str], Optional[str]]:
        """Normalize a raw SQL candidate: strip comments, sanitize, validate SELECT-only.

        Returns (normalized_sql, error_string_or_None).
        """
        # Step 1: Strip leading comments (handles all edge cases)
        t = _strip_sql_comments(s)
        if not t:
            return None, "empty_after_comment_strip"

        # Step 2: Apply model-output sanitization (backslash-escaped quotes, etc.)
        t = _sanitize_sql(t)

        # Step 3: Validate it's a SELECT (or WITH ... SELECT)
        low = t.lower().lstrip()
        if not (low.startswith("select") or low.startswith("with")):
            return None, "select_only_violation"

        # Step 4: Check for forbidden mutating tokens
        forbidden = (" pragma ", " attach ", " detach ", " insert ",
                      " update ", " delete ", " drop ", " create ", " alter ")
        padded = f" {low} "
        if any(tok in padded for tok in forbidden):
            return None, "forbidden_token_detected"

        return t, None

    per_sql: List[Dict[str, Any]] = []
    normalized_sqls: List[Optional[str]] = []
    results: List[Tuple[Optional[List[List[Any]]], Optional[List[str]], Optional[str]]] = []

    for idx, raw in enumerate(sqls):
        if not isinstance(raw, str) or not raw.strip():
            per_sql.append({"index": idx, "ok": False, "error": "empty_sql"})
            normalized_sqls.append(None)
            results.append((None, None, "empty_sql"))
            continue

        norm, guard_err = _normalize_sql(raw)
        if guard_err:
            per_sql.append({"index": idx, "ok": False, "error": guard_err})
            normalized_sqls.append(None)
            results.append((None, None, guard_err))
            continue

        rows = cols = None
        err = None
        try:
            r, c = _run_select(
                db_path,
                norm,
                hard_timeout_s=float(timeout_s),
                busy_ms=int(busy_timeout_ms),
                max_steps=int(vm_step_limit),
                cap_rows=int(max_return_rows) if max_return_rows is not None else None,
            )
            rows, cols = r, c
            per_sql.append({"index": idx, "ok": True, "rows": len(rows), "truncated": bool(max_return_rows and len(rows) >= max_return_rows)})
            normalized_sqls.append(norm)
        except sqlite3.OperationalError as e:
            err_str = str(e).lower()
            if "interrupted" in err_str:
                err = "timeout"
            elif "no such table" in err_str:
                err = f"table_not_found: {e}"
            elif "no such column" in err_str:
                err = f"column_not_found: {e}"
            elif "syntax error" in err_str:
                err = f"syntax_error: {e}"
            else:
                err = f"sqlite_error: {e}"
            per_sql.append({"index": idx, "ok": False, "error": err})
            normalized_sqls.append(norm)
        except ValueError as e:
            err = f"validation_error: {e}"
            per_sql.append({"index": idx, "ok": False, "error": err})
            normalized_sqls.append(norm)
        except Exception as e:
            err = f"unexpected_error: {e}"
            per_sql.append({"index": idx, "ok": False, "error": err})
            normalized_sqls.append(norm)
        results.append((rows, cols, err))

    import json as _json
    sig_to_members: Dict[str, List[int]] = {}
    sig_to_payload: Dict[str, Tuple[List[List[Any]], List[str]]] = {}
    memberships: List[int] = []

    for i, (rows, cols, err) in enumerate(results):
        if err or rows is None or cols is None:
            memberships.append(-1)
            continue
        sig = _json.dumps({"cols": cols, "rows": rows}, ensure_ascii=False, sort_keys=True)
        if sig not in sig_to_members:
            sig_to_members[sig] = []
            sig_to_payload[sig] = (rows, cols)
        sig_to_members[sig].append(i)
        memberships.append(-1)

    if not sig_to_members:
        first_err = next((e for (_, _, e) in results if e), "all_failed_or_empty")
        return {
            "error": "no_valid_clusters",
            "message": f"No candidate produced a valid result set: {first_err}",
            "per_sql": per_sql,
            "normalized_sqls": normalized_sqls,
        }

    ranked = sorted(sig_to_members.items(), key=lambda kv: (-len(kv[1]), min(kv[1])))
    best_sig, best_members = ranked[0]
    best_rows, best_cols = sig_to_payload[best_sig]
    largest_cluster_size = len(best_members)
    chosen_idx = min(best_members)
    consensus_sql = normalized_sqls[chosen_idx] or sqls[chosen_idx]

    sig_to_index = {sig: idx for idx, (sig, _) in enumerate(ranked)}
    for i, (rows, cols, err) in enumerate(results):
        if err or rows is None or cols is None:
            memberships[i] = -1
            continue
        sig = _json.dumps({"cols": cols, "rows": rows}, ensure_ascii=False, sort_keys=True)
        memberships[i] = sig_to_index.get(sig, -1)

    out: Dict[str, Any] = {
        "consensus_sql": consensus_sql,
        "columns": best_cols,
        "rows": best_rows,
        "largest_cluster_size": int(largest_cluster_size),
        "num_clusters": int(len(sig_to_members)),
        "memberships": memberships,
        "chosen_index": int(chosen_idx),
    }
    # Warn if consensus result is empty
    if not best_rows or (len(best_rows) == 1 and best_rows[0] in [[], [0], [None], [0.0]]):
        out["warning"] = ("All candidates returned empty or zero results. "
                         "This likely means your WHERE/JOIN conditions are wrong. "
                         "Check filter values with bm25_search_sqlite or sqlite_peek.")
    if isinstance(notes, list):
        out["notes"] = notes
    return out
