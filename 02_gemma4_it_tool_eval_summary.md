# BIRD Dev Execution Accuracy Summary

## Run Configuration

- bird_mode: `dev`
- inference_backend: `vllm_async`
- model_name_or_path: `google/gemma-4-31B-it`
- input_file: `data/bird_dev_data/raw/bird_dev.json`
- raw_input_file: `data/bird_dev_data/raw/bird_dev.json`
- build_prompts_at_runtime: `True`
- database_dir: `databases/dev_databases`
- diff_json_path: `data/bird_dev_data/raw/bird_dev.json`
- num_examples: `1534`
- loaded_rows: `1534`
- max_prompt_length: `34000`
- max_new_tokens: `8000`
- max_tool_rounds: `8`
- eval_timeout: `60.0`
- eval_workers: `16`
- vllm_tensor_parallel_size: `2`
- vllm_data_parallel_size: `1`
- vllm_async_concurrency: `8`
- vllm_gpu_memory_utilization: `0.93`
- vllm_max_model_len: `43000`

## Timing

- generation_seconds: `5152.00`
- evaluation_seconds: `94.15`
- total_seconds: `5344.78`

## Generation Stats

- generated_examples: `1534`
- filtered_examples: `0`
- stop_reason_counts: `{'finished': 1516, 'max_tool_rounds': 17, 'context_length_exceeded': 1}`
- tool_call_count_total: `1941`
- tool_round_count_total: `1941`
- avg_tool_calls_per_example: `1.265`
- avg_tool_rounds_per_example: `1.265`
- tool_name_counts: `{'sqlite_query': 1878, 'sqlite_peek': 21, 'bm25_search_sqlite': 42}`
- completion_token_total: `698279`
- avg_completion_tokens: `455.201`
- max_prompt_tokens: `30013`

## By Difficulty

| Group | Correct | Count | Accuracy |
| --- | ---: | ---: | ---: |
| simple | 708 | 925 | 76.54 |
| moderate | 302 | 464 | 65.09 |
| challenging | 90 | 145 | 62.07 |

## By Database

| Group | Correct | Count | Accuracy |
| --- | ---: | ---: | ---: |
| card_games | 129 | 191 | 67.54 |
| codebase_community | 134 | 186 | 72.04 |
| formula_1 | 114 | 174 | 65.52 |
| thrombosis_prediction | 96 | 163 | 58.90 |
| student_club | 131 | 158 | 82.91 |
| toxicology | 100 | 145 | 68.97 |
| european_football_2 | 92 | 129 | 71.32 |
| superhero | 115 | 129 | 89.15 |
| financial | 78 | 106 | 73.58 |
| california_schools | 65 | 89 | 73.03 |
| debit_card_specializing | 46 | 64 | 71.88 |

Overall EX Accuracy: 71.71% (1100/1534)

## Execution Stats

- pred_sql_extracted: 1516
- pred_sql_missing: 18
- gold_sql_extracted: 1534
- gold_sql_missing: 0
- pred_sql_executed: 1515
- pred_sql_execution_failed: 19
- gold_sql_executed: 1532
- gold_sql_execution_failed: 2
- both_sql_executed: 1513
