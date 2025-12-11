"""Sweep: bm25_top_k vs top_k

Compares different combinations of dense retrieval (top_k) and sparse BM25 retrieval (bm25_top_k).
BM25 results are added to dense retrieval for context expansion, NOT fused with scores.

Line parameter (categorical): bm25_top_k (0, 2, 4)
X-axis parameter (numerical): top_k (4, 8, 16)

Prerequisites:
    - An index with BM25 table built (run workflows/indexing/*.py with build_bm25_index=True)

Usage:
    python workflows/sweeps/bm25_top_k_vs_top_k.py
"""

import itertools
import json
from pathlib import Path
from typing import Any

from kohakuengine import Config, Script, capture_globals

# ============================================================================
# SWEEP PARAMETERS
# ============================================================================

# Line parameter: bm25_top_k (categorical - each value = one line)
LINE_PARAM = "bm25_top_k"
LINE_VALUES: list[int] = [0, 2, 4, 8]

# X-axis parameter: top_k (numerical - forms x-axis)
X_PARAM = "top_k"
X_VALUES: list[int] = [4, 8, 16]

# Multiple runs per config (for std dev calculation)
NUM_RUNS = 3

# ============================================================================
# SHARED SETTINGS
# ============================================================================

OUTPUT_DIR = Path("outputs/sweeps/bm25_top_k_vs_top_k")
QUESTIONS = "data/train_QA.csv"
METADATA = "data/metadata.csv"

# Base config using capture_globals
with capture_globals() as ctx:
    questions = QUESTIONS
    metadata = METADATA

    # Database settings - use an index with BM25 table
    # Options: wattbot_text_only.db, wattbot_with_images.db, wattbot_jinav4.db
    db = "artifacts/wattbot_text_only.db"
    table_prefix = "wattbot_text"

    # LLM settings
    llm_provider = "openai"  # Options: "openai", "openrouter"
    model = "openai/gpt-oss-120b"
    planner_model = None  # Falls back to model
    openrouter_api_key = None  # From env: OPENROUTER_API_KEY
    site_url = "https://github.com/KohakuBlueleaf/KohakuRAG"
    app_name = "KohakuRAG"

    # Fixed retrieval settings (not swept)
    planner_max_queries = 4
    deduplicate_retrieval = True
    rerank_strategy = "combined"  # Options: None, "frequency", "score", "combined"
    top_k_final = 32  # Truncate after dedup+rerank (None = no truncation)

    # Embedding settings
    embedding_model = "jina"  # Options: "jina" (v3), "jinav4"
    embedding_dim = None
    embedding_task = "retrieval"

    # Paragraph search mode (runtime toggle)
    paragraph_search_mode = "averaged"

    # Image settings (disabled for this sweep)
    with_images = False
    top_k_images = 0
    send_images_to_llm = False

    # Prompt ordering
    use_reordered_prompt = False

    # Other
    max_retries = 0
    max_concurrent = 32
    single_run_debug = False
    question_id = None

    # BM25 settings (will be overridden per config)
    bm25_top_k = 0


def create_config(line_val: int, x_val: int, output_path: str) -> Config:
    """Create config for a specific sweep point."""
    config = Config.from_context(ctx)

    # Apply sweep parameters
    config.globals_dict[LINE_PARAM] = line_val
    config.globals_dict[X_PARAM] = x_val
    config.globals_dict["output"] = output_path
    return config


def make_filename(line_val: int, x_val: int, run_idx: int) -> str:
    """Generate output filename for a sweep point."""
    return f"{LINE_PARAM}={line_val}_{X_PARAM}={x_val}_run{run_idx}_preds.csv"


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(LINE_VALUES) * len(X_VALUES) * NUM_RUNS
    print("=" * 70)
    print(f"Sweep: {LINE_PARAM} vs {X_PARAM}")
    print("=" * 70)
    print(f"Line values ({LINE_PARAM}): {LINE_VALUES}")
    print(f"X values ({X_PARAM}): {X_VALUES}")
    print(f"Runs per config: {NUM_RUNS}")
    print(f"Total runs: {total_runs}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)
    print(f"\nDatabase: {db}")
    print(f"Table prefix: {table_prefix}")
    print("=" * 70)

    # Save sweep metadata
    sweep_metadata = {
        "line_param": LINE_PARAM,
        "line_values": LINE_VALUES,
        "x_param": X_PARAM,
        "x_values": X_VALUES,
        "num_runs": NUM_RUNS,
        "questions": QUESTIONS,
        "db": db,
        "table_prefix": table_prefix,
        "fixed_settings": {
            "model": model,
            "planner_max_queries": planner_max_queries,
            "deduplicate_retrieval": deduplicate_retrieval,
            "rerank_strategy": rerank_strategy,
            "top_k_final": top_k_final,
        },
    }
    with (OUTPUT_DIR / "metadata.json").open("w") as f:
        json.dump(sweep_metadata, f, indent=2)

    # Run sweep sequentially
    run_count = 0
    skipped_count = 0
    for line_val, x_val in itertools.product(LINE_VALUES, X_VALUES):
        for run_idx in range(NUM_RUNS):
            run_count += 1
            filename = make_filename(line_val, x_val, run_idx)
            pred_path = (OUTPUT_DIR / filename).as_posix()

            # Skip if already exists
            if Path(pred_path).exists():
                print(f"[{run_count}/{total_runs}] Skipping (exists): {filename}")
                skipped_count += 1
                continue

            print(f"\n{'─' * 70}")
            print(
                f"[{run_count}/{total_runs}] {LINE_PARAM}={line_val}, {X_PARAM}={x_val}, run={run_idx}"
            )
            print(f"  Output: {pred_path}")
            print(f"{'─' * 70}")

            # Run answer script
            config = create_config(line_val, x_val, pred_path)
            answer_script = Script("scripts/wattbot_answer.py", config=config)
            answer_script.run()

    print("\n" + "=" * 70)
    print("Sweep Complete!")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(
        f"Total: {total_runs}, Skipped: {skipped_count}, Ran: {total_runs - skipped_count}"
    )
    print(f"Run 'python workflows/sweeps/sweep_plot.py {OUTPUT_DIR}' to plot results")
