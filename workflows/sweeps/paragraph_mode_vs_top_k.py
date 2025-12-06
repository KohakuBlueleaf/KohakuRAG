"""Sweep: paragraph_search_mode vs top_k

Compares different paragraph search modes across different top_k values.
Requires the database to be indexed with "both" mode to have both averaged
and full paragraph embeddings available.

Line parameter (categorical): paragraph_search_mode (averaged/full/both)
X-axis parameter (numerical): top_k

Paragraph Search Modes:
- averaged: Use sentence-averaged paragraph embeddings
- full: Use direct paragraph embeddings
- both: Search both tables and merge by score

Usage:
    python workflows/sweeps/paragraph_mode_vs_top_k.py
    python workflows/sweeps/paragraph_mode_vs_top_k.py --num-runs 5
"""

import argparse
import itertools
import json
from pathlib import Path

from kohakuengine import Config, Script, capture_globals

# ============================================================================
# SWEEP PARAMETERS
# ============================================================================

# Line parameter: paragraph_search_mode (categorical - each value = one line)
LINE_PARAM = "paragraph_search_mode"
LINE_VALUES: list[str] = ["averaged", "full", "both"]

# X-axis parameter: top_k (numerical - forms x-axis)
X_PARAM = "top_k"
X_VALUES: list[int] = [4, 8, 16]

# Multiple runs per config (for std dev calculation)
DEFAULT_NUM_RUNS = 3

# ============================================================================
# SHARED SETTINGS
# ============================================================================

OUTPUT_DIR = Path("outputs/sweeps/paragraph_mode_vs_top_k")
QUESTIONS = "data/train_QA.csv"
METADATA = "data/metadata.csv"

# Base config using capture_globals
with capture_globals() as ctx:
    questions = QUESTIONS
    metadata = METADATA

    # Database/embedding settings (must be indexed with "both" mode)
    db = "artifacts/wattbot_jinav4.db"
    table_prefix = "wattbot_jv4"
    embedding_model = "jinav4"
    embedding_dim = 512
    embedding_task = "retrieval"

    # LLM settings
    llm_provider = "openai"
    model = "openai/gpt-oss-120b"
    planner_model = None
    openrouter_api_key = None
    site_url = "https://github.com/KohakuBlueleaf/KohakuRAG"
    app_name = "KohakuRAG"

    # Fixed retrieval settings
    top_k = 16  # Will be overridden per x-value
    planner_max_queries = 4
    deduplicate_retrieval = True
    rerank_strategy = "combined"
    top_k_final = 32

    # Paragraph search mode (will be overridden per line value)
    paragraph_search_mode = "averaged"

    # Image settings
    with_images = True
    top_k_images = 2
    send_images_to_llm = False

    # Prompt ordering
    use_reordered_prompt = True

    # Other
    max_retries = 0
    max_concurrent = 32
    single_run_debug = False
    question_id = None


def create_config(line_val: str, x_val: int, output_path: str) -> Config:
    """Create config for a specific sweep point."""
    config = Config.from_context(ctx)

    # Apply paragraph_search_mode (line parameter)
    config.globals_dict["paragraph_search_mode"] = line_val

    # Apply top_k (x-axis parameter)
    config.globals_dict["top_k"] = x_val
    config.globals_dict["output"] = output_path
    return config


def make_filename(line_val: str, x_val: int, run_idx: int) -> str:
    """Generate output filename for a sweep point."""
    return f"{LINE_PARAM}={line_val}_{X_PARAM}={x_val}_run{run_idx}_preds.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep: paragraph_search_mode vs top_k"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help=f"Number of runs per config (default: {DEFAULT_NUM_RUNS})",
    )
    args = parser.parse_args()

    num_runs = args.num_runs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = len(LINE_VALUES) * len(X_VALUES) * num_runs
    print("=" * 70)
    print(f"Sweep: {LINE_PARAM} vs {X_PARAM}")
    print("=" * 70)
    print(f"Line values ({LINE_PARAM}): {LINE_VALUES}")
    print(f"X values ({X_PARAM}): {X_VALUES}")
    print(f"Runs per config: {num_runs}")
    print(f"Total runs: {total_runs}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  paragraph_search_mode (lines): {LINE_VALUES}")
    print(f"  top_k (x-axis): {X_VALUES}")
    print(f"  Model: {model}")
    print(f"  Database: {db}")
    print("\nNote: Database must be indexed with 'both' mode for full/both search")
    print("=" * 70)

    # Save sweep metadata
    sweep_metadata = {
        "line_param": LINE_PARAM,
        "line_values": LINE_VALUES,
        "x_param": X_PARAM,
        "x_values": X_VALUES,
        "num_runs": num_runs,
        "questions": QUESTIONS,
        "fixed_settings": {
            "db": db,
            "table_prefix": table_prefix,
            "embedding_model": embedding_model,
            "model": model,
            "planner_max_queries": planner_max_queries,
            "deduplicate_retrieval": deduplicate_retrieval,
            "rerank_strategy": rerank_strategy,
            "top_k_final": top_k_final,
            "with_images": with_images,
            "top_k_images": top_k_images,
        },
    }
    with (OUTPUT_DIR / "metadata.json").open("w") as f:
        json.dump(sweep_metadata, f, indent=2)

    # Run sweep sequentially
    run_count = 0
    for line_val, x_val in itertools.product(LINE_VALUES, X_VALUES):
        for run_idx in range(num_runs):
            run_count += 1
            filename = make_filename(line_val, x_val, run_idx)
            pred_path = (OUTPUT_DIR / filename).as_posix()

            # Skip if already exists
            if Path(pred_path).exists():
                print(f"[{run_count}/{total_runs}] Skipping (exists): {filename}")
                continue

            print(f"\n{'─' * 70}")
            print(
                f"[{run_count}/{total_runs}] {LINE_PARAM}={line_val}, {X_PARAM}={x_val}, run={run_idx}"
            )
            print(f"  paragraph_search_mode: {line_val}")
            print(f"  top_k: {x_val}")
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
    print(f"Run 'python workflows/sweeps/sweep_plot.py {OUTPUT_DIR}' to plot results")
