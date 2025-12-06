"""Sweep: send_images_to_llm vs top_k_images

Compares the effect of sending actual images to the LLM vs just using captions,
across different numbers of retrieved images.

Line parameter (categorical): send_images_to_llm (True/False)
X-axis parameter (numerical): top_k_images

Usage:
    python workflows/sweeps/send_images_vs_top_k_images.py
    python workflows/sweeps/send_images_vs_top_k_images.py --num-runs 5
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from kohakuengine import Config, Script, capture_globals

# ============================================================================
# SWEEP PARAMETERS
# ============================================================================

# Line parameter: send_images_to_llm (categorical - each value = one line)
LINE_PARAM = "send_images_to_llm"
LINE_VALUES: list[bool] = [True, False]

# X-axis parameter: top_k_images (numerical - forms x-axis)
X_PARAM = "top_k_images"
X_VALUES: list[int] = [0, 2, 4, 8]

# Multiple runs per config (for std dev calculation)
DEFAULT_NUM_RUNS = 3

# ============================================================================
# SHARED SETTINGS
# ============================================================================

OUTPUT_DIR = Path("outputs/sweeps/send_images_vs_top_k_images")
QUESTIONS = "data/train_QA.csv"
METADATA = "data/metadata.csv"

# Base config using capture_globals
with capture_globals() as ctx:
    questions = QUESTIONS
    metadata = METADATA

    # Database/embedding settings (using jina v4 with images)
    db = "artifacts/wattbot_jinav4.db"
    table_prefix = "wattbot_jv4"
    embedding_model = "jinav4"
    embedding_dim = 512
    embedding_task = "retrieval"

    # LLM settings (use vision-capable model)
    llm_provider = "openai"
    model = "x-ai/grok-4.1-fast"
    planner_model = None
    openrouter_api_key = None
    site_url = "https://github.com/KohakuBlueleaf/KohakuRAG"
    app_name = "KohakuRAG"

    # Fixed retrieval settings
    top_k = 16
    planner_max_queries = 4
    deduplicate_retrieval = True
    rerank_strategy = "combined"
    top_k_final = 32

    # Paragraph search mode
    paragraph_search_mode = "averaged"

    # Image settings (will be overridden per sweep point)
    with_images = True
    top_k_images = 0
    send_images_to_llm = False

    # Prompt ordering
    use_reordered_prompt = True

    # Other
    max_retries = 0
    max_concurrent = 32
    single_run_debug = False
    question_id = None


def create_config(line_val: bool, x_val: int, output_path: str) -> Config:
    """Create config for a specific sweep point."""
    config = Config.from_context(ctx)

    # Apply send_images_to_llm (line parameter)
    config.globals_dict["send_images_to_llm"] = line_val

    # Apply top_k_images (x-axis parameter)
    config.globals_dict["top_k_images"] = x_val
    config.globals_dict["output"] = output_path
    return config


def make_filename(line_val: bool, x_val: int, run_idx: int) -> str:
    """Generate output filename for a sweep point."""
    return f"{LINE_PARAM}={line_val}_{X_PARAM}={x_val}_run{run_idx}_preds.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep: send_images_to_llm vs top_k_images"
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
    print(f"  send_images_to_llm (lines): {LINE_VALUES}")
    print(f"  top_k_images (x-axis): {X_VALUES}")
    print(f"  Model: {model}")
    print(f"  Database: {db}")
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
            "top_k": top_k,
            "planner_max_queries": planner_max_queries,
            "deduplicate_retrieval": deduplicate_retrieval,
            "rerank_strategy": rerank_strategy,
            "top_k_final": top_k_final,
            "with_images": with_images,
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
            print(f"  send_images_to_llm: {line_val}")
            print(f"  top_k_images: {x_val}")
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
