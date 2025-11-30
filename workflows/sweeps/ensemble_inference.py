"""Ensemble Inference: Run N inferences for ensemble testing.

Runs N independent inferences that can be reused by different
ensemble aggregation sweeps.

Output:
    outputs/sweeps/ensemble/raw_runs/
    ├── run0_preds.csv
    ├── run1_preds.csv
    └── ...

Usage:
    python workflows/sweeps/ensemble_inference.py
    python workflows/sweeps/ensemble_inference.py --total-runs 12
"""

import argparse
import json
from pathlib import Path

from kohakuengine import Config, Script, capture_globals

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = Path("outputs/sweeps/ensemble")
RAW_DIR = OUTPUT_DIR / "raw_runs"
DB = "artifacts/wattbot_jinav4.db"
TABLE_PREFIX = "wattbot_jv4"
QUESTIONS = "data/train_QA.csv"
METADATA = "data/metadata.csv"

# Default total runs (can be overridden via CLI)
DEFAULT_TOTAL_RUNS = 16

# Base config using capture_globals
with capture_globals() as ctx:
    db = DB
    table_prefix = TABLE_PREFIX
    questions = QUESTIONS
    metadata = METADATA

    # LLM settings
    llm_provider = "openrouter"
    model = "openai/gpt-oss-120b"
    planner_model = None
    openrouter_api_key = None
    site_url = "https://github.com/KohakuBlueleaf/KohakuRAG"
    app_name = "KohakuRAG"

    # Retrieval settings
    top_k = 16
    planner_max_queries = 6
    deduplicate_retrieval = True
    rerank_strategy = "combined"  # Options: None, "frequency", "score", "combined"
    top_k_final = 48  # Truncate after dedup+rerank (None = no truncation)

    # JinaV4 embedding settings
    embedding_model = "jinav4"  # Options: "jina" (v3), "jinav4"
    embedding_dim = 512  # Matryoshka: 128, 256, 512, 1024, 2048
    embedding_task = "retrieval"  # Options: "retrieval", "text-matching", "code"

    # Paragraph search mode (runtime toggle, requires "both" mode during indexing)
    # Options: "averaged", "full"
    paragraph_search_mode = "averaged"

    # Other
    with_images = True
    top_k_images = 4
    send_images_to_llm = False
    use_reordered_prompt = True
    max_retries = 3
    max_concurrent = 32
    single_run_debug = False
    question_id = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run N inferences for ensemble testing"
    )
    parser.add_argument(
        "--total-runs",
        type=int,
        default=DEFAULT_TOTAL_RUNS,
        help=f"Number of inference runs (default: {DEFAULT_TOTAL_RUNS})",
    )
    args = parser.parse_args()

    total_runs = args.total_runs
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Ensemble Inference")
    print("=" * 70)
    print(f"Total runs: {total_runs}")
    print(f"Output directory: {RAW_DIR}")
    print(f"Model: {model}")
    print(f"top_k: {top_k}")
    print("=" * 70)

    for run_idx in range(total_runs):
        filename = f"run{run_idx}_preds.csv"
        pred_path = RAW_DIR / filename

        # Skip if already exists
        if pred_path.exists():
            print(
                f"[{run_idx + 1}/{total_runs}] Skipping run={run_idx} (already exists)"
            )
            continue

        print(f"\n{'─' * 70}")
        print(f"[{run_idx + 1}/{total_runs}] run={run_idx}")
        print(f"Output: {pred_path}")
        print(f"{'─' * 70}")

        # Create config
        config = Config.from_context(ctx)
        config.globals_dict["output"] = str(pred_path)

        # Run answer script
        answer_script = Script("scripts/wattbot_answer.py", config=config)
        answer_script.run()

    # Save metadata
    inference_metadata = {
        "total_runs": total_runs,
        "questions": QUESTIONS,
        "fixed_settings": {
            "model": model,
            "top_k": top_k,
            "planner_max_queries": planner_max_queries,
            "rerank_strategy": rerank_strategy,
            "top_k_final": top_k_final,
        },
    }
    with (OUTPUT_DIR / "inference_metadata.json").open("w") as f:
        json.dump(inference_metadata, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Inference complete! {total_runs} runs saved to {RAW_DIR}")
    print("=" * 70)
    print("\nNext steps:")
    print("  python workflows/sweeps/ensemble_vs_tiebreak.py --aggregate")
    print("  python workflows/sweeps/ensemble_vs_ref_vote.py --aggregate")
    print("=" * 70)
