"""Ensemble runner: parallel model comparison + aggregation.

Runs multiple models in parallel, then aggregates results using voting.

Usage:
    python configs/workflows/ensemble_runner.py
"""

from typing import Any
from pathlib import Path
from kohakuengine import Config, Script, Flow, capture_globals

# Shared settings
DB = "artifacts/wattbot_with_images.db"
TABLE_PREFIX = "wattbot_img"
QUESTIONS = "data/test_Q.csv"
METADATA = "data/metadata.csv"

MODEL = "openai/GPT-5-nano"
OUTPUT_DIR = Path("outputs/gpt-nano-dedup-rerank-3")
NUM_RUNS = 5

# Models to run in parallel
MODELS = [
    {
        "model": MODEL,
        "output": (OUTPUT_DIR / f"single_preds{i}.csv").as_posix(),
    }
    for i in range(NUM_RUNS)
]

# Aggregation settings
AGGREGATED_OUTPUT = (OUTPUT_DIR / "ensemble_preds.csv").as_posix()
REF_MODE = "intersection"
TIEBREAK = "first"

# Base config
with capture_globals() as ctx:
    db = DB
    table_prefix = TABLE_PREFIX
    questions = QUESTIONS
    metadata = METADATA
    planner_model = None
    planner_max_queries = 3
    top_k = 16
    max_retries = 3
    max_concurrent = -1
    with_images = True
    top_k_images = 2
    questions_id = None
    single_run_debug = False
    deduplicate_retrieval = True
    rerank_strategy = "combined"
    top_k_final = 32


def create_answer_config(cfg: dict[str, Any]) -> Config:
    """Create answer config for a specific model."""
    base_config = Config.from_context(ctx)
    base_config.globals_dict.update(cfg)
    return base_config


if __name__ == "__main__":
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create answer scripts for each model
    answer_scripts = [
        Script(
            "scripts/wattbot_answer.py",
            config=create_answer_config(cfg),
        )
        for cfg in MODELS
    ]

    # Run answer scripts in parallel
    print("=" * 60)
    print(f"Running {len(MODELS)} models in parallel...")
    print("=" * 60)

    answer_flow = Flow(answer_scripts, use_subprocess=True)
    answer_flow.run()

    # Aggregate results
    print("\n" + "=" * 60)
    print("Aggregating results...")
    print("=" * 60)

    aggregate_config = Config(
        globals_dict={
            "inputs": [cfg["output"] for cfg in MODELS],
            "output": AGGREGATED_OUTPUT,
            "ref_mode": REF_MODE,
            "tiebreak": TIEBREAK,
        }
    )

    aggregate_script = Script("scripts/wattbot_aggregate.py", config=aggregate_config)
    aggregate_script.run()

    print("\n" + "=" * 60)
    print("Ensemble complete!")
    print(f"Aggregated results: {AGGREGATED_OUTPUT}")
    print("=" * 60)

    print("setting brief")
    print(
        f"Model: {MODEL}\n"
        f"Ref mode: {REF_MODE}\n"
        f"Tiebreak: {TIEBREAK}\n"
        f"Num runs: {NUM_RUNS}\n"
        f"Planner: {planner_model}\n"
        f"Planner max queries: {planner_max_queries}\n"
        f"Top k: {top_k}\n"
        f"Max retries: {max_retries}\n"
        f"Max concurrent: {max_concurrent}\n"
        f"Top k images: {top_k_images}\n"
        f"Deduplicate retrieval: {deduplicate_retrieval}\n"
        f"Rerank strategy: {rerank_strategy}\n"
        f"Top k final: {top_k_final}\n"
    )
