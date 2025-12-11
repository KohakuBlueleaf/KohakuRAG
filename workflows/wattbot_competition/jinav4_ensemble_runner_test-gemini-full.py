"""JinaV4 Ensemble Runner: parallel model comparison + aggregation on train_QA.

Runs multiple models in parallel on JinaV4 multimodal index, then aggregates using voting.

Usage:
    python workflows/jinav4_ensemble_runner.py
"""

from pathlib import Path
from typing import Any

from kohakuengine import Config, Flow, Script, capture_globals

# Shared settings
DB = "artifacts/wattbot_jinav4.db"
TABLE_PREFIX = "wattbot_jv4"
QUESTIONS = "data/test_Q.csv"
METADATA = "data/metadata.csv"

MODEL = "google/gemini-3-pro-preview"
OUTPUT_DIR = Path("outputs/gemini3pro")
NUM_RUNS = 1

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
REF_MODE = "answer_priority"
TIEBREAK = "first"
IGNORE_BLANK = True

# Base config
with capture_globals() as ctx:
    db = DB
    table_prefix = TABLE_PREFIX
    questions = QUESTIONS
    metadata = METADATA

    # LLM settings
    llm_provider = (
        "openai"  # we use openai lib with openrouter api here, which is more stable
    )
    planner_model = None
    openrouter_api_key = None  # From env
    site_url = "https://github.com/KohakuBlueleaf/KohakuRAG"
    app_name = "KohakuRAG"

    # Retrieval settings
    top_k = 16
    bm25_top_k = 4
    planner_max_queries = 4
    deduplicate_retrieval = True
    rerank_strategy = "combined"
    top_k_final = None
    paragraph_search_mode = "full"

    # JinaV4 settings
    embedding_model = "jinav4"
    embedding_dim = 512
    embedding_task = "retrieval"

    # Image settings
    with_images = True
    top_k_images = 2
    send_images_to_llm = True

    # Prompt ordering (context before question to combat attention sink)
    use_reordered_prompt = True

    # Other
    max_retries = 2
    max_concurrent = 32
    single_run_debug = False
    question_id = None


def create_answer_config(cfg: dict[str, Any]) -> Config:
    """Create answer config for a specific model run."""
    base_config = Config.from_context(ctx)
    base_config.globals_dict.update(cfg)
    return base_config


if __name__ == "__main__":
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create answer scripts for each model run
    answer_scripts = [
        Script(
            "scripts/wattbot_answer.py",
            config=create_answer_config(cfg),
        )
        for cfg in MODELS
    ][1:]

    # Run answer scripts in parallel
    print("=" * 70)
    print(f"Running {len(MODELS)} ensemble runs in parallel (JinaV4)...")
    print("=" * 70)

    answer_flow = Flow(answer_scripts, use_subprocess=True)
    answer_flow.run()

    # Aggregate results
    print("\n" + "=" * 70)
    print("Aggregating results...")
    print("=" * 70)

    aggregate_config = Config(
        globals_dict={
            "inputs": [cfg["output"] for cfg in MODELS],
            "output": AGGREGATED_OUTPUT,
            "ref_mode": REF_MODE,
            "tiebreak": TIEBREAK,
            "ignore_blank": IGNORE_BLANK,
        }
    )

    aggregate_script = Script("scripts/wattbot_aggregate.py", config=aggregate_config)
    aggregate_script.run()

    print("\n" + "=" * 70)
    print("JinaV4 Ensemble Complete!")
    print("=" * 70)

    print("\nSettings Brief:")
    print(f"  Model: {MODEL}")
    print(f"  Num runs: {NUM_RUNS}")
    print(f"  Ref mode: {REF_MODE}")
    print(f"  Tiebreak: {TIEBREAK}")
    print(f"  LLM provider: {llm_provider}")
    print(f"  Planner max queries: {planner_max_queries}")
    print(f"  Top k: {top_k}")
    print(f"  Top k final: {top_k_final}")
    print(f"  Rerank strategy: {rerank_strategy}")
    print(f"  Deduplicate: {deduplicate_retrieval}")
    print(f"  Embedding: {embedding_model} (dim={embedding_dim})")
    print(f"  Top k images: {top_k_images}")
    print(f"  Send images to LLM: {send_images_to_llm}")
    print(f"  Reordered prompt: {use_reordered_prompt}")
    print(f"  Max retries: {max_retries}")
    print(f"  Ignore blank: {IGNORE_BLANK}")
    print(f"\n  Aggregated output: {AGGREGATED_OUTPUT}")
    print("=" * 70)
