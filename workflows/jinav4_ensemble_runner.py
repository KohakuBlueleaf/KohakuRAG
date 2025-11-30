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
QUESTIONS = "data/train_QA.csv"
METADATA = "data/metadata.csv"

MODEL = "openai/gpt-oss-120b"
OUTPUT_DIR = Path("outputs/jinav4-ensemble")
NUM_RUNS = 9

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

    # LLM settings
    llm_provider = "openrouter"  # Options: "openai", "openrouter"
    planner_model = None  # Falls back to model
    openrouter_api_key = None  # From env: OPENROUTER_API_KEY
    site_url = "https://github.com/KohakuBlueleaf/KohakuRAG"
    app_name = "KohakuRAG"

    # Retrieval settings
    top_k = 16
    planner_max_queries = 4
    deduplicate_retrieval = True
    rerank_strategy = "combined"  # Options: None, "frequency", "score", "combined"
    top_k_final = 24  # Truncate after dedup+rerank (None = no truncation)

    # JinaV4 embedding settings
    embedding_model = "jinav4"  # Options: "jina" (v3), "jinav4"
    embedding_dim = 512  # Matryoshka: 128, 256, 512, 1024, 2048
    embedding_task = "retrieval"  # Options: "retrieval", "text-matching", "code"

    # Paragraph search mode (runtime toggle, requires "both" mode during indexing)
    # Options: "averaged", "full"
    paragraph_search_mode = "averaged"

    # Image settings
    with_images = True
    top_k_images = 2
    send_images_to_llm = False

    # Prompt ordering (context before question to combat attention sink)
    use_reordered_prompt = False

    # Other
    max_retries = 3
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
    ]

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
        }
    )

    aggregate_script = Script("scripts/wattbot_aggregate.py", config=aggregate_config)
    aggregate_script.run()

    # Validate aggregated results
    print("\n" + "=" * 70)
    print("Validating ensemble results...")
    print("=" * 70)

    validate_config = Config(
        globals_dict={
            "truth": "data/train_QA.csv",
            "pred": AGGREGATED_OUTPUT,
            "show_errors": 0,
            "verbose": True,
        }
    )

    validate_script = Script("scripts/wattbot_validate.py", config=validate_config)
    validate_script.run()

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
    print(f"\n  Aggregated output: {AGGREGATED_OUTPUT}")
    print("=" * 70)
