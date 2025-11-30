"""Sweep: top_k vs embedding model

Compares different embedding configurations:
- jina v3 text-only
- jina v3 text + images
- jina v4 text + images

Line parameter (categorical): embedding configuration (switches db/table/settings)
X-axis parameter (numerical): top_k

Usage:
    python workflows/sweeps/top_k_vs_embedding.py
"""

import itertools
import json
from pathlib import Path
from typing import Any

from kohakuengine import Config, Script, capture_globals

# ============================================================================
# SWEEP PARAMETERS
# ============================================================================

# Line parameter: embedding config (categorical - each value = one line)
# Each config specifies: db, table_prefix, embedding_model, embedding_dim, with_images
LINE_PARAM = "embedding_config"
LINE_CONFIGS: dict[str, dict[str, Any]] = {
    "jina_v3_text": {
        "db": "artifacts/wattbot_text_only.db",
        "table_prefix": "wattbot_text",
        "embedding_model": "jina",
        "embedding_dim": None,
        "with_images": False,
        "top_k_images": 0,
    },
    "jina_v3_img": {
        "db": "artifacts/wattbot_with_images.db",
        "table_prefix": "wattbot_img",
        "embedding_model": "jina",
        "embedding_dim": None,
        "with_images": True,
        "top_k_images": 2,
    },
    "jina_v4_img": {
        "db": "artifacts/wattbot_jinav4.db",
        "table_prefix": "wattbot_jv4",
        "embedding_model": "jinav4",
        "embedding_dim": 512,
        "with_images": True,
        "top_k_images": 2,
    },
}
LINE_VALUES: list[str] = list(LINE_CONFIGS.keys())

# X-axis parameter: top_k (numerical - forms x-axis)
X_PARAM = "top_k"
X_VALUES: list[int] = [4, 8, 16]

# Multiple runs per config (for std dev calculation)
NUM_RUNS = 3

# ============================================================================
# SHARED SETTINGS
# ============================================================================

OUTPUT_DIR = Path("outputs/sweeps/top_k_vs_embedding")
QUESTIONS = "data/train_QA.csv"
METADATA = "data/metadata.csv"

# Base config using capture_globals
with capture_globals() as ctx:
    questions = QUESTIONS
    metadata = METADATA

    # LLM settings
    llm_provider = "openrouter"  # Options: "openai", "openrouter"
    model = "openai/gpt-oss-120b"
    planner_model = None  # Falls back to model
    openrouter_api_key = None  # From env: OPENROUTER_API_KEY
    site_url = "https://github.com/KohakuBlueleaf/KohakuRAG"
    app_name = "KohakuRAG"

    # Fixed retrieval settings (not swept)
    planner_max_queries = 4
    deduplicate_retrieval = True
    rerank_strategy = "combined"  # Options: None, "frequency", "score", "combined"
    top_k_final = 16  # Truncate after dedup+rerank (None = no truncation)

    # Embedding settings (will be overridden per config)
    embedding_model = "jina"
    embedding_dim = None
    embedding_task = "retrieval"

    # Paragraph search mode (runtime toggle)
    paragraph_search_mode = "averaged"

    # Image settings (will be overridden per config)
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


def create_config(line_val: str, x_val: Any, output_path: str) -> Config:
    """Create config for a specific sweep point."""
    config = Config.from_context(ctx)

    # Apply embedding-specific settings
    emb_config = LINE_CONFIGS[line_val]
    config.globals_dict["db"] = emb_config["db"]
    config.globals_dict["table_prefix"] = emb_config["table_prefix"]
    config.globals_dict["embedding_model"] = emb_config["embedding_model"]
    config.globals_dict["embedding_dim"] = emb_config["embedding_dim"]
    config.globals_dict["with_images"] = emb_config["with_images"]
    config.globals_dict["top_k_images"] = emb_config["top_k_images"]

    # Apply sweep parameters
    config.globals_dict[X_PARAM] = x_val
    config.globals_dict["output"] = output_path
    return config


def make_filename(line_val: str, x_val: Any, run_idx: int) -> str:
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
    print("\nEmbedding configurations:")
    for name, cfg in LINE_CONFIGS.items():
        print(
            f"  {name}: db={cfg['db']}, emb={cfg['embedding_model']}, images={cfg['with_images']}"
        )
    print("=" * 70)

    # Save sweep metadata
    sweep_metadata = {
        "line_param": LINE_PARAM,
        "line_values": LINE_VALUES,
        "line_configs": LINE_CONFIGS,
        "x_param": X_PARAM,
        "x_values": X_VALUES,
        "num_runs": NUM_RUNS,
        "questions": QUESTIONS,
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
    for line_val, x_val in itertools.product(LINE_VALUES, X_VALUES):
        for run_idx in range(NUM_RUNS):
            run_count += 1
            filename = make_filename(line_val, x_val, run_idx)
            pred_path = (OUTPUT_DIR / filename).as_posix()

            emb_cfg = LINE_CONFIGS[line_val]
            print(f"\n{'─' * 70}")
            print(
                f"[{run_count}/{total_runs}] {LINE_PARAM}={line_val}, {X_PARAM}={x_val}, run={run_idx}"
            )
            print(f"  db: {emb_cfg['db']}")
            print(
                f"  embedding: {emb_cfg['embedding_model']}, images: {emb_cfg['with_images']}"
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
    print(f"Run 'python workflows/sweeps/sweep_plot.py {OUTPUT_DIR}' to plot results")
