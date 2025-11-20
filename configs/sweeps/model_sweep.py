"""Sweep across different models for comparison.

Usage:
    kogine workflow parallel scripts/wattbot_answer.py --config configs/sweeps/model_sweep.py --workers 3
"""

from kohakuengine import Config

# Shared settings
db = "artifacts/wattbot.db"
table_prefix = "wattbot"
questions = "data/train_QA.csv"
metadata = "data/metadata.csv"
top_k = 6
planner_max_queries = 3
max_retries = 2
max_concurrent = 10
with_images = False
top_k_images = 0
single_run_debug = False
question_id = None

# Models to compare
MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
]


def config_gen():
    """Generate one config per model."""
    for model_name in MODELS:
        output_name = model_name.replace("-", "_").replace(".", "_")
        yield Config(
            globals_dict={
                "db": db,
                "table_prefix": table_prefix,
                "questions": questions,
                "output": f"artifacts/sweep_{output_name}.csv",
                "metadata": metadata,
                "model": model_name,
                "planner_model": None,
                "top_k": top_k,
                "planner_max_queries": planner_max_queries,
                "max_retries": max_retries,
                "max_concurrent": max_concurrent,
                "with_images": with_images,
                "top_k_images": top_k_images,
                "single_run_debug": single_run_debug,
                "question_id": question_id,
            },
            metadata={"model": model_name},
        )
