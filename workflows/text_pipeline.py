"""Full text-only pipeline workflow.

Runs: fetch -> index -> answer -> validate

Usage:
    python configs/workflows/text_pipeline.py
"""

from kohakuengine import Config, Script, Flow

# Shared paths
METADATA = "data/metadata.csv"
DB = "artifacts/wattbot_text_only.db"
TABLE_PREFIX = "wattbot_text"

# Stage configs
fetch_config = Config(
    globals_dict={
        "metadata": METADATA,
        "pdf_dir": "artifacts/raw_pdfs",
        "output_dir": "artifacts/docs",
        "force_download": False,
        "limit": 0,
    }
)

index_config = Config(
    globals_dict={
        "metadata": METADATA,
        "docs_dir": "artifacts/docs",
        "db": DB,
        "table_prefix": TABLE_PREFIX,
        "use_citations": False,
    }
)

answer_config = Config(
    globals_dict={
        "db": DB,
        "table_prefix": TABLE_PREFIX,
        "questions": "data/train_QA.csv",
        "output": "artifacts/text_pipeline_preds.csv",
        "metadata": METADATA,
        "model": "gpt-4o-mini",
        "top_k": 6,
        "planner_model": None,
        "planner_max_queries": 3,
        "max_retries": 2,
        "max_concurrent": 10,
        "with_images": False,
        "top_k_images": 0,
        "single_run_debug": False,
        "question_id": None,
    }
)

validate_config = Config(
    globals_dict={
        "truth": "data/train_QA.csv",
        "pred": "artifacts/text_pipeline_preds.csv",
        "show_errors": 5,
        "verbose": True,
    }
)


if __name__ == "__main__":
    scripts = [
        Script("scripts/wattbot_fetch_docs.py", config=fetch_config),
        Script("scripts/wattbot_build_index.py", config=index_config),
        Script("scripts/wattbot_answer.py", config=answer_config),
        Script("scripts/wattbot_validate.py", config=validate_config),
    ]

    flow = Flow(scripts, mode="sequential")
    results = flow.run()

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
