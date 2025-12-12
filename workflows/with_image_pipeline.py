"""Full image-enhanced pipeline workflow.

Runs: fetch -> caption -> index -> image_index -> answer -> validate

Usage:
    python configs/workflows/with_image_pipeline.py
"""

from kohakuengine import Config, Script, Flow

# Shared paths
METADATA = "data/metadata.csv"
DB = "artifacts/wattbot_with_images.db"
TABLE_PREFIX = "wattbot_img"

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

caption_config = Config(
    globals_dict={
        "docs_dir": "artifacts/docs",
        "pdf_dir": "artifacts/raw_pdfs",
        "output_dir": "artifacts/docs_with_images",
        "db": DB,
        "vision_model": "qwen/qwen3-vl-235b-a22b-instruct",
        "limit": 0,
        "max_concurrent": 5,
    }
)

index_config = Config(
    globals_dict={
        "metadata": METADATA,
        "docs_dir": "artifacts/docs_with_images",
        "db": DB,
        "table_prefix": TABLE_PREFIX,
        "use_citations": False,
    }
)

image_index_config = Config(
    globals_dict={
        "db": DB,
        "table_prefix": TABLE_PREFIX,
        "image_table": None,
    }
)

answer_config = Config(
    globals_dict={
        "db": DB,
        "table_prefix": TABLE_PREFIX,
        "questions": "data/train_QA.csv",
        "output": "artifacts/with_images_pipeline_preds.csv",
        "metadata": METADATA,
        "model": "gpt-4o-mini",
        "top_k": 6,
        "planner_model": None,
        "planner_max_queries": 3,
        "max_retries": 2,
        "max_concurrent": 10,
        "with_images": True,
        "top_k_images": 3,
        "single_run_debug": False,
        "question_id": None,
    }
)

validate_config = Config(
    globals_dict={
        "truth": "data/train_QA.csv",
        "pred": "artifacts/with_images_pipeline_preds.csv",
        "show_errors": 5,
        "verbose": True,
    }
)


if __name__ == "__main__":
    scripts = [
        Script("scripts/wattbot_fetch_docs.py", config=fetch_config),
        Script("scripts/wattbot_add_image_captions.py", config=caption_config),
        Script("scripts/wattbot_build_index.py", config=index_config),
        Script("scripts/wattbot_build_image_index.py", config=image_index_config),
        # Script("scripts/wattbot_answer.py", config=answer_config),
        # Script("scripts/wattbot_validate.py", config=validate_config),
    ]

    flow = Flow(scripts, mode="sequential")
    results = flow.run()

    print("\n" + "=" * 60)
    print("Image-Enhanced Pipeline Complete!")
    print("=" * 60)
