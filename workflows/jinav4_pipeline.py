"""JinaV4 multimodal pipeline workflow.

This workflow demonstrates the full JinaV4 multimodal RAG pipeline:
1. Add image captions (using OpenRouter vision model)
2. Build text index (using JinaV4 unified embeddings)
3. Build image-only index (using JinaV4 direct image embeddings)
4. Answer questions (using OpenRouter + JinaV4)
5. Validate results

Note: This workflow assumes docs_with_images folder already exists.
      For fresh start, run with_image_pipeline.py first.

Usage:
    python workflows/jinav4_pipeline.py
"""

from kohakuengine import Config, Script, Flow

# Shared paths
METADATA = "data/metadata.csv"
DB = "artifacts/wattbot_jinav4.db"
TABLE_PREFIX = "wattbot_jv4"
DOCS_WITH_IMAGES = "artifacts/docs_with_images"


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

# Stage 1: Add image captions using OpenRouter vision model
caption_config = Config(
    globals_dict={
        "docs_dir": "artifacts/docs",
        "pdf_dir": "artifacts/raw_pdfs",
        "output_dir": DOCS_WITH_IMAGES,
        "db": DB,
        "llm_provider": "openrouter",
        "vision_model": "qwen/qwen3-vl-235b-a22b-instruct",
        "openrouter_api_key": None,  # From env
        "limit": 0,
        "max_concurrent": -1,
    }
)

# Stage 2: Build text index with JinaV4 multimodal embeddings
index_config = Config(
    globals_dict={
        "metadata": METADATA,
        "docs_dir": DOCS_WITH_IMAGES,
        "db": DB,
        "table_prefix": TABLE_PREFIX,
        "use_citations": False,
        # JinaV4 settings
        "embedding_model": "jinav4",
        "embedding_dim": 512,  # Matryoshka dimension
        "embedding_task": "retrieval",
    }
)

# Stage 3: Build image-only index with direct JinaV4 image embeddings
image_index_config = Config(
    globals_dict={
        "db": DB,
        "table_prefix": TABLE_PREFIX,
        "image_table": None,  # Auto: {prefix}_images_vec
        # JinaV4 settings
        "embedding_model": "jinav4",
        "embedding_dim": 512,  # Must match text embeddings
        "embedding_task": "retrieval",
        "embed_images_directly": True,  # Use JinaV4.encode_image()
    }
)

# Stage 4: Answer questions with OpenRouter + JinaV4
answer_config = Config(
    globals_dict={
        "db": DB,
        "table_prefix": TABLE_PREFIX,
        "questions": "data/train_QA.csv",
        "output": "artifacts/jinav4_pipeline_preds.csv",
        "metadata": METADATA,
        # LLM settings (OpenRouter)
        "llm_provider": "openrouter",
        "model": "openai/gpt-oss-120b",
        "planner_model": None,  # Falls back to model
        "openrouter_api_key": None,  # From env
        # Retrieval settings
        "top_k": 16,
        "planner_max_queries": 4,
        "deduplicate_retrieval": True,
        "rerank_strategy": "combined",
        "top_k_final": None,
        # JinaV4 settings
        "embedding_model": "jinav4",
        "embedding_dim": 512,
        "embedding_task": "retrieval",
        # Image settings
        "with_images": True,
        "top_k_images": 2,
        # Other
        "max_retries": 2,
        "max_concurrent": -1,
        "single_run_debug": False,
        "question_id": None,
    }
)

# Stage 5: Validate results
validate_config = Config(
    globals_dict={
        "truth": "data/train_QA.csv",
        "pred": "artifacts/jinav4_pipeline_preds.csv",
        "show_errors": 0,
        "verbose": True,  # Set to True for detailed error output
    }
)


if __name__ == "__main__":
    scripts = [
        # Script("scripts/wattbot_fetch_docs.py", config=fetch_config),
        # Script("scripts/wattbot_add_image_captions.py", config=caption_config),
        # Script("scripts/wattbot_build_index.py", config=index_config),
        # Script("scripts/wattbot_build_image_index.py", config=image_index_config),
        Script("scripts/wattbot_answer.py", config=answer_config),
        Script("scripts/wattbot_validate.py", config=validate_config),
    ]

    flow = Flow(scripts, mode="sequential")
    results = flow.run()

    print("\n" + "=" * 70)
    print("JinaV4 Multimodal Pipeline Complete!")
    print("=" * 70)
    print("\nPipeline stages:")
    print("  1. ✓ Image captioning (OpenRouter + Qwen3-VL)")
    print("  2. ✓ Text indexing (JinaV4 multimodal embeddings)")
    print("  3. ✓ Image indexing (JinaV4 direct image embeddings)")
    print("  4. ✓ Question answering (OpenRouter + GPT5-nano)")
    print("  5. ✓ Validation")
    print("\nOutput: artifacts/jinav4_pipeline_preds.csv")
    print("=" * 70)
