"""JinaV4 pipeline workflow (skips captioning - assumes docs_with_images exists).

This workflow is for when you already have docs_with_images folder ready:
1. Build text index (using JinaV4 unified embeddings)
2. Build image-only index (using JinaV4 direct image embeddings)
3. Answer questions (using OpenRouter + JinaV4)
4. Validate results

For full pipeline including captioning, use jinav4_pipeline.py

Usage:
    python workflows/jinav4_pipeline_nocaption.py
"""

from kohakuengine import Config, Script, Flow

# Shared paths
METADATA = "data/metadata.csv"
DB = "artifacts/wattbot_jinav4.db"
TABLE_PREFIX = "wattbot_jv4"
DOCS_WITH_IMAGES = "artifacts/docs_with_images"

# Stage 1: Build text index with JinaV4 multimodal embeddings
index_config = Config(
    globals_dict={
        "metadata": METADATA,
        "docs_dir": DOCS_WITH_IMAGES,
        "db": DB,
        "table_prefix": TABLE_PREFIX,
        "use_citations": False,
        # JinaV4 settings
        "embedding_model": "jinav4",
        "embedding_dim": 1024,  # Matryoshka dimension
        "embedding_task": "retrieval",
    }
)

# Stage 2: Build image-only index with direct JinaV4 image embeddings
image_index_config = Config(
    globals_dict={
        "db": DB,
        "table_prefix": TABLE_PREFIX,
        "image_table": None,  # Auto: {prefix}_images_vec
        # JinaV4 direct image embedding
        "embedding_model": "jinav4",
        "embedding_dim": 1024,  # Must match text embeddings for unified search
        "embedding_task": "retrieval",
        "embed_images_directly": True,  # Use JinaV4.encode_image()
    }
)

# Stage 3: Answer questions with OpenRouter + JinaV4
answer_config = Config(
    globals_dict={
        "db": DB,
        "table_prefix": TABLE_PREFIX,
        "questions": "data/test_Q.csv",
        "output": "artifacts/jinav4_nocap_preds.csv",
        "metadata": METADATA,
        # LLM settings (OpenRouter)
        "llm_provider": "openrouter",
        "model": "openai/gpt-5-nano",
        "planner_model": None,
        "openrouter_api_key": None,  # From env: OPENROUTER_API_KEY
        "site_url": "https://github.com/KohakuBlueleaf/KohakuRAG",
        "app_name": "KohakuRAG",
        # Retrieval settings
        "top_k": 16,
        "planner_max_queries": 3,
        "deduplicate_retrieval": True,
        "rerank_strategy": "frequency",
        "top_k_final": 24,
        # JinaV4 settings
        "embedding_model": "jinav4",
        "embedding_dim": 1024,  # Must match index
        "embedding_task": "retrieval",
        # Image settings
        "with_images": True,
        "top_k_images": 4,
        # Other
        "max_retries": 3,
        "max_concurrent": 10,
        "single_run_debug": False,
        "question_id": None,
    }
)

# Stage 4: Validate results
validate_config = Config(
    globals_dict={
        "truth": "data/test_Q.csv",
        "pred": "artifacts/jinav4_nocap_preds.csv",
        "show_errors": 10,
        "verbose": False,
    }
)


if __name__ == "__main__":
    scripts = [
        Script("scripts/wattbot_build_index.py", config=index_config),
        Script("scripts/wattbot_build_image_index.py", config=image_index_config),
        Script("scripts/wattbot_answer.py", config=answer_config),
        Script("scripts/wattbot_validate.py", config=validate_config),
    ]

    flow = Flow(scripts, mode="sequential")
    results = flow.run()

    print("\n" + "=" * 70)
    print("JinaV4 Multimodal Pipeline Complete (No Captioning)!")
    print("=" * 70)
    print("\nPipeline stages:")
    print("  1. ✓ Text indexing (JinaV4 multimodal embeddings)")
    print("  2. ✓ Image indexing (JinaV4 direct image embeddings)")
    print("  3. ✓ Question answering (OpenRouter + GPT5-nano)")
    print("  4. ✓ Validation")
    print("\nOutput: artifacts/jinav4_nocap_preds.csv")
    print("=" * 70)
