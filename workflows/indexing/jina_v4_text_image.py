"""Indexing Pipeline: Jina V4 Text + Images

Builds an index with images using Jina V4 matryoshka embeddings.
Requires docs with image captions (run fetch_and_caption.py first).

Output: artifacts/wattbot_jinav4.db

Usage:
    python workflows/indexing/jina_v4_text_image.py
"""

from kohakuengine import Config, Script, capture_globals

# ============================================================================
# CONFIGURATION
# ============================================================================

with capture_globals() as ctx:
    # Document and database settings
    metadata = "data/metadata.csv"
    docs_dir = "artifacts/docs_with_images"  # From fetch_and_caption.py
    db = "artifacts/wattbot_jinav4.db"
    table_prefix = "wattbot_jv4"
    use_citations = False

    # Embedding settings (Jina V4 with matryoshka dimensions)
    embedding_model = "jinav4"  # Options: "jina" (v3), "jinav4"
    embedding_dim = 512  # Options: 128, 256, 512, 1024, 2048
    embedding_task = "retrieval"  # Options: "retrieval", "text-matching", "code"

    # Paragraph embedding mode
    # Options: "averaged", "full", "both"
    paragraph_embedding_mode = "both"


if __name__ == "__main__":
    print("=" * 70)
    print("Indexing Pipeline: Jina V4 Text + Images")
    print("=" * 70)
    print(f"Docs: {docs_dir}")
    print(f"Output DB: {db}")
    print(f"Table prefix: {table_prefix}")
    print(f"Embedding: {embedding_model} (dim={embedding_dim})")
    print(f"Paragraph mode: {paragraph_embedding_mode}")
    print("=" * 70)

    # Step 1: Build text index
    print("\n[1/2] Building text index...")
    index_config = Config.from_context(ctx)
    index_script = Script("scripts/wattbot_build_index.py", config=index_config)
    index_script.run()

    # Step 2: Build image-only index
    print("\n[2/2] Building image-only index...")
    image_index_config = Config.from_context(ctx)
    image_index_script = Script(
        "scripts/wattbot_build_image_index.py", config=image_index_config
    )
    image_index_script.run()

    print("\n" + "=" * 70)
    print("Indexing Complete!")
    print("=" * 70)
    print(f"Database: {db}")
    print(f"- Text index: {table_prefix}_vectors")
    print(f"- Image index: {table_prefix}_images_vec")
    print(f"- Embedding dimension: {embedding_dim}")
    print("=" * 70)
