"""Indexing Pipeline: Jina V3 Text Only

Builds a text-only index using Jina V3 embeddings.
No images, fastest option.

Output: artifacts/wattbot_text_only.db

Usage:
    python workflows/indexing/jina_v3_text_only.py
"""

from kohakuengine import Config, Script, capture_globals

# ============================================================================
# CONFIGURATION
# ============================================================================

with capture_globals() as ctx:
    # Document and database settings
    metadata = "data/metadata.csv"
    docs_dir = "artifacts/docs"  # Text-only docs (no images)
    db = "artifacts/wattbot_text_only.db"
    table_prefix = "wattbot_text"
    use_citations = False

    # Embedding settings (Jina V3 - faster)
    embedding_model = "jina"  # Options: "jina" (v3), "jinav4"
    embedding_dim = None  # Only for jinav4
    embedding_task = "retrieval"  # Options: "retrieval", "text-matching", "code"

    # Paragraph embedding mode
    # Options: "averaged", "full", "both"
    paragraph_embedding_mode = "both"


if __name__ == "__main__":
    print("=" * 70)
    print("Indexing Pipeline: Jina V3 Text Only")
    print("=" * 70)
    print(f"Docs: {docs_dir}")
    print(f"Output DB: {db}")
    print(f"Table prefix: {table_prefix}")
    print(f"Embedding: {embedding_model}")
    print(f"Paragraph mode: {paragraph_embedding_mode}")
    print("=" * 70)

    # Build index
    print("\n[1/1] Building index...")
    index_config = Config.from_context(ctx)
    index_script = Script("scripts/wattbot_build_index.py", config=index_config)
    index_script.run()

    print("\n" + "=" * 70)
    print("Indexing Complete!")
    print("=" * 70)
    print(f"Database: {db}")
    print("=" * 70)
