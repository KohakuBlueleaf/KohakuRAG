"""Indexing Pipeline: Jina V3 Text Only

Builds a text-only index using Jina V3 embeddings.
No images, fastest option.

Output: artifacts/wattbot_text_only.db

Usage:
    python workflows/indexing/jina_v3_text_only.py

Migration (add BM25 to existing index):
    If you already have an index without BM25, you can add it by running:
    python scripts/wattbot_build_bm25_index.py --db artifacts/wattbot_text_only.db --table-prefix wattbot_text
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

    # BM25 sparse search index (optional, for hybrid retrieval)
    build_bm25_index = True


if __name__ == "__main__":
    print("=" * 70)
    print("Indexing Pipeline: Jina V3 Text Only")
    print("=" * 70)
    print(f"Docs: {docs_dir}")
    print(f"Output DB: {db}")
    print(f"Table prefix: {table_prefix}")
    print(f"Embedding: {embedding_model}")
    print(f"Paragraph mode: {paragraph_embedding_mode}")
    print(f"Build BM25: {build_bm25_index}")
    print("=" * 70)

    # Step 1: Build dense vector index
    print("\n[1/2] Building dense vector index...")
    index_config = Config.from_context(ctx)
    index_script = Script("scripts/wattbot_build_index.py", config=index_config)
    index_script.run()

    # Step 2: Build BM25 sparse index (optional)
    if build_bm25_index:
        print("\n[2/2] Building BM25 sparse index...")
        bm25_config = Config.from_context(ctx)
        bm25_script = Script("scripts/wattbot_build_bm25_index.py", config=bm25_config)
        bm25_script.run()
    else:
        print("\n[2/2] Skipping BM25 index (disabled)")

    print("\n" + "=" * 70)
    print("Indexing Complete!")
    print("=" * 70)
    print(f"Database: {db}")
    print(f"- Dense index: {table_prefix}_vec")
    if build_bm25_index:
        print(f"- BM25 index: {table_prefix}_bm25")
    print("=" * 70)
