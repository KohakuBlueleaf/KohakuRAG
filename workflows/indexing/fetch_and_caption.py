"""Indexing Pipeline: Fetch Documents + Add Image Captions

Downloads PDFs and extracts images with AI-generated captions.
This is a preparation step before building the index.

Output:
- artifacts/raw_pdfs/       (downloaded PDFs)
- artifacts/docs/           (parsed JSON without captions)
- artifacts/docs_with_images/ (parsed JSON with image captions)
- artifacts/wattbot_with_images.db (image blob storage)

Usage:
    python workflows/indexing/fetch_and_caption.py
"""

from kohakuengine import Config, Script, capture_globals

# ============================================================================
# CONFIGURATION
# ============================================================================

with capture_globals() as ctx:
    # Document settings
    metadata = "data/metadata.csv"
    pdf_dir = "artifacts/raw_pdfs"
    docs_dir = "artifacts/docs"
    output_dir = "artifacts/docs_with_images"
    db = "artifacts/wattbot_with_images.db"

    # Vision model settings (for image captioning)
    llm_provider = "openrouter"  # Options: "openai", "openrouter"
    vision_model = "qwen/qwen3-vl-235b-a22b-instruct"
    openrouter_api_key = None  # From env: OPENROUTER_API_KEY
    max_concurrent = 5

    # Fetch settings
    force_download = False
    limit = 0


if __name__ == "__main__":
    print("=" * 70)
    print("Indexing Pipeline: Fetch Documents + Add Image Captions")
    print("=" * 70)
    print(f"Metadata: {metadata}")
    print(f"PDF dir: {pdf_dir}")
    print(f"Docs dir: {docs_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Database: {db}")
    print(f"Vision model: {vision_model}")
    print("=" * 70)

    # Step 1: Fetch and parse documents
    print("\n[1/2] Fetching and parsing documents...")
    fetch_config = Config.from_context(ctx)
    fetch_script = Script("scripts/wattbot_fetch_docs.py", config=fetch_config)
    fetch_script.run()

    # Step 2: Add image captions
    print("\n[2/2] Adding image captions...")
    caption_config = Config.from_context(ctx)
    caption_script = Script(
        "scripts/wattbot_add_image_captions.py", config=caption_config
    )
    caption_script.run()

    print("\n" + "=" * 70)
    print("Fetch + Caption Complete!")
    print("=" * 70)
    print(f"Parsed docs: {docs_dir}")
    print(f"Docs with captions: {output_dir}")
    print(f"Image storage: {db}")
    print("")
    print("Next steps:")
    print("  - Run jina_v3_text_only.py for text-only index")
    print("  - Run jina_v3_text_image.py for text+image index")
    print("  - Run jina_v4_text_image.py for JinaV4 text+image index")
    print("=" * 70)
