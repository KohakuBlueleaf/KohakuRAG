"""Generate captions for images using OpenRouter vision models.

Note: With JinaV4, captions are optional since images can be embedded directly.
However, captions are still useful for:
1. Backward compatibility
2. Fallback when vision LLM is not available
3. Better text-based search

Usage:
    kogine run scripts/wattbot_add_captions.py --config configs/jinav4/caption.py
"""

from kohakuengine import Config

# Input/Output
docs_dir = "artifacts/docs"
pdf_dir = "artifacts/raw_pdfs"
output_dir = "artifacts/docs_with_images"
db = "artifacts/wattbot_jinav4.db"

# Vision model settings (using OpenRouter)
llm_provider = "openrouter"
vision_model = "qwen/qwen3-vl-235b-a22b-instruct"  # Default vision model
openrouter_api_key = None  # From env: OPENROUTER_API_KEY

# Processing settings
limit = 0  # 0 = process all
max_concurrent = 5


def config_gen():
    return Config.from_globals()
