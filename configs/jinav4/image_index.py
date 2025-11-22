"""Build image-only index using JinaV4 direct image embeddings.

This creates a separate vector index for images using JinaV4's native image
embedding capability, avoiding the need for caption-based embedding.

Usage:
    kogine run scripts/wattbot_build_image_index.py --config configs/jinav4/image_index.py
"""

from kohakuengine import Config

# Database settings
db = "artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"
image_table = None  # Auto: {prefix}_images_vec

# JinaV4 embedding settings (must match text index for unified search)
embedding_model = "jinav4"
embedding_dim = 1024  # Same as text for unified vector space
embedding_task = "retrieval"

# Direct image embedding (use JinaV4's encode_image instead of captions)
embed_images_directly = True


def config_gen():
    return Config.from_globals()
