"""Config for wattbot_build_index.py (with-images path)

Usage:
    kogine run scripts/wattbot_build_index.py --config configs/with_images/index.py
"""

from kohakuengine import Config

# Document and database settings
metadata = "data/metadata.csv"
docs_dir = "artifacts/docs_with_images"
db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"
use_citations = False

# Embedding settings (using Jina v3 for text)
embedding_model = "jina"  # Options: "jina" (v3), "jinav4"
embedding_dim = None  # Only for jinav4: 128, 256, 512, 1024, 2048
embedding_task = "retrieval"  # Options: "retrieval", "text-matching", "code"

# Paragraph embedding mode
# Options:
#   - "averaged": Paragraph embedding = average of sentence embeddings
#   - "full": Paragraph embedding = direct embedding of paragraph text
#   - "both": Store both averaged (main) and full (separate table) - allows runtime toggle
paragraph_embedding_mode = "both"


def config_gen():
    return Config.from_globals()
