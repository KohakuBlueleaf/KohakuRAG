"""Build index with JinaV4 multimodal embeddings.

Usage:
    kogine run scripts/wattbot_build_index.py --config configs/jinav4/index.py
"""

from kohakuengine import Config

# Document and database settings
metadata = "data/metadata.csv"
docs_dir = "artifacts/docs_with_images"  # Must include images for multimodal
db = "artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"
use_citations = False

# JinaV4 embedding settings
embedding_model = "jinav4"  # Options: "jina" (v3), "jinav4"
embedding_dim = 1024  # Matryoshka dimensions: 128, 256, 512, 1024, 2048
embedding_task = "retrieval"  # Options: "retrieval", "text-matching", "code"

# Paragraph embedding mode
# Options:
#   - "averaged": Paragraph embedding = average of sentence embeddings
#   - "full": Paragraph embedding = direct embedding of paragraph text
#   - "both": Store both averaged (main) and full (separate table) - allows runtime toggle
paragraph_embedding_mode = "both"


def config_gen():
    return Config.from_globals()
