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
embedding_model = "jinav4"
embedding_dim = 1024  # Matryoshka: 128, 256, 512, 1024, 2048
embedding_task = "retrieval"  # or "text-matching", "code"


def config_gen():
    return Config.from_globals()
