"""Config for wattbot_build_index.py (text-only path)

Usage:
    kogine run scripts/wattbot_build_index.py --config configs/text_only/index.py
"""

from kohakuengine import Config

# Document and database settings
metadata = "data/metadata.csv"
docs_dir = "artifacts/docs"
db = "artifacts/wattbot_text_only.db"
table_prefix = "wattbot_text"
use_citations = False

# Embedding settings (using Jina v3 for text-only)
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
