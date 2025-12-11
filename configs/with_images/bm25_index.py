"""Config for wattbot_build_bm25_index.py (with-images path)

Builds BM25 sparse search index for the with-images database.
Requires the dense index to be built first.

Usage:
    kogine run scripts/wattbot_build_bm25_index.py --config configs/with_images/bm25_index.py
"""

from kohakuengine import Config

# Database settings (must match the dense index)
db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"

# Node types to index (sentence and paragraph match dense search behavior)
index_node_kinds = ["sentence", "paragraph"]


def config_gen():
    return Config.from_globals()
