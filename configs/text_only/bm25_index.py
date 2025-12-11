"""Config for wattbot_build_bm25_index.py (text-only path)

Builds BM25 sparse search index for the text-only database.
Requires the dense index to be built first.

Usage:
    kogine run scripts/wattbot_build_bm25_index.py --config configs/text_only/bm25_index.py
"""

from kohakuengine import Config

# Database settings (must match the dense index)
db = "artifacts/wattbot_text_only.db"
table_prefix = "wattbot_text"

# Node types to index (sentence and paragraph match dense search behavior)
index_node_kinds = ["sentence", "paragraph"]


def config_gen():
    return Config.from_globals()
