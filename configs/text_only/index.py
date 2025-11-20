"""Config for wattbot_build_index.py (text-only path)"""

from kohakuengine import Config

metadata = "data/metadata.csv"
docs_dir = "artifacts/docs"
db = "artifacts/wattbot_text_only.db"
table_prefix = "wattbot_text"
use_citations = False


def config_gen():
    return Config.from_globals()
