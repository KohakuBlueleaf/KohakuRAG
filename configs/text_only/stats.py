"""Config for wattbot_stats.py (text-only path)"""

from kohakuengine import Config

db = "artifacts/wattbot_text_only.db"
table_prefix = "wattbot_text"


def config_gen():
    return Config.from_globals()
