"""Config for wattbot_stats.py"""

from kohakuengine import Config

db = "artifacts/wattbot.db"
table_prefix = "wattbot"


def config_gen():
    return Config.from_globals()
