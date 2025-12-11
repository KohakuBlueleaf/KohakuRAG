"""Config for wattbot_stats.py"""

from kohakuengine import Config

db = "artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"


def config_gen():
    return Config.from_globals()
