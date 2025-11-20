"""Config for wattbot_inspect_node.py"""

from kohakuengine import Config

db = "artifacts/wattbot.db"
table_prefix = "wattbot"
node_id = "amazon2023:sec3:p12"  # Required
add_note = None  # Optional note to append


def config_gen():
    return Config.from_globals()
