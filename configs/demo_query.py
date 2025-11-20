"""Config for wattbot_demo_query.py"""

from kohakuengine import Config

db = "artifacts/wattbot.db"
table_prefix = "wattbot"
question = "How much water does GPT-3 training consume?"
top_k = 5
with_images = False
top_k_images = 0


def config_gen():
    return Config.from_globals()
