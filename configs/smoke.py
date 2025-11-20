"""Config for wattbot_smoke.py"""

from kohakuengine import Config

metadata = "data/metadata.csv"
question = "What is the ML.ENERGY benchmark?"


def config_gen():
    return Config.from_globals()
