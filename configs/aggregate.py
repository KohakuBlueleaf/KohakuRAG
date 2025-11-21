"""Config for wattbot_aggregate.py"""

from kohakuengine import Config

inputs = [
    "artifacts/results/*.csv",
]
output = "outputs/test.csv"
ref_mode = "intersection"  # or "intersection"
tiebreak = "first"  # or "blank"


def config_gen():
    return Config.from_globals()
