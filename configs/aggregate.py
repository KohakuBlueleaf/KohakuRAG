"""Config for wattbot_aggregate.py"""

from kohakuengine import Config

inputs = [
    "outputs/ensemble/*.csv",
]
output = "outputs/ensemble/ensemble-test.csv"
ref_mode = "answer_priority"  # or "intersection"
tiebreak = "first"  # or "blank"
ignore_blank = True


def config_gen():
    return Config.from_globals()
