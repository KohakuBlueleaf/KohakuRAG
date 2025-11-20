"""Config for wattbot_aggregate.py"""

from kohakuengine import Config

inputs = [
    "artifacts/results/model1.csv",
    "artifacts/results/model2.csv",
    "artifacts/results/model3.csv",
]
output = "artifacts/aggregated_preds.csv"
ref_mode = "union"  # or "intersection"
tiebreak = "first"  # or "blank"


def config_gen():
    return Config.from_globals()
