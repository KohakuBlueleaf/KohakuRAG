"""Config for wattbot_aggregate.py"""

from kohakuengine import Config

inputs = [
    "outputs/train-result-gpt-mini/single_preds*.csv",
]
output = "outputs/train-result-gpt-mini/ensemble_preds.csv"
ref_mode = "intersection"  # or "intersection"
tiebreak = "first"  # or "blank"


def config_gen():
    return Config.from_globals()
