"""Config for wattbot_validate.py"""

from kohakuengine import Config

truth = "data/train_QA.csv"
pred = "artifacts/wattbot_answers.csv"  # Required
pred = "outputs/train-result-gpt-oss/ensemble_preds.csv"
show_errors = 0
verbose = True


def config_gen():
    return Config.from_globals()
