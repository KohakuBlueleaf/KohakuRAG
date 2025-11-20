"""Config for wattbot_validate.py"""

from kohakuengine import Config

truth = "data/train_QA.csv"
pred = "artifacts/wattbot_answers.csv"  # Required
show_errors = 5
verbose = True


def config_gen():
    return Config.from_globals()
