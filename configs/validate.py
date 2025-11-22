"""Config for wattbot_validate.py"""

from kohakuengine import Config

truth = "data/train_QA.csv"
pred = "artifacts/with_images_train_preds3.csv"  # Required
show_errors = 0
verbose = True


def config_gen():
    return Config.from_globals()
