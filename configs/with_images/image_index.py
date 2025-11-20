"""Config for wattbot_build_image_index.py"""

from kohakuengine import Config

db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"
image_table = None  # Default: {prefix}_images_vec


def config_gen():
    return Config.from_globals()
