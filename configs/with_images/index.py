"""Config for wattbot_build_index.py (with-images path)"""

from kohakuengine import Config

metadata = "data/metadata.csv"
docs_dir = "artifacts/docs_with_images"
db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"
use_citations = False


def config_gen():
    return Config.from_globals()
