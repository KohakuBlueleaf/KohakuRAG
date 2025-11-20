"""Config for wattbot_add_image_captions.py"""

from kohakuengine import Config

docs_dir = "artifacts/docs"
pdf_dir = "artifacts/raw_pdfs"
output_dir = "artifacts/docs_with_images"
db = "artifacts/wattbot_with_images.db"
vision_model = "qwen/qwen3-vl-235b-a22b-instruct"
limit = 0
max_concurrent = 5


def config_gen():
    return Config.from_globals()
