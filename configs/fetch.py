"""Config for wattbot_fetch_docs.py"""

from kohakuengine import Config

metadata = "data/metadata.csv"
pdf_dir = "artifacts/raw_pdfs"
output_dir = "artifacts/docs"
force_download = False
limit = 0  # 0 = all documents


def config_gen():
    return Config.from_globals()
