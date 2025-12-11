"""Config for pdf_to_documents.py"""

from kohakuengine import Config

# Input PDF file
input = "document.pdf"

# Output directory (None = use input filename without extension)
output = None

# DPI for rendered page images (72=screen, 150=good, 300=print quality)
page_dpi = 150

# Image format: "png", "jpg", "webp"
image_format = "png"

# Whether to extract embedded images from PDF
extract_images = True

# Whether to render full page images
render_pages = True

# Minimum image dimension to extract (skip small icons/bullets)
min_image_size = 50


def config_gen():
    return Config.from_globals()
