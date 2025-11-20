"""Image-enhanced answer configuration for WattBot.

Usage:
    kogine run scripts/wattbot_answer.py --config configs/with_images/answer.py
"""

from kohakuengine import Config

# Database settings
db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"

# Input/output
questions = "data/test_Q.csv"
output = "artifacts/with_images_answers.csv"
metadata = "data/metadata.csv"

# Model settings
model = "gpt-4o-mini"
top_k = 6
planner_model = None
planner_max_queries = 3

# Execution settings
max_retries = 2
max_concurrent = 10

# Image settings (enabled)
with_images = True
top_k_images = 3  # Retrieve 3 additional images from image-only index

# Debug settings
single_run_debug = False
question_id = None


def config_gen():
    return Config.from_globals()
