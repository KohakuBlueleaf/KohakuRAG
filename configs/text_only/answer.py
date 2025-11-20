"""Text-only answer configuration for WattBot.

Usage:
    kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py
"""

from kohakuengine import Config

# Database settings
db = "artifacts/wattbot_text_only.db"
table_prefix = "wattbot_text"

# Input/output
questions = "data/test_Q.csv"
output = "artifacts/text_only_answers.csv"
metadata = "data/metadata.csv"

# Model settings
model = "gpt-4o-mini"
top_k = 6
planner_model = None  # Uses model
planner_max_queries = 3

# Execution settings
max_retries = 2
max_concurrent = 10

# Image settings (disabled for text-only)
with_images = False
top_k_images = 0

# Debug settings
single_run_debug = False
question_id = None


def config_gen():
    return Config.from_globals()
