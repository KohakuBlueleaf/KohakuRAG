"""Config for wattbot_answer.py (with-images path)"""

from kohakuengine import Config

db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"
questions = "data/test_Q.csv"
output = "artifacts/with_images_answers.csv"
model = "openai/gpt-oss-120b"
top_k = 16
planner_model = None
planner_max_queries = 3
metadata = "data/metadata.csv"
max_retries = 3
max_concurrent = -1
single_run_debug = False
question_id = None
with_images = True
top_k_images = 4


def config_gen():
    return Config.from_globals()
