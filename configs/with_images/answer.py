"""Config for wattbot_answer.py (with-images path)"""

from kohakuengine import Config

db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"
questions = "data/train_QA.csv"
output = "artifacts/with_images_train_preds3.csv"
model = "openai/GPT-5-mini"
top_k = 16
planner_model = None
planner_max_queries = 3
deduplicate_retrieval = True  # Deduplicate text results by node_id across queries
rerank_strategy = "combined"  # Options: None, "frequency", "score", "combined"
top_k_final = 24  # Optional: truncate to this many results after dedup+rerank (None = no truncation)
metadata = "data/metadata.csv"
max_retries = 3
max_concurrent = -1
single_run_debug = False
question_id = None
with_images = True
top_k_images = 4


def config_gen():
    return Config.from_globals()
