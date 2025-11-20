# WattBot 2025 Playbook

This guide ties the general KohakuRAG architecture to the specifics of the WattBot 2025 Kaggle competition.

## Prerequisites

Install [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine) for configuration management:

```bash
pip install kohakuengine
```

## Repository layout
- `data/metadata.csv` — bibliography of the reference documents.
- `data/train_QA.csv` — labeled examples showing the expected CSV output format.
- `data/test_Q.csv` — unlabeled questions to be answered for submission.
- `src/kohakurag/` — reusable library (datastore, indexing, RAG pipeline).
- `scripts/` — WattBot-focused utilities (document parsing, indexing, inference, submission helpers).
- `configs/` — KohakuEngine configuration files for all scripts.
- `workflows/` — runnable workflow scripts that orchestrate full pipelines.
- `docs/` — design and operations documentation.

## Quick Start: Run Full Pipeline

```bash
# Text-only pipeline (fetch → index → answer → validate)
python workflows/text_pipeline.py

# Image-enhanced pipeline (fetch → caption → index → answer → validate)
python workflows/with_image_pipeline.py

# Ensemble with voting (multiple parallel runs → aggregate)
python workflows/ensemble_runner.py
```

## Running Individual Scripts

All scripts are configured via Python config files. Edit the config, then run with kogine:

```bash
kogine run scripts/wattbot_fetch_docs.py --config configs/fetch.py
kogine run scripts/wattbot_build_index.py --config configs/text_only/index.py
kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py
kogine run scripts/wattbot_validate.py --config configs/validate.py
```

## Indexing flow

1. **Fetch documents**: Edit `configs/fetch.py` and run:
   ```bash
   kogine run scripts/wattbot_fetch_docs.py --config configs/fetch.py
   ```
   Downloads PDFs and converts them into structured JSON payloads under `artifacts/docs/`.

2. **Build index**: Edit `configs/text_only/index.py` (or `configs/with_images/index.py`) and run:
   ```bash
   kogine run scripts/wattbot_build_index.py --config configs/text_only/index.py
   ```
   Builds document → section → paragraph → sentence nodes with embeddings.

3. **Sanity check**: Edit `configs/demo_query.py` with your question and run:
   ```bash
   kogine run scripts/wattbot_demo_query.py --config configs/demo_query.py
   ```

## Answering questions

Edit `configs/text_only/answer.py` (or `configs/with_images/answer.py`):

```python
# configs/text_only/answer.py
from kohakuengine import Config

db = "artifacts/wattbot_text_only.db"
table_prefix = "wattbot_text"
questions = "data/test_Q.csv"
output = "artifacts/wattbot_answers.csv"
model = "gpt-4o-mini"
top_k = 6
max_retries = 2
max_concurrent = 10
# ... other settings

def config_gen():
    return Config.from_globals()
```

Then run:
```bash
kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py
```

## Validating against the training set

Edit `configs/validate.py`:

```python
from kohakuengine import Config

truth = "data/train_QA.csv"
pred = "artifacts/wattbot_answers.csv"
show_errors = 5
verbose = True

def config_gen():
    return Config.from_globals()
```

Then run:
```bash
kogine run scripts/wattbot_validate.py --config configs/validate.py
```

The validation script compares predictions to ground truth using the official WattBot score recipe (0.75 × answer_value, 0.15 × ref_id, 0.10 × NA handling).

## Aggregating multiple results

Edit `configs/aggregate.py`:

```python
from kohakuengine import Config

inputs = [
    "artifacts/results/run1.csv",
    "artifacts/results/run2.csv",
    "artifacts/results/run3.csv",
]
output = "artifacts/aggregated_preds.csv"
ref_mode = "union"  # or "intersection"
tiebreak = "first"  # or "blank"

def config_gen():
    return Config.from_globals()
```

Then run:
```bash
kogine run scripts/wattbot_aggregate.py --config configs/aggregate.py
```

## Configuring LLM and embeddings

### OpenAI Configuration
- Set `OPENAI_API_KEY` for production runs.
- Configure `max_concurrent` in your answer config to control rate limiting.
- Configure `max_retries` for automatic retry on rate limits.

**Example config for different TPM limits:**

```python
# configs/text_only/answer.py

# For 500K TPM accounts
max_concurrent = 5
top_k = 6

# For higher TPM accounts
max_concurrent = 20
top_k = 8

# For unlimited concurrency
max_concurrent = 0  # or -1
```

### Embedding Configuration
- Uses `jinaai/jina-embeddings-v3` via `JinaEmbeddingModel`.
- First run downloads ~2GB model from Hugging Face — set `HF_HOME` if you need a custom cache location.

## Testing checklist

1. Edit `configs/fetch.py` with `limit = 2`, then:
   ```bash
   kogine run scripts/wattbot_fetch_docs.py --config configs/fetch.py
   ```

2. Edit `configs/text_only/index.py`, then:
   ```bash
   kogine run scripts/wattbot_build_index.py --config configs/text_only/index.py
   ```

3. Edit `configs/demo_query.py` with your test question, then:
   ```bash
   kogine run scripts/wattbot_demo_query.py --config configs/demo_query.py
   ```

4. Run unit tests:
   ```bash
   python -m unittest tests.test_pipeline
   ```

These steps ensure the RAG pipeline works end-to-end before spending tokens on real OpenAI calls.
