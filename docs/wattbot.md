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

# JinaV4 multimodal pipeline (with direct image embedding)
python workflows/jinav4_pipeline.py

# Ensemble with voting (multiple parallel runs → aggregate)
python workflows/ensemble_runner.py
python workflows/jinav4_ensemble_runner.py  # JinaV4 version
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
ref_mode = "union"  # or "independent", "ref_priority", "answer_priority", "intersection"
tiebreak = "first"  # or "blank"
ignore_blank = False  # Set True to filter out "is_blank" before voting

def config_gen():
    return Config.from_globals()
```

Then run:
```bash
kogine run scripts/wattbot_aggregate.py --config configs/aggregate.py
```

### Aggregation Modes

| Mode | Description |
|------|-------------|
| `independent` | Vote ref_id and answer_value separately |
| `ref_priority` | Vote on ref_id first, then answer among matching rows |
| `answer_priority` | Vote on answer first, then ref among matching rows |
| `union` | Vote on answer, union all refs from matching rows |
| `intersection` | Vote on answer, intersect refs from matching rows |

### ignore_blank Option

When `ignore_blank=True`, the aggregator filters out `is_blank` answers before voting (if non-blank answers exist). This is useful for ensemble voting where some runs may fail to produce an answer due to rate limits or other errors.

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

Two embedding models are available:

| Model | Config Value | Dimension | Multimodal |
|-------|--------------|-----------|------------|
| Jina v3 | `embedding_model = "jina"` | 768 | ❌ |
| Jina v4 | `embedding_model = "jinav4"` | 128–2048 | ✅ |

**Jina v3** (default):
- Uses `jinaai/jina-embeddings-v3` via `JinaEmbeddingModel`
- First run downloads ~2GB model from Hugging Face — set `HF_HOME` if you need a custom cache location

**Jina v4** (multimodal):
- Uses `jinaai/jina-embeddings-v4` via `JinaV4EmbeddingModel`
- Configurable dimensions via `embedding_dim` (128, 256, 512, 1024, 2048)
- Task adapters via `embedding_task` (`retrieval`, `text-matching`, `code`)
- Direct image embedding support
- First run downloads ~4GB model

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

## Hyperparameter Sweeps

The `workflows/sweeps/` directory contains scripts for systematic hyperparameter exploration:

### Running a Sweep

```bash
# 1. Run the sweep (generates predictions for all parameter combinations)
python workflows/sweeps/top_k_vs_rerank.py

# 2. Plot results (validates each prediction and generates plots)
python workflows/sweeps/sweep_plot.py outputs/sweeps/top_k_vs_rerank
```

### Available Sweeps

| File | Line Parameter | X Parameter |
|------|----------------|-------------|
| `top_k_vs_rerank.py` | `rerank_strategy` | `top_k` |
| `top_k_vs_topk_final.py` | `top_k_final` | `top_k` |
| `top_k_final_vs_rerank.py` | `rerank_strategy` | `top_k_final` |
| `queries_vs_topk.py` | `planner_max_queries` | `top_k` |
| `ensemble_size_vs_ref_mode.py` | `ref_mode` | `ensemble_size` |
| `ensemble_vs_ignore_blank.py` | `ignore_blank` | `ensemble_size` |
| `llm_model_vs_embedding.py` | `embedding_config` | `llm_model` |

### Sweep Output

Each sweep produces in `outputs/sweeps/{sweep_name}/`:
- `metadata.json` – Sweep configuration
- `{params}_run{n}_preds.csv` – Predictions for each configuration
- `sweep_results.csv` – Validation scores (generated by sweep_plot.py)
- `plot_{metric}.png` – Line plots with mean, ±1 std dev shading, max dotted lines, and global max star marker

### Ensemble Sweeps

For ensemble sweeps, first run the inference to generate raw runs:

```bash
# Generate N raw inference runs
python workflows/sweeps/ensemble_inference.py

# Aggregate with different ensemble sizes and modes
python workflows/sweeps/ensemble_vs_ignore_blank.py
python workflows/sweeps/ensemble_size_vs_ref_mode.py

# Plot results
python workflows/sweeps/sweep_plot.py outputs/sweeps/ensemble_vs_ignore_blank
```
