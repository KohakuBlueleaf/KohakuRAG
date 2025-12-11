# Usage Workflow

All commands assume you've activated the local virtual environment (e.g., `source .venv/bin/activate`). The scripts live under `scripts/` and expect the repository root as the working directory.

## Prerequisites: Install KohakuEngine

All scripts use [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine) for configuration management. Install it first:

```bash
pip install kohakuengine
```

## Running Scripts with Configs

All scripts are configured via Python config files. No command-line arguments are supported.

```bash
# Run any script with its config
kogine run scripts/wattbot_answer.py --config configs/answer.py
```

## Available Config Files

Example configs are provided in the `configs/` directory:

### Common Configs (Root)

| Config | Script | Description |
|--------|--------|-------------|
| `configs/fetch.py` | `wattbot_fetch_docs.py` | Download and parse PDFs |
| `configs/validate.py` | `wattbot_validate.py` | Validate predictions |
| `configs/aggregate.py` | `wattbot_aggregate.py` | Aggregate multiple results |
| `configs/stats.py` | `wattbot_stats.py` | Print index statistics |
| `configs/demo_query.py` | `wattbot_demo_query.py` | Test retrieval |
| `configs/inspect_node.py` | `wattbot_inspect_node.py` | Inspect a node |
| `configs/smoke.py` | `wattbot_smoke.py` | Smoke test |

### Text-Only Path (`configs/text_only/`)

| Config | Script | Description |
|--------|--------|-------------|
| `configs/text_only/index.py` | `wattbot_build_index.py` | Build text-only index |
| `configs/text_only/bm25_index.py` | `wattbot_build_bm25_index.py` | Build BM25 sparse index |
| `configs/text_only/answer.py` | `wattbot_answer.py` | Generate answers (no images) |

### Image-Enhanced Path (`configs/with_images/`)

| Config | Script | Description |
|--------|--------|-------------|
| `configs/with_images/caption.py` | `wattbot_add_image_captions.py` | Add image captions |
| `configs/with_images/index.py` | `wattbot_build_index.py` | Build image-enhanced index |
| `configs/with_images/image_index.py` | `wattbot_build_image_index.py` | Build image-only retrieval index |
| `configs/with_images/bm25_index.py` | `wattbot_build_bm25_index.py` | Build BM25 sparse index |
| `configs/with_images/answer.py` | `wattbot_answer.py` | Generate answers (with images) |

### JinaV4 Multimodal Path (`configs/jinav4/`)

| Config | Script | Description |
|--------|--------|-------------|
| `configs/jinav4/index.py` | `wattbot_build_index.py` | Build JinaV4 index |
| `configs/jinav4/image_index.py` | `wattbot_build_image_index.py` | Build JinaV4 image index |
| `configs/jinav4/bm25_index.py` | `wattbot_build_bm25_index.py` | Build BM25 sparse index |
| `configs/jinav4/answer.py` | `wattbot_answer.py` | Generate answers (JinaV4) |
| `configs/jinav4/caption.py` | `wattbot_add_image_captions.py` | Add image captions |

### Workflows (`workflows/`)

Pre-built workflows that chain multiple scripts together:

| Workflow | Description |
|----------|-------------|
| `workflows/text_pipeline.py` | Full text-only pipeline: fetch → index → answer → validate |
| `workflows/with_image_pipeline.py` | Full image pipeline: fetch → caption → index → image_index → answer → validate |
| `workflows/jinav4_pipeline.py` | JinaV4 multimodal pipeline with direct image embeddings |
| `workflows/ensemble_runner.py` | Run multiple models in parallel, then aggregate results with voting |
| `workflows/jinav4_ensemble_runner.py` | JinaV4 ensemble runner |

**Running workflows:**

```bash
# Run text-only pipeline end-to-end
python workflows/text_pipeline.py

# Run image-enhanced pipeline end-to-end
python workflows/with_image_pipeline.py

# Run JinaV4 multimodal pipeline
python workflows/jinav4_pipeline.py

# Run ensemble with multiple parallel models + aggregation
python workflows/ensemble_runner.py
```

Workflows use KohakuEngine's `Flow` API to orchestrate multiple scripts sequentially or in parallel.

---

## Embedding Models

### Jina v3 (Default)

```python
embedding_model = "jina"  # 1024-dim, text-only
```

### Jina v4 (Multimodal)

```python
embedding_model = "jinav4"
embedding_dim = 1024      # Matryoshka: 128, 256, 512, 1024, 2048
embedding_task = "retrieval"  # Options: "retrieval", "text-matching", "code"
```

**JinaV4 Features:**
- **Matryoshka dimensions**: Flexible output dimensions (128-2048)
- **Task-aware**: Optimized for retrieval, text-matching, or code
- **Multimodal**: Direct image embedding (not just captions)
- **Longer context**: 32K tokens vs 8K for v3

See [jinav4_workflows.md](jinav4_workflows.md) for detailed JinaV4 usage.

---

## Three Retrieval Modes

KohakuRAG supports **three retrieval modes**:

| Mode | Description | Database | Config Setting |
|------|-------------|----------|----------------|
| **1. Text-Only** | Standard RAG with text content only | `wattbot_text_only.db` | None |
| **2. Text + Images (Tree)** | Images in main hierarchy, extracted from sections | `wattbot_with_images.db` | `with_images = True` |
| **3. Text + Images (Dedicated)** | Mode 2 + separate image-only retrieval | `wattbot_with_images.db` | `with_images = True`, `top_k_images = 3` |

**Mode Comparison**:

**Mode 1 (Text-Only)**:
- Fastest indexing
- Smallest database
- Works for text-heavy docs
- Misses visual information

**Mode 2 (Images in Tree)**:
- Images retrieved if their section is retrieved
- No guarantee images are included
- Single indexing step

**Mode 3 (Dedicated Image Retrieval)**:
- **Guarantees** top-k images in results
- Images retrieved independently via vector search
- Requires extra indexing step (wattbot_build_image_index.py)
- Best for visual-heavy Q&A

All modes can coexist for A/B testing!

---

## Step-by-Step Workflow

### 1a. Download and parse PDFs (Required for both paths)

Convert every WattBot source PDF into a structured JSON payload.

**Config** (`configs/fetch.py`):
```python
from kohakuengine import Config

metadata = "data/metadata.csv"
pdf_dir = "artifacts/raw_pdfs"
output_dir = "artifacts/docs"
force_download = False
limit = 10  # Set to 0 for all documents

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_fetch_docs.py --config configs/fetch.py
```

**What it does**:
- Downloads PDFs from URLs in metadata.csv
- Extracts text and creates hierarchical structure
- Detects images and creates placeholder entries (not yet captioned)
- Saves to `artifacts/docs/*.json`

Set `limit = 5` during dry runs to fetch only a few documents, and `force_download = True` if you want to refresh already downloaded PDFs.

---

### 1b. Add image captions (OPTIONAL - For Image-Enhanced Path Only)

> **Skip this step for text-only workflow!**

Generate AI captions for images in your PDFs:

#### Prerequisites

```bash
# Set up OpenRouter (recommended for vision models)
export OPENAI_API_KEY="sk-or-v1-your-openrouter-key"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

#### Run captioning

**Config** (`configs/with_images/caption.py`):
```python
from kohakuengine import Config

docs_dir = "artifacts/docs"
pdf_dir = "artifacts/raw_pdfs"
output_dir = "artifacts/docs_with_images"
db = "artifacts/wattbot_with_images.db"
vision_model = "qwen/qwen3-vl-235b-a22b-instruct"
max_concurrent = 5
limit = 10  # Start with 10 docs for testing

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_add_image_captions.py --config configs/with_images/caption.py
```

**What it does** (4-phase parallel processing):
- **Phase 1**: Reads ALL images from ALL PDFs concurrently (ThreadPoolExecutor)
- **Phase 2**: Compresses ALL images to JPEG (≤1024px, 95% quality) in parallel
- **Phase 3**: Generates captions for ALL images concurrently via vision API
- **Phase 4**: Stores compressed images + updates JSONs
- **Images stored in `wattbot_with_images.db` (table: image_blobs)**
- Caption format: `[img:name WxH] AI-generated caption...`
- Saves to `artifacts/docs_with_images/*.json`

---

### 2. Build the KohakuVault index

You'll create **separate database files** for each path to enable A/B testing.

#### Text-Only Path

**Config** (`configs/text_only/index.py`):
```python
from kohakuengine import Config

metadata = "data/metadata.csv"
docs_dir = "artifacts/docs"
db = "artifacts/wattbot_text_only.db"
table_prefix = "wattbot_text"
embedding_model = "jina"  # or "jinav4"

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_build_index.py --config configs/text_only/index.py
```

#### Image-Enhanced Path

**Config** (`configs/with_images/index.py`):
```python
from kohakuengine import Config

metadata = "data/metadata.csv"
docs_dir = "artifacts/docs_with_images"
db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"
embedding_model = "jina"

def config_gen():
    return Config.from_globals()
```

#### JinaV4 Multimodal Path

**Config** (`configs/jinav4/index.py`):
```python
from kohakuengine import Config

metadata = "data/metadata.csv"
docs_dir = "artifacts/docs_with_images"
db = "artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"
embedding_model = "jinav4"
embedding_dim = 1024
embedding_task = "retrieval"

def config_gen():
    return Config.from_globals()
```

---

### 2b. Build image-only index (OPTIONAL - For Mode 3 Only)

After building the image-enhanced index, optionally add a **dedicated image-only vector table** for guaranteed image retrieval.

**Config** (`configs/with_images/image_index.py`):
```python
from kohakuengine import Config

db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"
embedding_model = "jina"  # or "jinav4" for direct image embedding
embed_images_directly = False  # Set True for JinaV4 direct embedding

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_build_image_index.py --config configs/with_images/image_index.py
```

---

### 3. Run a retrieval sanity check

Test retrieval quality by printing top matches and context snippets.

**Config** (`configs/demo_query.py`):
```python
from kohakuengine import Config

db = "artifacts/wattbot_text_only.db"
table_prefix = "wattbot_text"
question = "How much water does GPT-3 training consume?"
top_k = 5

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_demo_query.py --config configs/demo_query.py
```

---

### 4. Generate WattBot answers

Run the full RAG pipeline (requires `OPENAI_API_KEY`) and produce a Kaggle-style CSV.

**Config** (`configs/text_only/answer.py`):
```python
from kohakuengine import Config

db = "artifacts/wattbot_text_only.db"
table_prefix = "wattbot_text"
questions = "data/test_Q.csv"
output = "artifacts/text_only_answers.csv"
metadata = "data/metadata.csv"

# LLM settings
llm_provider = "openai"  # or "openrouter"
model = "gpt-4o-mini"
planner_model = None  # Falls back to model

# Retrieval settings
top_k = 16
planner_max_queries = 4
deduplicate_retrieval = True
rerank_strategy = "combined"  # Options: None, "frequency", "score", "combined"
top_k_final = 24

# Embedding settings (must match index)
embedding_model = "jina"  # or "jinav4"
embedding_dim = None  # Required for jinav4
embedding_task = "retrieval"

# Image settings
with_images = False
top_k_images = 0

# Other
max_concurrent = 10
max_retries = 3
use_reordered_prompt = True

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py
```

---

### 5. Validate against the labeled training set

**Config** (`configs/validate.py`):
```python
from kohakuengine import Config

truth = "data/train_QA.csv"
pred = "artifacts/text_only_train_preds.csv"
show_errors = 5
verbose = True

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_validate.py --config configs/validate.py

# Example output:
# WattBot score: 0.7812
```

---

## Aggregation: Combining Multiple Results

When you have multiple result CSVs from different runs (e.g., different models, parameters, or random seeds), aggregate them using majority voting.

### Aggregation Modes

| Mode | Description |
|------|-------------|
| `independent` | Vote ref_id and answer_value separately (simple majority) |
| `ref_priority` | First vote on ref_id, then vote answer among rows with winning ref |
| `answer_priority` | First vote on answer, then vote ref among rows with winning answer |
| `union` | Vote on answer, then union all ref_ids from matching rows |
| `intersection` | Vote on answer, then intersect ref_ids from matching rows |

### Configuration

**Config** (`configs/aggregate.py`):
```python
from kohakuengine import Config

inputs = [
    "artifacts/results/run1.csv",
    "artifacts/results/run2.csv",
    "artifacts/results/run3.csv",
]
output = "artifacts/aggregated_preds.csv"
ref_mode = "union"         # Aggregation mode
tiebreak = "first"         # or "blank"
ignore_blank = True        # Filter out is_blank before voting

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_aggregate.py --config configs/aggregate.py
```

### Options

| Setting | Values | Description |
|---------|--------|-------------|
| `ref_mode` | `"independent"`, `"ref_priority"`, `"answer_priority"`, `"union"`, `"intersection"` | How to combine ref_ids |
| `tiebreak` | `"first"` (default), `"blank"` | What to do when all answers differ |
| `ignore_blank` | `True`, `False` | Filter out "is_blank" answers before voting |

### ignore_blank Option

When `ignore_blank = True`:
- Before majority voting, filter out "is_blank" values
- Only filter if there are non-blank values (fallback to "is_blank" if all are blank)
- Useful for ensemble voting where some runs may fail to produce an answer

Example: If 3 runs produce `["42", "is_blank", "42"]`, with `ignore_blank=True`, the result is "42" (ignoring the blank).

---

## Hyperparameter Sweeps

KohakuRAG includes comprehensive sweep workflows for systematic optimization.

### Available Sweeps

| Sweep File | Line Parameter | X Parameter | Description |
|------------|---------------|-------------|-------------|
| `top_k_vs_embedding.py` | embedding_config | top_k | Compare embedding models |
| `top_k_vs_rerank.py` | rerank_strategy | top_k | Compare reranking strategies |
| `top_k_vs_reorder.py` | use_reordered_prompt | top_k | Compare prompt ordering |
| `top_k_vs_max_retries.py` | max_retries | top_k | Compare retry strategies |
| `top_k_vs_top_k_final.py` | top_k_final | top_k | Compare truncation limits |
| `planner_queries_vs_top_k.py` | planner_max_queries | top_k | Compare query planning |
| `bm25_top_k_vs_top_k.py` | bm25_top_k | top_k | Compare BM25 hybrid retrieval |
| `llm_model_vs_embedding.py` | embedding_config | llm_model | Compare LLM models |
| `ensemble_inference.py` | - | - | Run N inferences for ensemble |
| `ensemble_vs_ref_vote.py` | ref_vote_mode | ensemble_size | Compare aggregation modes |
| `ensemble_vs_tiebreak.py` | tiebreak_mode | ensemble_size | Compare tiebreak strategies |
| `ensemble_vs_ignore_blank.py` | ignore_blank | ensemble_size | Compare ignore_blank |

### Running a Sweep

```bash
# Run a parameter sweep
python workflows/sweeps/top_k_vs_embedding.py

# Plot results with mean, std dev, and max
python workflows/sweeps/sweep_plot.py outputs/sweeps/top_k_vs_embedding
```

### Sweep Output

Each sweep creates:
- `metadata.json`: Sweep configuration
- `*_preds.csv`: Prediction files for each config
- `sweep_results.csv`: Validation scores for all runs
- `plot_*.png`: Line plots with error bars

### Sweep Plot Features

- **Solid line**: Mean score across runs
- **Shaded area**: ±1 standard deviation
- **Dashed line**: Maximum score per config
- **Star marker**: Global maximum with label

### Ensemble Sweeps

For ensemble testing, first run inferences once, then aggregate with different strategies:

```bash
# Step 1: Run N inferences (only once)
python workflows/sweeps/ensemble_inference.py --total-runs 16

# Step 2: Run aggregation sweeps (reuses inference results)
python workflows/sweeps/ensemble_vs_ref_vote.py --max-combinations 32
python workflows/sweeps/ensemble_vs_ignore_blank.py --max-combinations 32
python workflows/sweeps/ensemble_vs_tiebreak.py --max-combinations 32

# Step 3: Plot results
python workflows/sweeps/sweep_plot.py outputs/sweeps/ensemble_vs_ref_vote
```

### Custom Sweeps

Create your own sweep by copying an existing one:

```python
# workflows/sweeps/my_sweep.py
from kohakuengine import Config, Script, capture_globals

# Define sweep parameters
LINE_PARAM = "my_param"
LINE_VALUES = ["value1", "value2", "value3"]
X_PARAM = "top_k"
X_VALUES = [4, 8, 16]
NUM_RUNS = 3

# ... rest of sweep logic
```

---

## Rate Limit Handling

**KohakuRAG automatically handles OpenAI rate limits** without requiring manual intervention:

### How It Works

1. **Server-recommended delays**: When OpenAI returns a rate limit error, the system parses the suggested wait time
2. **Exponential backoff**: Falls back to exponential backoff if no delay is provided
3. **Semaphore-based concurrency**: Limits concurrent API requests

### Tuning for Your Rate Limits

**Low TPM accounts (e.g., 500K TPM):**
```python
max_concurrent = 5  # Limit concurrent requests
top_k = 4           # Reduce tokens per request
```

**Higher TPM accounts (e.g., 2M+ TPM):**
```python
max_concurrent = 20  # More concurrent requests
top_k = 10
```

**Self-hosted or unlimited endpoints:**
```python
max_concurrent = 0  # Unlimited concurrency
```

---

## Single-Question Debug Mode

For debugging prompt/parse issues on a single row:

**Config:**
```python
# Add to your answer config
single_run_debug = True
question_id = "q054"  # Optional: specific question to debug
```

**Run:**
```bash
kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py
```

This mode:
- Processes only one question from the input CSV
- Shows the full prompt including "Referenced media" section
- Logs raw model output and parsed structured answer
- Automatically handles context overflow by reducing `top_k`

---

## KohakuEngine Config Reference

### Config File Structure

Config files are pure Python. Define variables at module level, then use `Config.from_globals()`:

```python
# configs/my_config.py
from kohakuengine import Config

# All settings as module-level variables
db = "artifacts/wattbot.db"
model = "gpt-4o-mini"
top_k = 6
output = "artifacts/my_results.csv"

def config_gen():
    return Config.from_globals()
```

### Workflow Orchestration

Chain multiple scripts with the Flow API:

```python
# workflows/my_workflow.py
from kohakuengine import Config, Script, Flow

fetch_config = Config(globals_dict={
    "metadata": "data/metadata.csv",
    "pdf_dir": "artifacts/raw_pdfs",
    "output_dir": "artifacts/docs",
})

answer_config = Config(globals_dict={
    "db": "artifacts/wattbot.db",
    "questions": "data/test_Q.csv",
    "output": "artifacts/answers.csv",
})

if __name__ == "__main__":
    scripts = [
        Script("scripts/wattbot_fetch_docs.py", config=fetch_config),
        Script("scripts/wattbot_answer.py", config=answer_config),
    ]

    # Use use_subprocess=True for scripts that use asyncio
    flow = Flow(scripts, mode="sequential", use_subprocess=True)
    flow.run()
```

**Important notes:**

- **Use `use_subprocess=True` for asyncio scripts**: KohakuRAG scripts use `asyncio`. When running them via `Flow` or `Script.run()`, set `use_subprocess=True` to avoid "event loop is closed" errors.

- **`max_workers` controls parallelism**: When using `mode="parallel"`, the `max_workers` parameter limits concurrent subprocess execution.

---

## Complete Config Parameter Reference

### Indexing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | str | required | Path to database file |
| `table_prefix` | str | required | Prefix for database tables |
| `docs_dir` | str | required | Path to document JSONs |
| `metadata` | str | required | Path to metadata.csv |
| `embedding_model` | str | `"jina"` | `"jina"` or `"jinav4"` |
| `embedding_dim` | int | None | Required for jinav4 (128-2048) |
| `embedding_task` | str | `"retrieval"` | Task for jinav4 |

### Answer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_provider` | str | `"openai"` | `"openai"` or `"openrouter"` |
| `model` | str | required | LLM model name |
| `planner_model` | str | None | Model for query planning (defaults to model) |
| `top_k` | int | 6 | Results per query |
| `planner_max_queries` | int | 1 | Total queries per question |
| `deduplicate_retrieval` | bool | False | Remove duplicate nodes |
| `rerank_strategy` | str | None | `"frequency"`, `"score"`, `"combined"` |
| `top_k_final` | int | None | Truncate after dedup+rerank |
| `with_images` | bool | False | Enable image retrieval |
| `top_k_images` | int | 0 | Images from dedicated index |
| `bm25_top_k` | int | 0 | Additional BM25 results (0 = disabled) |
| `max_concurrent` | int | 10 | Max concurrent API requests |
| `max_retries` | int | 3 | Retry attempts for blank answers |
| `use_reordered_prompt` | bool | False | Reorder context in prompt |

### Aggregation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | list[str] | required | Input CSV files |
| `output` | str | required | Output CSV path |
| `ref_mode` | str | `"union"` | Aggregation mode |
| `tiebreak` | str | `"first"` | Tiebreak strategy |
| `ignore_blank` | bool | False | Filter is_blank before voting |
