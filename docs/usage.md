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
| `configs/text_only/answer.py` | `wattbot_answer.py` | Generate answers (no images) |

### Image-Enhanced Path (`configs/with_images/`)

| Config | Script | Description |
|--------|--------|-------------|
| `configs/with_images/caption.py` | `wattbot_add_image_captions.py` | Add image captions |
| `configs/with_images/index.py` | `wattbot_build_index.py` | Build image-enhanced index |
| `configs/with_images/image_index.py` | `wattbot_build_image_index.py` | Build image-only retrieval index |
| `configs/with_images/answer.py` | `wattbot_answer.py` | Generate answers (with images) |

### Workflows (`workflows/`)

Pre-built workflows that chain multiple scripts together. These are runnable scripts (not configs) that orchestrate the full pipeline:

| Workflow | Description |
|----------|-------------|
| `workflows/text_pipeline.py` | Full text-only pipeline: fetch → index → answer → validate |
| `workflows/with_image_pipeline.py` | Full image pipeline: fetch → caption → index → image_index → answer → validate |
| `workflows/ensemble_runner.py` | Run multiple models in parallel, then aggregate results with voting |

**Running workflows:**

```bash
# Run text-only pipeline end-to-end
python workflows/text_pipeline.py

# Run image-enhanced pipeline end-to-end
python workflows/with_image_pipeline.py

# Run ensemble with multiple parallel models + aggregation
python workflows/ensemble_runner.py
```

Workflows use KohakuEngine's `Flow` API to orchestrate multiple scripts sequentially or in parallel.

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

## 1a. Download and parse PDFs (Required for both paths)

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

## 1b. Add image captions (OPTIONAL - For Image-Enhanced Path Only)

> **Skip this step for text-only workflow!**

Generate AI captions for images in your PDFs:

### Prerequisites

```bash
# Set up OpenRouter (recommended for vision models)
export OPENAI_API_KEY="sk-or-v1-your-openrouter-key"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# Or create .env file (see .env.example)
cp .env.example .env
# Edit .env with your OpenRouter credentials
```

### Run captioning

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

**Important**: Use the SAME `db` path for `wattbot_build_index.py` in step 2. Images and RAG nodes share the same database file (different tables).

**Progress output**:
```
PHASE 1: Reading images from all documents (parallel)
[1/32] amazon2023... ✓ 5 images
[2/32] google2024... ✓ 3 images
[3/32] nvidia2024... ✓ 7 images
...
✓ Collected 143 images from 32 docs (3.2s)

PHASE 2: Compressing all images (parallel)
  [1/143] ✓ amazon2023 p1:i1 - 234.5KB→78.2KB (67% saved)
  [2/143] ✓ amazon2023 p3:i2 - 189.3KB→62.1KB (67% saved)
  ...
✓ Compressed 143/143 images (4.1s)

PHASE 3: Captioning all images (parallel)
  [1/143] ✓ amazon2023 p1:i1
  [2/143] ✓ amazon2023 p3:i2
  ...
✓ Captioned 143/143 images (125.7s)

PHASE 4: Storing images and updating documents
[1/32] amazon2023
  ✓ Updated with 5 captions
...
✓ Stored images and updated documents (2.8s)

FINAL SUMMARY
Documents updated:     32
Captions added:        143
Images stored:         143
Errors:                0
```

**Recommended models** (via OpenRouter):
- `qwen/qwen3-vl-235b-a22b-instruct` - Best quality/cost (~$0.50 per 1K images)
- `gpt-4o-mini` - Fastest, cheapest (~$0.15 per 1K images)
- `gpt-4o` - Highest quality (~$2.50 per 1K images)

**Cost estimate**: ~$0.13 for 50 documents with 5 images each (using qwen model)

See [image_rag_example.md](image_rag_example.md) for detailed configuration and troubleshooting.

---

## 2. Build the KohakuVault index

You'll create **separate database files** for each path to enable A/B testing.

### Text-Only Path

Embed the structured payloads (from step 1a) and store in a text-only database.

**Config** (`configs/text_only/index.py`):
```python
from kohakuengine import Config

metadata = "data/metadata.csv"
docs_dir = "artifacts/docs"
db = "artifacts/wattbot_text_only.db"
table_prefix = "wattbot_text"

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_build_index.py --config configs/text_only/index.py
```

**Output**: `artifacts/wattbot_text_only.db` (~130 MB for 50 docs)

### Image-Enhanced Path

Embed the image-captioned payloads (from step 1b) and store in a separate database.

**Config** (`configs/with_images/index.py`):
```python
from kohakuengine import Config

metadata = "data/metadata.csv"
docs_dir = "artifacts/docs_with_images"
db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_build_index.py --config configs/with_images/index.py
```

**Output**: `artifacts/wattbot_with_images.db` (~145 MB for 50 docs, +13% for captions)

### Database Architecture

**IMPORTANT**: Images are stored in the SAME `.db` file as RAG nodes!

```bash
ls -lh artifacts/*.db

# Expected (after both indexing steps):
# wattbot_text_only.db      130 MB  (RAG nodes only - no images)
# wattbot_with_images.db    168 MB  (RAG nodes + compressed JPEG images)
#   ├─ wattbot_img_kv      (node metadata - includes image caption nodes)
#   ├─ wattbot_img_vec     (node embeddings - includes image caption embeddings)
#   └─ image_blobs         (compressed JPEG blobs)
```

**Key points**:
- **Single file**: One `.db` file contains BOTH nodes AND images
- **Separate databases**: Text-only vs. with-images for A/B comparison
- **Same table prefix**: Use consistent prefix (e.g., `wattbot_img`) for both nodes and images
- **Different tables**: `image_blobs` table coexists with `_kv` and `_vec` tables in same file

**Why same file?**
- Single-file distribution (easy to share/deploy)
- Atomic transactions (nodes + images updated together)
- No path management issues
- KohakuVault supports multiple tables per file

**Note**: If you only want to test the pipeline without PDFs, add `use_citations = True` in your config to index the citation text from `metadata.csv`.

---

## 2b. Build image-only index (OPTIONAL - For Mode 3 Only)

> **Skip this for Mode 1 (text-only) and Mode 2 (images in tree)!**

After building the image-enhanced index (step 2, image path), optionally add a **dedicated image-only vector table** for guaranteed image retrieval.

**Config** (`configs/with_images/image_index.py`):
```python
from kohakuengine import Config

db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_build_image_index.py --config configs/with_images/image_index.py
```

**What it does**:
- Scans the existing database for image caption nodes
- Creates a separate vector table (`wattbot_img_images_vec`) containing ONLY image embeddings
- Enables fast top-k image retrieval independent of text sections
- Guarantees images in retrieval results (Mode 3)

**Output**:
```
Building Image-Only Vector Index
============================================================
Image embeddings: 143
Table: wattbot_img_images_vec

Now you can use top_k_images in your answer config
```

**Database structure after this step**:
```
wattbot_with_images.db:
  ├─ wattbot_img_kv         (node metadata)
  ├─ wattbot_img_vec        (all node embeddings)
  ├─ wattbot_img_images_vec (image-only embeddings) ← NEW
  └─ image_blobs            (compressed JPEG blobs)
```

---

## 3. Run a retrieval sanity check

Test retrieval quality by printing top matches and context snippets.

### Mode 1: Text-Only

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

### Mode 2: Text + Images (Tree)

Update `configs/demo_query.py`:
```python
db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"
question = "What does Figure 3 show about GPU power consumption?"
top_k = 5
with_images = True
```

### Mode 3: Text + Images (Dedicated)

Update `configs/demo_query.py`:
```python
db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"
question = "What does Figure 3 show about GPU power consumption?"
top_k = 5
with_images = True
top_k_images = 3
```

**Mode 3 output includes**:
```
Referenced media (3 images):
  [1] nvidia2024 page 3, image 1
      [img:Fig3 800x600] Line graph comparing power consumption across GPU generations...
```

## 4. Inspect a stored node

Fetch the raw text/metadata for any node ID (e.g., a paragraph). Works with both paths.

**Config** (`configs/inspect_node.py`):
```python
from kohakuengine import Config

db = "artifacts/wattbot_text_only.db"  # or wattbot_with_images.db
table_prefix = "wattbot_text"          # or wattbot_img
node_id = "amazon2023:sec3:p12"
add_note = None  # Set to "text" to append a developer note

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_inspect_node.py --config configs/inspect_node.py
```

## 5. Snapshot index statistics

Summarize document, paragraph, and sentence counts. Works with both paths.

**Config** (`configs/stats.py`):
```python
from kohakuengine import Config

db = "artifacts/wattbot_text_only.db"  # or wattbot_with_images.db
table_prefix = "wattbot_text"          # or wattbot_img

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_stats.py --config configs/stats.py
```

## 6. Generate WattBot answers

Run the full RAG pipeline (requires `OPENAI_API_KEY`) and produce a Kaggle-style CSV. The script:
- Reads questions from `data/test_Q.csv` (or any compatible file)
- Uses the `answer_unit` column as known metadata (not predicted)
- Calls the core RAG pipeline to get `answer`, `answer_value`, `ref_id`, `explanation`
- Resolves `ref_url` and `supporting_materials` from `data/metadata.csv` using the chosen `ref_id` values
- **Automatically handles OpenAI rate limits** with intelligent retry logic

### Text-Only Path

**Config** (`configs/text_only/answer.py`):
```python
from kohakuengine import Config

db = "artifacts/wattbot_text_only.db"
table_prefix = "wattbot_text"
questions = "data/test_Q.csv"
output = "artifacts/text_only_answers.csv"
metadata = "data/metadata.csv"
model = "gpt-4o-mini"
top_k = 6
max_concurrent = 10
max_retries = 2

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py
```

### Mode 2: Text + Images (Tree)

**Config** (`configs/with_images/answer.py`):
```python
from kohakuengine import Config

db = "artifacts/wattbot_with_images.db"
table_prefix = "wattbot_img"
questions = "data/test_Q.csv"
output = "artifacts/mode2_answers.csv"
metadata = "data/metadata.csv"
model = "gpt-4o-mini"
top_k = 6
max_concurrent = 10
max_retries = 2
with_images = True  # ← Extract images from retrieved sections

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_answer.py --config configs/with_images/answer.py
```

### Mode 3: Text + Images (Dedicated)

Update `configs/with_images/answer.py`:
```python
with_images = True
top_k_images = 3  # ← Additionally retrieve 3 images from image-only index
```

**Mode differences**:
- **Mode 2**: Images retrieved only if their section is in top-k text results
- **Mode 3**: **Guarantees** 3 most relevant images, even if their sections weren't retrieved

**What `with_images = True` does**:
- Retrieves images from sections containing matched text (always)
- If `top_k_images > 0`: additionally searches image-only index
- Adds separate "Referenced media" section to prompt:
  ```
  Context snippets:
  [ref_id=doc1] Text content...
  ---

  Referenced media:
  [ref_id=doc1] [img:Fig3 800x600] Bar chart showing GPU power consumption...
  ```
- **Note**: Images include `[ref_id=doc_id]` to show which document they're from
- LLM can now reference visual information in answers

### Comparing All Three Modes

Create separate answer configs for each mode and compare accuracy:

```bash
# Mode 1: Text-only
kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py

# Mode 2: Images from sections
kogine run scripts/wattbot_answer.py --config configs/with_images/answer_mode2.py

# Mode 3: Images from sections + dedicated image index
kogine run scripts/wattbot_answer.py --config configs/with_images/answer_mode3.py

# Compare accuracy (update configs/validate.py for each)
kogine run scripts/wattbot_validate.py --config configs/validate.py

# Expected results:
# Mode 1: 0.78 (baseline)
# Mode 2: 0.82 (+5%)
# Mode 3: 0.84 (+8%) - best for visual-heavy questions
```

### Key Config Parameters

- **`with_images`**: Enable image-aware retrieval (extracts images from retrieved sections). Works with image-captioned databases.
- **`top_k_images`**: Number of ADDITIONAL images from image-only index (default: 0). Requires running `wattbot_build_image_index.py` first.
  - `0`: Only extract images from sections (Mode 2)
  - `3`: Guarantee 3 most relevant images (Mode 3)
- `max_concurrent`: Maximum concurrent API requests (default: 10). Set to 0 for unlimited.
- `planner_model`: Model for generating additional retrieval queries (defaults to `model`).
- `planner_max_queries`: Total retrieval queries per question.
- `metadata`: Path to `metadata.csv` for resolving citations.
- `max_retries`: Extra attempts when model returns `is_blank`.

The script writes each answered row to `output` as soon as it finishes (streaming results via async generator), so you can inspect partial results while a long run is still in progress.

### Rate Limit Handling

**KohakuRAG automatically handles OpenAI rate limits** without requiring manual intervention:

#### How It Works

1. **Server-recommended delays**: When OpenAI returns a rate limit error like:
   ```
   Rate limit reached for gpt-4o-mini in organization org-xxx on tokens per min (TPM):
   Limit 500000, Used 500000, Requested 193. Please try again in 23ms.
   ```
   The system parses "23ms" and waits exactly that long (plus a small buffer).

2. **Exponential backoff**: If no specific delay is provided, uses exponential backoff:
   - Attempt 1: wait 1 second
   - Attempt 2: wait 2 seconds
   - Attempt 3: wait 4 seconds
   - Attempt 4: wait 8 seconds
   - Attempt 5: wait 16 seconds

3. **Transparent logging**: You'll see messages like:
   ```
   Rate limit hit (attempt 1/6). Waiting 0.12s before retry...
   Rate limit hit (attempt 2/6). Waiting 2.00s before retry...
   ```

#### Tuning for Your Rate Limits

**Low TPM accounts (e.g., 500K TPM):**
```python
# configs/text_only/answer.py
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

**Customizing retry behavior in code:**
```python
import asyncio
from kohakurag.llm import OpenAIChatModel

async def main():
    # More aggressive retries for restrictive limits
    chat = OpenAIChatModel(
        model="gpt-4o-mini",
        max_concurrent=5,        # Limit concurrent requests
        max_retries=10,          # Retry up to 10 times
        base_retry_delay=2.0,    # Start with 2s instead of 3s
    )

    response = await chat.complete("What is RAG?")

asyncio.run(main())
```

#### Best Practices

1. **Start conservative**: Use `max_concurrent = 5` for your first run to understand your rate limits
2. **Monitor the logs**: Watch for retry messages to gauge how often you're hitting limits
3. **Scale up gradually**: Increase `max_concurrent` until you start seeing frequent retries, then back off
4. **Use batch processing windows**: Run large jobs during off-peak hours to maximize throughput
5. **Leverage async concurrency**: All scripts use `asyncio.gather()` for efficient concurrent processing
6. **Switch backends via config**: To move from OpenAI-hosted models to self-hosted vLLM/llama.cpp or OpenAI-compatible proxies for Anthropic/Gemini, configure `OPENAI_BASE_URL` / `OPENAI_API_KEY` as described in `docs/deployment.md`.

## 7. Validate against the labeled training set

Because the public `data/test_Q.csv` file has no answers, use `data/train_QA.csv` as a proxy leaderboard to measure progress locally.

### Text-Only Path

**Generate predictions config** (`configs/text_only/answer.py`):
```python
questions = "data/train_QA.csv"
output = "artifacts/text_only_train_preds.csv"
# ... other settings
```

**Run:**
```bash
kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py
```

**Validate config** (`configs/validate.py`):
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

### Image-Enhanced Path

Update answer config:
```python
questions = "data/train_QA.csv"
output = "artifacts/with_images_train_preds.csv"
with_images = True
```

Update validate config:
```python
pred = "artifacts/with_images_train_preds.csv"
```

**Run:**
```bash
kogine run scripts/wattbot_answer.py --config configs/with_images/answer.py
kogine run scripts/wattbot_validate.py --config configs/validate.py

# Example output:
# WattBot score: 0.8245 (+5.4% improvement!)
```

### Comparing Results

```bash
# Show side-by-side comparison
echo "Text-only:"
kogine run scripts/wattbot_validate.py --config configs/validate_text.py

echo -e "\nWith images:"
kogine run scripts/wattbot_validate.py --config configs/validate_images.py
```

The validation command compares predictions to ground truth using the official WattBot scoring recipe (answer accuracy, citation overlap, and NA handling). Iterate here until the validation score looks good, then switch the `questions` field back to `data/test_Q.csv` to produce the submission file.

All questions are processed concurrently via `asyncio.gather()`, with results streaming to the output file as they complete.

## 8. Single-question debug mode

For debugging prompt/parse issues on a single row (for example, a specific id in `train_QA.csv`), use the `single_run_debug` setting.

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
- Processes only one question from the input CSV (by default the first row, or the row whose `id` matches `question_id`)
- Logs every retry attempt with its `top_k` value
- **Shows the full prompt** including any "Referenced media" section (when using `with_images = True`)
- Logs the exact user prompt sent to the model for each attempt
- Logs the raw model output and the parsed structured answer for each attempt
- Automatically handles context overflow (400 errors) by reducing `top_k` recursively
- Writes a single prediction row to `output` so you can inspect all intermediate steps end-to-end

**Use this to**:
- Debug why a specific question fails
- Inspect exactly what context (text + images) the LLM sees
- Verify image captions are being retrieved correctly

## 9. Aggregate multiple result files

When you have multiple result CSVs from different runs (e.g., different models, parameters, or random seeds), aggregate them using majority voting.

**Config** (`configs/aggregate.py`):
```python
from kohakuengine import Config

inputs = [
    "artifacts/results/run1.csv",
    "artifacts/results/run2.csv",
    "artifacts/results/run3.csv",
]
output = "artifacts/aggregated_preds.csv"
ref_mode = "union"     # or "intersection"
tiebreak = "first"     # or "blank"

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_aggregate.py --config configs/aggregate.py
```

**What it does**:
- Loads all input CSVs and groups rows by question ID
- For each question, selects the most frequent `answer_value` across all files
- Aggregates `ref_id` from rows with the winning answer

### Options

| Setting | Values | Description |
|---------|--------|-------------|
| `ref_mode` | `"union"` (default), `"intersection"` | How to combine ref_ids from matching answers |
| `tiebreak` | `"first"` (default), `"blank"` | What to do when all answers differ |

### Tiebreak modes

**`tiebreak = "first"`** (default):
- When all CSVs give different answers, use the first CSV's answer
- Useful when you trust earlier runs more

**`tiebreak = "blank"`**:
- When all CSVs give different answers, set all fields to `is_blank`
- Conservative approach when uncertain

### Example workflow

```python
# Create multiple answer configs with different models
# configs/sweeps/gpt4o_mini.py, configs/sweeps/gpt4o.py, configs/sweeps/claude.py

# Run them (or use workflow for parallel execution)
kogine run scripts/wattbot_answer.py --config configs/sweeps/gpt4o_mini.py
kogine run scripts/wattbot_answer.py --config configs/sweeps/gpt4o.py
kogine run scripts/wattbot_answer.py --config configs/sweeps/claude.py

# Aggregate results
kogine run scripts/wattbot_aggregate.py --config configs/aggregate.py

# Validate aggregated results
kogine run scripts/wattbot_validate.py --config configs/validate.py
```

## 10. Using KohakuEngine Configs

All scripts support KohakuEngine configuration files for reproducible, version-controlled experiments.

### Quick Start

```bash
# Run with a config file
kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py

# Run a workflow
python configs/workflows/text_pipeline.py
```

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

### Available Config Examples

```
configs/
├── text_only/
│   ├── index.py            # Text-only indexing config
│   └── answer.py           # Text-only answer config
├── with_images/
│   ├── caption.py          # Image captioning config
│   ├── index.py            # Image-enhanced indexing config
│   ├── image_index.py      # Image-only index config
│   └── answer.py           # Image-enhanced answer config
├── fetch.py                # Document fetching config
├── validate.py             # Validation config
├── aggregate.py            # Aggregation config
├── stats.py                # Statistics config
├── demo_query.py           # Demo query config
├── inspect_node.py         # Node inspection config
└── smoke.py                # Smoke test config

workflows/                      # Runnable workflow scripts
├── text_pipeline.py            # Full fetch→index→answer→validate
├── with_image_pipeline.py      # Full image pipeline
└── ensemble_runner.py          # Parallel models + aggregation
```

### Running Sweeps

Generate multiple configs from one file for hyperparameter sweeps:

```python
# configs/sweeps/model_sweep.py
from kohakuengine import Config

base_config = {
    "db": "artifacts/wattbot.db",
    "questions": "data/test_Q.csv",
    "top_k": 6,
}

models = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"]

def config_gen():
    for model in models:
        config = base_config.copy()
        config["model"] = model
        config["output"] = f"artifacts/results/{model.replace('/', '_')}.csv"
        yield Config(globals_dict=config)
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

    # Use use_subprocess=True for scripts that use asyncio to avoid event loop errors
    flow = Flow(scripts, mode="sequential", use_subprocess=True)
    flow.run()
```

**Important notes:**

- **Use `use_subprocess=True` for asyncio scripts**: KohakuRAG scripts use `asyncio`. When running them via `Flow` or `Script.run()`, set `use_subprocess=True` to avoid "event loop is closed" errors. This runs each script in a separate Python process.

- **`max_workers` controls parallelism**: When using `mode="parallel"`, the `max_workers` parameter limits concurrent subprocess execution. Defaults to CPU count if not specified.

### Ensemble/Voting Workflow

Run multiple models and aggregate with majority voting:

```bash
# This runs:
# 1. Multiple models in parallel
# 2. Aggregates results with wattbot_aggregate.py
# 3. Validates aggregated predictions
python workflows/ensemble_runner.py
```

### Benefits

- **Reproducible**: Config files are version-controlled Python
- **Composable**: Chain scripts into workflows
- **Parallel**: Run sweeps and ensembles concurrently
- **No code changes**: Scripts work with configs only
