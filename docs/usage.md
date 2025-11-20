# Usage Workflow

All commands assume you've activated the local virtual environment (e.g., `source .venv/bin/activate`). The scripts live under `scripts/` and expect the repository root as the working directory.

## 🔀 Three Retrieval Modes

KohakuRAG supports **three retrieval modes**:

| Mode | Description | Database | Flags |
|------|-------------|----------|-------|
| **1️⃣ Text-Only** | Standard RAG with text content only | `wattbot_text_only.db` | None |
| **2️⃣ Text + Images (Tree)** | Images in main hierarchy, extracted from sections | `wattbot_with_images.db` | `--with-images` |
| **3️⃣ Text + Images (Dedicated)** | Mode 2 + separate image-only retrieval | `wattbot_with_images.db` | `--with-images --top-k-images 3` |

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

Convert every WattBot source PDF into a structured JSON payload:
```bash
python scripts/wattbot_fetch_docs.py --metadata data/metadata.csv --pdf-dir artifacts/raw_pdfs --output-dir artifacts/docs --limit 10  # Optional: test with 10 docs first
```

**What it does**:
- Downloads PDFs from URLs in metadata.csv
- Extracts text and creates hierarchical structure
- Detects images and creates placeholder entries (not yet captioned)
- Saves to `artifacts/docs/*.json`

Use `--limit 5` during dry runs to fetch only a few documents, and add `--force-download` if you want to refresh already downloaded PDFs.

---

## 1b. Add image captions (🖼️ OPTIONAL - For Image-Enhanced Path Only)

> **📝 Skip this step for text-only workflow!**

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

```bash
python scripts/wattbot_add_image_captions.py --docs-dir artifacts/docs --pdf-dir artifacts/raw_pdfs --output-dir artifacts/docs_with_images --db artifacts/wattbot_with_images.db --vision-model qwen/qwen3-vl-235b-a22b-instruct --max-concurrent 5 --limit 10  # Start with 10 docs for testing
```

**What it does** (4-phase parallel processing):
- **Phase 1**: Reads ALL images from ALL PDFs concurrently (ThreadPoolExecutor)
- **Phase 2**: Compresses ALL images to JPEG (≤1024px, 95% quality) in parallel
- **Phase 3**: Generates captions for ALL images concurrently via vision API
- **Phase 4**: Stores compressed images + updates JSONs
- **Images stored in `wattbot_with_images.db` (table: image_blobs)**
- Caption format: `[img:name WxH] AI-generated caption...`
- Saves to `artifacts/docs_with_images/*.json`

**Important**: Use the SAME `--db` path for `wattbot_build_index.py` in step 2. Images and RAG nodes share the same database file (different tables).

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

### 📝 Text-Only Path

Embed the structured payloads (from step 1a) and store in a text-only database:
```bash
python scripts/wattbot_build_index.py --metadata data/metadata.csv --docs-dir artifacts/docs --db artifacts/wattbot_text_only.db --table-prefix wattbot_text
```

**Output**: `artifacts/wattbot_text_only.db` (~130 MB for 50 docs)

### 🖼️ Image-Enhanced Path

Embed the image-captioned payloads (from step 1b) and store in a separate database:
```bash
python scripts/wattbot_build_index.py --metadata data/metadata.csv --docs-dir artifacts/docs_with_images --db artifacts/wattbot_with_images.db --table-prefix wattbot_img
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

**Note**: If you only want to test the pipeline without PDFs, add `--use-citations` to index the citation text from `metadata.csv`.

---

## 2b. Build image-only index (🖼️ OPTIONAL - For Mode 3 Only)

> **Skip this for Mode 1 (text-only) and Mode 2 (images in tree)!**

After building the image-enhanced index (step 2, image path), optionally add a **dedicated image-only vector table** for guaranteed image retrieval:

```bash
python scripts/wattbot_build_image_index.py --db artifacts/wattbot_with_images.db --table-prefix wattbot_img
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

Now you can use --top-k-images flag with wattbot_answer.py
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

### 1️⃣ Mode 1: Text-Only
```bash
python scripts/wattbot_demo_query.py --db artifacts/wattbot_text_only.db --table-prefix wattbot_text --question "How much water does GPT-3 training consume?" --top-k 5
```

### 2️⃣ Mode 2: Text + Images (Tree)
```bash
python scripts/wattbot_demo_query.py --db artifacts/wattbot_with_images.db --table-prefix wattbot_img --question "What does Figure 3 show about GPU power consumption?" --top-k 5 --with-images
```

### 3️⃣ Mode 3: Text + Images (Dedicated)
```bash
python scripts/wattbot_demo_query.py --db artifacts/wattbot_with_images.db --table-prefix wattbot_img --question "What does Figure 3 show about GPU power consumption?" --top-k 5 --with-images --top-k-images 3
```

**Mode 3 output includes**:
```
Referenced media (3 images):
  [1] nvidia2024 page 3, image 1
      [img:Fig3 800x600] Line graph comparing power consumption across GPU generations...
```

## 4. Inspect a stored node

Fetch the raw text/metadata for any node ID (e.g., a paragraph). Works with both paths:

```bash
# Text-only path
python scripts/wattbot_inspect_node.py --db artifacts/wattbot_text_only.db --table-prefix wattbot_text --node-id amazon2023:sec3:p12

# Image-enhanced path
python scripts/wattbot_inspect_node.py --db artifacts/wattbot_with_images.db --table-prefix wattbot_img --node-id amazon2023:sec3:p12
```

Use `--add-note "text"` to append a developer note into the node metadata.

## 5. Snapshot index statistics

Summarize document, paragraph, and sentence counts. Works with both paths:

```bash
# Text-only path
python scripts/wattbot_stats.py --db artifacts/wattbot_text_only.db --table-prefix wattbot_text

# Image-enhanced path
python scripts/wattbot_stats.py --db artifacts/wattbot_with_images.db --table-prefix wattbot_img
```

## 6. Generate WattBot answers

Run the full RAG pipeline (requires `OPENAI_API_KEY`) and produce a Kaggle-style CSV. The script:
- Reads questions from `data/test_Q.csv` (or any compatible file)
- Uses the `answer_unit` column as known metadata (not predicted)
- Calls the core RAG pipeline to get `answer`, `answer_value`, `ref_id`, `explanation`
- Resolves `ref_url` and `supporting_materials` from `data/metadata.csv` using the chosen `ref_id` values
- **Automatically handles OpenAI rate limits** with intelligent retry logic

### 📝 Text-Only Path

Standard retrieval without image captions:

```bash
python scripts/wattbot_answer.py --db artifacts/wattbot_text_only.db --table-prefix wattbot_text --questions data/test_Q.csv --output artifacts/text_only_answers.csv --model gpt-4o-mini --top-k 6 --max-concurrent 10 --max-retries 2
```

### 🖼️ Mode 2: Text + Images (Tree)

**Add the `--with-images` flag** to extract images from retrieved sections:

```bash
python scripts/wattbot_answer.py --db artifacts/wattbot_with_images.db --table-prefix wattbot_img --questions data/test_Q.csv --output artifacts/mode2_answers.csv --model gpt-4o-mini --top-k 6 --max-concurrent 10 --max-retries 2 --with-images  # ← Extract images from retrieved sections
```

### 🖼️ Mode 3: Text + Images (Dedicated)

**Add `--with-images --top-k-images 3`** to guarantee image retrieval:

```bash
python scripts/wattbot_answer.py --db artifacts/wattbot_with_images.db --table-prefix wattbot_img --questions data/test_Q.csv --output artifacts/mode3_answers.csv --model gpt-4o-mini --top-k 6 --max-concurrent 10 --max-retries 2 --with-images --top-k-images 3  # ← Additionally retrieve 3 images from image-only index
```

**Mode differences**:
- **Mode 2**: Images retrieved only if their section is in top-k text results
- **Mode 3**: **Guarantees** 3 most relevant images, even if their sections weren't retrieved

**What `--with-images` does**:
- Retrieves images from sections containing matched text (always)
- If `--top-k-images > 0`: additionally searches image-only index
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

Run all three and compare accuracy:

```bash
# Mode 1: Text-only
python scripts/wattbot_answer.py --db artifacts/wattbot_text_only.db --output artifacts/mode1_answers.csv

# Mode 2: Images from sections
python scripts/wattbot_answer.py --db artifacts/wattbot_with_images.db --output artifacts/mode2_answers.csv --with-images

# Mode 3: Images from sections + dedicated image index
python scripts/wattbot_answer.py --db artifacts/wattbot_with_images.db --output artifacts/mode3_answers.csv --with-images --top-k-images 3  # Guarantees 3 images

# Compare accuracy
python scripts/wattbot_validate.py --pred artifacts/mode1_answers.csv
python scripts/wattbot_validate.py --pred artifacts/mode2_answers.csv
python scripts/wattbot_validate.py --pred artifacts/mode3_answers.csv

# Expected results:
# Mode 1: 0.78 (baseline)
# Mode 2: 0.82 (+5%)
# Mode 3: 0.84 (+8%) - best for visual-heavy questions
```

### Key Flags

- **`--with-images`**: Enable image-aware retrieval (extracts images from retrieved sections). Works with image-captioned databases.
- **`--top-k-images`**: Number of ADDITIONAL images from image-only index (default: 0). Requires running `wattbot_build_image_index.py` first.
  - `0`: Only extract images from sections (Mode 2)
  - `3`: Guarantee 3 most relevant images (Mode 3)
- `--max-concurrent`: Maximum concurrent API requests (default: 10). Set to 0 for unlimited.
- `--planner-model`: Model for generating additional retrieval queries (defaults to `--model`).
- `--planner-max-queries`: Total retrieval queries per question.
- `--metadata`: Path to `metadata.csv` for resolving citations.
- `--max-retries`: Extra attempts when model returns `is_blank`.

The script writes each answered row to `--output` as soon as it finishes (streaming results via async generator), so you can inspect partial results while a long run is still in progress.

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
```bash
python scripts/wattbot_answer.py --max-concurrent 5 \   # Limit concurrent requests
    --model gpt-4o-mini --top-k 4              # Reduce tokens per request
```

**Higher TPM accounts (e.g., 2M+ TPM):**
```bash
python scripts/wattbot_answer.py --max-concurrent 20 \  # More concurrent requests
    --model gpt-4o --top-k 10
```

**Self-hosted or unlimited endpoints:**
```bash
python scripts/wattbot_answer.py --max-concurrent 0 \   # Unlimited concurrency
    --model local-llama --top-k 10
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

1. **Start conservative**: Use `--max-concurrent 5` for your first run to understand your rate limits
2. **Monitor the logs**: Watch for retry messages to gauge how often you're hitting limits
3. **Scale up gradually**: Increase `--max-concurrent` until you start seeing frequent retries, then back off
4. **Use batch processing windows**: Run large jobs during off-peak hours to maximize throughput
5. **Leverage async concurrency**: All scripts use `asyncio.gather()` for efficient concurrent processing
6. **Switch backends via config**: To move from OpenAI-hosted models to self-hosted vLLM/llama.cpp or OpenAI-compatible proxies for Anthropic/Gemini, configure `OPENAI_BASE_URL` / `OPENAI_API_KEY` as described in `docs/deployment.md`.

## 7. Validate against the labeled training set

Because the public `data/test_Q.csv` file has no answers, use `data/train_QA.csv` as a proxy leaderboard to measure progress locally.

### 📝 Text-Only Path

```bash
# Generate predictions
python scripts/wattbot_answer.py --db artifacts/wattbot_text_only.db --table-prefix wattbot_text --questions data/train_QA.csv --output artifacts/text_only_train_preds.csv --model gpt-4o-mini --top-k 6 --max-concurrent 10 --max-retries 2

# Validate
python scripts/wattbot_validate.py --pred artifacts/text_only_train_preds.csv --show-errors 5 --verbose

# Example output:
# WattBot score: 0.7812
```

### 🖼️ Image-Enhanced Path

```bash
# Generate predictions with images
python scripts/wattbot_answer.py --db artifacts/wattbot_with_images.db --table-prefix wattbot_img --questions data/train_QA.csv --output artifacts/with_images_train_preds.csv --model gpt-4o-mini --top-k 6 --max-concurrent 10 --max-retries 2 --with-images  # ← Important!

# Validate
python scripts/wattbot_validate.py --pred artifacts/with_images_train_preds.csv --show-errors 5 --verbose

# Example output:
# WattBot score: 0.8245 (+5.4% improvement!)
```

### Comparing Results

```bash
# Show side-by-side comparison
echo "Text-only:"
python scripts/wattbot_validate.py --pred artifacts/text_only_train_preds.csv --verbose

echo -e "\nWith images:"
python scripts/wattbot_validate.py --pred artifacts/with_images_train_preds.csv --verbose
```

The validation command compares predictions to ground truth using the official WattBot scoring recipe (answer accuracy, citation overlap, and NA handling). Iterate here until the validation score looks good, then switch the `--questions` flag back to `data/test_Q.csv` to produce the submission file.

All questions are processed concurrently via `asyncio.gather()`, with results streaming to the output file as they complete.

## 8. Single-question debug mode

For debugging prompt/parse issues on a single row (for example, a specific id in `train_QA.csv`), use the `--single-run-debug` flag.

### 📝 Text-Only Path

```bash
python scripts/wattbot_answer.py --db artifacts/wattbot_text_only.db --table-prefix wattbot_text --questions data/train_QA.csv --output artifacts/debug_single.csv --model gpt-4o-mini --top-k 6 --max-retries 2 --single-run-debug --question-id q054
```

### 🖼️ Image-Enhanced Path

```bash
python scripts/wattbot_answer.py --db artifacts/wattbot_with_images.db --table-prefix wattbot_img --questions data/train_QA.csv --output artifacts/debug_single_with_images.csv --model gpt-4o-mini --top-k 6 --max-retries 2 --single-run-debug --question-id q054 --with-images  # ← Add for image-aware debugging
```

This mode:
- Processes only one question from the input CSV (by default the first row, or the row whose `id` matches `--question-id`)
- Logs every retry attempt with its `top_k` value
- **Shows the full prompt** including any "Referenced media" section (when using `--with-images`)
- Logs the exact user prompt sent to the model for each attempt
- Logs the raw model output and the parsed structured answer for each attempt
- Automatically handles context overflow (400 errors) by reducing `top_k` recursively
- Writes a single prediction row to `--output` so you can inspect all intermediate steps end-to-end

**Use this to**:
- Debug why a specific question fails
- Inspect exactly what context (text + images) the LLM sees
- Verify image captions are being retrieved correctly

## 9. Aggregate multiple result files

When you have multiple result CSVs from different runs (e.g., different models, parameters, or random seeds), aggregate them using majority voting:

```bash
python scripts/wattbot_aggregate.py \
    artifacts/results/*.csv \
    -o artifacts/aggregated_preds.csv \
    --ref-mode union \
    --tiebreak first
```

**What it does**:
- Loads all input CSVs and groups rows by question ID
- For each question, selects the most frequent `answer_value` across all files
- Aggregates `ref_id` from rows with the winning answer

### Options

| Flag | Values | Description |
|------|--------|-------------|
| `--ref-mode` | `union` (default), `intersection` | How to combine ref_ids from matching answers |
| `--tiebreak` | `first` (default), `blank` | What to do when all answers differ |

### Tiebreak modes

**`--tiebreak first`** (default):
- When all CSVs give different answers, use the first CSV's answer
- Useful when you trust earlier runs more

**`--tiebreak blank`**:
- When all CSVs give different answers, set all fields to `is_blank`
- Conservative approach when uncertain

### Example workflow

```bash
# Run multiple models
python scripts/wattbot_answer.py --model gpt-4o-mini --output artifacts/results/gpt4o_mini.csv ...
python scripts/wattbot_answer.py --model gpt-4o --output artifacts/results/gpt4o.csv ...
python scripts/wattbot_answer.py --model claude-3-5-sonnet --output artifacts/results/claude.csv ...

# Aggregate results
python scripts/wattbot_aggregate.py \
    artifacts/results/*.csv \
    -o artifacts/ensemble_preds.csv \
    --ref-mode union

# Validate aggregated results
python scripts/wattbot_validate.py --pred artifacts/ensemble_preds.csv --verbose
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
│   └── answer.py           # Text-only answer config
├── with_images/
│   └── answer.py           # Image-enhanced answer config
├── sweeps/
│   └── model_sweep.py      # Compare multiple models
└── workflows/
    ├── text_pipeline.py    # Full fetch→index→answer→validate
    └── ensemble_runner.py  # Parallel models + aggregation
```

### Running Sweeps

Generate multiple configs from one file for hyperparameter sweeps:

```bash
# Run model comparison in parallel
kogine workflow parallel scripts/wattbot_answer.py \
    --config configs/sweeps/model_sweep.py \
    --workers 3
```

### Workflow Orchestration

Chain multiple scripts with the Flow API:

```python
# configs/workflows/my_workflow.py
from kohakuengine import Config, Script, Flow

fetch_config = Config(globals_dict={"metadata": "data/metadata.csv", ...})
answer_config = Config(globals_dict={"db": "artifacts/wattbot.db", ...})

if __name__ == "__main__":
    scripts = [
        Script("scripts/wattbot_fetch_docs.py", config=fetch_config),
        Script("scripts/wattbot_answer.py", config=answer_config),
    ]

    flow = Flow(scripts, mode="sequential")
    flow.run()
```

### Ensemble/Voting Workflow

Run multiple models and aggregate with majority voting:

```bash
# This runs:
# 1. Multiple models in parallel
# 2. Aggregates results with wattbot_aggregate.py
# 3. Validates aggregated predictions
python configs/workflows/ensemble_runner.py
```

### Benefits

- **Reproducible**: Config files are version-controlled Python
- **Composable**: Chain scripts into workflows
- **Parallel**: Run sweeps and ensembles concurrently
- **No code changes**: Scripts work with both CLI args and configs
