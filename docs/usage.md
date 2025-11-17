# Usage Workflow

All commands assume you've activated the local virtual environment (e.g., `source .venv/bin/activate`). The scripts live under `scripts/` and expect the repository root as the working directory.

## 1. Download and parse PDFs
Convert every WattBot source PDF into a structured JSON payload:
```bash
python scripts/wattbot_fetch_docs.py --metadata data/metadata.csv --pdf-dir artifacts/raw_pdfs --output-dir artifacts/docs
```
Use `--limit 5` during dry runs to fetch only a few documents, and add `--force-download` if you want to refresh already downloaded PDFs.

## 2. Build the KohakuVault index
Embed the structured payloads and store them in `artifacts/wattbot.db`:
```bash
python scripts/wattbot_build_index.py --metadata data/metadata.csv --docs-dir artifacts/docs --db artifacts/wattbot.db --table-prefix wattbot
```
If you only want to test the pipeline without PDFs, add `--use-citations` to index the citation text from `metadata.csv`.

## 3. Run a retrieval sanity check
Given a question, print the top matches and context snippets:
```bash
python scripts/wattbot_demo_query.py --db artifacts/wattbot.db --table-prefix wattbot --question "How much water does GPT-3 training consume?"
```

## 4. Inspect a stored node
Fetch the raw text/metadata for any node ID (e.g., a paragraph):
```bash
python scripts/wattbot_inspect_node.py --db artifacts/wattbot.db --table-prefix wattbot --node-id amazon2023:sec3:p12
```
Use `--add-note "text"` to append a developer note into the node metadata.

## 5. Snapshot index statistics
Summarize document, paragraph, and sentence counts:
```bash
python scripts/wattbot_stats.py --db artifacts/wattbot.db --table-prefix wattbot
```

## 6. Generate WattBot answers
Run the full RAG pipeline (requires `OPENAI_API_KEY`) and produce a Kaggle-style CSV. The script:
- Reads questions from `data/test_Q.csv` (or any compatible file),
- Uses the `answer_unit` column as known metadata (not predicted),
- Calls the core RAG pipeline to get `answer`, `answer_value`, `ref_id`, `explanation`,
- Resolves `ref_url` and `supporting_materials` from `data/metadata.csv` using the chosen `ref_id` values.
- **Automatically handles OpenAI rate limits** with intelligent retry logic

```bash
python scripts/wattbot_answer.py --db artifacts/wattbot.db --table-prefix wattbot --questions data/test_Q.csv --output artifacts/wattbot_answers.csv --model gpt-4o-mini --top-k 6 --max-concurrent 10 --max-retries 2
```

### Key Flags

- `--max-concurrent`: Maximum number of concurrent API requests (default: 10). **Tune based on your OpenAI rate limits**. Set to 0 for unlimited concurrency (self-hosted endpoints).
- `--planner-model`: Optional model used to generate additional retrieval queries (defaults to `--model`).
- `--planner-max-queries`: Total number of retrieval queries per question (original user question + LLM-generated queries).
- `--metadata`: Path to `metadata.csv` (defaults to `data/metadata.csv`) for resolving `ref_url` and `supporting_materials`.
- `--max-retries`: Number of extra attempts to make per question when the model returns `is_blank`, each time retrieving a larger set of snippets.

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
python scripts/wattbot_answer.py \
    --max-concurrent 5 \   # Limit concurrent requests
    --model gpt-4o-mini \
    --top-k 4              # Reduce tokens per request
```

**Higher TPM accounts (e.g., 2M+ TPM):**
```bash
python scripts/wattbot_answer.py \
    --max-concurrent 20 \  # More concurrent requests
    --model gpt-4o \
    --top-k 10
```

**Self-hosted or unlimited endpoints:**
```bash
python scripts/wattbot_answer.py \
    --max-concurrent 0 \   # Unlimited concurrency
    --model local-llama \
    --top-k 10
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
Because the public `data/test_Q.csv` file has no answers, use `data/train_QA.csv` as a proxy leaderboard to measure progress locally:
```bash
python scripts/wattbot_answer.py --db artifacts/wattbot.db --table-prefix wattbot --questions data/train_QA.csv --output artifacts/wattbot_train_preds.csv --model gpt-4o-mini --top-k 6 --max-concurrent 10 --max-retries 2

python scripts/wattbot_validate.py --pred artifacts/wattbot_train_preds.csv --show-errors 5
```
The first command runs the exact same pipeline as a test submission but over the labeled questions; the second command compares the predictions to the ground truth using the official WattBot scoring recipe (answer accuracy, citation overlap, and NA handling). Iterate here until the validation score looks good, then switch the `--questions` flag back to `data/test_Q.csv` to produce the submission file.

All questions are processed concurrently via `asyncio.gather()`, with results streaming to the output file as they complete.

## 8. Single-question debug mode
For debugging prompt/parse issues on a single row (for example, a specific id in `train_QA.csv`), use the `--single-run-debug` flag:
```bash
python scripts/wattbot_answer.py --db artifacts/wattbot.db --table-prefix wattbot --questions data/train_QA.csv --output artifacts/wattbot_train_single.csv --model gpt-4o-mini --top-k 6 --max-retries 2 --single-run-debug --question-id q054
```
This mode:
- Processes only one question from the input CSV (by default the first row, or the row whose `id` matches `--question-id`),
- Logs every retry attempt with its `top_k` value,
- Logs the exact user prompt sent to the model for each attempt,
- Logs the raw model output and the parsed structured answer for each attempt,
- Automatically handles context overflow (400 errors) by reducing `top_k` recursively,
- Writes a single prediction row to `--output` so you can inspect all intermediate steps end-to-end.
