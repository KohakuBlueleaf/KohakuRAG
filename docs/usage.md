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

```bash
python scripts/wattbot_answer.py --db artifacts/wattbot.db --table-prefix wattbot --questions data/test_Q.csv --output artifacts/wattbot_answers.csv --model gpt-5-mini --top-k 6 --max-workers 4 --max-retries 2
```
Key flags:
- `--max-workers`: how many questions to process in parallel (each worker has its own datastore + LLM client). Tune based on your OpenAI rate limits; `--max-workers 1` runs sequentially.
- `--planner-model`: optional model used to generate additional retrieval queries (defaults to `--model`).
- `--planner-max-queries`: total number of retrieval queries per question (original user question + LLM-generated queries).
- `--metadata`: path to `metadata.csv` (defaults to `data/metadata.csv`) for resolving `ref_url` and `supporting_materials`.
- `--max-retries`: number of extra attempts to make per question when the model returns `is_blank`, each time retrieving a larger set of snippets.

The script writes each answered row to `--output` as soon as it finishes, so you can inspect partial results while a long run is still in progress.

## 7. Validate against the labeled training set
Because the public `data/test_Q.csv` file has no answers, use `data/train_QA.csv` as a proxy leaderboard to measure progress locally:
```bash
python scripts/wattbot_answer.py --db artifacts/wattbot.db --table-prefix wattbot --questions data/train_QA.csv --output artifacts/wattbot_train_preds.csv --model gpt-5-mini --top-k 6 --max-workers 4 --max-retries 2

python scripts/wattbot_validate.py --pred artifacts/wattbot_train_preds.csv --show-errors 5
```
The first command runs the exact same pipeline as a test submission but over the labeled questions; the second command compares the predictions to the ground truth using the official WattBot scoring recipe (answer accuracy, citation overlap, and NA handling). Iterate here until the validation score looks good, then switch the `--questions` flag back to `data/test_Q.csv` to produce the submission file.

## 8. Single-question debug mode
For debugging prompt/parse issues on a single row (for example, a specific id in `train_QA.csv`), use the `--single-run-debug` flag:
```bash
python scripts/wattbot_answer.py --db artifacts/wattbot.db --table-prefix wattbot --questions data/train_QA.csv --output artifacts/wattbot_train_single.csv --model gpt-5-mini --top-k 6 --max-retries 2 --single-run-debug --question-id q054
```
This mode:
- Processes only one question from the input CSV (by default the first row, or the row whose `id` matches `--question-id`),
- Logs every retry attempt with its `top_k` value,
- Logs the exact user prompt sent to the model for each attempt,
- Logs the raw model output and the parsed structured answer for each attempt,
- Writes a single prediction row to `--output` so you can inspect all intermediate steps end-to-end.
