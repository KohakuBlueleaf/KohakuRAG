# Usage Workflow

All commands assume you've activated the local virtual environment (e.g., `source .venv/bin/activate`). The scripts live under `scripts/` and expect the repository root as the working directory.

## 1. Download and parse PDFs
Convert every WattBot source PDF into a structured JSON payload:
```bash
python scripts/wattbot_fetch_docs.py \
  --metadata data/metadata.csv \
  --pdf-dir artifacts/raw_pdfs \
  --output-dir artifacts/docs
```
Use `--limit 5` during dry runs to fetch only a few documents, and add `--force-download` if you want to refresh already downloaded PDFs.

## 2. Build the KohakuVault index
Embed the structured payloads and store them in `artifacts/wattbot.db`:
```bash
python scripts/wattbot_build_index.py \
  --metadata data/metadata.csv \
  --docs-dir artifacts/docs \
  --db artifacts/wattbot.db \
  --table-prefix wattbot
```
If you only want to test the pipeline without PDFs, add `--use-citations` to index the citation text from `metadata.csv`.

## 3. Run a retrieval sanity check
Given a question, print the top matches and context snippets:
```bash
python scripts/wattbot_demo_query.py \
  --db artifacts/wattbot.db \
  --table-prefix wattbot \
  --question "How much water does GPT-3 training consume?"
```

## 4. Inspect a stored node
Fetch the raw text/metadata for any node ID (e.g., a paragraph):
```bash
python scripts/wattbot_inspect_node.py \
  --db artifacts/wattbot.db \
  --table-prefix wattbot \
  --node-id amazon2023:sec3:p12
```
Use `--add-note "text"` to append a developer note into the node metadata.

## 5. Snapshot index statistics
Summarize document, paragraph, and sentence counts:
```bash
python scripts/wattbot_stats.py \
  --db artifacts/wattbot.db \
  --table-prefix wattbot
```

## 6. Generate WattBot answers
Run the full RAG pipeline (requires `OPENAI_API_KEY`) and produce a Kaggle-style CSV:
```bash
python scripts/wattbot_answer.py \
  --db artifacts/wattbot.db \
  --table-prefix wattbot \
  --questions data/test_Q.csv \
  --output artifacts/wattbot_answers.csv \
  --model gpt-4o-mini \
  --top-k 6
```
The script streams progress and writes the final answers to `artifacts/wattbot_answers.csv`.
