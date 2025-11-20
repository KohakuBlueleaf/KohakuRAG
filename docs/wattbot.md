# WattBot 2025 Playbook

This guide ties the general KohakuRAG architecture to the specifics of the WattBot 2025 Kaggle competition.

## Repository layout
- `data/metadata.csv` — bibliography of the reference documents.
- `data/train_QA.csv` — labeled examples showing the expected CSV output format.
- `data/test_Q.csv` — unlabeled questions to be answered for submission.
- `src/kohakurag/` — reusable library (datastore, indexing, RAG pipeline).
- `scripts/` — WattBot-focused utilities (document parsing, indexing, inference, submission helpers).
- `docs/` — design and operations documentation.

## Indexing flow
1. Run `scripts/wattbot_fetch_docs.py` to download PDFs (where permissible) and convert them into structured JSON payloads under `artifacts/docs/`. Each payload stores per-page sections, paragraphs, sentences, and placeholder captions for figures.
2. Run `scripts/wattbot_build_index.py` to read those JSON files (or fall back to citation text when `--use-citations` is set). The script:
   - Loads the metadata table to enrich payloads with titles/URLs.
   - Uses the `DocumentIndexer` to build document → section → paragraph → sentence nodes with embeddings.
   - Stores the nodes in `artifacts/wattbot.db` via `KVaultNodeStore`.
3. Run `scripts/wattbot_demo_query.py` to sanity check that questions retrieve relevant snippets.

## Answering questions
1. Point `scripts/wattbot_answer.py` at the built index and a CSV of questions (`data/train_QA.csv` or `data/test_Q.csv`).
2. The script loads the datastore, spins up the `RAGPipeline`, and processes each row:
   - Plans queries/keywords for the question.
   - Retrieves top-k snippets with hierarchical expansion.
   - Calls the configured LLM (OpenAI by default) with a structured prompt and the retrieved context.
   - Parses the JSON response, fills in missing fields (e.g., `answer_value`), and writes out a Kaggle-ready CSV.
3. Review the generated explanations/supporting materials to ensure the answers are grounded.

## Validating against the training set
Use the labeled `data/train_QA.csv` file to sanity-check your pipeline before submitting to Kaggle:

```bash
python scripts/wattbot_answer.py \
    --db artifacts/wattbot.db \
    --table-prefix wattbot \
    --questions data/train_QA.csv \
    --output artifacts/wattbot_train_preds.csv \
    --model gpt-5.1 \
    --top-k 6

python scripts/wattbot_validate.py \
    --pred artifacts/wattbot_train_preds.csv \
    --verbose
```

The validation script compares your predictions to the ground truth using the official WattBot score recipe (0.75 × answer_value, 0.15 × ref_id, 0.10 × NA handling). Use `--show-errors 5` to print the lowest-scoring rows and inspect which answers, citations, or NA flags need attention. Add `--verbose` for detailed per-question output.

## Aggregating multiple results

When you have multiple result CSVs (e.g., from different models or runs), use the aggregation script to combine them using majority voting:

```bash
python scripts/wattbot_aggregate.py \
    artifacts/results/*.csv \
    -o artifacts/aggregated_preds.csv \
    --ref-mode union \
    --tiebreak first
```

**Options:**
- `--ref-mode union` (default): Combine ref_ids from all matching answers
- `--ref-mode intersection`: Only keep ref_ids that appear in all matching answers
- `--tiebreak first` (default): When all answers differ, use the first CSV's answer
- `--tiebreak blank`: When all answers differ, output `is_blank`

The script selects the most frequent `answer_value` for each question across all input CSVs, then aggregates reference IDs from the rows that had the winning answer.

## Configuring LLM and embeddings

### OpenAI Configuration
- Set `OPENAI_API_KEY` for production runs. The scripts automatically choose the OpenAI chat backend when the key is available; otherwise they fall back to a lightweight mock useful for unit tests.
- **Async/await architecture** — all I/O operations (API calls, embeddings, database) use async for efficient concurrent processing
- **Rate limit handling is automatic** — the `OpenAIChatModel` class includes:
  - Semaphore-based concurrency control via `max_concurrent` parameter
  - Intelligent retry logic that parses server-recommended wait times
  - Falls back to exponential backoff if no specific delay is provided
  - Prints clear retry messages for monitoring
  - Continues processing without manual intervention

**Recommended configuration for WattBot 2025:**
```bash
# For 500K TPM accounts (common for gpt-4o-mini)
python scripts/wattbot_answer.py \
    --max-concurrent 5 \      # Limit concurrent API requests
    --model gpt-4o-mini \
    --top-k 6

# For higher TPM accounts or self-hosted endpoints
python scripts/wattbot_answer.py \
    --max-concurrent 20 \     # More concurrent requests
    --model gpt-4o-mini \
    --top-k 8

# For unlimited concurrency (use with caution)
python scripts/wattbot_answer.py \
    --max-concurrent 0 \      # No rate limiting
    --model gpt-4o-mini \
    --top-k 6
```

### Embedding Configuration
- Every environment uses `jinaai/jina-embeddings-v3` via `JinaEmbeddingModel`. Alternate encoders can implement the `EmbeddingModel` protocol, but we recommend sticking with Jina to keep embedding behavior aligned.
- First run downloads ~2GB model from Hugging Face — set `HF_HOME` if you need a custom cache location.

## Testing checklist
- `scripts/wattbot_fetch_docs.py --limit 2`: downloads/converts a couple of PDFs and emits JSON payloads into `artifacts/docs/`.
- `scripts/wattbot_build_index.py --docs-dir artifacts/docs --db artifacts/dev_index.db`: builds a mini index from those structured files.
- `scripts/wattbot_demo_query.py --db artifacts/dev_index.db --question "How much water does GPT-3 training consume?"`: prints retrieved snippets and the mock answer.
- `python -m unittest tests.test_pipeline`: runs the regression tests over the mock LLM stack while still exercising the Jina embedding pipeline (patched via fixtures when needed).

These steps ensure the RAG pipeline works end-to-end before spending tokens on real OpenAI calls.
