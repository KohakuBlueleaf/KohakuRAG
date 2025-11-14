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
    --pred artifacts/wattbot_train_preds.csv
```

The validation script compares your predictions to the ground truth using the official WattBot score recipe (0.75 × answer_value, 0.15 × ref_id, 0.10 × NA handling). Use `--show-errors 5` to print the lowest-scoring rows and inspect which answers, citations, or NA flags need attention.

## Configuring LLM and embeddings
- Set `OPENAI_API_KEY` for production runs. The scripts automatically choose the OpenAI chat backend when the key is available; otherwise they fall back to a lightweight mock useful for unit tests.
- Every environment uses `jinaai/jina-embeddings-v3` via `JinaEmbeddingModel`. Alternate encoders can implement the `EmbeddingModel` protocol, but we recommend sticking with Jina to keep embedding behavior aligned.

## Testing checklist
- `scripts/wattbot_fetch_docs.py --limit 2`: downloads/converts a couple of PDFs and emits JSON payloads into `artifacts/docs/`.
- `scripts/wattbot_build_index.py --docs-dir artifacts/docs --db artifacts/dev_index.db`: builds a mini index from those structured files.
- `scripts/wattbot_demo_query.py --db artifacts/dev_index.db --question "How much water does GPT-3 training consume?"`: prints retrieved snippets and the mock answer.
- `python -m unittest tests.test_pipeline`: runs the regression tests over the mock LLM stack while still exercising the Jina embedding pipeline (patched via fixtures when needed).

These steps ensure the RAG pipeline works end-to-end before spending tokens on real OpenAI calls.
