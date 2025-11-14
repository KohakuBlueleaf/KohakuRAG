# KohakuRAG — Simple Hierarchical RAG Framework

KohakuRAG is a general-purpose Retrieval-Augmented Generation (RAG) framework. It ingests long-form documents (PDF, Markdown, or raw text), converts them into hierarchical trees (document → section → paragraph → sentence), and stores them in a single-file SQLite + sqlite-vec datastore via [KohakuVault](https://github.com/KohakuBlueleaf/KohakuVault). The stack is domain-agnostic: we demonstrate it with Kaggle’s WattBot 2025 energy dataset, but the same components apply to any corpus that benefits from structured retrieval.

## Key features
- **Structured ingestion** – parse PDFs (or Markdown/plain text) into `DocumentPayload` objects with per-page sections, paragraph metadata, and sentence-level granularity. Image placeholders preserve figure positioning even when captions are missing.
- **Tree-based embeddings** – leaves (sentences) are embedded with `jinaai/jina-embeddings-v3` and parent nodes inherit averaged vectors so queries can hit any level while keeping context intact.
- **Single-file datastore** – `KVaultNodeStore` stores metadata blobs and sqlite-vec entries side-by-side in `artifacts/wattbot.db`, making it easy to ship or inspect the entire index without extra services.
- **Pluggable LLM orchestration** – `RAGPipeline` handles query planning, retrieval expansion, and prompting. You can start with the mock chat model and swap in OpenAI (or any other provider) for final answers.
- **WattBot utilities** – scripts under `scripts/` download PDFs, build the index, run retrieval sanity checks, inspect nodes, summarize stats, and answer Kaggle questions end-to-end.


# WattBot 2025 Workflow (Example Integration)
```bash
# 1. Download PDFs and convert them into structured JSON
python scripts/wattbot_fetch_docs.py --metadata data/metadata.csv --pdf-dir artifacts/raw_pdfs --output-dir artifacts/docs

# 2. Build the KohakuVault index with hierarchical nodes + embeddings
python scripts/wattbot_build_index.py --metadata data/metadata.csv --docs-dir artifacts/docs --db artifacts/wattbot.db

# 3. Inspect the corpus
python scripts/wattbot_stats.py --db artifacts/wattbot.db
python scripts/wattbot_demo_query.py --db artifacts/wattbot.db --question "How much water does GPT-3 training consume?"

# 4. Answer Kaggle questions with OpenAI
export OPENAI_API_KEY=...  # set once
python scripts/wattbot_answer.py --db artifacts/wattbot.db --questions data/test_Q.csv --output artifacts/wattbot_answers.csv
```
(See `docs/usage.md` for more scenarios: node inspection, dry runs with `--limit`, fallback to citation-only indexing, etc.)

## Architecture highlights
1. **Parsing** – `pdf_to_document_payload` extracts per-page text and image placeholders; `markdown_to_payload` and `text_to_payload` offer alternative entry points. Every payload carries `metadata` (document ids, URLs, years, reference ids).
2. **Indexing** – `DocumentIndexer` walks the tree, embeds sentence leaves via `JinaEmbeddingModel`, averages parent embeddings, and emits `StoredNode` objects.
3. **Storage** – `KVaultNodeStore` writes node blobs into KohakuVault’s key-value table and stage embeddings in sqlite-vec (`VectorKVault`). Retrieval is just brute-force cosine search on the stored vectors.
4. **Retrieval + LLM** – `RAGPipeline` plans queries, retrieves top-k nodes (with parent/child context expansion), and prompts a chat model (mock or OpenAI) to produce structured answers.

## Development notes
- Python 3.10+ (repo uses PEP 563-style `list[str]`/`dict[str, Any]` annotations).
- Dependencies: `torch`, `transformers`, `kohakuvault`, `pypdf`, `requests`, `openai`.
- Jina embeddings are downloaded on first run; set `HF_HOME` if you need a custom cache path.
- Use `.venv` (or another virtual environment) and follow `docs/usage.md` to reproduce the pipeline locally.

## License
Apache-2.0 (see `LICENSE`).
