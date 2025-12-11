# KohakuRAG — Simple Hierarchical RAG Framework

<div align="center">

**A simple RAG framework with hierarchical document indexing**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

[Features](#-key-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [WattBot 2025](#-wattbot-2025-example) • [Architecture](#-architecture-overview)

</div>

---

## Overview

KohakuRAG is a **domain-agnostic Retrieval-Augmented Generation (RAG) framework** designed for production use. It transforms long-form documents (PDFs, Markdown, or plain text) into hierarchical knowledge trees and enables intelligent retrieval with context-aware search.

**What makes KohakuRAG different:**
- **Hierarchical structure** preserves document organization (document → section → paragraph → sentence)
- **Smart context expansion** returns not just matched sentences, but their surrounding paragraphs and sections
- **Single-file storage** using SQLite + [KohakuVault](https://github.com/KohakuBlueleaf/KohakuVault) — no external services required
- **Multimodal support** with Jina v3 and Jina v4 embeddings (text + direct image embedding)
- **Rate-limit resilient** with automatic retry and exponential backoff for LLM APIs
- **Ensemble & sweeps** for hyperparameter optimization and model voting
- **Production-tested** on Kaggle's WattBot 2025 competition (energy research corpus)
- **Python-based configuration** via [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine) — no YAML/JSON, fully reproducible experiments

While we demonstrate KohakuRAG with the WattBot 2025 dataset, **the core library is completely domain-agnostic** and can be applied to any document corpus.

---

## Key Features

### Structured Document Ingestion
- Parse **PDFs**, **Markdown**, or **plain text** into structured `DocumentPayload` objects
- Preserve document hierarchy with per-page sections, paragraph metadata, and sentence-level granularity
- Maintain image placeholders to preserve figure positioning even when captions are missing

### Tree-Based Embeddings
- **Jina v3**: 1024-dim text embeddings
- **Jina v4**: Multimodal embeddings with Matryoshka dimensions (128-2048), task-aware modes, and direct image embedding
- **Leaf nodes** (sentences) embedded directly; **parent nodes** inherit averaged vectors from children
- **Multi-level retrieval** — queries can match at any level while preserving full context

### Single-File Datastore
- Built on **SQLite + sqlite-vec** via [KohakuVault](https://github.com/KohakuBlueleaf/KohakuVault)
- **No external dependencies** — entire index stored in one `.db` file
- Easy to version control, backup, and deploy

### Pluggable LLM Orchestration
- **Modular RAG pipeline** with swappable components (planner, retriever, answerer)
- Built-in **OpenAI** and **OpenRouter** integration with automatic rate limit handling
- **Mock chat model** for testing without API costs
- Add your own LLM backend by implementing the `ChatModel` protocol

### Advanced Retrieval Features
- **Multi-query retrieval** with LLM-powered query planning
- **Deduplication** removes duplicate nodes across queries
- **Reranking strategies**: frequency, score, or combined
- **Final truncation** to control context window size

### Ensemble & Hyperparameter Sweeps
- Run **N parallel inferences** and aggregate with majority voting
- **5 aggregation modes**: independent, ref_priority, answer_priority, union, intersection
- **ignore_blank** option to filter failed answers before voting
- **Sweep workflows** for systematic hyperparameter optimization
- **Plotting with std dev** for multi-run experiments

### Production-Ready Features
- **Async/await architecture** for efficient concurrent I/O
- **Automatic rate limit handling** with intelligent retry logic and semaphore-based concurrency control
- **Thread-safe operations** via single-worker executors for embedding and datastore access
- **Structured logging** for debugging and monitoring
- **Validation scripts** for measuring accuracy before deployment

### KohakuEngine Configuration
- **Python-based configs** via [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine) — no YAML/JSON
- **Reproducible experiments** with version-controlled configuration files
- **Workflow orchestration** for chaining multiple scripts (use `use_subprocess=True` for asyncio scripts)
- **Parallel execution** with `max_workers` control for hyperparameter sweeps and model ensembles

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/KohakuBlueleaf/KohakuRAG.git
cd KohakuRAG

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Install KohakuEngine for configuration management
pip install kohakuengine
```

### Basic Usage

#### Programmatic Usage (Async)

```python
import asyncio
from kohakurag import RAGPipeline, OpenAIChatModel, JinaEmbeddingModel, InMemoryNodeStore

async def main():
    # Initialize components
    chat = OpenAIChatModel(model="gpt-4o-mini", max_concurrent=10)
    embedder = JinaEmbeddingModel()
    store = InMemoryNodeStore()
    pipeline = RAGPipeline(chat=chat, embedder=embedder, store=store)

    # Index documents (async I/O)
    await pipeline.index_documents(documents)

    # Single query
    result = await pipeline.run_qa(
        query="What is RAG?",
        system_prompt="You are a helpful assistant.",
        user_template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
    )
    print(result)

    # Batch queries with concurrent execution
    questions = ["Q1", "Q2", "Q3", ...]
    results = await asyncio.gather(*[
        pipeline.run_qa(query=q, system_prompt="...", user_template="...")
        for q in questions
    ])

asyncio.run(main())
```

#### Running Scripts with KohakuEngine

All scripts are configured via Python config files using [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine). No command-line arguments needed.

```bash
# 1. Prepare your documents (PDF/Markdown/Text)
# Place them in a directory or use the WattBot example below

# 2. Build the index (edit configs/text_only/index.py first)
kogine run scripts/wattbot_build_index.py --config configs/text_only/index.py

# 3. Query the index (edit configs/demo_query.py first)
kogine run scripts/wattbot_demo_query.py --config configs/demo_query.py

# 4. Generate answers with OpenAI (edit configs/text_only/answer.py first)
export OPENAI_API_KEY=your_key_here
kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py
```

**Example config file** (`configs/text_only/answer.py`):
```python
from kohakuengine import Config

db = "artifacts/wattbot.db"
table_prefix = "wattbot"
questions = "data/test_Q.csv"
output = "artifacts/answers.csv"
model = "gpt-4o-mini"
top_k = 6
max_concurrent = 10  # Control API rate (0 = unlimited)
max_retries = 2

def config_gen():
    return Config.from_globals()
```

---

## WattBot 2025 Example

KohakuRAG was developed for the [Kaggle WattBot 2025 competition](https://www.kaggle.com/competitions/wattbot-2025), which challenges participants to build a RAG system for answering questions about energy research papers.

### Complete WattBot Workflow

The easiest way to run the full pipeline is using the pre-built workflows:

```bash
# Text-only pipeline (fetch → index → answer → validate)
python workflows/text_pipeline.py

# Image-enhanced pipeline (fetch → caption → index → answer → validate)
python workflows/with_image_pipeline.py

# JinaV4 multimodal pipeline (direct image embeddings)
python workflows/jinav4_pipeline.py

# Ensemble with voting (multiple parallel runs → aggregate)
python workflows/ensemble_runner.py
python workflows/jinav4_ensemble_runner.py
```

### Step-by-Step with Individual Configs

```bash
# 1. Download and parse PDFs into structured JSON
# Edit configs/fetch.py, then:
kogine run scripts/wattbot_fetch_docs.py --config configs/fetch.py

# 2. Build the hierarchical index
# Edit configs/text_only/index.py, then:
kogine run scripts/wattbot_build_index.py --config configs/text_only/index.py

# 3. Verify the index
# Edit configs/stats.py, then:
kogine run scripts/wattbot_stats.py --config configs/stats.py

# Edit configs/demo_query.py, then:
kogine run scripts/wattbot_demo_query.py --config configs/demo_query.py

# 4. Generate answers for Kaggle submission
export OPENAI_API_KEY=sk-...
# Edit configs/text_only/answer.py, then:
kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py

# 5. Validate against training set (optional)
# Edit configs/validate.py, then:
kogine run scripts/wattbot_validate.py --config configs/validate.py
```

**Key Config Parameters:**
- `top_k`: Number of context snippets to retrieve per query
- `max_retries`: Extra attempts when model returns blank answers
- `planner_max_queries`: Total retrieval queries per question (original + LLM-generated)
- `max_concurrent`: Maximum concurrent API requests (default: 10, set to 0 for unlimited)
- `deduplicate_retrieval`: Remove duplicate nodes across multi-query results
- `rerank_strategy`: Rank results by "frequency", "score", or "combined"
- `top_k_final`: Truncate after deduplication and reranking

See [`docs/wattbot.md`](docs/wattbot.md) and [`docs/usage.md`](docs/usage.md) for advanced usage patterns.

---

## Embedding Models

### Jina v3 (Default)
- **Dimensions**: 1024 (fixed)
- **Use case**: Text-only retrieval
- **Config**:
  ```python
  embedding_model = "jina"
  ```

### Jina v4 (Multimodal)
- **Dimensions**: 128, 256, 512, 1024, 2048 (Matryoshka)
- **Tasks**: "retrieval", "text-matching", "code"
- **Features**: Direct image embedding, longer context (32K tokens)
- **Config**:
  ```python
  embedding_model = "jinav4"
  embedding_dim = 1024
  embedding_task = "retrieval"
  ```

See [`docs/jinav4_workflows.md`](docs/jinav4_workflows.md) for detailed JinaV4 usage.

---

## Ensemble & Aggregation

KohakuRAG supports running multiple inferences and aggregating results with majority voting.

### Basic Ensemble Workflow

```bash
# 1. Run N inferences
python workflows/sweeps/ensemble_inference.py --total-runs 16

# 2. Aggregate with different strategies
python workflows/sweeps/ensemble_vs_ref_vote.py
python workflows/sweeps/ensemble_vs_ignore_blank.py

# 3. Plot results with std dev
python workflows/sweeps/sweep_plot.py outputs/sweeps/ensemble_vs_ref_vote
```

### Aggregation Modes

| Mode | Description |
|------|-------------|
| `independent` | Vote ref_id and answer_value separately |
| `ref_priority` | First vote on ref_id, then answer among matching refs |
| `answer_priority` | First vote on answer, then ref among matching answers |
| `union` | Vote on answer, then union all ref_ids from matching rows |
| `intersection` | Vote on answer, then intersect ref_ids from matching rows |

### Aggregation Script

```python
# configs/aggregate.py
inputs = ["run1.csv", "run2.csv", "run3.csv"]
output = "aggregated.csv"
ref_mode = "union"        # Aggregation mode
tiebreak = "first"        # or "blank"
ignore_blank = True       # Filter out is_blank before voting
```

```bash
kogine run scripts/wattbot_aggregate.py --config configs/aggregate.py
```

---

## Hyperparameter Sweeps

KohakuRAG includes sweep workflows for systematic optimization:

| Sweep | Line Parameter | X Parameter |
|-------|---------------|-------------|
| `top_k_vs_embedding.py` | embedding_config | top_k |
| `top_k_vs_rerank.py` | rerank_strategy | top_k |
| `top_k_vs_reorder.py` | use_reordered_prompt | top_k |
| `top_k_vs_max_retries.py` | max_retries | top_k |
| `top_k_vs_top_k_final.py` | top_k_final | top_k |
| `planner_queries_vs_top_k.py` | planner_max_queries | top_k |
| `llm_model_vs_embedding.py` | embedding_config | llm_model |
| `ensemble_vs_ref_vote.py` | ref_vote_mode | ensemble_size |
| `ensemble_vs_tiebreak.py` | tiebreak_mode | ensemble_size |
| `ensemble_vs_ignore_blank.py` | ignore_blank | ensemble_size |

### Running a Sweep

```bash
# Run the sweep
python workflows/sweeps/top_k_vs_embedding.py

# Plot results with mean, std dev, and max lines
python workflows/sweeps/sweep_plot.py outputs/sweeps/top_k_vs_embedding
```

### Sweep Plot Features
- **Solid line**: Mean score across runs
- **Shaded area**: ±1 standard deviation
- **Dashed line**: Maximum score per config
- **Star marker**: Global maximum with label

---

## Image Captioning for Multimodal RAG

KohakuRAG supports **vision model integration** to extract and caption images from PDFs.

### Quick Start

```bash
# 1. Set up OpenRouter
export OPENAI_API_KEY="sk-or-v1-..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# 2. Generate image captions
kogine run scripts/wattbot_add_image_captions.py --config configs/with_images/caption.py

# 3. Build image-enhanced index
kogine run scripts/wattbot_build_index.py --config configs/with_images/index.py

# 4. Build separate image index (for guaranteed image retrieval)
kogine run scripts/wattbot_build_image_index.py --config configs/with_images/image_index.py
```

### Retrieval Modes

| Mode | Description | Config |
|------|-------------|--------|
| **Text-Only** | Standard RAG | `with_images = False` |
| **Text + Images (Tree)** | Images from retrieved sections | `with_images = True` |
| **Text + Images (Dedicated)** | Guaranteed top-k images | `with_images = True, top_k_images = 3` |

See [`docs/image_rag_example.md`](docs/image_rag_example.md) for detailed examples.

---

## Architecture Overview

### High-Level Pipeline

```
Documents (PDF/MD/TXT)
    ↓
Parse into hierarchical payload
    ↓
Build tree structure (doc → section → paragraph → sentence)
    ↓
Embed leaves with Jina, average for parents
    ↓
Store in SQLite + sqlite-vec (KohakuVault)
    ↓
Query → Plan → Retrieve → Dedupe → Rerank → Truncate
    ↓
LLM generates structured answer
```

### Core Components

1. **Parsers** (`src/kohakurag/parsers.py`, `pdf_utils.py`)
   - `pdf_to_document_payload`: Extract text, sections, and image placeholders from PDFs
   - `markdown_to_payload`: Parse Markdown with heading-based structure
   - `text_to_payload`: Simple text ingestion with heuristic segmentation

2. **Embeddings** (`src/kohakurag/embeddings.py`)
   - `JinaEmbeddingModel`: Jina v3 (1024-dim)
   - `JinaV4EmbeddingModel`: Jina v4 (Matryoshka, multimodal)

3. **Indexer** (`src/kohakurag/indexer.py`)
   - Walks document tree and creates nodes for each level
   - Embeds sentences, averages child embeddings for parent nodes

4. **Datastore** (`src/kohakurag/datastore.py`)
   - `KVaultNodeStore`: SQLite-backed storage with metadata and embeddings
   - `ImageStore`: Compressed image blob storage

5. **RAG Pipeline** (`src/kohakurag/pipeline.py`)
   - **Planner**: Generates additional retrieval queries
   - **Retriever**: Fetches top-k nodes with context expansion
   - **Deduplication & Reranking**: Removes duplicates, ranks by frequency/score
   - **Answerer**: Prompts LLM with context and parses structured responses

6. **LLM Integration** (`src/kohakurag/llm.py`)
   - `OpenAIChatModel`: OpenAI API with automatic retry
   - `OpenRouterChatModel`: OpenRouter API integration

For detailed architecture documentation, see [`docs/architecture.md`](docs/architecture.md).

---

## Documentation

- **[Architecture Guide](docs/architecture.md)** — Detailed design decisions and component interactions
- **[Usage Guide](docs/usage.md)** — Complete workflow examples and config reference
- **[WattBot Playbook](docs/wattbot.md)** — Competition-specific setup and validation
- **[JinaV4 Workflows](docs/jinav4_workflows.md)** — Multimodal embedding guide
- **[BM25 Hybrid Search](docs/bm25_hybrid_search.md)** — Sparse + dense hybrid retrieval
- **[Dedup & Rerank](docs/dedup_rerank.md)** — Multi-query retrieval optimization
- **[Image RAG Examples](docs/image_rag_example.md)** — Multimodal RAG with vision models
- **[API Reference](docs/api_reference.md)** — Detailed API documentation
- **[Deployment Guide](docs/deployment.md)** — Production deployment options

---

## Project Structure

```
KohakuRAG/
├── src/kohakurag/          # Core library
│   ├── parsers.py          # Document parsing (PDF/MD/TXT)
│   ├── indexer.py          # Tree building and embedding
│   ├── datastore.py        # Storage abstractions
│   ├── embeddings.py       # Jina v3 & v4 embedding models
│   ├── pipeline.py         # RAG orchestration
│   └── llm.py              # LLM integrations (OpenAI, OpenRouter)
├── scripts/                # WattBot utilities
│   ├── wattbot_fetch_docs.py
│   ├── wattbot_build_index.py
│   ├── wattbot_add_image_captions.py
│   ├── wattbot_build_image_index.py
│   ├── wattbot_answer.py
│   ├── wattbot_validate.py
│   ├── wattbot_aggregate.py
│   └── ...
├── configs/                # KohakuEngine configuration files
│   ├── text_only/          # Text-only pipeline configs
│   ├── with_images/        # Image-enhanced configs
│   └── jinav4/             # JinaV4 multimodal configs
├── workflows/              # Multi-script workflow runners
│   ├── text_pipeline.py
│   ├── with_image_pipeline.py
│   ├── jinav4_pipeline.py
│   ├── ensemble_runner.py
│   ├── indexing/           # Specialized indexing workflows
│   └── sweeps/             # Hyperparameter sweep experiments
├── docs/                   # Documentation
├── data/                   # WattBot dataset
│   ├── metadata.csv
│   ├── train_QA.csv
│   └── test_Q.csv
└── artifacts/              # Generated files (gitignored)
```

---

## Development

### Requirements
- Python 3.10+ (uses modern type hints: `list[str]`, `dict[str, Any]`)
- Dependencies: `torch`, `transformers`, `kohakuvault`, `pypdf`, `httpx`, `openai`, `kohakuengine`
- Jina embeddings (~2GB for v3, ~8GB for v4) downloaded on first run — set `HF_HOME` for custom cache location
- All core operations use async/await for efficient I/O

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_integration.py -v
```

---

## Troubleshooting

### Rate Limit Errors
**Problem:** `openai.RateLimitError: Rate limit reached for gpt-4o-mini`

**Solution:** The retry mechanism handles this automatically. If you still see errors:
1. Reduce `max_concurrent` parameter in your config (default: 10)
2. Increase `max_retries` in your config (default: 5)
3. Consider using a higher-tier OpenAI plan for increased TPM limits

### Embedding Model Download Issues
**Problem:** Slow or failed Jina model download

**Solution:**
```bash
# Set custom Hugging Face cache
export HF_HOME=/path/to/large/disk
kogine run scripts/wattbot_build_index.py --config configs/text_only/index.py
```

### Out of Memory
**Problem:** CUDA OOM during embedding

**Solution:**
- For JinaV4: Use smaller `embedding_dim` (512 instead of 1024)
- Use CPU-only mode: Set `CUDA_VISIBLE_DEVICES=-1`

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Commit with clear messages
5. Push and open a Pull Request

---

## License

Apache-2.0 — See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Built with [KohakuVault](https://github.com/KohakuBlueleaf/KohakuVault) for vector storage
- Configuration management via [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine)
- Embeddings powered by [Jina AI](https://huggingface.co/jinaai/jina-embeddings-v3)
- Developed for [Kaggle WattBot 2025](https://www.kaggle.com/competitions/wattbot-2025)

---

<div align="center">
Made with care by <a href="https://github.com/KohakuBlueleaf">KohakuBlueLeaf</a>
</div>
