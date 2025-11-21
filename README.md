# KohakuRAG — Simple Hierarchical RAG Framework

<div align="center">

**A simple RAG framework with hierarchical document indexing**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

[Features](#-key-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [WattBot 2025](#-wattbot-2025-example) • [Architecture](#-architecture-overview)

</div>

---

## 📖 Overview

KohakuRAG is a **domain-agnostic Retrieval-Augmented Generation (RAG) framework** designed for production use. It transforms long-form documents (PDFs, Markdown, or plain text) into hierarchical knowledge trees and enables intelligent retrieval with context-aware search.

**What makes KohakuRAG different:**
- **Hierarchical structure** preserves document organization (document → section → paragraph → sentence)
- **Smart context expansion** returns not just matched sentences, but their surrounding paragraphs and sections
- **Single-file storage** using SQLite + [KohakuVault](https://github.com/KohakuBlueleaf/KohakuVault) — no external services required
- **Rate-limit resilient** with automatic retry and exponential backoff for LLM APIs
- **Production-tested** on Kaggle's WattBot 2025 competition (energy research corpus)
- **Python-based configuration** via [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine) — no YAML/JSON, fully reproducible experiments

While we demonstrate KohakuRAG with the WattBot 2025 dataset, **the core library is completely domain-agnostic** and can be applied to any document corpus.

---

## ✨ Key Features

### 📄 Structured Document Ingestion
- Parse **PDFs**, **Markdown**, or **plain text** into structured `DocumentPayload` objects
- Preserve document hierarchy with per-page sections, paragraph metadata, and sentence-level granularity
- Maintain image placeholders to preserve figure positioning even when captions are missing

### 🌳 Tree-Based Embeddings
- **Leaf nodes** (sentences) embedded using `jinaai/jina-embeddings-v3` (1024-dim)
- **Parent nodes** inherit averaged vectors from children
- **Multi-level retrieval** — queries can match at any level while preserving full context

### 💾 Single-File Datastore
- Built on **SQLite + sqlite-vec** via [KohakuVault](https://github.com/KohakuBlueleaf/KohakuVault)
- **No external dependencies** — entire index stored in one `.db` file
- Easy to version control, backup, and deploy

### 🔌 Pluggable LLM Orchestration
- **Modular RAG pipeline** with swappable components (planner, retriever, answerer)
- Built-in **OpenAI integration** with automatic rate limit handling
- **Mock chat model** for testing without API costs
- Add your own LLM backend by implementing the `ChatModel` protocol

### 🛡️ Production-Ready Features
- **Async/await architecture** for efficient concurrent I/O
- **Automatic rate limit handling** with intelligent retry logic and semaphore-based concurrency control
- **Thread-safe operations** via single-worker executors for embedding and datastore access
- **Structured logging** for debugging and monitoring
- **Validation scripts** for measuring accuracy before deployment

### ⚙️ KohakuEngine Configuration
- **Python-based configs** via [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine) — no YAML/JSON
- **Reproducible experiments** with version-controlled configuration files
- **Workflow orchestration** for chaining multiple scripts (use `use_subprocess=True` for asyncio scripts)
- **Parallel execution** with `max_workers` control for hyperparameter sweeps and model ensembles

---

## 🚀 Quick Start

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

### Rate Limit Handling & Async Concurrency

KohakuRAG uses **async/await** for efficient concurrent I/O and **automatically handles OpenAI rate limits**:

**Semaphore-based rate limiting:**
- Built-in `asyncio.Semaphore` limits concurrent API requests
- Configure via `max_concurrent` parameter (default: 10)
- No complex threading or manual locks needed

**Intelligent retry logic:**
- Parses server-recommended retry delays from error messages
- Falls back to exponential backoff (1s, 2s, 4s, 8s, 16s...)
- Configurable via `max_retries` and `base_retry_delay` parameters
- Works with restrictive TPM (tokens per minute) limits

**OpenAI-compatible endpoints:**
- Can be pointed at any OpenAI-compatible endpoint (vLLM/llama.cpp/Anthropic/Gemini proxies)
- Configure via `OPENAI_BASE_URL` environment variable or `base_url` argument

```python
import asyncio
from kohakurag.llm import OpenAIChatModel

async def main():
    # Configure concurrency and retry behavior
    chat = OpenAIChatModel(
        model="gpt-4o-mini",
        max_concurrent=10,       # Max 10 concurrent requests
        max_retries=5,           # Retry up to 5 times on rate limit
        base_retry_delay=1.0     # Start with 1s delay
    )

    # Disable rate limiting for unlimited concurrency
    chat_unlimited = OpenAIChatModel(
        model="gpt-4o-mini",
        max_concurrent=0         # 0 or negative = no rate limit
    )

    # All API calls are async
    response = await chat.complete("What is RAG?")

    # Concurrent batch processing with asyncio.gather()
    questions = ["Q1", "Q2", "Q3"]
    responses = await asyncio.gather(*[
        chat.complete(q) for q in questions
    ])

asyncio.run(main())
```

For details on configuring different backends (OpenAI, vLLM, llama.cpp, or OpenAI-compatible proxies), see `docs/deployment.md`.

---

## 🤖 WattBot 2025 Example

KohakuRAG was developed for the [Kaggle WattBot 2025 competition](https://www.kaggle.com/competitions/wattbot-2025), which challenges participants to build a RAG system for answering questions about energy research papers.

### Complete WattBot Workflow

The easiest way to run the full pipeline is using the pre-built workflows:

```bash
# Text-only pipeline (fetch → index → answer → validate)
python workflows/text_pipeline.py

# Image-enhanced pipeline (fetch → caption → index → answer → validate)
python workflows/with_image_pipeline.py

# Ensemble with voting (multiple parallel runs → aggregate)
python workflows/ensemble_runner.py
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
  - Controls OpenAI API rate limiting via semaphore
  - All scripts use `asyncio.gather()` for efficient concurrent processing

See [`docs/wattbot.md`](docs/wattbot.md) and [`docs/usage.md`](docs/usage.md) for advanced usage patterns.

---

## 🖼️ Image Captioning for Multimodal RAG

KohakuRAG supports **vision model integration** to extract and caption images from PDFs, enabling multimodal retrieval.

### Why Add Image Captions?

Many technical documents (research papers, reports, presentations) contain critical information in figures, charts, and diagrams. By generating AI captions for these images, you can:
- **Improve answer accuracy** for questions about visual data
- **Retrieve figure context** alongside text
- **Compare performance** of text-only vs. image-enhanced RAG

### Quick Start

#### 1. Set Up OpenRouter (Recommended)

```bash
# Get API key from https://openrouter.ai
export OPENAI_API_KEY="sk-or-v1-..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

**Recommended model**: `qwen/qwen3-vl-235b-a22b-instruct` (cost-effective, good quality)

#### 2. Generate Image Captions

Edit `configs/with_images/caption.py` with your settings:

```python
from kohakuengine import Config

docs_dir = "artifacts/docs"
pdf_dir = "artifacts/raw_pdfs"
output_dir = "artifacts/docs_with_images"
db = "artifacts/wattbot_with_images.db"
vision_model = "qwen/qwen3-vl-235b-a22b-instruct"
max_concurrent = 5
limit = 10  # Test with 10 documents first

def config_gen():
    return Config.from_globals()
```

Then run:
```bash
kogine run scripts/wattbot_add_image_captions.py --config configs/with_images/caption.py
```

**What it does** (3-phase parallel processing):
- **Phase 1**: Reads ALL images from ALL PDFs concurrently
- **Phase 2**: Compresses ALL images to JPEG (≤1024px, 95% quality) in parallel
- **Phase 3**: Generates captions for ALL images concurrently via vision API
- **Phase 4**: Stores images + updates JSONs with format: `[img:name WxH] caption...`
- All images stored in SAME database that will hold RAG nodes

#### 3. Build Separate Indices

```bash
# Text-only index (baseline)
kogine run scripts/wattbot_build_index.py --config configs/text_only/index.py

# Image-enhanced index
kogine run scripts/wattbot_build_index.py --config configs/with_images/index.py
```

#### 4. Compare Performance

```bash
# Query with text-only
kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py

# Query with images (set with_images=True in config)
kogine run scripts/wattbot_answer.py --config configs/with_images/answer.py

# Validate both
kogine run scripts/wattbot_validate.py --config configs/validate.py
```

### Image Format in Prompts

When `with_images=True` is enabled in config, retrieved context includes a separate "Referenced media" section:

```
Context snippets:
[ref_id=amazon2023] Text about AWS sustainability...
---
[ref_id=google2024] Information about data center cooling...

Referenced media:
[ref_id=nvidia2024] [img:Figure3 800x600] Bar chart showing GPU power consumption trends from 2020-2024, with NVIDIA A100 at 400W and H100 at 700W peak.

[ref_id=amazon2023] [img:Fig5 1200x900] Diagram of water cooling system architecture with labeled components: heat exchangers, cooling towers, and water treatment facilities.
```

### Configuration Options

**Vision Models** (via OpenRouter or OpenAI):
- `qwen/qwen3-vl-235b-a22b-instruct` (recommended, ~$0.50 per 1K images)
- `gpt-4o` (best quality, ~$2.50 per 1K images)
- `gpt-4o-mini` (fast, ~$0.15 per 1K images)

**Storage**:
- **Same database file**: Images stored in same `.db` as RAG nodes (table: `image_blobs`)
  - `wattbot_with_images.db` contains BOTH nodes AND compressed images
  - Single-file deployment, no separate image database needed
- Compressed WebP format saves ~70% storage vs. original
- Original images always available in source PDFs

See [`docs/image_rag_example.md`](docs/image_rag_example.md) for detailed examples and performance analysis.

---

## 🏗️ Architecture Overview

### High-Level Pipeline

```
Documents (PDF/MD/TXT)
    ↓
📄 Parse into hierarchical payload
    ↓
🌳 Build tree structure (doc → section → paragraph → sentence)
    ↓
🔢 Embed leaves with Jina, average for parents
    ↓
💾 Store in SQLite + sqlite-vec (KohakuVault)
    ↓
🔍 Query → Retrieve top-k nodes + context
    ↓
🤖 LLM generates structured answer
```

### Core Components

1. **Parsers** (`src/kohakurag/parsers.py`, `pdf_utils.py`)
   - `pdf_to_document_payload`: Extract text, sections, and image placeholders from PDFs
   - `markdown_to_payload`: Parse Markdown with heading-based structure
   - `text_to_payload`: Simple text ingestion with heuristic segmentation

2. **Indexer** (`src/kohakurag/indexer.py`)
   - Walks document tree and creates nodes for each level
   - Embeds sentences using `JinaEmbeddingModel`
   - Averages child embeddings for parent nodes (weighted by token length)

3. **Datastore** (`src/kohakurag/datastore.py`)
   - `KVaultNodeStore`: SQLite-backed storage with metadata and embeddings
   - `VectorKVault`: Vector similarity search using sqlite-vec
   - Thread-safe for concurrent access

4. **RAG Pipeline** (`src/kohakurag/pipeline.py`)
   - **Planner**: Generates additional retrieval queries (LLM-powered or rule-based)
   - **Retriever**: Fetches top-k nodes with hierarchical context expansion
   - **Answerer**: Prompts LLM with context and parses structured responses

5. **LLM Integration** (`src/kohakurag/llm.py`)
   - `OpenAIChatModel`: Chat completions API with automatic retry logic
   - Supports custom system prompts and structured output parsing
   - Graceful degradation with mock chat model for testing

For detailed architecture documentation, see [`docs/architecture.md`](docs/architecture.md).

---

## 📚 Documentation

- **[Architecture Guide](docs/architecture.md)** — Detailed design decisions and component interactions
- **[Usage Guide](docs/usage.md)** — Complete workflow examples and config reference
- **[WattBot Playbook](docs/wattbot.md)** — Competition-specific setup and validation
- **[API Reference](docs/api_reference.md)** — Detailed API documentation for all components

---

## 🔧 Development

### Requirements
- Python 3.10+ (uses modern type hints: `list[str]`, `dict[str, Any]`)
- Dependencies: `torch`, `transformers`, `kohakuvault`, `pypdf`, `httpx`, `openai`, `kohakuengine`
- Jina embeddings (~2GB) downloaded on first run — set `HF_HOME` for custom cache location
- All core operations use async/await for efficient I/O

### Running Tests
```bash
# Run all tests
python -m unittest discover tests

# Run specific test
python -m unittest tests.test_pipeline
```

### Project Structure
```
KohakuRAG/
├── src/kohakurag/          # Core library
│   ├── parsers.py          # Document parsing (PDF/MD/TXT)
│   ├── indexer.py          # Tree building and embedding
│   ├── datastore.py        # Storage abstractions
│   ├── embeddings.py       # Jina embedding model
│   ├── pipeline.py         # RAG orchestration
│   └── llm.py              # LLM integrations (OpenAI)
├── scripts/                # WattBot utilities
│   ├── wattbot_fetch_docs.py
│   ├── wattbot_build_index.py
│   ├── wattbot_answer.py
│   └── ...
├── configs/                # KohakuEngine configuration files
│   ├── text_only/          # Text-only pipeline configs
│   └── with_images/        # Image-enhanced configs
├── workflows/              # Multi-script workflow runners
├── docs/                   # Documentation
│   ├── architecture.md
│   ├── usage.md
│   └── wattbot.md
├── data/                   # WattBot dataset
│   ├── metadata.csv
│   ├── train_QA.csv
│   └── test_Q.csv
└── artifacts/              # Generated files (gitignored)
    ├── raw_pdfs/
    ├── docs/               # Parsed JSON payloads
    └── wattbot.db          # Built index
```

---

## 🐛 Troubleshooting

### Rate Limit Errors
**Problem:** `openai.RateLimitError: Rate limit reached for gpt-4o-mini`

**Solution:** The retry mechanism handles this automatically. If you still see errors:
1. Reduce `max_concurrent` parameter in your config (default: 10)
2. Increase `max_retries` in your config (default: 5)
3. Consider using a higher-tier OpenAI plan for increased TPM limits

Example config:
```python
# configs/text_only/answer.py
max_concurrent = 5   # Reduce concurrent requests
max_retries = 10     # More retry attempts
```

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
- Reduce batch size in `JinaEmbeddingModel` (edit `src/kohakurag/embeddings.py`)
- Use CPU-only mode: Set `CUDA_VISIBLE_DEVICES=-1`

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Commit with clear messages
5. Push and open a Pull Request

---

## 📄 License

Apache-2.0 — See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Built with [KohakuVault](https://github.com/KohakuBlueleaf/KohakuVault) for vector storage
- Configuration management via [KohakuEngine](https://github.com/KohakuBlueleaf/KohakuEngine)
- Embeddings powered by [Jina AI](https://huggingface.co/jinaai/jina-embeddings-v3)
- Developed for [Kaggle WattBot 2025](https://www.kaggle.com/competitions/wattbot-2025)

---

<div align="center">
Made with ❤️ by <a href="https://github.com/KohakuBlueleaf">KohakuBlueLeaf</a>
</div>
