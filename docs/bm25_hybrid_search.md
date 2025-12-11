# BM25 Hybrid Search

KohakuRAG supports **hybrid retrieval** combining dense vector search (semantic) with sparse BM25 search (keyword-based). This provides complementary retrieval where dense search captures semantic meaning while BM25 catches exact keyword matches that semantic search might miss.

## Overview

**Key Design Decision:** BM25 results are **added to** dense retrieval results for context expansion, **NOT fused** with dense scores. This is because our dense search (Jina embeddings) already achieves state-of-the-art performance, and we use BM25 purely to add complementary keyword-matched content.

### How It Works

1. **Dense retrieval** runs first, returning top-k semantically similar nodes
2. **BM25 search** then adds additional keyword-matched nodes (if `bm25_top_k > 0`)
3. BM25 results are **deduplicated** against dense results (no duplicates)
4. Combined results go through context expansion as usual

This approach ensures dense search quality is preserved while BM25 adds potentially useful exact-match content.

---

## Quick Start

### 1. Build BM25 Index

After building your dense vector index, add the BM25 index:

```bash
# Using KohakuEngine configs
kogine run scripts/wattbot_build_bm25_index.py --config configs/text_only/bm25_index.py
kogine run scripts/wattbot_build_bm25_index.py --config configs/with_images/bm25_index.py
kogine run scripts/wattbot_build_bm25_index.py --config configs/jinav4/bm25_index.py
```

The BM25 table is added to the **same database file** as your dense index.

### 2. Enable BM25 in Answering

Add `bm25_top_k` to your answer config:

```python
# configs/text_only/answer.py
from kohakuengine import Config

db = "artifacts/wattbot_text_only.db"
table_prefix = "wattbot_text"
top_k = 8                # Dense retrieval results per query
bm25_top_k = 4           # Additional BM25 results (0 = disabled)
# ... other settings

def config_gen():
    return Config.from_globals()
```

```bash
kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py
```

---

## Migration Guide

### Adding BM25 to Existing Index

If you already have a dense index without BM25, simply run the BM25 indexing script:

```bash
# Text-only database
kogine run scripts/wattbot_build_bm25_index.py --config configs/text_only/bm25_index.py

# With-images database
kogine run scripts/wattbot_build_bm25_index.py --config configs/with_images/bm25_index.py

# JinaV4 database
kogine run scripts/wattbot_build_bm25_index.py --config configs/jinav4/bm25_index.py
```

This will:
- Add a new `{table_prefix}_bm25` FTS5 table to your existing `.db` file
- Index all sentence and paragraph nodes
- **Not affect** your existing dense vector index

### Verifying BM25 Index

Check that BM25 was added successfully:

```python
from kohakurag.datastore import KVaultNodeStore
from pathlib import Path

store = KVaultNodeStore(Path("artifacts/wattbot_text_only.db"), table_prefix="wattbot_text")
print(f"Has BM25 index: {store.has_bm25_index()}")
```

---

## Indexing Workflows

The indexing workflows now include BM25 by default:

### Text-Only Pipeline

```bash
python workflows/indexing/jina_v3_text_only.py
```

This runs:
1. `wattbot_build_index.py` — Dense vector index
2. `wattbot_build_bm25_index.py` — BM25 sparse index

### Image-Enhanced Pipeline

```bash
python workflows/indexing/jina_v3_text_image.py
python workflows/indexing/jina_v4_text_image.py
```

These run:
1. `wattbot_build_index.py` — Dense vector index
2. `wattbot_build_image_index.py` — Image-only vector index
3. `wattbot_build_bm25_index.py` — BM25 sparse index

### Disabling BM25 in Workflows

To skip BM25 indexing, edit the workflow file:

```python
# In workflows/indexing/*.py
build_bm25_index = False  # Change from True to False
```

---

## Configuration Reference

### BM25 Index Config

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | str | required | Path to database file (must exist) |
| `table_prefix` | str | required | Must match dense index prefix |
| `index_node_kinds` | list | `["sentence", "paragraph"]` | Node types to index |

**Example** (`configs/text_only/bm25_index.py`):

```python
from kohakuengine import Config

db = "artifacts/wattbot_text_only.db"
table_prefix = "wattbot_text"
index_node_kinds = ["sentence", "paragraph"]

def config_gen():
    return Config.from_globals()
```

### Answer Config (BM25 Settings)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bm25_top_k` | int | `0` | Additional BM25 results (0 = disabled) |

**Interaction with other settings:**

- `bm25_top_k` results are added **after** dense retrieval + deduplication + reranking
- They are **not** included in `top_k_final` truncation (applied before BM25)
- Total context = dense results + up to `bm25_top_k` unique BM25 results

---

## Hyperparameter Sweep

A sweep is provided to find optimal `bm25_top_k` and `top_k` combinations:

```bash
# Run the sweep
python workflows/sweeps/bm25_top_k_vs_top_k.py

# Plot results
python workflows/sweeps/sweep_plot.py outputs/sweeps/bm25_top_k_vs_top_k
```

### Sweep Parameters

| Line Parameter | Values |
|---------------|--------|
| `bm25_top_k` | 0, 2, 4, 8 |

| X-axis Parameter | Values |
|-----------------|--------|
| `top_k` | 4, 8, 16 |

### Expected Results

- `bm25_top_k=0` serves as baseline (dense-only)
- Small `bm25_top_k` values (2-4) often help for keyword-heavy queries
- Larger values may add noise without improving accuracy

---

## API Reference

### KVaultNodeStore Methods

```python
from kohakurag.datastore import KVaultNodeStore

store = KVaultNodeStore(path, table_prefix="wattbot")

# Check if BM25 index exists
has_bm25 = store.has_bm25_index()  # Returns bool

# Search BM25 (async)
matches = await store.search_bm25(
    query="renewable energy",
    k=5,
    kinds={NodeKind.SENTENCE, NodeKind.PARAGRAPH}
)
# Returns: list[RetrievalMatch]
```

### RAGPipeline Parameters

```python
from kohakurag.pipeline import RAGPipeline

pipeline = RAGPipeline(
    store=store,
    embedder=embedder,
    chat_model=chat,
    bm25_top_k=4,  # Set default BM25 results
    # ... other settings
)

# Override per-query
result = await pipeline.retrieve(
    question="What is solar panel efficiency?",
    top_k=8,
    bm25_top_k=2  # Override default
)
```

---

## Implementation Details

### Storage

- BM25 uses SQLite FTS5 (Full-Text Search) via [KohakuVault TextVault](https://github.com/KohakuBlueleaf/KohakuVault)
- Table name: `{table_prefix}_bm25`
- Stored in the **same `.db` file** as dense index
- Single-column index (concatenated text)
- Value stored: `node_id` (for lookup in main KVault)

### What Gets Indexed

By default, **sentence** and **paragraph** nodes are indexed (matching dense search behavior):

- Document and section nodes are **not** indexed (too coarse)
- Attachment nodes are **not** indexed

### Score Normalization

BM25 scores are normalized for consistency:
- FTS5 returns negative scores (lower = better match)
- Scores are transformed to 0-1 range: `score = max(0, min(1, (bm25_score + 20) / 20))`

---

## Best Practices

### When to Use BM25

BM25 helps when:
- Questions contain **specific technical terms** or **acronyms**
- Documents have **unique identifiers** (model names, product codes)
- Dense search misses **exact phrase matches**

### Recommended Settings

| Use Case | `top_k` | `bm25_top_k` |
|----------|---------|--------------|
| General QA | 8 | 2-4 |
| Technical docs | 8 | 4 |
| Dense-only baseline | 8 | 0 |

### Monitoring

Check if BM25 is being used:

```python
# In your answer script, after creating pipeline
print(f"BM25 enabled: {pipeline._store.has_bm25_index()}")
print(f"BM25 top_k: {pipeline._bm25_top_k}")
```

---

## Troubleshooting

### "BM25 table doesn't exist"

The BM25 index wasn't built or the table prefix doesn't match:

```bash
# Rebuild BM25 index
kogine run scripts/wattbot_build_bm25_index.py --config configs/text_only/bm25_index.py
```

### "Only indexed N nodes (expected more)"

The `keys()` method has a default limit. The script uses `limit=1_000_000` to handle large indexes. If you have more than 1M nodes, increase this in the script.

### BM25 Not Improving Results

- BM25 is complementary, not a replacement for dense search
- Try smaller `bm25_top_k` values (2-4)
- BM25 helps more with keyword-heavy queries than semantic queries
- Run the sweep to find optimal values for your dataset
