# KohakuRAG Pipeline Architecture

This document explains how KohakuRAG works from end to end: from indexing documents to retrieving relevant context to generating answers. We follow the natural workflow of the system, introducing configurable parameters as they become relevant.

---

## Overview: The Three Phases

KohakuRAG operates in three distinct phases:

1. **Indexing**: Parse documents into a hierarchical structure and compute embeddings
2. **Retrieval**: Find relevant content using multi-query vector search
3. **Answering**: Generate structured answers using an LLM with retrieved context

```
Documents → [Indexing] → Vector DB → [Retrieval] → Context → [Answering] → Response
```

---

## Phase 1: Indexing

### 1.1 The 4-Level Hierarchy

KohakuRAG organizes documents into a strict 4-level hierarchy:

```
Document
└── Section(s)
    └── Paragraph(s)
        └── Sentence(s)
```

Each level serves a purpose:

| Level | Purpose | Example |
|-------|---------|---------|
| **Document** | Top-level container | "Amazon 2023 Annual Report" |
| **Section** | Logical groupings (pages, chapters) | "Financial Highlights" |
| **Paragraph** | Coherent text blocks | A single paragraph of text |
| **Sentence** | Atomic retrieval units | Individual sentences |

**Node IDs** follow a hierarchical naming convention for easy parent lookup:
- Document: `amazon2023`
- Section: `amazon2023:sec1`
- Paragraph: `amazon2023:sec1:p2`
- Sentence: `amazon2023:sec1:p2:s3`

### 1.2 Embedding Strategy: Bottom-Up Propagation

The indexer computes embeddings from the bottom up:

1. **Sentences are embedded directly** using the embedding model (e.g., Jina v3/v4)
2. **Paragraphs inherit from sentences** via weighted averaging (or direct embedding)
3. **Sections inherit from paragraphs** via weighted averaging
4. **Documents inherit from sections** via weighted averaging

```python
# Pseudocode for embedding propagation
def compute_embedding(node):
    if node.is_sentence:
        return embedding_model.embed(node.text)
    else:
        child_embeddings = [compute_embedding(child) for child in node.children]
        return normalize(mean(child_embeddings))
```

This approach ensures that:
- Fine-grained queries match sentences
- Broader queries match paragraphs/sections
- Parent nodes represent semantic summaries of their children

### 1.3 Paragraph Embedding Modes

Paragraphs get special treatment because they're the primary retrieval unit. The `paragraph_embedding_mode` parameter (set at indexing time) controls how paragraph embeddings are computed:

| Mode | Description | Use Case |
|------|-------------|----------|
| `"averaged"` | Paragraph embedding = mean of sentence embeddings | Default, memory-efficient |
| `"full"` | Paragraph embedding = direct embedding of paragraph text | Better semantic matching |
| `"both"` | Store both averaged AND full embeddings | Maximum flexibility |

**Why does this matter?**

- **Averaged embeddings** capture the compositional meaning of sentences. Good for queries that match specific facts within a paragraph.
- **Full embeddings** capture the holistic meaning of the paragraph as a unit. Good for queries about the paragraph's overall topic.
- **Both** lets you choose at query time (see Phase 2).

**Configuration:**
```python
# In indexing config
paragraph_embedding_mode = "both"  # Options: "averaged", "full", "both"
```

### 1.4 Embedding Models

KohakuRAG supports two embedding models:

**Jina v3** (`embedding_model = "jina"`)
- 1024-dimensional embeddings
- Text-only
- Well-suited for general-purpose retrieval

**Jina v4** (`embedding_model = "jinav4"`)
- Configurable dimensions via Matryoshka truncation: 128, 256, 512, 1024, 2048
- Multimodal (text + images in unified space)
- Required for image retrieval features

```python
# Configuration
embedding_model = "jinav4"
embedding_dim = 512        # Matryoshka dimension (jinav4 only)
embedding_task = "retrieval"  # Options: "retrieval", "text-matching", "code"
```

### 1.5 Storage Architecture

All data is stored in a single SQLite database with multiple tables:

| Table | Content |
|-------|---------|
| `{prefix}_kv` | Node metadata (text, hierarchy, references) |
| `{prefix}_vec` | Main vector table (averaged paragraph embeddings) |
| `{prefix}_para_full_vec` | Full paragraph embeddings (if "both"/"full" mode) |
| `{prefix}_images_vec` | Image embeddings (if image index built) |
| `image_blobs` | Compressed image bytes (WebP) |

```python
# Configuration
db = "artifacts/my_index.db"
table_prefix = "rag_nodes"
```

---

## Phase 2: Retrieval

### 2.1 Query Planning

Instead of searching with just the user's question, KohakuRAG generates multiple retrieval queries using an LLM-based planner. This improves recall by covering different phrasings and entity mentions.

```
User Question: "What was Amazon's revenue in 2023?"
              ↓
        [Query Planner]
              ↓
Generated Queries:
  1. "What was Amazon's revenue in 2023?" (original)
  2. "Amazon total revenue fiscal year 2023"
  3. "Amazon net sales 2023 financial results"
```

**Configuration:**
```python
planner_max_queries = 4  # Max queries to generate (including original)
planner_model = None     # LLM for planning (defaults to main model)
```

### 2.2 Vector Search

For each query, the pipeline:

1. **Embeds the query** using the same embedding model as indexing
2. **Searches the vector database** for similar nodes
3. **Filters by node type**: Only **SENTENCE** and **PARAGRAPH** nodes are searched

```python
matches = store.search(
    query_vector,
    k=top_k,
    kinds={NodeKind.SENTENCE, NodeKind.PARAGRAPH}  # Skip documents/sections
)
```

**Why skip documents and sections?**
- Documents are too broad—their embeddings are averaged over everything
- Sections are still quite large and may dilute relevance
- Sentences and paragraphs provide the right granularity for accurate retrieval

**Configuration:**
```python
top_k = 16  # Results per query
```

### 2.3 Paragraph Search Mode (Runtime)

At query time, `paragraph_search_mode` controls which paragraph embeddings are used:

| Mode | Behavior |
|------|----------|
| `"averaged"` | Search only the main vector table (averaged embeddings) |
| `"full"` | Search the full paragraph table (direct embeddings) |
| `"both"` | Search both tables, merge results by highest score |

**"Both" mode explained:**
```
Query: "climate change mitigation strategies"
        ↓
[Search averaged table] → [Result A: score 0.82, Result B: score 0.78]
[Search full table]     → [Result A: score 0.85, Result C: score 0.80]
        ↓
[Merge by max score]
        ↓
Final: [Result A: 0.85, Result C: 0.80, Result B: 0.78]
```

This combines the strengths of both embedding strategies: compositional matching (averaged) and holistic matching (full).

**Configuration:**
```python
paragraph_search_mode = "averaged"  # Options: "averaged", "full", "both"
```

### 2.4 Deduplication

When using multiple queries, the same node might be retrieved multiple times. Deduplication removes these duplicates:

```python
deduplicate_retrieval = True  # Remove duplicate nodes across queries
```

### 2.5 Reranking

After collecting results from all queries, optional reranking improves result quality:

| Strategy | Description |
|----------|-------------|
| `"frequency"` | Nodes appearing in more queries rank higher |
| `"score"` | Rank by sum of similarity scores |
| `"combined"` | 0.4 × normalized_frequency + 0.6 × normalized_score |

**Intuition:** A node that appears in multiple query results is likely more relevant than one that appears only once.

```python
rerank_strategy = "combined"  # Options: None, "frequency", "score", "combined"
top_k_final = 32              # Truncate after reranking (None = no truncation)
```

### 2.6 Context Expansion

After retrieval, each matched node is expanded with hierarchical context:

```
Matched: sentence "amazon2023:sec1:p2:s3"
         ↓
[Context Expansion: parent_depth=1, child_depth=1]
         ↓
Expanded Context:
  - Parent paragraph: "amazon2023:sec1:p2" (full paragraph text)
  - Sibling sentences: s1, s2, s4, s5 (nearby context)
  - The matched sentence itself
```

This ensures the LLM sees complete, coherent context rather than isolated fragments.

### 2.7 Image Retrieval

KohakuRAG supports two image retrieval strategies:

**1. Images from text sections** (always active when `with_images=True`)
- Extracts image nodes from retrieved sections
- Images are included as captions in the prompt

**2. Images from vision search** (when `top_k_images > 0`)
- Directly searches the image embedding index
- Uses Jina v4's multimodal embeddings
- Can send actual image bytes to vision-capable LLMs

```python
with_images = True        # Extract images from retrieved sections
top_k_images = 4          # Additional images from image-only index
send_images_to_llm = True # Send image bytes (not just captions) to LLM
```

---

## Phase 3: Answering

### 3.1 Prompt Construction

The retrieved context is formatted into a prompt for the LLM:

```
System: You must answer strictly based on the provided context snippets.
        Do NOT use external knowledge or assumptions.
        If the context does not clearly support an answer, output "is_blank".

User: Additional info (JSON): {"answer_unit": "USD millions", ...}

      Question: What was Amazon's revenue in 2023?

      Context:
      [ref_id=amazon2023:sec1:p2]
      Amazon reported total net sales of $574.8 billion for fiscal year 2023...

      [ref_id=amazon2023:sec3:p1]
      Revenue growth was driven primarily by AWS and advertising services...

      Return STRICT JSON with keys: explanation, answer, answer_value, ref_id
```

**Prompt ordering** (`use_reordered_prompt`) can place context before the question to combat "attention sink" effects where LLMs focus too heavily on the question.

```python
use_reordered_prompt = True  # Put context before question
```

### 3.2 Multimodal Content

When `send_images_to_llm=True`, the prompt becomes a multimodal content list:

```python
[
    {"type": "text", "text": "Question: ..."},
    {"type": "image_url", "image_url": {"url": "data:image/webp;base64,..."}},
    {"type": "text", "text": "Context: ..."},
]
```

This allows vision-capable LLMs (GPT-4V, Claude 3, etc.) to analyze charts, diagrams, and other visual content directly.

### 3.3 LLM Configuration

```python
llm_provider = "openrouter"  # Options: "openai", "openrouter"
model = "openai/gpt-4o-mini"
max_concurrent = 32          # Concurrent API requests
max_retries = 3              # Retry on blank answers (increases top_k each retry)
```

### 3.4 Retry Logic

The pipeline automatically retries when answers are unsatisfactory:

1. **Blank answer retry**: If the LLM returns "is_blank", retry with increased `top_k`
2. **Context overflow retry**: If context exceeds LLM limits, reduce `top_k` and retry
3. **Rate limit retry**: Automatic exponential backoff on 429/5xx errors

```
Attempt 1: top_k=16 → "is_blank"
Attempt 2: top_k=32 → "is_blank"
Attempt 3: top_k=48 → Valid answer!
```

### 3.5 Response Parsing

The LLM's JSON response is parsed into a structured answer:

```python
@dataclass
class StructuredAnswer:
    answer: str           # Natural language answer
    answer_value: str     # Structured value (number, category, "is_blank")
    ref_id: list[str]     # Source document IDs
    explanation: str      # 1-3 sentence reasoning
```

---

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INDEXING PHASE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Raw Documents (PDF/Markdown/Text)                                      │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ DocumentIndexer                                                  │   │
│  │                                                                  │   │
│  │  1. Parse into hierarchy:                                        │   │
│  │     Document → Sections → Paragraphs → Sentences                │   │
│  │                                                                  │   │
│  │  2. Embed sentences (Jina v3/v4)                                │   │
│  │                                                                  │   │
│  │  3. Propagate embeddings upward:                                │   │
│  │     - paragraph_embedding_mode: "averaged" | "full" | "both"    │   │
│  │     - Sections/Documents: always averaged                       │   │
│  │                                                                  │   │
│  │  4. Flatten to StoredNode list                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ KVaultNodeStore (SQLite)                                        │   │
│  │                                                                  │   │
│  │  Tables:                                                         │   │
│  │  • {prefix}_kv          → Node metadata                         │   │
│  │  • {prefix}_vec         → Main vectors (averaged paragraphs)    │   │
│  │  • {prefix}_para_full_vec → Full paragraph vectors              │   │
│  │  • {prefix}_images_vec  → Image embeddings                      │   │
│  │  • image_blobs          → Compressed image bytes                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          RETRIEVAL PHASE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  User Question: "What was Amazon's revenue in 2023?"                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ LLMQueryPlanner                                                  │   │
│  │                                                                  │   │
│  │  • planner_max_queries: 4                                       │   │
│  │                                                                  │   │
│  │  Generated queries:                                              │   │
│  │  1. "What was Amazon's revenue in 2023?"                        │   │
│  │  2. "Amazon total revenue fiscal year 2023"                     │   │
│  │  3. "Amazon net sales 2023 financial results"                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Vector Search (per query)                                        │   │
│  │                                                                  │   │
│  │  • top_k: 16                                                    │   │
│  │  • kinds: {SENTENCE, PARAGRAPH}                                 │   │
│  │  • paragraph_search_mode: "averaged" | "full" | "both"          │   │
│  │                                                                  │   │
│  │  For "both" mode:                                                │   │
│  │  - Search averaged table → Results A                            │   │
│  │  - Search full table → Results B                                │   │
│  │  - Merge by max score per node                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Post-Processing                                                  │   │
│  │                                                                  │   │
│  │  1. Deduplication (deduplicate_retrieval: true)                 │   │
│  │     - Remove duplicate node_ids across queries                  │   │
│  │                                                                  │   │
│  │  2. Reranking (rerank_strategy: "combined")                     │   │
│  │     - "frequency": nodes in multiple queries rank higher        │   │
│  │     - "score": sum of similarity scores                         │   │
│  │     - "combined": 0.4×freq + 0.6×score                          │   │
│  │                                                                  │   │
│  │  3. Truncation (top_k_final: 32)                                │   │
│  │     - Keep top N after reranking                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Context Expansion                                                │   │
│  │                                                                  │   │
│  │  For each matched node:                                          │   │
│  │  - Fetch parent (paragraph → section)                           │   │
│  │  - Fetch children (paragraph → sentences)                       │   │
│  │  - Create ContextSnippet with full text                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Image Retrieval (optional)                                       │   │
│  │                                                                  │   │
│  │  • with_images: true                                            │   │
│  │    → Extract images from retrieved sections (captions)          │   │
│  │                                                                  │   │
│  │  • top_k_images: 4                                              │   │
│  │    → Search image-only index for additional images              │   │
│  │                                                                  │   │
│  │  • send_images_to_llm: true                                     │   │
│  │    → Include actual image bytes in prompt                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          ANSWERING PHASE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Prompt Construction                                              │   │
│  │                                                                  │   │
│  │  • use_reordered_prompt: true                                   │   │
│  │    → Place context BEFORE question (combats attention sink)     │   │
│  │                                                                  │   │
│  │  System: "Answer based on context only. Output is_blank if      │   │
│  │           context doesn't support an answer."                   │   │
│  │                                                                  │   │
│  │  User:                                                           │   │
│  │    Context:                                                      │   │
│  │    [ref_id=amazon2023:sec1:p2]                                  │   │
│  │    Amazon reported total net sales of $574.8 billion...         │   │
│  │                                                                  │   │
│  │    Question: What was Amazon's revenue in 2023?                 │   │
│  │                                                                  │   │
│  │    Return JSON: {explanation, answer, answer_value, ref_id}     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ LLM Completion                                                   │   │
│  │                                                                  │   │
│  │  • llm_provider: "openrouter"                                   │   │
│  │  • model: "openai/gpt-4o-mini"                                  │   │
│  │  • max_concurrent: 32                                           │   │
│  │                                                                  │   │
│  │  Built-in handling:                                              │   │
│  │  - Rate limit retry (429) with exponential backoff              │   │
│  │  - Server error retry (500/502/503/504)                         │   │
│  │  - Empty response retry                                          │   │
│  │  - Context overflow → reduce top_k and retry                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Retry Logic                                                      │   │
│  │                                                                  │   │
│  │  • max_retries: 3                                               │   │
│  │                                                                  │   │
│  │  If answer == "is_blank":                                        │   │
│  │    Retry with top_k × 2                                         │   │
│  │                                                                  │   │
│  │  If context overflow:                                            │   │
│  │    Retry with top_k - 1                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Response Parsing                                                 │   │
│  │                                                                  │   │
│  │  LLM Response:                                                   │   │
│  │  {                                                               │   │
│  │    "explanation": "The context states Amazon's net sales...",   │   │
│  │    "answer": "Amazon's revenue in 2023 was $574.8 billion",     │   │
│  │    "answer_value": "574.8",                                     │   │
│  │    "ref_id": "amazon2023:sec1:p2"                               │   │
│  │  }                                                               │   │
│  │                                                                  │   │
│  │  Normalization:                                                  │   │
│  │  - True/False → 1/0                                             │   │
│  │  - Ranges → [lower, upper]                                      │   │
│  │  - ref_id lookup → source URLs                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Reference

### Indexing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `paragraph_embedding_mode` | `"averaged"` | How to embed paragraphs: `"averaged"`, `"full"`, `"both"` |
| `embedding_model` | `"jina"` | Embedding model: `"jina"` (v3) or `"jinav4"` |
| `embedding_dim` | `None` | Dimension for Jina v4 (128, 256, 512, 1024, 2048) |
| `embedding_task` | `"retrieval"` | Jina v4 task: `"retrieval"`, `"text-matching"`, `"code"` |
| `db` | - | Path to SQLite database file |
| `table_prefix` | `"rag_nodes"` | Prefix for database tables |

### Retrieval Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | `5` | Results per query |
| `planner_max_queries` | `3` | Maximum queries from planner |
| `paragraph_search_mode` | `"averaged"` | Which embeddings to search: `"averaged"`, `"full"`, `"both"` |
| `deduplicate_retrieval` | `False` | Remove duplicate nodes across queries |
| `rerank_strategy` | `None` | Reranking: `None`, `"frequency"`, `"score"`, `"combined"` |
| `top_k_final` | `None` | Truncate after reranking |

### Image Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `with_images` | `False` | Extract images from retrieved sections |
| `top_k_images` | `0` | Additional images from image index |
| `send_images_to_llm` | `False` | Send image bytes to vision LLM |

### LLM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_provider` | `"openrouter"` | Provider: `"openai"`, `"openrouter"` |
| `model` | `"openai/gpt-4o-mini"` | Model identifier |
| `planner_model` | `None` | Model for query planning (defaults to `model`) |
| `max_concurrent` | `32` | Maximum concurrent API requests |
| `max_retries` | `0` | Retry attempts on blank answers |
| `use_reordered_prompt` | `False` | Place context before question |

---

## Key Design Decisions

### Why 4 levels?

- **Documents** provide attribution and source tracking
- **Sections** enable efficient context expansion without loading entire documents
- **Paragraphs** are the ideal retrieval unit—coherent, self-contained, not too long
- **Sentences** enable fine-grained matching for specific facts

### Why averaged vs full embeddings?

**Averaged embeddings** are computed by averaging sentence embeddings. They excel at:
- Matching queries that target specific facts within a paragraph
- Compositional understanding (the whole = sum of parts)

**Full embeddings** are computed by embedding the entire paragraph text. They excel at:
- Matching queries about the paragraph's overall topic
- Holistic understanding (the paragraph as a semantic unit)

**Both mode** gives you the best of both worlds at the cost of storage space.

### Why multi-query retrieval?

A single query might miss relevant content due to:
- Vocabulary mismatch (user says "revenue", document says "net sales")
- Entity mentions (user assumes context, document is explicit)
- Paraphrasing (many ways to express the same concept)

Multi-query retrieval with LLM planning generates diverse queries that cover more ground.

### Why reranking?

When a node appears in multiple query results, it's likely more relevant than a node appearing only once. Reranking leverages this signal to improve result quality.

---

## Example: Full Pipeline Run

```python
# 1. INDEXING (run once)
indexer = DocumentIndexer(
    embedding_model=JinaV4EmbeddingModel(truncate_dim=512),
    paragraph_embedding_mode="both"
)
nodes = await indexer.index(document)
store = KVaultNodeStore("index.db", dimensions=512)
await store.upsert_nodes(nodes)

# 2. RETRIEVAL + ANSWERING (run per question)
pipeline = RAGPipeline(
    store=store,
    embedder=JinaV4EmbeddingModel(truncate_dim=512),
    chat_model=OpenRouterChatModel(model="openai/gpt-4o-mini"),
    planner=LLMQueryPlanner(chat_model, max_queries=4),
    top_k=16,
    deduplicate_retrieval=True,
    rerank_strategy="combined",
    top_k_final=32,
)

# Set runtime search mode
store.set_paragraph_search_mode("both")

# Run QA
result = await pipeline.run_qa(
    question="What was Amazon's revenue in 2023?",
    with_images=True,
    top_k_images=4,
    send_images_to_llm=False,
)

print(result.answer.answer_value)  # "574.8"
print(result.answer.ref_id)        # ["amazon2023:sec1:p2"]
```

---

## Sweep Experiments

KohakuRAG includes sweep workflows for systematic evaluation:

| Sweep | Line Parameter | X-Axis | Purpose |
|-------|---------------|--------|---------|
| `llm_model_vs_top_k.py` | LLM model | top_k | Compare models across retrieval depths |
| `paragraph_mode_vs_top_k.py` | paragraph_search_mode | top_k | Compare embedding strategies |
| `send_images_vs_top_k_images.py` | send_images_to_llm | top_k_images | Evaluate vision features |
| `top_k_vs_embedding.py` | embedding_config | top_k | Compare embedding models |

Run sweeps with:
```bash
python workflows/sweeps/paragraph_mode_vs_top_k.py --num-runs 3
```
