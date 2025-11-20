# KohakuRAG API Reference

This document provides detailed API documentation for the core KohakuRAG library components.

---

## Table of Contents

- [LLM Integration](#llm-integration)
  - [OpenAIChatModel](#openaichatmodel)
- [Embeddings](#embeddings)
  - [JinaEmbeddingModel](#jinaembeddingmodel)
- [Datastore](#datastore)
  - [KVaultNodeStore](#kvaultnodestore)
- [RAG Pipeline](#rag-pipeline)
  - [RAGPipeline](#ragpipeline)
- [Document Parsing](#document-parsing)
  - [pdf_to_document_payload](#pdf_to_document_payload)
  - [markdown_to_payload](#markdown_to_payload)
  - [text_to_payload](#text_to_payload)
- [Indexing](#indexing)
  - [DocumentIndexer](#documentindexer)

---

## LLM Integration

### OpenAIChatModel

**Location:** `src/kohakurag/llm.py`

Chat backend powered by OpenAI's Chat Completions API with automatic rate limit handling.

#### Constructor

```python
OpenAIChatModel(
    *,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
    system_prompt: str | None = None,
    max_retries: int = 5,
    base_retry_delay: float = 3.0,
    base_url: Optional[str] = None,
    max_concurrent: int = 10,
)
```

**Parameters:**

- `model` (str, default: `"gpt-4o-mini"`): OpenAI model identifier
- `api_key` (Optional[str], default: `None`): OpenAI API key. If `None`, reads from `OPENAI_API_KEY` environment variable or `.env` file
- `organization` (Optional[str], default: `None`): OpenAI organization ID
- `system_prompt` (str | None, default: `None`): Default system prompt for all completions
- `max_retries` (int, default: `5`): Maximum number of retry attempts on rate limit errors
- `base_retry_delay` (float, default: `3.0`): Base delay in seconds for exponential backoff
- `base_url` (Optional[str], default: `None`): Optional override for the API base URL (for example, `http://localhost:8000/v1` for self-hosted vLLM/llama.cpp or an OpenAI-compatible proxy). If omitted, falls back to the `OPENAI_BASE_URL` environment variable or `.env` file when present.
- `max_concurrent` (int, default: `10`): Maximum number of concurrent API requests. Set to 0 or negative to disable rate limiting (unlimited concurrency).

**Raises:**
- `ImportError`: If `openai>=1.0.0` is not installed
- `ValueError`: If no API key is found

#### Methods

##### `async complete(prompt: str, *, system_prompt: str | None = None) -> str`

Execute a chat completion request with automatic rate limit retry (async).

**Parameters:**
- `prompt` (str): User prompt to send to the model
- `system_prompt` (str | None, optional): Override the default system prompt for this request

**Returns:**
- `str`: Model's response content

**Raises:**
- `openai.RateLimitError`: If rate limit persists after all retries
- `openai.BadRequestError`: For context overflow or other 400 errors
- `openai.OpenAIError`: For other API errors

**Retry Behavior:**

The method automatically handles rate limit errors using an intelligent retry strategy:

1. **Semaphore control**: Limits concurrent requests via `asyncio.Semaphore(max_concurrent)`
2. **Server-recommended delays**: Parses error messages for suggested wait times (e.g., "Please try again in 23ms")
3. **Exponential backoff**: Falls back to 3s, 6s, 12s, 24s, 48s... by default if no specific delay is provided (scaled by `base_retry_delay`)
4. **Automatic retry**: Continues until success or `max_retries` is exhausted

**Example:**

```python
import asyncio
from kohakurag.llm import OpenAIChatModel

async def main():
    # Basic usage
    chat = OpenAIChatModel(model="gpt-4o-mini")
    response = await chat.complete("Explain quantum computing in one sentence.")

    # Configure concurrency and retry behavior
    chat = OpenAIChatModel(
        model="gpt-4o-mini",
        max_concurrent=10,       # Max 10 concurrent requests
        max_retries=10,          # More retries for TPM-constrained accounts
        base_retry_delay=2.0,    # Longer initial delay
    )
    response = await chat.complete("What is the capital of France?")

    # Override system prompt per request
    chat = OpenAIChatModel(
        system_prompt="You are a helpful assistant."
    )
    response = await chat.complete(
        "Explain RAG systems",
        system_prompt="You are an expert in information retrieval."
    )

    # Batch processing with asyncio.gather()
    questions = ["Q1", "Q2", "Q3"]
    responses = await asyncio.gather(*[
        chat.complete(q) for q in questions
    ])

asyncio.run(main())
```

#### Rate Limit Handling Details

The retry mechanism is designed to work seamlessly with OpenAI's rate limits:

**Supported Error Formats:**
- `"Please try again in 23ms"` → waits 0.023s + 0.1s buffer
- `"Please try again in 1.5s"` → waits 1.5s + 0.1s buffer
- `"Please try again in 2m"` → waits 120s + 0.1s buffer

**Exponential Backoff Schedule:**
| Attempt | Wait Time (seconds) |
|---------|---------------------|
| 1       | 3.0                 |
| 2       | 6.0                 |
| 3       | 12.0                |
| 4       | 24.0                |
| 5       | 48.0                |

**Console Output Example:**
```
Rate limit hit (attempt 1/6). Waiting 0.12s before retry...
Rate limit hit (attempt 2/6). Waiting 2.00s before retry...
```

---

## Embeddings

### JinaEmbeddingModel

**Location:** `src/kohakurag/embeddings.py`

Sentence embedding model using `jinaai/jina-embeddings-v3` from Hugging Face.

#### Constructor

```python
JinaEmbeddingModel(
    model_name: str = "jinaai/jina-embeddings-v3",
    device: str | None = None,
    trust_remote_code: bool = True,
)
```

**Parameters:**
- `model_name` (str, default: `"jinaai/jina-embeddings-v3"`): Hugging Face model identifier
- `device` (str | None, default: `None`): Device to use (`"cuda"`, `"mps"`, `"cpu"`). Auto-detected if `None`
- `trust_remote_code` (bool, default: `True`): Allow remote code execution (required for Jina models)

#### Properties

##### `dimension -> int`

Returns the embedding dimension (1024 for Jina v3).

#### Methods

##### `async embed(texts: Sequence[str]) -> numpy.ndarray`

Generate embeddings for a batch of texts (async).

**Parameters:**
- `texts` (Sequence[str]): List of text strings to embed

**Returns:**
- `numpy.ndarray`: Array of shape `(len(texts), dimension)` with float32 dtype

**Example:**

```python
import asyncio
from kohakurag.embeddings import JinaEmbeddingModel

async def main():
    embedder = JinaEmbeddingModel()

    # Single text
    embedding = await embedder.embed(["Hello, world!"])
    print(embedding.shape)  # (1, 1024)

    # Batch embedding
    texts = [
        "This is the first sentence.",
        "This is the second sentence.",
        "And a third one for good measure."
    ]
    embeddings = await embedder.embed(texts)
    print(embeddings.shape)  # (3, 1024)

asyncio.run(main())
```

**Performance Notes:**
- First call downloads ~2GB model from Hugging Face
- Automatically uses FP16 on CUDA/MPS for 2x speedup
- Batch processing is more efficient than individual calls
- Thread-safe via single-worker `ThreadPoolExecutor` (no manual locking needed)

---

## Datastore

### KVaultNodeStore

**Location:** `src/kohakurag/datastore.py`

SQLite-backed hierarchical vector store using KohakuVault.

#### Constructor

```python
KVaultNodeStore(
    db_path: str | Path,
    table_prefix: str = "nodes",
    dimensions: int | None = None,
)
```

**Parameters:**
- `db_path` (str | Path): Path to SQLite database file (created if doesn't exist)
- `table_prefix` (str, default: `"nodes"`): Prefix for KohakuVault tables
- `dimensions` (int | None, default: `None`): Embedding dimension (auto-detected from first insert if `None`)

#### Methods

##### `async upsert_nodes(nodes: Sequence[StoredNode]) -> None`

Insert or update nodes in the datastore (async).

**Parameters:**
- `nodes` (Sequence[StoredNode]): List of nodes to upsert

##### `async search(query_vector: np.ndarray, k: int = 5, kinds: set[NodeKind] | None = None) -> list[RetrievalMatch]`

Search for nearest neighbors (async).

**Parameters:**
- `query_vector` (np.ndarray): Query embedding vector
- `k` (int, default: 5): Number of results to return
- `kinds` (set[NodeKind] | None): Filter by node types

**Returns:**
- `list[RetrievalMatch]`: List of matches with nodes and scores

##### `async get_node(node_id: str) -> StoredNode`

Retrieve a node by ID (async).

**Parameters:**
- `node_id` (str): Node identifier

**Returns:**
- `StoredNode`: Node object

**Raises:**
- `KeyError`: If node not found

**Example:**

```python
import asyncio
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel

async def main():
    # Create datastore
    store = KVaultNodeStore(
        db_path="artifacts/my_index.db",
        table_prefix="docs",
        dimensions=1024,
    )

    # Create embeddings
    embedder = JinaEmbeddingModel()
    query_embedding = (await embedder.embed(["How does RAG work?"]))[0]

    # Search
    results = await store.search(query_embedding, k=5)
    for match in results:
        print(f"[{match.score:.3f}] {match.node.text[:100]}...")

asyncio.run(main())
```

---

## RAG Pipeline

### RAGPipeline

**Location:** `src/kohakurag/pipeline.py`

End-to-end RAG orchestration with query planning, retrieval, and answer generation.

#### Constructor

```python
RAGPipeline(
    store: HierarchicalVectorStore,
    embedder: EmbeddingModel,
    chat_model: ChatModel,
    planner: QueryPlanner | None = None,
)
```

**Parameters:**
- `store` (HierarchicalVectorStore): Datastore for retrieval
- `embedder` (EmbeddingModel): Embedding model for queries
- `chat_model` (ChatModel): LLM for answer generation
- `planner` (QueryPlanner | None, optional): Query expansion planner

#### Methods

##### `async index_documents(documents: Iterable[StoredNode]) -> None`

Bulk insert pre-built nodes into the store (async).

##### `async retrieve(question: str, *, top_k: int | None = None) -> RetrievalResult`

Execute multi-query retrieval with hierarchical context expansion (async).

##### `async answer(question: str) -> dict`

Simple QA: retrieve + prompt + generate (async).

##### `async run_qa(...) -> StructuredAnswerResult`

Execute a complete question-answering pipeline (async).

**Parameters:**
- `question` (str): User question
- `system_prompt` (str): System prompt for the LLM
- `user_template` (str): Template for formatting context + question
- `additional_info` (dict[str, Any], optional): Extra metadata for the prompt
- `top_k` (int, default: 5): Number of snippets to retrieve

**Returns:**
- `StructuredAnswerResult`: Object containing:
  - `answer`: Structured answer object
  - `raw_response`: Raw LLM output
  - `prompt`: Final prompt sent to LLM
  - `retrieval`: Retrieval result with matches and snippets

**Example:**

```python
import asyncio
from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel
from kohakurag.llm import OpenAIChatModel

async def main():
    # Initialize components
    store = KVaultNodeStore("artifacts/index.db")
    embedder = JinaEmbeddingModel()
    chat = OpenAIChatModel(model="gpt-4o-mini", max_concurrent=10, max_retries=5)

    # Create pipeline
    pipeline = RAGPipeline(
        store=store,
        embedder=embedder,
        chat_model=chat,
    )

    # Run Q&A
    result = await pipeline.run_qa(
        question="What is the water consumption of GPT-3 training?",
        system_prompt="Answer based only on the provided context.",
        user_template="Question: {question}\n\nContext:\n{context}\n\nAnswer:",
        top_k=6,
    )

    print(result.answer.answer_value)
    print(result.answer.explanation)

    # Batch processing with asyncio.gather()
    questions = ["Q1", "Q2", "Q3"]
    results = await asyncio.gather(*[
        pipeline.run_qa(
            question=q,
            system_prompt="Answer based on context.",
            user_template="Q: {question}\nContext: {context}\nA:",
        )
        for q in questions
    ])

asyncio.run(main())
```

---

## Document Parsing

### pdf_to_document_payload

**Location:** `src/kohakurag/pdf_utils.py`

Extract structured payload from PDF files.

**Signature:**
```python
def pdf_to_document_payload(
    pdf_path: str | Path,
    metadata: dict[str, Any],
) -> DocumentPayload
```

**Parameters:**
- `pdf_path` (str | Path): Path to PDF file
- `metadata` (dict[str, Any]): Document metadata (title, author, URL, etc.)

**Returns:**
- `DocumentPayload`: Structured document with sections, paragraphs, and sentences

**Example:**

```python
from kohakurag.pdf_utils import pdf_to_document_payload

payload = pdf_to_document_payload(
    pdf_path="papers/attention_is_all_you_need.pdf",
    metadata={
        "id": "vaswani2017",
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al.",
        "year": 2017,
    }
)

print(f"Pages: {len(payload.sections)}")
print(f"First paragraph: {payload.sections[0].paragraphs[0].text[:100]}...")
```

### markdown_to_payload

**Location:** `src/kohakurag/parsers.py`

Parse Markdown files with heading-based structure.

**Signature:**
```python
def markdown_to_payload(
    markdown_text: str,
    metadata: dict[str, Any],
) -> DocumentPayload
```

### text_to_payload

**Location:** `src/kohakurag/parsers.py`

Convert plain text to structured payload with heuristic segmentation.

**Signature:**
```python
def text_to_payload(
    text: str,
    metadata: dict[str, Any],
) -> DocumentPayload
```

---

## Indexing

### DocumentIndexer

**Location:** `src/kohakurag/indexer.py`

Build hierarchical tree and compute embeddings for documents.

#### Constructor

```python
DocumentIndexer(
    embedder: EmbeddingModel,
    store: HierarchicalVectorStore,
)
```

**Parameters:**
- `embedder` (EmbeddingModel): Model for generating embeddings
- `store` (HierarchicalVectorStore): Datastore for persistence

#### Methods

##### `async index(payload: DocumentPayload) -> list[StoredNode]`

Index a single document payload (async).

**Parameters:**
- `payload` (DocumentPayload): Structured document to index

**Returns:**
- `list[StoredNode]`: List of storable nodes with embeddings

**Example:**

```python
import asyncio
from kohakurag.indexer import DocumentIndexer
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel
from kohakurag.pdf_utils import pdf_to_document_payload

async def main():
    # Setup
    embedder = JinaEmbeddingModel()
    store = KVaultNodeStore("artifacts/index.db", dimensions=1024)
    indexer = DocumentIndexer(embedder)

    # Index a document
    payload = pdf_to_document_payload(
        pdf_path="papers/bert.pdf",
        doc_id="bert2018",
        title="BERT",
        metadata={"year": 2018},
    )
    nodes = await indexer.index(payload)
    await store.upsert_nodes(nodes)

asyncio.run(main())
```

---

## Error Handling

### Common Exceptions

#### Rate Limit Errors

```python
from kohakurag.llm import OpenAIChatModel
import openai

chat = OpenAIChatModel(max_retries=3)

try:
    response = chat.complete("Hello!")
except openai.RateLimitError as e:
    print(f"Rate limit exceeded after all retries: {e}")
```

#### Missing API Key

```python
from kohakurag.llm import OpenAIChatModel

try:
    chat = OpenAIChatModel()
except ValueError as e:
    print(f"API key not found: {e}")
```

---

## Async Concurrency

### Async-Safe Components

All I/O operations in KohakuRAG are async:

- `JinaEmbeddingModel`: Thread-safe via single-worker `ThreadPoolExecutor`
- `KVaultNodeStore`: Thread-safe via single-worker `ThreadPoolExecutor` for SQLite operations
- `OpenAIChatModel`: Async with semaphore-based concurrency control

### Concurrent Processing Example

```python
import asyncio
from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel
from kohakurag.llm import OpenAIChatModel

async def main():
    # Create shared pipeline (all components are async-safe)
    store = KVaultNodeStore("artifacts/index.db")
    embedder = JinaEmbeddingModel()
    chat = OpenAIChatModel(max_concurrent=10, max_retries=5)
    pipeline = RAGPipeline(store=store, embedder=embedder, chat_model=chat)

    # Concurrent batch processing
    questions = ["Q1", "Q2", "Q3", ...]
    results = await asyncio.gather(*[
        pipeline.run_qa(
            question=q,
            system_prompt="You are a helpful assistant.",
            user_template="Question: {question}\nContext: {context}\nAnswer:",
        )
        for q in questions
    ])

    for result in results:
        print(result.answer.answer_value)

asyncio.run(main())
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `HF_HOME` | Hugging Face cache directory | `~/.cache/huggingface` |
| `CUDA_VISIBLE_DEVICES` | GPU devices to use | All available |

---

## Best Practices

### Rate Limit Management

1. **Start with conservative settings:**
   ```python
   chat = OpenAIChatModel(
       max_concurrent=5,        # Limit concurrent requests
       max_retries=10,
       base_retry_delay=2.0,
   )
   ```

2. **Adjust concurrency when hitting limits:**
   ```python
   # In your config file (e.g., configs/text_only/answer.py)
   max_concurrent = 5  # Reduce from default 10
   ```

3. **Disable rate limiting for self-hosted endpoints:**
   ```python
   # In your config file
   max_concurrent = 0  # Unlimited
   ```

4. **Monitor retry messages in logs:**
   ```
   Rate limit hit (attempt 1/11). Waiting 0.12s before retry...
   ```

### Embedding Performance

1. **Use GPU when available:**
   ```python
   embedder = JinaEmbeddingModel(device="cuda")
   ```

2. **Batch embed for efficiency:**
   ```python
   # Good: batch embedding
   embeddings = await embedder.embed(all_texts)

   # Bad: individual calls in sequence
   embeddings = [await embedder.embed([text])[0] for text in all_texts]

   # Better: concurrent individual calls (if needed)
   embeddings = await asyncio.gather(*[
       embedder.embed([text]) for text in all_texts
   ])
   ```

### Datastore Management

1. **Use consistent table prefixes:**
   ```python
   store = KVaultNodeStore("index.db", table_prefix="v2")  # Isolate versions
   ```

2. **Backup before re-indexing:**
   ```bash
   cp artifacts/wattbot.db artifacts/wattbot_backup.db
   ```

---

For more examples, see the [Usage Guide](usage.md) and [WattBot Playbook](wattbot.md).
