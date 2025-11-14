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
    base_retry_delay: float = 1.0,
)
```

**Parameters:**

- `model` (str, default: `"gpt-4o-mini"`): OpenAI model identifier
- `api_key` (Optional[str], default: `None`): OpenAI API key. If `None`, reads from `OPENAI_API_KEY` environment variable or `.env` file
- `organization` (Optional[str], default: `None`): OpenAI organization ID
- `system_prompt` (str | None, default: `None`): Default system prompt for all completions
- `max_retries` (int, default: `5`): Maximum number of retry attempts on rate limit errors
- `base_retry_delay` (float, default: `1.0`): Base delay in seconds for exponential backoff

**Raises:**
- `ImportError`: If `openai>=1.0.0` is not installed
- `ValueError`: If no API key is found

#### Methods

##### `complete(prompt: str, *, system_prompt: str | None = None) -> str`

Execute a chat completion request with automatic rate limit retry.

**Parameters:**
- `prompt` (str): User prompt to send to the model
- `system_prompt` (str | None, optional): Override the default system prompt for this request

**Returns:**
- `str`: Model's response content

**Raises:**
- `openai.RateLimitError`: If rate limit persists after all retries
- `openai.OpenAIError`: For other API errors

**Retry Behavior:**

The method automatically handles rate limit errors using an intelligent retry strategy:

1. **Server-recommended delays**: Parses error messages for suggested wait times (e.g., "Please try again in 23ms")
2. **Exponential backoff**: Falls back to 1s, 2s, 4s, 8s, 16s... if no specific delay is provided
3. **Automatic retry**: Continues until success or `max_retries` is exhausted

**Example:**

```python
from kohakurag.llm import OpenAIChatModel

# Basic usage
chat = OpenAIChatModel(model="gpt-4o-mini")
response = chat.complete("Explain quantum computing in one sentence.")

# Configure retry behavior for restrictive rate limits
chat = OpenAIChatModel(
    model="gpt-4o-mini",
    max_retries=10,          # More retries for TPM-constrained accounts
    base_retry_delay=2.0,    # Longer initial delay
)
response = chat.complete("What is the capital of France?")

# Override system prompt per request
chat = OpenAIChatModel(
    system_prompt="You are a helpful assistant."
)
response = chat.complete(
    "Explain RAG systems",
    system_prompt="You are an expert in information retrieval."
)
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
| 1       | 1.0                 |
| 2       | 2.0                 |
| 3       | 4.0                 |
| 4       | 8.0                 |
| 5       | 16.0                |

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

##### `embed(texts: Sequence[str]) -> numpy.ndarray`

Generate embeddings for a batch of texts.

**Parameters:**
- `texts` (Sequence[str]): List of text strings to embed

**Returns:**
- `numpy.ndarray`: Array of shape `(len(texts), dimension)` with float32 dtype

**Example:**

```python
from kohakurag.embeddings import JinaEmbeddingModel

embedder = JinaEmbeddingModel()

# Single text
embedding = embedder.embed(["Hello, world!"])
print(embedding.shape)  # (1, 1024)

# Batch embedding
texts = [
    "This is the first sentence.",
    "This is the second sentence.",
    "And a third one for good measure."
]
embeddings = embedder.embed(texts)
print(embeddings.shape)  # (3, 1024)
```

**Performance Notes:**
- First call downloads ~2GB model from Hugging Face
- Automatically uses FP16 on CUDA/MPS for 2x speedup
- Batch processing is more efficient than individual calls

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

##### `insert(nodes: Sequence[StoredNode]) -> None`

Insert nodes into the datastore.

**Parameters:**
- `nodes` (Sequence[StoredNode]): List of nodes to insert

##### `search(query_vector: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]`

Search for nearest neighbors.

**Parameters:**
- `query_vector` (np.ndarray): Query embedding vector
- `top_k` (int, default: 10): Number of results to return

**Returns:**
- `list[tuple[str, float]]`: List of (node_id, similarity_score) pairs

##### `get(node_id: str) -> StoredNode | None`

Retrieve a node by ID.

**Parameters:**
- `node_id` (str): Node identifier

**Returns:**
- `StoredNode | None`: Node object or `None` if not found

**Example:**

```python
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel

# Create datastore
store = KVaultNodeStore(
    db_path="artifacts/my_index.db",
    table_prefix="docs",
    dimensions=1024,
)

# Create embeddings
embedder = JinaEmbeddingModel()
query_embedding = embedder.embed(["How does RAG work?"])[0]

# Search
results = store.search(query_embedding, top_k=5)
for node_id, score in results:
    node = store.get(node_id)
    print(f"[{score:.3f}] {node.text[:100]}...")
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

##### `run_qa(...) -> QAResult`

Execute a complete question-answering pipeline.

**Parameters:**
- `question` (str): User question
- `system_prompt` (str): System prompt for the LLM
- `user_template` (str): Template for formatting context + question
- `additional_info` (dict[str, Any], optional): Extra metadata for the prompt
- `top_k` (int, default: 5): Number of snippets to retrieve

**Returns:**
- `QAResult`: Object containing:
  - `answer`: Structured answer object
  - `raw_response`: Raw LLM output
  - `prompt`: Final prompt sent to LLM

**Example:**

```python
from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel
from kohakurag.llm import OpenAIChatModel

# Initialize components
store = KVaultNodeStore("artifacts/index.db")
embedder = JinaEmbeddingModel()
chat = OpenAIChatModel(model="gpt-4o-mini", max_retries=5)

# Create pipeline
pipeline = RAGPipeline(
    store=store,
    embedder=embedder,
    chat_model=chat,
)

# Run Q&A
result = pipeline.run_qa(
    question="What is the water consumption of GPT-3 training?",
    system_prompt="Answer based only on the provided context.",
    user_template="Question: {question}\n\nContext:\n{context}\n\nAnswer:",
    top_k=6,
)

print(result.answer.answer_value)
print(result.answer.explanation)
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

##### `index_document(payload: DocumentPayload) -> None`

Index a single document payload.

**Parameters:**
- `payload` (DocumentPayload): Structured document to index

**Example:**

```python
from kohakurag.indexer import DocumentIndexer
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel
from kohakurag.pdf_utils import pdf_to_document_payload

# Setup
embedder = JinaEmbeddingModel()
store = KVaultNodeStore("artifacts/index.db", dimensions=1024)
indexer = DocumentIndexer(embedder=embedder, store=store)

# Index a document
payload = pdf_to_document_payload(
    pdf_path="papers/bert.pdf",
    metadata={"id": "bert2018", "title": "BERT"},
)
indexer.index_document(payload)
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

## Thread Safety

### Thread-Safe Components

- `JinaEmbeddingModel`: Uses locks for concurrent embedding calls
- `KVaultNodeStore`: SQLite with thread-safe connections
- `OpenAIChatModel`: Each instance maintains its own client (thread-safe)

### Concurrent Processing Example

```python
import concurrent.futures
from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel
from kohakurag.llm import OpenAIChatModel

# Shared embedder with thread-safe access
embedder = JinaEmbeddingModel()

def process_question(question: str) -> str:
    # Each worker gets its own store and chat client
    store = KVaultNodeStore("artifacts/index.db")
    chat = OpenAIChatModel(max_retries=5)
    pipeline = RAGPipeline(store=store, embedder=embedder, chat_model=chat)

    result = pipeline.run_qa(question, ...)
    return result.answer.answer_value

questions = ["Q1", "Q2", "Q3", ...]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    answers = list(executor.map(process_question, questions))
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
       max_retries=10,
       base_retry_delay=2.0,
   )
   ```

2. **Reduce parallel workers when hitting limits:**
   ```bash
   python scripts/wattbot_answer.py --max-workers 2  # Instead of 4
   ```

3. **Monitor retry messages in logs:**
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
   embeddings = embedder.embed(all_texts)

   # Bad: individual calls
   embeddings = [embedder.embed([text])[0] for text in all_texts]
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
