# KohakuRAG Architecture

KohakuRAG is a general-purpose Retrieval-Augmented Generation (RAG) stack. The core library exposes reusable abstractions for hierarchical document storage, embedding, retrieval, and LLM orchestration. Project-specific logic (such as the WattBot 2025 CSV schema) lives in thin adapters so that other datasets can reuse the exact same core without modification.

## Goals
- **Project-agnostic foundation** – datastore and pipeline APIs are neutral; WattBot integrations plug in through adapters (question loaders, answer serializers, metadata resolvers).
- **Hierarchical retrieval** – documents are parsed into trees (`document → section → paragraph → sentence`). Queries can land on any level and still return an extended context window (e.g., the entire paragraph when a sentence matches).
- **Efficient vector storage** – the default backend targets KohakuVault (SQLite + sqlite-vec) but also supports in-memory or JSON dumps for quick experiments.
- **LLM-friendly orchestration** – the query planner, retriever, and answerer are modular so that developers can swap in different planners (rule-based, LLM) or models (OpenAI, local) without touching storage code.
- **Consistent modeling** – every environment (dev, prod, tests) uses the same `jinaai/jina-embeddings-v3` backbone so retrieval quality stays predictable. When needed, tests can patch the embedding layer with a fixture but we avoid alternate models.

## Module overview

### Embeddings
- `JinaEmbeddingModel` – wraps `jinaai/jina-embeddings-v3` via `AutoModel.from_pretrained(..., trust_remote_code=True)`. It automatically selects CUDA or MPS when available, defaults to FP16 on accelerators, and exposes a simple `embed(list[str]) -> numpy.ndarray` interface.
- If a future project needs a different encoder, it can subclass the `EmbeddingModel` protocol, but the default builds, scripts, and tests all rely on the Jina model to keep behavior consistent.

### Datastore
- **Nodes**: Each node stores ids, parent/child pointers, raw text, metadata (source, title, citation ids), and an embedding vector.
- **Interpolation**: Leaf vectors come directly from the embedding model; parents use weighted averaging (default weight: child token length). Custom interpolators can be registered.
- **Storage adapters**: `HierarchicalVectorStore` defines CRUD and nearest-neighbor search. Two KohakuVault-backed adapters are planned:
  - `KVaultStore`: a key-value store (`{id: data}`) for metadata and tree nodes. Data must be msgpack-packable (dict/list/primitive). Non-packable objects fall back to pickle (slower), so serializers should stick to JSON-like structures.
  - `VectorKVaultStore`: a vector-only store (`{embedding: id}`) where embeddings are indexed in sqlite-vec and the ids reference the rich node payload in the kv store. This separation keeps lookup fast while allowing the metadata document to evolve without re-writing large blobs.
- An in-memory implementation ships with the library for quick iteration.
- **Search expansion**: Retrieval always returns both the matched node and optional sibling/parent context so prompts can include broader evidence when needed.

### Indexing
- **Input format**: `DocumentPayload` objects with nested sections/paragraphs/sentences. Helpers convert plain text, Markdown, or PDFs into that structure so downstream components remain agnostic to the original source format.
- **Segmentation**: When no structure is provided, the indexer heuristically splits text into paragraphs and sentences. Structured inputs can provide their own sentence boundaries.
- **Metadata**: The indexer attaches arbitrary metadata dicts (e.g., WattBot reference ids, URLs) to each node.
- **Output**: The resulting tree is fed to the datastore where embeddings are computed and stored.

### RAG pipeline
1. **Planner** – takes a question and emits retrieval intents (keywords, boolean filters, etc.). The default planner simply forwards the raw question; advanced planners can be LLM-powered.
2. **Retriever** – executes each intent against the datastore, returning top-k nodes plus expanded context snippets.
3. **Answerer** – sends the question + snippets + project-specific instructions to a chat model. The chat backend is pluggable (OpenAI, Azure, OSS). Responses are validated against a schema interface so different projects can define custom answer shapes.
4. **Post-processing** – converts the structured answer object into whatever output format the caller expects (CSV row, JSON payload, etc.).

### LLM Integration with Automatic Rate Limit Handling

The `OpenAIChatModel` class provides production-ready OpenAI integration with intelligent rate limit management:

**OpenAI-Compatible Backends:**
- The implementation uses the official `openai` Python client and can talk to **any endpoint that implements the OpenAI Chat Completions protocol**, not just `api.openai.com`.
- Use the `base_url` argument or `OPENAI_BASE_URL` environment variable to point at self-hosted or proxy servers (for example, vLLM, llama.cpp, or an OpenAI-compatible gateway that fronts Anthropic/Gemini).
- This keeps the core RAG pipeline dependency surface small while allowing you to swap providers without changing pipeline code—only the configuration and model name change.

**Rate Limit Resilience:**
- **Automatic retry** – catches `openai.RateLimitError` and retries with configurable backoff
- **Server-guided delays** – parses error messages like "Please try again in 23ms" and respects the suggested wait time
- **Exponential backoff** – falls back to `base_retry_delay * (2 ** attempt)` (for example, 3s, 6s, 12s, 24s, 48s with the default settings) if no specific delay is provided
- **Configurable behavior** – adjust `max_retries` and `base_retry_delay` based on your rate limits

**Why This Matters:**
- OpenAI's TPM (tokens per minute) limits can be restrictive for batch processing
- The WattBot 2025 competition example uses `gpt-4o-mini` with 500K TPM limits
- Automatic retry prevents pipeline failures during long-running batch jobs
- Transparent logging shows retry attempts without requiring manual intervention

**Implementation Details:**
```python
# In src/kohakurag/llm.py
def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
    for attempt in range(self._max_retries + 1):
        try:
            return self._client.chat.completions.create(...)
        except RateLimitError as e:
            wait_time = self._parse_retry_after(error_msg) or (self._base_retry_delay * (2 ** attempt))
            print(f"Rate limit hit (attempt {attempt + 1}). Waiting {wait_time:.2f}s...")
            time.sleep(wait_time)
```

This design ensures that:
1. **No manual intervention** – scripts continue running despite rate limits
2. **Optimal throughput** – uses server-recommended delays instead of arbitrary waits
3. **Cost efficiency** – avoids token waste from failed requests
4. **Monitoring friendly** – prints clear retry messages for observability

## Development workflow
1. **Document staging** – convert PDFs or external data into structured payloads (optionally via Markdown/JSON intermediates) with per-section text and captions.
2. **Index build** – run the project-specific ingestion script (for example, the WattBot helpers under `scripts/`) to feed payloads through the core indexer and populate the datastore.
3. **Retrieval sanity check** – call the demo CLI (e.g., `scripts/wattbot_demo_query.py`) or your own equivalent to make sure queries hit relevant snippets.
4. **Answer generation** – connect the `RAGPipeline` to your preferred chat model and output format (CSV, JSON, API response, etc.) via a thin adapter.

While WattBot currently supplies the main adapters and scripts, future projects can replicate the same pattern—only the thin ingestion/answer wrappers change; the `kohakurag` package stays untouched.
