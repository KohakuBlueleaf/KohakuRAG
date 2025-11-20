# Deployment and LLM Backends

KohakuRAG's core library depends only on the official `openai` Python client, but it can talk to **any server that implements the OpenAI Chat Completions protocol**. This includes:

- OpenAI's own `api.openai.com` (e.g., `gpt-4o-mini`, `gpt-4.1`, `gpt-5.1-mini`)
- Self-hosted OpenAI-compatible servers (vLLM, llama.cpp)
- Proxies/aggregators that front other providers (Anthropic, Gemini, etc.) behind an OpenAI-style `/v1/chat/completions` API

The goal is to keep the RAG pipeline agnostic to the actual provider: you configure one OpenAI-compatible endpoint and model name, and everything else stays the same.

---

## 1. Configuration Overview

`OpenAIChatModel` resolves credentials and endpoints as follows:

1. **API key**
   - Explicit `api_key` argument (highest priority), or
   - `OPENAI_API_KEY` from the environment, or
   - `OPENAI_API_KEY` from a local `.env` file.

2. **Base URL (OpenAI-compatible endpoint)**
   - Explicit `base_url` argument (highest priority), or
   - `OPENAI_BASE_URL` from the environment, or
   - `OPENAI_BASE_URL` from a local `.env` file.

If `OPENAI_API_KEY` is missing everywhere, `OpenAIChatModel` raises a `ValueError`. `OPENAI_BASE_URL` is optional; if it is not set, the client defaults to OpenAI's hosted API.

This means you can switch between OpenAI, self-hosted vLLM/llama.cpp, or a multi-provider proxy **without changing application code**—only environment variables and the config settings.

---

## 2. Direct OpenAI (default)

For standard OpenAI usage, no base URL override is required.

```bash
export OPENAI_API_KEY=sk-...
```

**Config** (`configs/text_only/answer.py`):
```python
from kohakuengine import Config

db = "artifacts/wattbot.db"
table_prefix = "wattbot"
questions = "data/test_Q.csv"
output = "artifacts/wattbot_answers.csv"
model = "gpt-4o-mini"
top_k = 6
max_concurrent = 10
max_retries = 3

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_answer.py --config configs/text_only/answer.py
```

**Notes on TPM limits (e.g., `gpt-4o-mini`):**

- Lower TPM → use smaller `max_concurrent` (e.g., `5`), smaller `top_k`, and possibly larger `max_retries` in config.
- Higher TPM → you can safely increase `max_concurrent` and `top_k` for throughput.
- Self-hosted → set `max_concurrent = 0` for unlimited concurrency.

---

## 3. Self-hosted vLLM (OpenAI-compatible)

vLLM exposes an OpenAI-style API when started in OpenAI server mode. A typical launch command looks like:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <hf-model-id-or-path> \
    --host 0.0.0.0 \
    --port 8000
```

Then point KohakuRAG at this server:

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"  # vLLM often ignores the key but the client requires one
```

**Config:**
```python
from kohakuengine import Config

db = "artifacts/wattbot.db"
table_prefix = "wattbot"
questions = "data/test_Q.csv"
output = "artifacts/wattbot_answers.csv"
model = "<your-vllm-model-name>"
top_k = 6
max_concurrent = 0  # No rate limiting for local endpoint

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_answer.py --config configs/vllm_answer.py
```

As long as vLLM implements the OpenAI Chat Completions protocol, `OpenAIChatModel` will work unchanged. All scripts use async for efficient concurrent processing.

---

## 4. Self-hosted llama.cpp (OpenAI-compatible)

The llama.cpp server also exposes an OpenAI-style API when run with the `--api` flag. For example:

```bash
./llama-server \
    -m /path/to/model.gguf \
    --port 8000 \
    --host 0.0.0.0 \
    --api
```

Configure KohakuRAG:

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"
```

**Config:**
```python
from kohakuengine import Config

db = "artifacts/wattbot.db"
table_prefix = "wattbot"
questions = "data/test_Q.csv"
output = "artifacts/wattbot_answers.csv"
model = "<llama-model-name>"
top_k = 6
max_concurrent = 0  # No rate limiting for local endpoint

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_answer.py --config configs/llama_answer.py
```

Again, no code changes are required—only the endpoint and model name differ. The async architecture efficiently handles concurrent requests.

---

## 5. Anthropic, Gemini, and Other Providers via Proxies

Anthropic and Gemini expose their own HTTP APIs and SDKs; KohakuRAG does **not** talk to them directly. Instead, you can run (or use) an OpenAI-compatible proxy that forwards requests to these providers. Common patterns include:

- Self-hosted gateways (for example, a small proxy built with tools like LiteLLM or similar) that expose `/v1/chat/completions` and translate model names such as `claude-3-opus` or `gemini-1.5-pro` to the provider-specific API.
- Hosted aggregators that present a single OpenAI-style endpoint and route traffic to multiple backends based on the `model` string.

From KohakuRAG's perspective, such a proxy looks like "just another OpenAI-compatible server":

```bash
export OPENAI_API_KEY="your-proxy-api-key"
export OPENAI_BASE_URL="https://your-proxy.example.com/v1"
```

**Config:**
```python
from kohakuengine import Config

db = "artifacts/wattbot.db"
table_prefix = "wattbot"
questions = "data/test_Q.csv"
output = "artifacts/wattbot_answers.csv"
model = "claude-3-opus"  # or gemini-1.5-pro, etc.
top_k = 6
max_concurrent = 10

def config_gen():
    return Config.from_globals()
```

**Run:**
```bash
kogine run scripts/wattbot_answer.py --config configs/proxy_answer.py
```

The proxy is responsible for holding provider-specific secrets (`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, etc.), enforcing per-provider rate limits, and translating between the OpenAI schema and the provider's native API.

This pattern lets you mix and match:

- OpenAI models (direct or via proxy)
- Anthropic models
- Gemini models
- Local OSS models (vLLM / llama.cpp)

all through a single OpenAI-compatible interface.

---

## 6. Mixing Providers in One Project

The RAG pipeline only depends on the `ChatModel` protocol. In your own scripts, you can instantiate multiple `OpenAIChatModel` instances with different endpoints:

```python
import asyncio
from kohakurag.llm import OpenAIChatModel
from kohakurag import RAGPipeline

async def main():
    # Planner uses a fast/cheap model (maybe OpenAI or a small local model)
    planner_chat = OpenAIChatModel(
        model="gpt-4o-mini",
        max_concurrent=20,
    )

    # Answerer uses a larger or different backend via an OpenAI-compatible proxy
    answer_chat = OpenAIChatModel(
        model="claude-3-opus",
        base_url="https://your-proxy.example.com/v1",
        api_key="proxy-api-key",
        max_concurrent=10,
    )

    # Use in pipeline
    pipeline = RAGPipeline(chat_model=answer_chat, ...)
    result = await pipeline.run_qa(...)

asyncio.run(main())
```

You can wire these into `RAGPipeline` as separate components, or continue to use the existing WattBot scripts and switch providers globally via `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and the config settings. All operations are async for efficient concurrent processing.

For more details on how the pipeline uses `ChatModel`, see `docs/architecture.md` and `docs/api_reference.md`.

