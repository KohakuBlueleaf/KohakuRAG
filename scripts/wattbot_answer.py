"""Answer WattBot questions using the KohakuRAG pipeline + OpenAI.

This script demonstrates end-to-end RAG usage:
- Loads questions from CSV
- Retrieves relevant context from the index
- Generates structured answers via OpenAI
- Handles rate limits automatically
- Supports concurrent processing with asyncio.gather()

Usage (CLI):
    python scripts/wattbot_answer.py \\
        --db artifacts/wattbot.db \\
        --questions data/test_Q.csv \\
        --output artifacts/answers.csv \\
        --model gpt-4o-mini

Usage (KohakuEngine):
    kogine run scripts/wattbot_answer.py --config configs/answer_config.py
"""

import asyncio
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from openai import BadRequestError

from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.embeddings import JinaEmbeddingModel
from kohakurag.llm import OpenAIChatModel


Row = dict[str, Any]
BLANK_TOKEN = "is_blank"

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

db = "artifacts/wattbot.db"
table_prefix = "wattbot"
questions = "data/test_Q.csv"
output = "artifacts/wattbot_answers.csv"
model = "gpt-4o-mini"
top_k = 5
planner_model = None  # Falls back to model
planner_max_queries = 3
metadata = "data/metadata.csv"
max_retries = 3
max_concurrent = 10
single_run_debug = False
question_id = None
with_images = False
top_k_images = 0

# ============================================================================
# PROMPT TEMPLATES
# WattBot-specific prompts live in this script; core library stays generic.
# ============================================================================
ANSWER_SYSTEM_PROMPT = """
You must answer strictly based on the provided context snippets.
Do NOT use external knowledge or assumptions.
If the context does not clearly support an answer, you must output the literal string "is_blank" for both answer_value and ref_id.
The additional info JSON contains an "answer_unit" field indicating the unit for the final answer_value.
You MUST reason about this unit explicitly in your explanation (e.g., what the unit means and how it is applied or converted) and ensure answer_value is expressed in that unit with no unit name included.
""".strip()

PLANNER_SYSTEM_PROMPT = """
Rewrite the user question into focused document search queries.
- Keep the first query identical to the original question.
- Optionally add a few short queries that highlight key entities, numbers, or model names.
- Respond with JSON: {"queries": ["query 1", "query 2", ...]}.
""".strip()

USER_PROMPT_TEMPLATE = """
You will be given a question and context snippets taken from documents.
You must follow these rules:
- Use only the provided context; do not rely on external knowledge.
- If the context does not clearly support an answer, use "is_blank".
- The additional info JSON contains an "answer_unit" field; you MUST interpret what this unit means and explain how you apply it in your reasoning.
- Express answer_value in that unit and do NOT include the unit name in answer_value.
- If the answer is a numeric range, format it as [lower,upper] using the requested unit.

Additional info (JSON): {additional_info_json}

Question: {question}

Context:
{context}

Return STRICT JSON with the following keys, in this order:
- explanation          (1–3 sentences explaining how the context supports the answer AND how you use the answer_unit; or "is_blank")
- answer               (short sentence in natural language)
- answer_value         (string with ONLY the numeric or categorical value in the requested unit, or "is_blank")
- ref_id               (list of document ids from the context used as evidence; or "is_blank")

JSON Answer:
""".strip()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_metadata_records(path: Path) -> dict[str, dict[str, str]]:
    """Load document metadata CSV into a lookup dict."""
    records: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            doc_id = row.get("id")
            if not doc_id:
                continue
            records[doc_id] = row
    if not records:
        raise ValueError(f"No metadata rows found in {path}")
    return records


def build_ref_details(
    ref_ids: Sequence[str],
    metadata: Mapping[str, Mapping[str, str]],
) -> tuple[str, str]:
    """Convert reference document IDs into URLs and citations.

    Returns:
        (ref_url, supporting_materials) tuple in WattBot CSV format
    """
    urls: list[str] = []
    snippets: list[str] = []

    # Extract URLs and citations from metadata
    for ref_id in ref_ids:
        key = str(ref_id).strip()
        if not key:
            continue
        row = metadata.get(key)
        if not row:
            continue
        url = (row.get("url") or "").strip()
        if url:
            urls.append(url)
        snippet = (row.get("citation") or row.get("title") or key).strip()
        if snippet:
            snippets.append(snippet)

    # Format as WattBot CSV expects: ['url1','url2']
    if urls:
        joined = ",".join(f"'{u}'" for u in urls)
        ref_url = f"[{joined}]"
    else:
        ref_url = BLANK_TOKEN

    supporting = " | ".join(snippets) if snippets else BLANK_TOKEN
    return ref_url, supporting


def normalize_answer_value(raw: str, question: str) -> str:
    """Apply domain-specific normalization to answer values.

    - True/False questions → 1/0
    - Numeric ranges → [lower,upper] format
    """
    value = (raw or "").strip()
    if not value or value.lower() == BLANK_TOKEN:
        return BLANK_TOKEN

    q_lower = question.strip().lower()

    # Normalize True/False questions to 1/0.
    if q_lower.startswith("true or false"):
        v_lower = value.lower()
        if v_lower in {"true", "1"}:
            return "1"
        if v_lower in {"false", "0"}:
            return "0"
        if "true" in v_lower and "false" not in v_lower:
            return "1"
        if "false" in v_lower and "true" not in v_lower:
            return "0"

    # Normalize simple numeric ranges to [lower,upper].
    # Heuristic: if there are exactly two numbers and the text contains a range marker.
    nums = re.findall(r"-?\d+(?:\.\d+)?", value)
    if len(nums) == 2 and any(
        marker in value for marker in ["-", "–", "—", " to ", "–"]
    ):
        try:
            a = float(nums[0])
            b = float(nums[1])
            lo, hi = sorted((a, b))
            return f"[{lo},{hi}]"
        except ValueError:
            pass

    return value


# ============================================================================
# GLOBAL SHARED EMBEDDER
# Pre-load the Jina model once and share across concurrent tasks
# ============================================================================

# Create a single shared embedder (thread-safe via async executor)
GLOBAL_EMBEDDER = JinaEmbeddingModel()


# ============================================================================
# QUERY PLANNER
# ============================================================================


class LLMQueryPlanner:
    """LLM-backed planner that proposes follow-up retrieval queries."""

    def __init__(self, chat: OpenAIChatModel, max_queries: int = 3) -> None:
        self._chat = chat
        self._max_queries = max(1, max_queries)

    async def plan(self, question: str) -> Sequence[str]:
        """Generate multiple retrieval queries from a single question.

        Strategy:
        1. Always include the original question
        2. Ask LLM to generate paraphrases/entity-focused queries
        3. Fall back to simple reformulation if LLM fails
        """
        base = [question.strip()]
        prompt = f"""
You convert a WattBot question into targeted document search queries.
- The first retrieval query should remain the original question.
- Generate up to {self._max_queries - 1} additional short queries that highlight key entities, units, or paraphrases.
- Respond with JSON: {{"queries": ["query 1", "query 2"]}}
- Return an empty list if the question is already precise.

Question: {question.strip()}

JSON:
""".strip()

        # Ask LLM to generate query variations
        raw = await self._chat.complete(prompt)

        # Parse JSON response
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            extracted = raw[start:end]
            data = json.loads(extracted)
            items = data.get("queries")
            extra = [str(item).strip() for item in items or [] if str(item).strip()]
        except Exception:
            extra = []  # If LLM returns invalid JSON, just use original question

        # Deduplicate and enforce max_queries limit
        seen = {q.lower() for q in base if q}
        for query in extra:
            key = query.lower()
            if key in seen:
                continue
            base.append(query)
            seen.add(key)
            if len(base) >= self._max_queries:
                break

        # Fallback: add simple reformulation if LLM provided nothing useful
        if len(base) == 1:
            reformulation = question.strip().split("?", 1)[0].strip()
            if reformulation and reformulation.lower() not in seen:
                base.append(reformulation)
        return base


# ============================================================================
# DATA LOADING
# ============================================================================


def ensure_columns(row: Row, columns: Sequence[str]) -> Row:
    """Ensure row has all required columns (filling missing with empty strings)."""
    out = {col: row.get(col, "") for col in columns}
    return out


def load_questions(path: Path) -> tuple[list[Row], list[str]]:
    """Load questions CSV and infer column names."""
    with path.open(newline="", encoding="utf-8-sig") as f_in:
        reader = csv.DictReader(f_in)
        rows = [dict(row) for row in reader]
        if not rows:
            raise ValueError("Question CSV is empty.")
        columns = reader.fieldnames or [
            "id",
            "question",
            "answer",
            "answer_value",
            "answer_unit",
            "ref_id",
            "ref_url",
            "supporting_materials",
            "explanation",
        ]
    return rows, list(columns)


# ============================================================================
# SHARED RESOURCES
# All async tasks share the same pipeline (thread-safe via async)
# ============================================================================


@dataclass(frozen=True)
class AppConfig:
    """Immutable config for the application."""

    db: Path
    table_prefix: str
    model: str
    top_k: int
    planner_model: str
    planner_max_queries: int
    metadata: Mapping[str, Mapping[str, str]]
    max_retries: int
    max_concurrent: int
    with_images: bool = False
    top_k_images: int = 0


@dataclass(frozen=True)
class AnswerResult:
    """Result from answering a single question."""

    position: int
    row: Row
    message: str


def create_pipeline(config: AppConfig) -> RAGPipeline:
    """Create a shared RAG pipeline for all async tasks."""
    # Datastore (thread-safe via async executor)
    store = KVaultNodeStore(
        config.db,
        table_prefix=config.table_prefix,
        dimensions=None,
    )

    # Query planner LLM (generates retrieval queries)
    planner_chat = OpenAIChatModel(
        model=config.planner_model,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        max_concurrent=config.max_concurrent,
    )
    planner = LLMQueryPlanner(
        chat=planner_chat,
        max_queries=config.planner_max_queries,
    )

    # Answer LLM (generates final structured answers)
    chat = OpenAIChatModel(
        model=config.model,
        system_prompt=ANSWER_SYSTEM_PROMPT,
        max_concurrent=config.max_concurrent,
    )

    # Assemble the full RAG pipeline
    pipeline = RAGPipeline(
        store=store,
        embedder=GLOBAL_EMBEDDER,  # Shared embedder
        chat_model=chat,
        planner=planner,
    )

    return pipeline


# ============================================================================
# QUESTION ANSWERING
# ============================================================================


async def _answer_single_row(
    idx: int,
    row: Row,
    columns: Sequence[str],
    config: AppConfig,
    pipeline: RAGPipeline,
    retry_count: int | None = None,
    override_top_k: int | None = None,
) -> AnswerResult:
    """Answer a single question with retry logic for blank responses and context overflow handling.

    Strategy:
    - Start with top_k context snippets (or override_top_k if provided)
    - If answer is blank, retry with 2*top_k, then 3*top_k, etc. (unless retry_count=0)
    - If context overflow (400 error), recursively retry with retry_count=0 and k=k-1
    - Return immediately after successful context overflow recovery

    Args:
        retry_count: If 0, skip blank retries. If None, use config.max_retries.
        override_top_k: If provided, use this instead of config.top_k as base.
    """
    if override_top_k is not None and override_top_k <= 0:
        raise ValueError(f"override_top_k={override_top_k} must be > 0")

    question = row["question"]
    additional_info: dict[str, Any] = {
        "answer_unit": (row.get("answer_unit") or "").strip(),
        "question_id": row.get("id", "").strip(),
    }

    # Determine retry behavior
    max_retries = 0 if retry_count == 0 else (retry_count or config.max_retries)
    base_top_k = override_top_k or config.top_k

    # Retry loop: increase context window each iteration if answer is blank
    qa_result = None
    structured = None

    for attempt in range(max_retries + 1):
        current_top_k = base_top_k * (attempt + 1)

        try:
            qa_result = await pipeline.run_qa(
                question,
                system_prompt=ANSWER_SYSTEM_PROMPT,
                user_template=USER_PROMPT_TEMPLATE,
                additional_info=additional_info,
                top_k=current_top_k,
                with_images=config.with_images,
                top_k_images=config.top_k_images,
            )
            structured = qa_result.answer
            is_blank = (
                structured.answer_value.strip().lower() == BLANK_TOKEN
                or not structured.ref_id
            )

            if not is_blank:
                break  # Got a valid answer, stop retrying

        except BadRequestError as e:
            # Check if it's a context length overflow error
            error_msg = str(e).lower()
            if "context length" in error_msg or "maximum context" in error_msg:
                reduced_top_k = current_top_k - 1
                return await _answer_single_row(
                    idx,
                    row,
                    columns,
                    config,
                    pipeline,
                    retry_count=0,
                    override_top_k=reduced_top_k,
                )
            else:
                raise  # Not a context overflow error
        except json.decoder.JSONDecodeError:
            reduced_top_k = current_top_k - 1
            return await _answer_single_row(
                idx,
                row,
                columns,
                config,
                pipeline,
                retry_count=0,
                override_top_k=reduced_top_k,
            )

    # Create blank result if we don't have a valid answer
    if qa_result is None or structured is None:
        structured = SimpleNamespace(
            answer="", answer_value=BLANK_TOKEN, ref_id=[], explanation=""
        )

    is_blank = (
        structured.answer_value.strip().lower() == BLANK_TOKEN or not structured.ref_id
    )

    # Format output row based on answer status
    result = dict(row)

    if is_blank:
        result["answer"] = BLANK_TOKEN
        result["answer_value"] = BLANK_TOKEN
        result["ref_id"] = BLANK_TOKEN
        result["ref_url"] = BLANK_TOKEN
        result["supporting_materials"] = BLANK_TOKEN
        result["explanation"] = BLANK_TOKEN

    else:
        # Populate with structured answer fields
        result["answer"] = structured.answer or BLANK_TOKEN

        normalized_value = normalize_answer_value(structured.answer_value, question)
        result["answer_value"] = normalized_value or BLANK_TOKEN

        # Format ref_id as list: ['doc1','doc2']
        if structured.ref_id:
            joined_ids = ",".join(f"'{rid}'" for rid in structured.ref_id)
            result["ref_id"] = f"[{joined_ids}]"
        else:
            result["ref_id"] = BLANK_TOKEN

        # Resolve URLs and citations from metadata
        ref_url, supporting = build_ref_details(structured.ref_id, config.metadata)
        result["ref_url"] = ref_url
        result["supporting_materials"] = supporting
        result["explanation"] = structured.explanation or BLANK_TOKEN

    # Build progress message
    row_id = row.get("id") or row.get("question", "")[:24] or f"row-{idx}"
    message = f"Answered {row_id} - {question[:60]}..."

    return AnswerResult(
        position=idx,
        row=ensure_columns(result, columns),
        message=message,
    )


async def answer_questions(
    rows: Sequence[Row],
    columns: Sequence[str],
    config: AppConfig,
    pipeline: RAGPipeline,
):
    """Process all questions concurrently, yielding results as they complete."""
    # Create async tasks for all questions
    tasks = [
        asyncio.create_task(_answer_single_row(idx, row, columns, config, pipeline))
        for idx, row in enumerate(rows)
    ]

    # Yield results as they complete
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(result.message)
        yield result


# ============================================================================
# SINGLE-RUN DEBUG MODE
# ============================================================================


async def run_single_question_debug(
    first_row: Row,
    columns: Sequence[str],
    config: AppConfig,
    pipeline: RAGPipeline,
    writer: csv.DictWriter,
    f_out,
) -> None:
    """Run a single question with detailed debugging output."""
    question = first_row["question"]
    additional_info: dict[str, Any] = {
        "answer_unit": (first_row.get("answer_unit") or "").strip(),
        "question_id": first_row.get("id", "").strip(),
    }

    attempts_log: list[dict[str, Any]] = []
    qa_result = None
    structured = None
    is_blank = True

    # Retry with increasing context window
    for attempt in range(config.max_retries + 1):
        current_top_k = config.top_k * (attempt + 1)
        qa_result = await pipeline.run_qa(
            question,
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_template=USER_PROMPT_TEMPLATE,
            additional_info=additional_info,
            top_k=current_top_k,
            with_images=config.with_images,
            top_k_images=config.top_k_images,
        )
        structured = qa_result.answer
        is_blank = (
            structured.answer_value.strip().lower() == BLANK_TOKEN
            or not structured.ref_id
        )
        attempts_log.append(
            {
                "attempt": attempt + 1,
                "top_k": current_top_k,
                "is_blank": is_blank,
                "prompt": qa_result.prompt,
                "raw": qa_result.raw_response,
                "parsed": {
                    "answer": structured.answer,
                    "answer_value": structured.answer_value,
                    "ref_id": structured.ref_id,
                    "explanation": structured.explanation,
                },
            }
        )

        if not is_blank:
            break  # Got answer, stop retrying

    assert qa_result is not None and structured is not None

    # Format output row
    result_row: Row = dict(first_row)

    if is_blank:
        result_row["answer"] = BLANK_TOKEN
        result_row["answer_value"] = BLANK_TOKEN
        result_row["ref_id"] = BLANK_TOKEN
        result_row["ref_url"] = BLANK_TOKEN
        result_row["supporting_materials"] = BLANK_TOKEN
        result_row["explanation"] = BLANK_TOKEN
    else:
        result_row["answer"] = structured.answer or BLANK_TOKEN
        normalized_value = normalize_answer_value(structured.answer_value, question)
        result_row["answer_value"] = normalized_value or BLANK_TOKEN
        if structured.ref_id:
            joined_ids = ",".join(f"'{rid}'" for rid in structured.ref_id)
            result_row["ref_id"] = f"[{joined_ids}]"
        else:
            result_row["ref_id"] = BLANK_TOKEN
        ref_url, supporting = build_ref_details(structured.ref_id, config.metadata)
        result_row["ref_url"] = ref_url
        result_row["supporting_materials"] = supporting
        result_row["explanation"] = structured.explanation or BLANK_TOKEN

    # Write result and print debug info
    writer.writerow(ensure_columns(result_row, columns))
    f_out.flush()

    print("=== Single-run debug ===")
    print(f"Question ID: {first_row.get('id')}")
    print(f"Question: {question}")

    for entry in attempts_log:
        print(f"\n--- Attempt {entry['attempt']} (top_k={entry['top_k']}) ---")
        print(f"is_blank: {entry['is_blank']}")
        print("\nPrompt:\n")
        print(entry["prompt"])
        print("\nRaw model output:\n")
        print(entry["raw"])
        print("\nParsed structured answer:\n")
        print(json.dumps(entry["parsed"], ensure_ascii=False, indent=2))


# ============================================================================
# STANDARD RUN MODE
# ============================================================================


async def run_all_questions(
    rows: Sequence[Row],
    columns: Sequence[str],
    config: AppConfig,
    pipeline: RAGPipeline,
    writer: csv.DictWriter,
    f_out,
) -> None:
    """Process all questions and write results as they complete."""
    async for answer in answer_questions(rows, columns, config, pipeline):
        writer.writerow(answer.row)
        f_out.flush()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


async def main() -> None:
    """Main entry point: load data, create pipeline, process questions."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing output to: {output_path}")

    # Load input data
    rows, columns = load_questions(Path(questions))
    metadata_records = load_metadata_records(Path(metadata))

    # Build immutable config
    config = AppConfig(
        db=Path(db),
        table_prefix=table_prefix,
        model=model,
        top_k=top_k,
        planner_model=planner_model or model,
        planner_max_queries=planner_max_queries,
        metadata=metadata_records,
        max_retries=max(0, max_retries),
        max_concurrent=max_concurrent,
        with_images=with_images,
        top_k_images=top_k_images,
    )

    # Create shared pipeline
    pipeline = create_pipeline(config)

    with output_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=list(columns))
        writer.writeheader()

        # DEBUG MODE: Process one question with detailed logging
        if single_run_debug:
            if not rows:
                raise ValueError("No questions found for single-run debug.")

            # Select question to debug (by ID or first row)
            if question_id:
                target_row: Row | None = None
                for row in rows:
                    if row.get("id") == question_id:
                        target_row = row
                        break
                if target_row is None:
                    raise ValueError(
                        f"Question id {question_id} not found in {questions}"
                    )
                first_row = target_row
            else:
                first_row = rows[0]

            await run_single_question_debug(
                first_row, columns, config, pipeline, writer, f_out
            )

        # STANDARD MODE: Process all questions concurrently, streaming results
        else:
            await run_all_questions(rows, columns, config, pipeline, writer, f_out)


def entry_point():
    print(output)
    asyncio.run(main())


if __name__ == "__main__":
    entry_point()
