"""High-level RAG pipeline orchestration."""

import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

from .datastore import HierarchicalNodeStore, InMemoryNodeStore, matches_to_snippets
from .embeddings import EmbeddingModel, JinaEmbeddingModel
from .types import ContextSnippet, NodeKind, RetrievalMatch, StoredNode


# ============================================================================
# PROTOCOLS
# ============================================================================


class ChatModel(Protocol):
    """Protocol for chat backends."""

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:  # pragma: no cover
        raise NotImplementedError


class QueryPlanner(Protocol):
    """Protocol for query expansion/rewriting."""

    def plan(self, question: str) -> Sequence[str]:  # pragma: no cover
        raise NotImplementedError


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class RetrievalResult:
    """Container for retrieval outputs."""

    question: str
    matches: list[RetrievalMatch]  # Direct vector search results
    snippets: list[ContextSnippet]  # Expanded with parent/child context


@dataclass
class StructuredAnswer:
    """Structured answer format (for WattBot and similar tasks)."""

    answer: str
    answer_value: str
    ref_id: list[str]
    explanation: str


@dataclass
class StructuredAnswerResult:
    """Complete result from structured QA pipeline."""

    answer: StructuredAnswer
    retrieval: RetrievalResult
    raw_response: str
    prompt: str


@dataclass
class PromptTemplate:
    """Template for building LLM prompts with dynamic context."""

    system_prompt: str
    user_template: str  # Must have {question}, {context}, {additional_info_json}
    additional_info: Mapping[str, object] | None = None

    def render(self, *, question: str, snippets: Sequence[ContextSnippet]) -> str:
        """Fill template with question and retrieved context."""
        context = format_snippets(snippets)
        extras = self.additional_info or {}
        extras_json = json.dumps(extras, ensure_ascii=False)

        return self.user_template.format(
            question=question,
            context=context,
            additional_info_json=extras_json,
            additional_info=extras,
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def format_snippets(snippets: Sequence[ContextSnippet]) -> str:
    """Render snippets as formatted context string for LLM prompt.

    Format: [ref_id=doc node=sec1:p2 score=0.95] snippet text
    Snippets separated by --- lines for readability.
    """
    blocks: list[str] = []

    for snippet in snippets:
        meta = snippet.metadata or {}
        doc_id = str(meta.get("document_id", "unknown"))
        full_node_id = snippet.node_id

        # Compact node ID: amazon2023:sec1:p2 → sec1:p2
        node_label = (
            full_node_id.split(":", 1)[1] if ":" in full_node_id else full_node_id
        )

        header = f"[ref_id={doc_id} node={node_label} score={snippet.score:.3f}] "
        text = snippet.text.strip()
        blocks.append(header + text)

    return "\n---\n".join(blocks)


# ============================================================================
# DEFAULT IMPLEMENTATIONS
# ============================================================================


class SimpleQueryPlanner:
    """Pass-through planner that uses the raw question without expansion."""

    def plan(self, question: str) -> Sequence[str]:
        """Return single-element list containing the original question."""
        return [question]


class MockChatModel:
    """Dummy LLM for testing (returns truncated context)."""

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Extract context from prompt and return as mock answer."""
        return "Mock response:\n" + prompt.split("Context:", 1)[-1].strip()[:200]


# ============================================================================
# RAG PIPELINE
# ============================================================================


class RAGPipeline:
    """Coordinates query planning, retrieval, and LLM answering."""

    def __init__(
        self,
        *,
        store: HierarchicalNodeStore | None = None,
        embedder: EmbeddingModel | None = None,
        chat_model: ChatModel | None = None,
        planner: QueryPlanner | None = None,
        top_k: int = 5,
    ) -> None:
        """Initialize RAG pipeline with pluggable components.

        All components default to in-memory/mock implementations for testing.
        """
        self._store = store or InMemoryNodeStore()
        self._embedder = embedder or JinaEmbeddingModel()
        self._chat = chat_model or MockChatModel()
        self._planner = planner or SimpleQueryPlanner()
        self._top_k = top_k

    @property
    def store(self) -> HierarchicalNodeStore:
        return self._store

    def index_documents(self, documents: Iterable[StoredNode]) -> None:
        """Bulk insert pre-built nodes into the store."""
        self._store.upsert_nodes(list(documents))

    def retrieve(self, question: str, *, top_k: int | None = None) -> RetrievalResult:
        """Execute multi-query retrieval with hierarchical context expansion.

        For each planner-generated query, we independently search the vector
        store for the top-k matching nodes (sentences or paragraphs). Results
        from different queries are not re-sorted or merged by score; they are
        simply concatenated in planner order so each query contributes its own
        neighborhood of matches when top_k > 0.
        """
        # Generate multiple retrieval queries (or just one if simple planner)
        queries = list(self._planner.plan(question))
        if not queries:
            raise ValueError("Planner returned no queries.")

        # Embed all queries
        query_vectors = self._embedder.embed(queries)
        k = top_k or self._top_k

        # Execute each query independently
        all_matches: list[RetrievalMatch] = []
        for vector in query_vectors:
            matches = self._store.search(
                vector,
                k=k,
                kinds={
                    NodeKind.SENTENCE,
                    NodeKind.PARAGRAPH,
                },  # Skip documents/sections
            )
            all_matches.extend(matches)

        # Expand each match with hierarchical context
        snippets = matches_to_snippets(
            all_matches,
            self._store,
            parent_depth=1,  # Include parent paragraph/section
            child_depth=1,  # Include child sentences
        )

        return RetrievalResult(
            question=question,
            matches=all_matches,
            snippets=snippets,
        )

    def answer(self, question: str) -> dict:
        """Simple QA: retrieve + prompt + generate (returns unstructured dict)."""
        retrieval = self.retrieve(question)
        prompt = self._build_prompt(question, retrieval.snippets)
        response = self._chat.complete(prompt)

        return {
            "question": question,
            "response": response,
            "snippets": retrieval.snippets,
        }

    def structured_answer(
        self,
        question: str,
        prompt: PromptTemplate,
        *,
        top_k: int | None = None,
    ) -> StructuredAnswerResult:
        """QA with custom prompt template and structured JSON parsing."""
        retrieval = self.retrieve(question, top_k=top_k)

        # Render user prompt with context
        rendered_prompt = prompt.render(question=question, snippets=retrieval.snippets)

        # Get LLM response
        raw = self._chat.complete(rendered_prompt, system_prompt=prompt.system_prompt)

        # Parse JSON structure
        parsed = self._parse_structured_response(raw)

        return StructuredAnswerResult(
            answer=parsed,
            retrieval=retrieval,
            raw_response=raw,
            prompt=rendered_prompt,
        )

    def run_qa(
        self,
        question: str,
        *,
        system_prompt: str,
        user_template: str,
        additional_info: Mapping[str, object] | None = None,
        top_k: int | None = None,
    ) -> StructuredAnswerResult:
        """High-level entry point for structured question answering.

        This method keeps the RAG core generic by requiring callers to supply
        their own system prompt, user prompt template, and any per-call metadata
        via ``additional_info``.
        """
        template = PromptTemplate(
            system_prompt=system_prompt,
            user_template=user_template,
            additional_info=additional_info,
        )
        return self.structured_answer(
            question=question,
            prompt=template,
            top_k=top_k,
        )

    def _build_prompt(
        self,
        question: str,
        snippets: Sequence[ContextSnippet],
    ) -> str:
        """Build simple prompt for answer() method."""
        context_blocks = []
        for snippet in snippets:
            context_blocks.append(
                f"[{snippet.document_title} | node={snippet.node_id} | score={snippet.score:.3f}]\n{snippet.text}"
            )

        context_text = "\n\n".join(context_blocks) if context_blocks else "None"

        return (
            "You are an assistant.\n"
            "Use only the provided context to answer the question.\n"
            "If the context is insufficient, respond with 'NOT ENOUGH DATA'.\n\n"
            f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer:"
        )

    def _parse_structured_response(self, raw: str) -> StructuredAnswer:
        """Extract JSON from LLM response and validate fields."""
        try:
            # Find JSON block in response
            start = raw.index("{")
            end = raw.rindex("}") + 1
            snippet = raw[start:end]
            data = json.loads(snippet)

        except Exception:
            # Return empty structure if parsing fails
            return StructuredAnswer(
                answer="",
                answer_value="",
                ref_id=[],
                explanation="",
            )

        # Extract and normalize fields
        answer = str(data.get("answer", "")).strip()
        answer_value = str(data.get("answer_value", "")).strip()
        explanation = str(data.get("explanation", "")).strip()

        # Parse ref_id (can be string or list)
        ref_ids_raw = data.get("ref_id", [])
        ref_ids: list[str] = []

        if isinstance(ref_ids_raw, str):
            ref_ids_raw = [ref_ids_raw]

        if isinstance(ref_ids_raw, Sequence):
            for item in ref_ids_raw:
                text = str(item).strip()
                if text:
                    ref_ids.append(text)

        return StructuredAnswer(
            answer=answer,
            answer_value=answer_value,
            ref_id=ref_ids,
            explanation=explanation,
        )
