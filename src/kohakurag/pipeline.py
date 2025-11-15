"""High-level RAG pipeline orchestration."""

import json
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence

from .datastore import HierarchicalNodeStore, InMemoryNodeStore, matches_to_snippets
from .embeddings import EmbeddingModel, JinaEmbeddingModel
from .types import ContextSnippet, NodeKind, RetrievalMatch, StoredNode


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
    def plan(self, question: str) -> Sequence[str]:  # pragma: no cover
        raise NotImplementedError


class SimpleQueryPlanner:
    """Default planner that uses the raw question as the sole retrieval query."""

    def plan(self, question: str) -> Sequence[str]:
        return [question]


@dataclass
class RetrievalResult:
    question: str
    matches: list[RetrievalMatch]
    snippets: list[ContextSnippet]


class MockChatModel:
    """Rule-based fallback LLM used for local tests."""

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        return "Mock response:\n" + prompt.split("Context:", 1)[-1].strip()[:200]


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
        self._store = store or InMemoryNodeStore()
        self._embedder = embedder or JinaEmbeddingModel()
        self._chat = chat_model or MockChatModel()
        self._planner = planner or SimpleQueryPlanner()
        self._top_k = top_k

    @property
    def store(self) -> HierarchicalNodeStore:
        return self._store

    def index_documents(self, documents: Iterable[StoredNode]) -> None:
        self._store.upsert_nodes(list(documents))

    def retrieve(self, question: str, *, top_k: int | None = None) -> RetrievalResult:
        """Naive per-query top-k retrieval over the store.

        For each planner-generated query, we independently search the vector
        store for the top-k matching nodes (sentences or paragraphs). Results
        from different queries are not re-sorted or merged by score; they are
        simply concatenated in planner order so each query contributes its own
        neighborhood of matches when top_k > 0.
        """
        queries = list(self._planner.plan(question))
        if not queries:
            raise ValueError("Planner returned no queries.")
        query_vectors = self._embedder.embed(queries)
        k = top_k or self._top_k
        all_matches: list[RetrievalMatch] = []
        for vector in query_vectors:
            matches = self._store.search(
                vector,
                k=k,
                kinds={NodeKind.SENTENCE, NodeKind.PARAGRAPH},
            )
            all_matches.extend(matches)
        snippets = matches_to_snippets(
            all_matches,
            self._store,
            parent_depth=1,
            child_depth=1,
        )
        return RetrievalResult(
            question=question,
            matches=all_matches,
            snippets=snippets,
        )

    def answer(self, question: str) -> dict:
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
        retrieval = self.retrieve(question, top_k=top_k)
        rendered_prompt = prompt.render(question=question, snippets=retrieval.snippets)
        raw = self._chat.complete(rendered_prompt, system_prompt=prompt.system_prompt)
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
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            snippet = raw[start:end]
            data = json.loads(snippet)
        except Exception:
            return StructuredAnswer(
                answer="",
                answer_value="",
                ref_id=[],
                explanation="",
            )
        answer = str(data.get("answer", "")).strip()
        answer_value = str(data.get("answer_value", "")).strip()
        explanation = str(data.get("explanation", "")).strip()
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


def format_snippets(snippets: Sequence[ContextSnippet]) -> str:
    """Render retrieval snippets into a compact context block.

    Each snippet becomes one block. Blocks are joined by ``---`` lines for
    readability. All snippets passed in are included; ``max_chars`` is
    accepted for backward compatibility but not enforced.
    """
    blocks: list[str] = []
    for snippet in snippets:
        meta = snippet.metadata or {}
        doc_id = str(meta.get("document_id", "unknown"))
        full_node_id = snippet.node_id
        # Strip the document prefix so headers are shorter: doc:sec:p -> sec:p.
        node_label = (
            full_node_id.split(":", 1)[1] if ":" in full_node_id else full_node_id
        )
        header = f"[ref_id={doc_id} node={node_label} score={snippet.score:.3f}] "
        text = snippet.text.strip()
        blocks.append(header + text)
    return "\n---\n".join(blocks)


@dataclass
class PromptTemplate:
    """Container describing how to build the LLM user prompt."""

    system_prompt: str
    user_template: str
    additional_info: Mapping[str, object] | None = None

    def render(self, *, question: str, snippets: Sequence[ContextSnippet]) -> str:
        context = format_snippets(snippets)
        extras = self.additional_info or {}
        extras_json = json.dumps(extras, ensure_ascii=False)
        return self.user_template.format(
            question=question,
            context=context,
            additional_info_json=extras_json,
            additional_info=extras,
        )


@dataclass
class StructuredAnswer:
    answer: str
    answer_value: str
    ref_id: list[str]
    explanation: str


@dataclass
class StructuredAnswerResult:
    answer: StructuredAnswer
    retrieval: RetrievalResult
    raw_response: str
    prompt: str
