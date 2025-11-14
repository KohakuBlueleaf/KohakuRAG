"""High-level RAG pipeline orchestration."""

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from .datastore import HierarchicalNodeStore, InMemoryNodeStore, matches_to_snippets
from .embeddings import EmbeddingModel, JinaEmbeddingModel
from .types import ContextSnippet, NodeKind, RetrievalMatch, StoredNode


class ChatModel(Protocol):
    """Protocol for chat backends."""

    def complete(self, prompt: str) -> str:  # pragma: no cover
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

    def complete(self, prompt: str) -> str:
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
        queries = list(self._planner.plan(question))
        if not queries:
            raise ValueError("Planner returned no queries.")
        query_vectors = self._embedder.embed(queries)
        combined: dict[str, RetrievalMatch] = {}
        for idx, vector in enumerate(query_vectors):
            matches = self._store.search(
                vector,
                k=top_k or self._top_k,
                kinds={NodeKind.SENTENCE, NodeKind.PARAGRAPH},
            )
            for match in matches:
                existing = combined.get(match.node.node_id)
                if existing is None or match.score > existing.score:
                    combined[match.node.node_id] = match
        ordered_matches = sorted(
            combined.values(), key=lambda item: item.score, reverse=True
        )[: top_k or self._top_k]
        snippets = matches_to_snippets(
            ordered_matches,
            self._store,
            parent_depth=1,
            child_depth=1,
        )
        return RetrievalResult(
            question=question,
            matches=ordered_matches,
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
            "You are an assistant focusing on energy and sustainability topics.\n"
            "Use only the provided context to answer the question.\n"
            "If the context is insufficient, respond with 'NOT ENOUGH DATA'.\n\n"
            f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer:"
        )
