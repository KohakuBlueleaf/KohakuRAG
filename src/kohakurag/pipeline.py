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

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:  # pragma: no cover
        raise NotImplementedError


class QueryPlanner(Protocol):
    """Protocol for query expansion/rewriting."""

    async def plan(self, question: str) -> Sequence[str]:  # pragma: no cover
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
    image_nodes: list[StoredNode] | None = None  # Images from retrieved sections


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

    def render(
        self,
        *,
        question: str,
        snippets: Sequence[ContextSnippet],
        image_nodes: Sequence[StoredNode] | None = None,
    ) -> str:
        """Fill template with question and retrieved context.

        Args:
            question: User question
            snippets: Retrieved context snippets
            image_nodes: Optional image nodes from sections (for image-aware RAG)

        Returns:
            Rendered prompt string
        """
        # Format context with optional images
        if image_nodes:
            context = format_context_with_images(snippets, image_nodes)
        else:
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

        # header = f"[ref_id={doc_id} node={node_label} score={snippet.score:.3f}] "
        header = f"[ref_id={doc_id}] "  # only necessary info to avoid LLM hallucination and waste tokens
        text = snippet.text.strip()
        blocks.append(header + text)

    return "\n---\n".join(blocks)


def format_image_nodes(image_nodes: Sequence[StoredNode]) -> str:
    """Format image nodes for LLM prompt.

    Format:
        [ref_id=doc1] [img:name WxH] Caption text...

        [ref_id=doc2] [img:name2 WxH2] Caption text 2...

    Returns empty string if no images.
    """
    if not image_nodes:
        return ""

    blocks: list[str] = []
    for node in image_nodes:
        # Get document ID from metadata
        doc_id = node.metadata.get("document_id", "unknown")

        # Image text is already in format: [img:name WxH] caption...
        # Add ref_id prefix to match text snippet format
        formatted = f"[ref_id={doc_id}] {node.text.strip()}"
        blocks.append(formatted)

    return "\n\n".join(blocks)


def format_context_with_images(
    snippets: Sequence[ContextSnippet],
    image_nodes: Sequence[StoredNode] | None = None,
) -> str:
    """Format context with separate sections for text and images.

    Format:
        Context snippets:
        [ref_id=doc1] Text...
        ---
        [ref_id=doc2] More text...

        Referenced media:
        [img:Fig1 800x600] Bar chart showing...

        [img:Fig2 1200x900] Diagram of system...
    """
    context = format_snippets(snippets)

    if image_nodes:
        image_text = format_image_nodes(image_nodes)
        if image_text:
            context += "\n\nReferenced media:\n" + image_text

    return context


# ============================================================================
# DEFAULT IMPLEMENTATIONS
# ============================================================================


class SimpleQueryPlanner:
    """Pass-through planner that uses the raw question without expansion."""

    async def plan(self, question: str) -> Sequence[str]:
        """Return single-element list containing the original question."""
        return [question]


class MockChatModel:
    """Dummy LLM for testing (returns truncated context)."""

    async def complete(
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

    async def index_documents(self, documents: Iterable[StoredNode]) -> None:
        """Bulk insert pre-built nodes into the store."""
        await self._store.upsert_nodes(list(documents))

    async def retrieve(
        self, question: str, *, top_k: int | None = None
    ) -> RetrievalResult:
        """Execute multi-query retrieval with hierarchical context expansion.

        For each planner-generated query, we independently search the vector
        store for the top-k matching nodes (sentences or paragraphs). Results
        from different queries are not re-sorted or merged by score; they are
        simply concatenated in planner order so each query contributes its own
        neighborhood of matches when top_k > 0.
        """
        # Generate multiple retrieval queries (or just one if simple planner)
        queries = list(await self._planner.plan(question))
        if not queries:
            raise ValueError("Planner returned no queries.")

        # Embed all queries
        query_vectors = await self._embedder.embed(queries)
        k = top_k or self._top_k

        # Execute each query independently
        all_matches: list[RetrievalMatch] = []
        for vector in query_vectors:
            matches = await self._store.search(
                vector,
                k=k,
                kinds={
                    NodeKind.SENTENCE,
                    NodeKind.PARAGRAPH,
                },  # Skip documents/sections
            )
            all_matches.extend(matches)

        # Expand each match with hierarchical context
        snippets = await matches_to_snippets(
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

    async def _extract_images_from_snippets(
        self, snippets: Sequence[ContextSnippet]
    ) -> list[StoredNode]:
        """Extract image nodes from retrieved sections.

        Looks at all sections containing retrieved snippets and collects
        their image children (paragraphs with attachment_type='image').

        Args:
            snippets: Retrieved context snippets

        Returns:
            List of image nodes
        """
        image_nodes: list[StoredNode] = []
        seen_sections: set[str] = set()

        for snippet in snippets:
            # Get section ID from node ID (format: doc:sec:p:s → doc:sec)
            parts = snippet.node_id.split(":")
            if len(parts) >= 2:
                section_id = ":".join(parts[:2])
            else:
                continue  # Not a hierarchical node

            # Skip if we already processed this section
            if section_id in seen_sections:
                continue
            seen_sections.add(section_id)

            try:
                # Get the section node
                section_node = await self._store.get_node(section_id)

                # Check all children for images
                for child_id in section_node.child_ids:
                    try:
                        child_node = await self._store.get_node(child_id)

                        # Check if this is an image node
                        if child_node.metadata.get("attachment_type") == "image":
                            image_nodes.append(child_node)

                    except KeyError:
                        continue  # Child node not found

            except KeyError:
                continue  # Section node not found

        return image_nodes

    async def retrieve_with_images(
        self, question: str, *, top_k: int | None = None, top_k_images: int = 0
    ) -> RetrievalResult:
        """Execute multi-query retrieval with image extraction.

        Image retrieval strategy:
        1. Always extract images from retrieved text sections (default behavior)
        2. Additionally retrieve from image-only index if top_k_images > 0
        3. Combine and deduplicate both sources

        Args:
            question: User question
            top_k: Number of text results per query (uses default if None)
            top_k_images: Number of ADDITIONAL images from image-only index
                         (0 = only extract from sections, >0 = also search image index)

        Returns:
            RetrievalResult with image_nodes populated
        """
        # Standard text retrieval
        result = await self.retrieve(question, top_k=top_k)

        # Image retrieval: always extract from sections
        images_from_sections = await self._extract_images_from_snippets(result.snippets)

        # Additionally retrieve from image-only index if requested
        if top_k_images > 0:
            images_from_index = await self._retrieve_images_only(question, top_k_images)
        else:
            images_from_index = []

        # Combine and deduplicate images
        all_images = []
        seen_ids = set()

        # Prioritize images from sections (more contextually relevant)
        for node in images_from_sections:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                all_images.append(node)

        # Add images from dedicated index
        for node in images_from_index:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                all_images.append(node)

        return RetrievalResult(
            question=result.question,
            matches=result.matches,
            snippets=result.snippets,
            image_nodes=all_images if all_images else None,
        )

    async def _retrieve_images_only(self, question: str, k: int) -> list[StoredNode]:
        """Retrieve top-k images using dedicated image-only vector index.

        Args:
            question: User question
            k: Number of images to retrieve

        Returns:
            List of image nodes (empty if image index doesn't exist)
        """
        # Check if image-only index exists
        if not hasattr(self._store, "search_images"):
            return []

        # Generate retrieval queries
        queries = list(await self._planner.plan(question))
        if not queries:
            return []

        # Embed all queries
        query_vectors = await self._embedder.embed(queries)

        # Search image-only index for each query
        all_image_matches: list[StoredNode] = []
        for vector in query_vectors:
            matches = await self._store.search_images(vector, k=k)
            all_image_matches.extend([m.node for m in matches])

        # Deduplicate by node_id (in case same image matched multiple queries)
        seen_ids = set()
        unique_images = []
        for node in all_image_matches:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                unique_images.append(node)

        return unique_images[:k]  # Limit to top-k overall

    async def answer(self, question: str) -> dict:
        """Simple QA: retrieve + prompt + generate (returns unstructured dict)."""
        retrieval = await self.retrieve(question)
        prompt = self._build_prompt(question, retrieval.snippets)
        response = await self._chat.complete(prompt)

        return {
            "question": question,
            "response": response,
            "snippets": retrieval.snippets,
        }

    async def structured_answer(
        self,
        question: str,
        prompt: PromptTemplate,
        *,
        top_k: int | None = None,
        with_images: bool = False,
        top_k_images: int = 0,
    ) -> StructuredAnswerResult:
        """QA with custom prompt template and structured JSON parsing.

        Args:
            question: User question
            prompt: Prompt template
            top_k: Number of text results per query
            with_images: Whether to include images from retrieved sections
            top_k_images: Number of images from image-only index (0 = extract from sections)

        Returns:
            Complete structured answer result
        """
        # Use image-aware retrieval if requested
        if with_images:
            retrieval = await self.retrieve_with_images(
                question, top_k=top_k, top_k_images=top_k_images
            )
        else:
            retrieval = await self.retrieve(question, top_k=top_k)

        # Render user prompt with context (and images if present)
        rendered_prompt = prompt.render(
            question=question,
            snippets=retrieval.snippets,
            image_nodes=retrieval.image_nodes,
        )

        # Get LLM response
        raw = await self._chat.complete(
            rendered_prompt, system_prompt=prompt.system_prompt
        )

        # Parse JSON structure
        parsed = self._parse_structured_response(raw)

        return StructuredAnswerResult(
            answer=parsed,
            retrieval=retrieval,
            raw_response=raw,
            prompt=rendered_prompt,
        )

    async def run_qa(
        self,
        question: str,
        *,
        system_prompt: str,
        user_template: str,
        additional_info: Mapping[str, object] | None = None,
        top_k: int | None = None,
        with_images: bool = False,
        top_k_images: int = 0,
    ) -> StructuredAnswerResult:
        """High-level entry point for structured question answering.

        This method keeps the RAG core generic by requiring callers to supply
        their own system prompt, user prompt template, and any per-call metadata
        via ``additional_info``.

        Args:
            question: User question
            system_prompt: System prompt for LLM
            user_template: User prompt template
            additional_info: Extra metadata for template
            top_k: Number of text results per query
            with_images: Whether to include images from retrieved sections
            top_k_images: Number of images from image-only index (requires wattbot_build_image_index.py)

        Returns:
            Structured answer result
        """
        template = PromptTemplate(
            system_prompt=system_prompt,
            user_template=user_template,
            additional_info=additional_info,
        )
        return await self.structured_answer(
            question=question,
            prompt=template,
            top_k=top_k,
            with_images=with_images,
            top_k_images=top_k_images,
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
