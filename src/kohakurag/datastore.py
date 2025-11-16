"""Simple hierarchical vector store implementations."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
from kohakuvault import KVault, VectorKVault

from .types import ContextSnippet, NodeKind, RetrievalMatch, StoredNode


class HierarchicalNodeStore:
    """Abstract interface for node stores."""

    async def upsert_nodes(
        self, nodes: Sequence[StoredNode]
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    async def get_node(self, node_id: str) -> StoredNode:  # pragma: no cover
        raise NotImplementedError

    async def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
    ) -> list[RetrievalMatch]:  # pragma: no cover
        raise NotImplementedError

    async def get_context(
        self,
        node_id: str,
        *,
        parent_depth: int = 1,
        child_depth: int = 0,
    ) -> list[StoredNode]:  # pragma: no cover
        raise NotImplementedError


class InMemoryNodeStore(HierarchicalNodeStore):
    """In-memory store with brute-force cosine search (for testing/development)."""

    def __init__(self) -> None:
        self._nodes: dict[str, StoredNode] = {}

    async def upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:
        """Store nodes in memory (overwrites existing)."""
        for node in nodes:
            self._nodes[node.node_id] = node

    async def get_node(self, node_id: str) -> StoredNode:
        return self._nodes[node_id]

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize to unit length for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    async def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
    ) -> list[RetrievalMatch]:
        """Brute-force linear scan with cosine similarity."""
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be a 1D numpy array.")

        normalized_query = self._normalize(query_vector)
        matches: list[RetrievalMatch] = []

        # Compare query against all nodes
        for node in self._nodes.values():
            # Filter by node type if specified
            if kinds is not None and node.kind not in kinds:
                continue

            node_vec = self._normalize(node.embedding)
            score = float(np.dot(normalized_query, node_vec))
            matches.append(RetrievalMatch(node=node, score=score))

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:k]

    async def get_context(
        self,
        node_id: str,
        *,
        parent_depth: int = 1,
        child_depth: int = 0,
    ) -> list[StoredNode]:
        """Retrieve node with hierarchical context (parents and/or children)."""
        node = await self.get_node(node_id)
        context: list[StoredNode] = [node]

        # Walk up the parent chain
        if parent_depth > 0:
            parent = node
            for _ in range(parent_depth):
                if parent.parent_id is None:
                    break
                parent = await self.get_node(parent.parent_id)
                context.append(parent)

        # Walk down to children
        if child_depth > 0:
            await self._collect_children(
                node, depth=child_depth, accumulator=context, seen=set()
            )

        # Deduplicate
        unique = []
        seen_ids: set[str] = set()
        for item in context:
            if item.node_id in seen_ids:
                continue
            seen_ids.add(item.node_id)
            unique.append(item)

        return unique

    async def _collect_children(
        self,
        node: StoredNode,
        *,
        depth: int,
        accumulator: list[StoredNode],
        seen: set[str],
    ) -> None:
        """Recursively collect children up to specified depth."""
        if depth <= 0:
            return

        for child_id in node.child_ids:
            if child_id in seen:
                continue

            seen.add(child_id)
            child = await self.get_node(child_id)
            accumulator.append(child)

            # Recurse to grandchildren
            await self._collect_children(
                child,
                depth=depth - 1,
                accumulator=accumulator,
                seen=seen,
            )


class KVaultNodeStore(HierarchicalNodeStore):
    """SQLite-backed store using KohakuVault (key-value) + sqlite-vec (vectors).

    - Metadata stored in KohakuVault table
    - Embeddings indexed in sqlite-vec for fast similarity search
    - Both tables live in the same .db file
    """

    META_KEY = "__kohakurag_meta__"

    def __init__(
        self,
        path: str | Path,
        *,
        table_prefix: str = "rag_nodes",
        dimensions: int | None = None,
        metric: str = "cosine",
    ) -> None:
        """Initialize or open existing datastore.

        Args:
            path: SQLite database file path
            table_prefix: Logical namespace for tables
            dimensions: Embedding dimension (auto-detected if None and DB exists)
            metric: Distance metric ("cosine" or "l2")
        """
        self._path = str(path)

        # Open key-value table for metadata
        self._kv = KVault(self._path, table=f"{table_prefix}_kv")
        self._kv.enable_auto_pack()

        # Validate or infer dimensions
        stored_meta = self._kv.get(self.META_KEY, None)
        if dimensions is None:
            if stored_meta is None:
                raise ValueError(
                    "Embedding dimension required for new store. Pass dimensions=... "
                    "when creating the index."
                )
            dimensions = int(stored_meta.get("dimensions"))

        self._dimensions = int(dimensions)

        # Check dimension consistency with existing store
        if (
            stored_meta
            and int(stored_meta.get("dimensions", self._dimensions)) != self._dimensions
        ):
            raise ValueError(
                f"Existing store was built with dimension {stored_meta['dimensions']}, "
                f"but {self._dimensions} was requested."
            )

        # Store metadata
        self._kv[self.META_KEY] = {"dimensions": self._dimensions, "metric": metric}

        # Open vector table
        self._vectors = VectorKVault(
            self._path,
            table=f"{table_prefix}_vec",
            dimensions=self._dimensions,
            metric=metric,
        )
        self._vectors.enable_auto_pack()
        self._metric = metric

        # Single-worker executor for thread-safe async SQLite operations
        self._executor = ThreadPoolExecutor(max_workers=1)

    def __del__(self) -> None:
        """Cleanup executor on deletion."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _sync_upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:
        """Synchronous upsert logic (called via executor)."""
        for node in nodes:
            record = self._serialize_node(node)
            row_id = record.get("vector_row_id")

            # Insert or update vector
            if row_id is None:
                # New node: insert into vector table
                vector_row = self._vectors.insert(
                    node.embedding.astype(np.float32),
                    node.node_id,
                )
            else:
                # Existing node: update vector
                self._vectors.update(
                    row_id,
                    vector=node.embedding.astype(np.float32),
                    value=node.node_id,
                )
                vector_row = row_id

            # Store metadata with vector row reference
            record["vector_row_id"] = vector_row
            self._kv[node.node_id] = record

    async def upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:
        """Insert or update nodes (metadata + embeddings)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._sync_upsert_nodes, nodes)

    def _sync_get_node(self, node_id: str) -> StoredNode:
        """Synchronous get_node logic (called via executor)."""
        record = self._kv[node_id]
        return self._deserialize_node(record)

    async def get_node(self, node_id: str) -> StoredNode:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_get_node, node_id)

    def _sync_search(
        self,
        query_vector: np.ndarray,
        k: int,
        kinds: set[NodeKind] | None,
    ) -> list[RetrievalMatch]:
        """Synchronous search logic (called via executor)."""
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be a 1D numpy array.")

        # Query sqlite-vec index
        results = self._vectors.search(
            query_vector.astype(np.float32),
            k=k,
        )

        # Fetch full nodes and apply filters
        matches: list[RetrievalMatch] = []
        for row_id, distance, node_id in results:
            node = self._sync_get_node(node_id)

            # Skip if node type doesn't match filter
            if kinds is not None and node.kind not in kinds:
                continue

            # Convert distance to similarity score
            score = (
                1.0 - float(distance) if self._metric == "cosine" else -float(distance)
            )
            matches.append(RetrievalMatch(node=node, score=score))

        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:k]

    async def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
    ) -> list[RetrievalMatch]:
        """Vector similarity search with optional node type filtering."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._sync_search, query_vector, k, kinds
        )

    def _sync_get_context(
        self,
        node_id: str,
        parent_depth: int,
        child_depth: int,
    ) -> list[StoredNode]:
        """Synchronous get_context logic (called via executor)."""
        node = self._sync_get_node(node_id)
        context: list[StoredNode] = [node]

        # Walk up the parent chain (sentence → paragraph → section → document)
        if parent_depth > 0:
            parent = node
            for _ in range(parent_depth):
                if parent.parent_id is None:
                    break
                parent = self._sync_get_node(parent.parent_id)
                context.append(parent)

        # Walk down to children (paragraph → sentences)
        if child_depth > 0:
            self._sync_collect_children(
                node,
                depth=child_depth,
                accumulator=context,
                seen={node.node_id},
            )

        # Deduplicate in case parent/child overlap
        unique: list[StoredNode] = []
        seen_ids: set[str] = set()
        for item in context:
            if item.node_id in seen_ids:
                continue
            seen_ids.add(item.node_id)
            unique.append(item)

        return unique

    async def get_context(
        self,
        node_id: str,
        *,
        parent_depth: int = 1,
        child_depth: int = 0,
    ) -> list[StoredNode]:
        """Retrieve node with hierarchical context (parents and/or children).

        Example: For a matched sentence, get its paragraph and section too.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._sync_get_context, node_id, parent_depth, child_depth
        )

    def _sync_collect_children(
        self,
        node: StoredNode,
        *,
        depth: int,
        accumulator: list[StoredNode],
        seen: set[str],
    ) -> None:
        """Recursively collect children up to specified depth."""
        if depth <= 0:
            return

        for child_id in node.child_ids:
            if child_id in seen:
                continue

            seen.add(child_id)
            child = self._sync_get_node(child_id)
            accumulator.append(child)

            # Recurse to grandchildren
            self._sync_collect_children(
                child,
                depth=depth - 1,
                accumulator=accumulator,
                seen=seen,
            )

    def _serialize_node(self, node: StoredNode) -> dict:
        """Convert StoredNode to KohakuVault-compatible dict."""
        record = {
            "node_id": node.node_id,
            "parent_id": node.parent_id,
            "kind": node.kind.value,
            "title": node.title,
            "text": node.text,
            "metadata": node.metadata,
            "child_ids": node.child_ids,
        }

        # Preserve vector row ID if updating existing node
        existing = None
        try:
            existing = self._kv[node.node_id]
        except KeyError:
            existing = None

        if existing:
            record["vector_row_id"] = existing.get("vector_row_id")

        return record

    def _deserialize_node(self, record: dict) -> StoredNode:
        """Reconstruct StoredNode from KohakuVault record + vector lookup."""
        vector_row = record.get("vector_row_id")
        if vector_row is None:
            raise ValueError(f"Node {record['node_id']} missing vector_row_id.")

        # Fetch embedding from vector table
        embedding, _ = self._vectors.get_by_id(vector_row)
        embedding_arr = np.array(embedding, dtype=np.float32, copy=True)

        return StoredNode(
            node_id=record["node_id"],
            parent_id=record.get("parent_id"),
            kind=NodeKind(record["kind"]),
            title=record["title"],
            text=record["text"],
            metadata=record.get("metadata", {}),
            embedding=embedding_arr,
            child_ids=list(record.get("child_ids", [])),
        )


async def matches_to_snippets(
    matches: Sequence[RetrievalMatch],
    store: HierarchicalNodeStore,
    *,
    parent_depth: int = 1,
    child_depth: int = 0,
) -> list[ContextSnippet]:
    """Convert retrieval matches into context snippets using hierarchical context.

    For each matched node, this helper pulls a small neighborhood of parents
    and children via ``get_context`` and flattens them into ``ContextSnippet``
    objects. It deliberately keeps duplicates and both sentences and paragraphs
    so callers see the raw local context around each hit.
    """
    snippets: list[ContextSnippet] = []
    for rank, match in enumerate(matches, 1):
        nodes = await store.get_context(
            match.node.node_id,
            parent_depth=parent_depth,
            child_depth=child_depth,
        )
        for context_node in nodes:
            snippets.append(
                ContextSnippet(
                    node_id=context_node.node_id,
                    document_title=context_node.metadata.get(
                        "document_title", context_node.title
                    ),
                    text=context_node.text,
                    metadata=context_node.metadata,
                    rank=rank,
                    score=match.score,
                )
            )
    return snippets
