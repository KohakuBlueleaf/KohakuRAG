"""Simple hierarchical vector store implementations."""

from pathlib import Path
from typing import Sequence

import numpy as np
from kohakuvault import KVault, VectorKVault

from .types import ContextSnippet, NodeKind, RetrievalMatch, StoredNode


class HierarchicalNodeStore:
    """Abstract interface for node stores."""

    def upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:  # pragma: no cover
        raise NotImplementedError

    def get_node(self, node_id: str) -> StoredNode:  # pragma: no cover
        raise NotImplementedError

    def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
    ) -> list[RetrievalMatch]:  # pragma: no cover
        raise NotImplementedError

    def get_context(
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

    def upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:
        """Store nodes in memory (overwrites existing)."""
        for node in nodes:
            self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> StoredNode:
        return self._nodes[node_id]

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize to unit length for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def search(
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

    def get_context(
        self,
        node_id: str,
        *,
        parent_depth: int = 1,
        child_depth: int = 0,
    ) -> list[StoredNode]:
        """Retrieve node with hierarchical context (parents and/or children)."""
        node = self.get_node(node_id)
        context: list[StoredNode] = [node]

        # Walk up the parent chain
        if parent_depth > 0:
            parent = node
            for _ in range(parent_depth):
                if parent.parent_id is None:
                    break
                parent = self.get_node(parent.parent_id)
                context.append(parent)

        # Walk down to children
        if child_depth > 0:
            self._collect_children(
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

    def _collect_children(
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
            child = self.get_node(child_id)
            accumulator.append(child)

            # Recurse to grandchildren
            self._collect_children(
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

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def upsert_nodes(self, nodes: Sequence[StoredNode]) -> None:
        """Insert or update nodes (metadata + embeddings)."""
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

    def get_node(self, node_id: str) -> StoredNode:
        record = self._kv[node_id]
        return self._deserialize_node(record)

    def search(
        self,
        query_vector: np.ndarray,
        *,
        k: int = 5,
        kinds: set[NodeKind] | None = None,
    ) -> list[RetrievalMatch]:
        """Vector similarity search with optional node type filtering."""
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
            node = self.get_node(node_id)

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

    def get_context(
        self,
        node_id: str,
        *,
        parent_depth: int = 1,
        child_depth: int = 0,
    ) -> list[StoredNode]:
        """Retrieve node with hierarchical context (parents and/or children).

        Example: For a matched sentence, get its paragraph and section too.
        """
        node = self.get_node(node_id)
        context: list[StoredNode] = [node]

        # Walk up the parent chain (sentence → paragraph → section → document)
        if parent_depth > 0:
            parent = node
            for _ in range(parent_depth):
                if parent.parent_id is None:
                    break
                parent = self.get_node(parent.parent_id)
                context.append(parent)

        # Walk down to children (paragraph → sentences)
        if child_depth > 0:
            self._collect_children(
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

    def _collect_children(
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
            child = self.get_node(child_id)
            accumulator.append(child)

            # Recurse to grandchildren
            self._collect_children(
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


def matches_to_snippets(
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
        nodes = store.get_context(
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
