from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class NodeKind(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    ATTACHMENT = "attachment"


@dataclass
class SentencePayload:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParagraphPayload:
    text: str
    sentences: list[SentencePayload] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SectionPayload:
    title: str
    paragraphs: list[ParagraphPayload]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentPayload:
    """Raw document text/structure and metadata before parsing."""

    document_id: str
    title: str
    text: str
    metadata: dict[str, Any]
    sections: list[SectionPayload] | None = None


@dataclass
class TreeNode:
    """Intermediate node used during parsing/indexing."""

    node_id: str
    kind: NodeKind
    text: str
    title: str
    metadata: dict[str, Any]
    parent_id: str | None = None
    children: list["TreeNode"] = field(default_factory=list)
    embedding: np.ndarray | None = None


@dataclass
class StoredNode:
    """Node persisted in the datastore."""

    node_id: str
    parent_id: str | None
    kind: NodeKind
    title: str
    text: str
    metadata: dict[str, Any]
    embedding: np.ndarray
    child_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievalMatch:
    node: StoredNode
    score: float


@dataclass
class ContextSnippet:
    node_id: str
    document_title: str
    text: str
    metadata: dict[str, Any]
    rank: int
    score: float
