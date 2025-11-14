"""Shared text parsing utilities."""

import re

SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")
PARAGRAPH_RE = re.compile(r"\n\s*\n+")


def split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    parts = [
        segment.strip() for segment in SENTENCE_RE.split(stripped) if segment.strip()
    ]
    return parts if parts else [stripped]


def split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    raw_paragraphs = PARAGRAPH_RE.split(text)
    paragraphs = [
        paragraph.strip() for paragraph in raw_paragraphs if paragraph.strip()
    ]
    return paragraphs if paragraphs else [text.strip()]
