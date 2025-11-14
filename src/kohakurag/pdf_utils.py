"""Utilities for parsing PDFs (arXiv papers, reports) into structured payloads."""

from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.generic import DictionaryObject, IndirectObject

from .text_utils import split_paragraphs, split_sentences
from .types import (
    DocumentPayload,
    ParagraphPayload,
    SectionPayload,
    SentencePayload,
)


def _resolve(obj):
    return obj.get_object() if isinstance(obj, IndirectObject) else obj


def _extract_images(page) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    resources = page.get("/Resources")
    if not resources:
        return images
    resources = _resolve(resources)
    xobject = (
        resources.get("/XObject") if isinstance(resources, DictionaryObject) else None
    )
    if xobject is None:
        return images
    xobject = _resolve(xobject)
    if not isinstance(xobject, DictionaryObject):
        return images
    for name, obj in xobject.items():
        resolved = _resolve(obj)
        if not isinstance(resolved, DictionaryObject):
            continue
        subtype = resolved.get("/Subtype")
        if subtype == "/Image":
            images.append(
                {
                    "name": str(name),
                    "width": resolved.get("/Width"),
                    "height": resolved.get("/Height"),
                    "color_space": resolved.get("/ColorSpace"),
                }
            )
    return images


def pdf_to_document_payload(
    pdf_path: Path,
    *,
    doc_id: str,
    title: str,
    metadata: dict[str, Any],
) -> DocumentPayload:
    reader = PdfReader(str(pdf_path))
    sections: list[SectionPayload] = []
    all_paragraph_texts: list[str] = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        paragraphs = []
        for paragraph_text in split_paragraphs(raw_text):
            sentences = [
                SentencePayload(text=sentence)
                for sentence in split_sentences(paragraph_text)
            ]
            paragraphs.append(
                ParagraphPayload(
                    text=paragraph_text,
                    sentences=sentences or None,
                    metadata={"page": page_num},
                )
            )
            all_paragraph_texts.append(paragraph_text)
        images = _extract_images(page)
        for idx, info in enumerate(images, start=1):
            caption = (
                f"[Image page={page_num} idx={idx}] Placeholder for "
                f"{info.get('width')}x{info.get('height')} {info.get('color_space')} graphic."
            )
            sentences = [SentencePayload(text=caption)]
            paragraphs.append(
                ParagraphPayload(
                    text=caption,
                    sentences=sentences,
                    metadata={
                        "page": page_num,
                        "image_index": idx,
                        "placeholder": True,
                    },
                )
            )
            all_paragraph_texts.append(caption)
        if paragraphs:
            sections.append(
                SectionPayload(
                    title=f"Page {page_num}",
                    paragraphs=paragraphs,
                    metadata={"page": page_num},
                )
            )
    combined_text = "\n\n".join(all_paragraph_texts)
    return DocumentPayload(
        document_id=doc_id,
        title=title,
        text=combined_text,
        metadata=metadata,
        sections=sections,
    )


def pdf_to_markdown(
    pdf_path: Path,
    *,
    doc_id: str,
    title: str,
    metadata: dict[str, Any],
) -> str:
    payload = pdf_to_document_payload(
        pdf_path, doc_id=doc_id, title=title, metadata=metadata
    )
    lines = [f"# {title}", ""]
    for section in payload.sections or []:
        lines.append(f"## {section.title}")
        lines.append("")
        for paragraph in section.paragraphs:
            lines.append(paragraph.text)
            lines.append("")
    return "\n".join(lines)
