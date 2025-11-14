#!/usr/bin/env python3
"""Minimal smoke test that indexes metadata citations and answers one question."""

import argparse
import csv
from pathlib import Path

from kohakurag import DocumentIndexer, RAGPipeline, text_to_payload


def load_documents(metadata_path: Path):
    documents = []
    with metadata_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            citation = row.get("citation") or ""
            title = row.get("title") or row["id"]
            metadata = {
                "year": row.get("year"),
                "url": row.get("url"),
                "type": row.get("type"),
            }
            documents.append(
                text_to_payload(
                    document_id=row["id"],
                    title=title,
                    text=citation,
                    metadata=metadata,
                )
            )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="WattBot smoke test.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/metadata.csv"),
        help="Path to metadata CSV.",
    )
    parser.add_argument(
        "--question",
        default="What is the ML.ENERGY benchmark?",
        help="Question to ask.",
    )
    args = parser.parse_args()

    documents = load_documents(args.metadata)
    indexer = DocumentIndexer()
    pipeline = RAGPipeline()
    for payload in documents:
        nodes = indexer.index(payload)
        pipeline.index_documents(nodes)

    answer = pipeline.answer(args.question)
    print("Question:", answer["question"])
    print("Response:\n", answer["response"])
    print("\nTop snippets:")
    for snippet in answer["snippets"][:3]:
        print(
            f"- {snippet.document_title} ({snippet.node_id}) -> {snippet.text[:120]}..."
        )


if __name__ == "__main__":
    main()
