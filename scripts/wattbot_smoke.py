"""Minimal smoke test that indexes metadata citations and answers one question.

Usage (CLI):
    python scripts/wattbot_smoke.py --metadata data/metadata.csv

Usage (KohakuEngine):
    kogine run scripts/wattbot_smoke.py --config configs/smoke_config.py
"""

import asyncio
import csv
from pathlib import Path

from kohakurag import DocumentIndexer, RAGPipeline, text_to_payload

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

metadata = "data/metadata.csv"
question = "What is the ML.ENERGY benchmark?"


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


async def main() -> None:
    documents = load_documents(Path(metadata))
    indexer = DocumentIndexer()
    pipeline = RAGPipeline()
    for payload in documents:
        nodes = await indexer.index(payload)
        await pipeline.index_documents(nodes)

    answer = await pipeline.answer(question)
    print("Question:", answer["question"])
    print("Response:\n", answer["response"])
    print("\nTop snippets:")
    for snippet in answer["snippets"][:3]:
        print(
            f"- {snippet.document_title} ({snippet.node_id}) -> {snippet.text[:120]}..."
        )


if __name__ == "__main__":
    asyncio.run(main())
