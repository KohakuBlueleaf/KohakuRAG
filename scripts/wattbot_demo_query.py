"""Query the WattBot index and show retrieved snippets.

This script demonstrates RAG retrieval without LLM generation:
- Takes a question as input
- Retrieves top-k matching nodes from the index
- Shows hierarchical context expansion
- Displays results in formatted tables

Usage:
    python scripts/wattbot_demo_query.py \\
        --db artifacts/wattbot.db \\
        --question "How much water does GPT-3 training consume?" \\
        --top-k 10
"""

import argparse
import asyncio
import textwrap
from pathlib import Path
from typing import Sequence

from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore


# ============================================================================
# FORMATTING HELPERS
# ============================================================================


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Format data as aligned ASCII table for terminal."""
    normalized_rows: list[list[str]] = [
        [str(cell) if cell is not None else "-" for cell in row] for row in rows
    ]
    col_widths = [len(header) for header in headers]
    for row in normalized_rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def _format_line(values: Sequence[str]) -> str:
        return " | ".join(
            f"{value:<{col_widths[idx]}}" for idx, value in enumerate(values)
        )

    lines = [
        _format_line(headers),
        "-+-".join("-" * width for width in col_widths),
        *(_format_line(row) for row in normalized_rows),
    ]
    return "\n".join(lines)


def _preview_text(text: str, width: int = 80) -> str:
    """Truncate text to single line for table display."""
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder="…")


def _compact_node_id(node_id: str, doc_id: str | None) -> str:
    """Strip document prefix from node ID for readability.

    Example: amazon2023:sec1:p5:s12 → sec1:p5:s12
    """
    if doc_id:
        prefix = str(doc_id)
        if not node_id.startswith(prefix):
            return node_id
        suffix = node_id[len(prefix) :]
        suffix = suffix.lstrip(":-_/")
        if suffix:
            return suffix
    return node_id


# ============================================================================
# MAIN QUERY DEMO
# ============================================================================


async def main() -> None:
    """Run a demo query and display results."""
    parser = argparse.ArgumentParser(description="Query the WattBot KohakuVault index.")
    parser.add_argument("--db", type=Path, default=Path("artifacts/wattbot.db"))
    parser.add_argument("--table-prefix", default="wattbot")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--with-images",
        action="store_true",
        help="Enable image-aware retrieval (extract images from sections)",
    )
    parser.add_argument(
        "--top-k-images",
        type=int,
        default=0,
        help="Number of images from image-only index (requires wattbot_build_image_index.py)",
    )
    args = parser.parse_args()

    # Load datastore and create pipeline
    store = KVaultNodeStore(
        args.db,
        table_prefix=args.table_prefix,
        dimensions=None,
    )
    pipeline = RAGPipeline(store=store)

    # Execute retrieval
    if args.with_images:
        result = await pipeline.retrieve_with_images(
            args.question, top_k=args.top_k, top_k_images=args.top_k_images
        )
    else:
        result = await pipeline.retrieve(args.question, top_k=args.top_k)

    print(f"Question: {args.question}")

    # Format match results
    match_rows = []
    for idx, match in enumerate(result.matches, start=1):
        meta = match.node.metadata
        doc_id = meta.get("document_id")
        match_rows.append(
            [
                str(idx),
                f"{match.score:.3f}",
                _compact_node_id(match.node.node_id, doc_id),
                doc_id,
                match.node.title or "-",
            ]
        )

    print("\nTop matches:")
    if match_rows:
        print(
            _format_table(
                headers=("Rank", "Score", "Node ID", "Doc ID", "Title"),
                rows=match_rows,
            )
        )
    else:
        print("No matches found.")

    # Format expanded context snippets
    snippet_rows = []
    for snippet in result.snippets[: args.top_k]:
        doc_id = snippet.metadata.get("document_id")
        snippet_rows.append(
            [
                str(snippet.rank),
                f"{snippet.score:.3f}",
                doc_id,
                _compact_node_id(snippet.node_id, doc_id),
                _preview_text(snippet.text),
            ]
        )

    print("\nContext snippets:")
    if snippet_rows:
        print(
            _format_table(
                headers=("Rank", "Score", "Doc ID", "Node ID", "Preview"),
                rows=snippet_rows,
            )
        )
    else:
        print("No snippets available.")

    # Show image results if image-aware retrieval was used
    if args.with_images and result.image_nodes:
        print(f"\nReferenced media ({len(result.image_nodes)} images):")
        for idx, img_node in enumerate(result.image_nodes, 1):
            doc_id = img_node.metadata.get("document_id", "unknown")
            page = img_node.metadata.get("page", "?")
            img_idx = img_node.metadata.get("image_index", "?")
            print(f"\n  [{idx}] {doc_id} page {page}, image {img_idx}")
            print(f"      {img_node.text[:150]}...")


if __name__ == "__main__":
    asyncio.run(main())
