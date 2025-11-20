"""Query the WattBot index and show retrieved snippets.

This script demonstrates RAG retrieval without LLM generation:
- Takes a question as input
- Retrieves top-k matching nodes from the index
- Shows hierarchical context expansion
- Displays results in formatted tables

Usage (CLI):
    python scripts/wattbot_demo_query.py \\
        --db artifacts/wattbot.db \\
        --question "How much water does GPT-3 training consume?" \\
        --top-k 10

Usage (KohakuEngine):
    kogine run scripts/wattbot_demo_query.py --config configs/demo_query_config.py
"""

import asyncio
import textwrap
from pathlib import Path
from typing import Sequence

from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

db = "artifacts/wattbot.db"
table_prefix = "wattbot"
question = ""  # Required
top_k = 5
with_images = False
top_k_images = 0


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
    if not question:
        raise ValueError("question must be set in config")

    # Load datastore and create pipeline
    store = KVaultNodeStore(
        Path(db),
        table_prefix=table_prefix,
        dimensions=None,
    )
    pipeline = RAGPipeline(store=store)

    # Execute retrieval
    if with_images:
        result = await pipeline.retrieve_with_images(
            question, top_k=top_k, top_k_images=top_k_images
        )
    else:
        result = await pipeline.retrieve(question, top_k=top_k)

    print(f"Question: {question}")

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
