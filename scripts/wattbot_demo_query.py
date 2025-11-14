#!/usr/bin/env python3
"""Query the WattBot index and show retrieved snippets."""

import argparse
from pathlib import Path
import textwrap
from typing import List, Sequence

from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Return a simple aligned table for terminal output."""
    normalized_rows: List[List[str]] = [
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
    """Return a single-line preview of snippet text."""
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder="…")


def _compact_node_id(node_id: str, doc_id: str | None) -> str:
    """Drop the document prefix from a node id when possible."""
    if doc_id:
        prefix = str(doc_id)
        if not node_id.startswith(prefix):
            return node_id
        suffix = node_id[len(prefix) :]
        suffix = suffix.lstrip(":-_/")
        if suffix:
            return suffix
    return node_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the WattBot KohakuVault index.")
    parser.add_argument("--db", type=Path, default=Path("artifacts/wattbot.db"))
    parser.add_argument("--table-prefix", default="wattbot")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    store = KVaultNodeStore(
        args.db,
        table_prefix=args.table_prefix,
        dimensions=None,
    )
    pipeline = RAGPipeline(store=store)
    result = pipeline.retrieve(args.question, top_k=args.top_k)

    print(f"Question: {args.question}")
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


if __name__ == "__main__":
    main()
