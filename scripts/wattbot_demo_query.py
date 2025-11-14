#!/usr/bin/env python3
"""Query the WattBot index and show retrieved snippets."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore


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
    print("\nTop matches:")
    for match in result.matches:
        meta = match.node.metadata
        doc_id = meta.get("document_id")
        print(
            f"- score={match.score:.3f} node={match.node.node_id} doc={doc_id} "
            f"title={match.node.title}"
        )
    print("\nContext snippets:")
    for snippet in result.snippets[: args.top_k]:
        doc_id = snippet.metadata.get("document_id")
        print(
            f"[rank={snippet.rank} score={snippet.score:.3f} doc={doc_id} id={snippet.node_id}] "
            f"{snippet.text[:300]}"
        )


if __name__ == "__main__":
    main()
