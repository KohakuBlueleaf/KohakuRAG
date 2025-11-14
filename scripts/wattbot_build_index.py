#!/usr/bin/env python3
"""Build a KohakuVault-backed index for WattBot documents."""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

from kohakurag import (
    DocumentIndexer,
    dict_to_payload,
    text_to_payload,
)
from kohakurag.datastore import KVaultNodeStore


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records[row["id"]] = row
    return records


def iter_structured_docs(docs_dir: Path) -> Iterable[dict]:
    if not docs_dir.exists():
        return []
    for json_path in sorted(docs_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        yield data


def iter_documents(
    docs_dir: Path | None,
    metadata: dict[str, dict[str, str]],
    use_citations: bool,
):
    if docs_dir and docs_dir.exists():
        for data in iter_structured_docs(docs_dir):
            yield dict_to_payload(data)
        return
    if use_citations:
        for doc_id, info in metadata.items():
            citation = info.get("citation") or info.get("title") or doc_id
            yield text_to_payload(
                document_id=doc_id,
                title=info.get("title", doc_id),
                text=citation,
                metadata={
                    "document_id": doc_id,
                    "document_title": info.get("title", doc_id),
                    "url": info.get("url"),
                    "type": info.get("type"),
                    "year": info.get("year"),
                },
            )
        return
    raise SystemExit(
        "Provide --docs-dir with structured JSON files or use --use-citations."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WattBot KohakuVault index.")
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("artifacts/docs"),
        help="Directory of structured JSON documents produced by wattbot_fetch_docs.py.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("artifacts/wattbot.db"),
        help="SQLite database file for KohakuVault.",
    )
    parser.add_argument(
        "--table-prefix",
        default="wattbot",
        help="Logical prefix for KohakuVault tables.",
    )
    parser.add_argument(
        "--use-citations",
        action="store_true",
        help="Fallback to metadata citation text when structured docs are unavailable.",
    )
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    documents = list(iter_documents(args.docs_dir, metadata, args.use_citations))
    total_docs = len(documents)
    if not total_docs:
        raise SystemExit("No documents found to index.")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    indexer = DocumentIndexer()
    store: KVaultNodeStore | None = None
    total_nodes = 0
    for idx, payload in enumerate(documents, start=1):
        print(f"[{idx}/{total_docs}] indexing {payload.document_id}...", flush=True)
        nodes = indexer.index(payload)
        if not nodes:
            print(f"  -> no nodes generated, skipping.", flush=True)
            continue
        if store is None:
            store = KVaultNodeStore(
                args.db,
                table_prefix=args.table_prefix,
                dimensions=nodes[0].embedding.shape[0],
            )
        store.upsert_nodes(nodes)
        total_nodes += len(nodes)
        print(
            f"  -> added {len(nodes)} nodes (running total {total_nodes})", flush=True
        )

    print(f"Indexed {len(documents)} documents with {total_nodes} nodes into {args.db}")


if __name__ == "__main__":
    main()
