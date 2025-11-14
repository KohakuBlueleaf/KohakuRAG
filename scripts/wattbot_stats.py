#!/usr/bin/env python3
"""Report basic statistics for a KohakuVault-backed index."""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kohakurag import NodeKind
from kohakurag.datastore import KVaultNodeStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Print stats for the WattBot index.")
    parser.add_argument("--db", type=Path, default=Path("artifacts/wattbot.db"))
    parser.add_argument("--table-prefix", default="wattbot")
    args = parser.parse_args()

    store = KVaultNodeStore(args.db, table_prefix=args.table_prefix, dimensions=None)
    counters = Counter()
    paragraph_per_doc = defaultdict(int)
    sentence_per_doc = defaultdict(int)
    attachment_count = 0

    info = store._vectors.info()  # type: ignore[attr-defined]
    total_entries = info.get("count", 0)

    for row_id in range(1, total_entries + 1):
        try:
            _, node_id = store._vectors.get_by_id(row_id)  # type: ignore[attr-defined]
            if isinstance(node_id, bytes):
                node_id = node_id.decode()
            node = store.get_node(node_id)
        except Exception:
            continue
        counters[node.kind] += 1
        doc_id = node.metadata.get("document_id", node.node_id.split(":")[0])
        if node.kind == NodeKind.PARAGRAPH:
            paragraph_per_doc[doc_id] += 1
        if node.kind == NodeKind.SENTENCE:
            sentence_per_doc[doc_id] += 1
        if node.kind == NodeKind.ATTACHMENT:
            attachment_count += 1

    total_docs = counters[NodeKind.DOCUMENT]
    print(f"Database: {args.db}")
    print(f"Total vectors: {total_entries}")
    print(f"Total documents: {total_docs}")
    print(f"Total nodes: {sum(counters.values())}")
    for kind in NodeKind:
        print(f"  {kind.value:>10}: {counters[kind]}")

    if total_docs:
        avg_paragraphs = sum(paragraph_per_doc.values()) / total_docs
        avg_sentences = sum(sentence_per_doc.values()) / total_docs
        print(f"\nAverage paragraphs per document: {avg_paragraphs:.2f}")
        print(f"Average sentences per document: {avg_sentences:.2f}")
    print(f"Attachment nodes: {attachment_count}")


if __name__ == "__main__":
    main()
