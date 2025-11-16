"""Inspect and optionally update a node stored in the WattBot index."""

import argparse
import asyncio
from pathlib import Path

from kohakurag import StoredNode
from kohakurag.datastore import KVaultNodeStore


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a node from the WattBot index."
    )
    parser.add_argument("--db", type=Path, default=Path("artifacts/wattbot.db"))
    parser.add_argument("--table-prefix", default="wattbot")
    parser.add_argument("--node-id", required=True)
    parser.add_argument(
        "--add-note", help="Append a developer note to the node metadata."
    )
    args = parser.parse_args()

    store = KVaultNodeStore(args.db, table_prefix=args.table_prefix, dimensions=None)
    node = await store.get_node(args.node_id)

    print(f"Node: {node.node_id}")
    print(f"Kind: {node.kind.value}")
    print(f"Title: {node.title}")
    print(f"Parent: {node.parent_id}")
    print(f"Children: {len(node.child_ids)}")
    print(f"Metadata: {node.metadata}")
    print(f"Text preview:\n{node.text[:500]}")

    if args.add_note:
        metadata = dict(node.metadata)
        notes = list(metadata.get("dev_notes", []))
        notes.append(args.add_note)
        metadata["dev_notes"] = notes
        updated = StoredNode(
            node_id=node.node_id,
            parent_id=node.parent_id,
            kind=node.kind,
            title=node.title,
            text=node.text,
            metadata=metadata,
            embedding=node.embedding,
            child_ids=node.child_ids,
        )
        await store.upsert_nodes([updated])
        print("Appended note to metadata.dev_notes.")


if __name__ == "__main__":
    asyncio.run(main())
