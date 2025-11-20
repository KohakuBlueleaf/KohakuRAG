"""Inspect and optionally update a node stored in the WattBot index.

Usage (CLI):
    python scripts/wattbot_inspect_node.py --node-id amazon2023:sec3:p12

Usage (KohakuEngine):
    kogine run scripts/wattbot_inspect_node.py --config configs/inspect_config.py
"""

import asyncio
from pathlib import Path

from kohakurag import StoredNode
from kohakurag.datastore import KVaultNodeStore

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

db = "artifacts/wattbot.db"
table_prefix = "wattbot"
node_id = ""  # Required
add_note = None


async def main() -> None:
    if not node_id:
        raise ValueError("node_id must be set in config")

    store = KVaultNodeStore(Path(db), table_prefix=table_prefix, dimensions=None)
    node = await store.get_node(node_id)

    print(f"Node: {node.node_id}")
    print(f"Kind: {node.kind.value}")
    print(f"Title: {node.title}")
    print(f"Parent: {node.parent_id}")
    print(f"Children: {len(node.child_ids)}")
    print(f"Metadata: {node.metadata}")
    print(f"Text preview:\n{node.text[:500]}")

    if add_note:
        metadata = dict(node.metadata)
        notes = list(metadata.get("dev_notes", []))
        notes.append(add_note)
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
