"""Download WattBot PDFs and convert them into structured JSON payloads.

Usage (CLI):
    python scripts/wattbot_fetch_docs.py --metadata data/metadata.csv --limit 10

Usage (KohakuEngine):
    kogine run scripts/wattbot_fetch_docs.py --config configs/fetch_config.py
"""

import asyncio
import csv
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from kohakurag.parsers import payload_to_dict
from kohakurag.pdf_utils import pdf_to_document_payload

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

metadata = "data/metadata.csv"
pdf_dir = "artifacts/raw_pdfs"
output_dir = "artifacts/docs"
force_download = False
limit = 0


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def clean_url(url: str) -> str:
    return (url or "").strip()


def is_pdf_url(url: str) -> bool:
    cleaned = clean_url(url)
    if not cleaned:
        return False
    parts = urlsplit(cleaned)
    base = f"{parts.scheme}://{parts.netloc}{parts.path}".rstrip("/")
    path_lower = parts.path.lower().rstrip("/")
    if base.lower().endswith(".pdf"):
        return True
    # arXiv-style URLs sometimes omit the .pdf suffix but still live under /pdf/.
    if parts.netloc.endswith("arxiv.org") and path_lower.startswith("/pdf/"):
        return True
    return False


async def has_pdf_mime(url: str, client: httpx.AsyncClient) -> bool:
    try:
        resp = await client.head(url, follow_redirects=True, timeout=30)
        ctype = resp.headers.get("Content-Type", "").lower()
        return "pdf" in ctype
    except httpx.HTTPError:
        return False


async def download_pdf(
    url: str, dest: Path, client: httpx.AsyncClient, *, force: bool = False
) -> None:
    if dest.exists() and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "KohakuRAG/0.0.1"}
    resp = await client.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


async def main() -> None:
    # Load document metadata
    rows = load_metadata(Path(metadata))
    processed = 0
    skipped = 0

    # Create async HTTP client
    async with httpx.AsyncClient() as client:
        # Process each document
        for row in rows:
            doc_id = row["id"]
            title = row.get("title") or doc_id
            url = clean_url(row.get("url") or "")

            # Validate URL
            if not url:
                skipped += 1
                print(f"[skip] {doc_id}: missing URL", file=sys.stderr)
                continue

            if not is_pdf_url(url) and not await has_pdf_mime(url, client):
                skipped += 1
                print(
                    f"[skip] {doc_id}: URL does not look like PDF ({url})",
                    file=sys.stderr,
                )
                continue

            pdf_path = Path(pdf_dir) / f"{doc_id}.pdf"
            json_path = Path(output_dir) / f"{doc_id}.json"

            try:
                # Download PDF
                await download_pdf(url, pdf_path, client, force=force_download)

                # Parse PDF into structured payload
                payload = pdf_to_document_payload(
                    pdf_path,
                    doc_id=doc_id,
                    title=title,
                    metadata={
                        "url": url,
                        "type": row.get("type"),
                        "year": row.get("year"),
                    },
                )

                # Save as JSON
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_path.write_text(
                    json.dumps(payload_to_dict(payload), ensure_ascii=False),
                    encoding="utf-8",
                )

                processed += 1
                print(f"[ok] {doc_id} -> {json_path}")

            except httpx.HTTPError as err:
                skipped += 1
                print(f"[error] {doc_id}: download failed ({err})", file=sys.stderr)
            except Exception as exc:
                skipped += 1
                print(f"[error] {doc_id}: conversion failed ({exc})", file=sys.stderr)

            # Respect limit for testing
            if limit and processed >= limit:
                break

    print(
        f"Processed {processed} documents, skipped {skipped}. "
        f"Structured docs saved under {output_dir}"
    )


if __name__ == "__main__":
    asyncio.run(main())
