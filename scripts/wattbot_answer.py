#!/usr/bin/env python3
"""Answer WattBot questions using the KohakuRAG pipeline + OpenAI."""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kohakurag import RAGPipeline
from kohakurag.datastore import KVaultNodeStore
from kohakurag.llm import OpenAIChatModel


def format_context(snippets, max_chars: int = 6000) -> str:
    blocks = []
    total = 0
    for snippet in snippets:
        doc_id = snippet.metadata.get("document_id")
        block = (
            f"[doc={doc_id} node={snippet.node_id} score={snippet.score:.3f}] "
            f"{snippet.text.strip()}"
        )
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n".join(blocks)


def build_prompt(question: str, snippets) -> str:
    context = format_context(snippets)
    return f"""
You are WattBot, an assistant that answers electricity/energy questions.
Use ONLY the provided context snippets. If insufficient evidence exists, reply with nulls in every field.
Return a JSON object with the following fields:
- answer (string)
- answer_value (number or null)
- answer_unit (string or null)
- ref_id (list of source ids)
- ref_url (list of urls, aligned with ref_id)
- supporting_materials (short quote or derivation)
- explanation (why the answer is correct)

Question: {question}

Context:
{context}

JSON Answer:
""".strip()


def parse_response(raw: str) -> dict[str, Any]:
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        snippet = raw[start:end]
        data = json.loads(snippet)
        if not isinstance(data.get("ref_id"), list):
            data["ref_id"] = []
        if not isinstance(data.get("ref_url"), list):
            data["ref_url"] = []
        return data
    except Exception:
        return {
            "answer": "NOT ENOUGH DATA",
            "answer_value": None,
            "answer_unit": None,
            "ref_id": [],
            "ref_url": [],
            "supporting_materials": "",
            "explanation": "LLM response could not be parsed.",
        }


def ensure_columns(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    out = {col: row.get(col, "") for col in columns}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer WattBot questions.")
    parser.add_argument("--db", type=Path, default=Path("artifacts/wattbot.db"))
    parser.add_argument("--table-prefix", default="wattbot")
    parser.add_argument("--questions", type=Path, default=Path("data/test_Q.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/wattbot_answers.csv"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    store = KVaultNodeStore(args.db, table_prefix=args.table_prefix, dimensions=None)
    chat = OpenAIChatModel(model=args.model)
    pipeline = RAGPipeline(store=store, chat_model=chat)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.questions.open(newline="", encoding="utf-8") as f_in, args.output.open(
        "w", newline="", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        columns = reader.fieldnames or [
            "id",
            "question",
            "answer",
            "answer_value",
            "answer_unit",
            "ref_id",
            "ref_url",
            "supporting_materials",
            "explanation",
        ]
        writer = csv.DictWriter(f_out, fieldnames=columns)
        writer.writeheader()

        for row in reader:
            question = row["question"]
            retrieval = pipeline.retrieve(question, top_k=args.top_k)
            prompt = build_prompt(question, retrieval.snippets)
            llm_output = chat.complete(prompt)
            parsed = parse_response(llm_output)
            result = dict(row)
            result["answer"] = parsed.get("answer", "")
            result["answer_value"] = parsed.get("answer_value")
            result["answer_unit"] = parsed.get("answer_unit")
            result["ref_id"] = json.dumps(parsed.get("ref_id", []))
            result["ref_url"] = json.dumps(parsed.get("ref_url", []))
            result["supporting_materials"] = parsed.get("supporting_materials", "")
            result["explanation"] = parsed.get("explanation", "")
            writer.writerow(ensure_columns(result, columns))
            print(f"Answered {row['id']} - {question[:60]}...")

    print(f"Wrote answers to {args.output}")


if __name__ == "__main__":
    main()
