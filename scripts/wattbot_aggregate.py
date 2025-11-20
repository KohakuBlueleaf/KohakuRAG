#!/usr/bin/env python3
"""
Aggregate multiple result CSVs using majority voting.

For each question, selects the most frequent answer_value as the final answer.
Reference IDs can be aggregated using union or intersection of selected answers.
"""

import argparse
import csv
import ast
import sys
import glob
from pathlib import Path
from collections import Counter
from typing import Literal

# Column names matching existing scripts
COLUMNS = [
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


def parse_ref_ids(ref_str: str) -> set[str]:
    """Parse ref_id string to a set of document IDs."""
    if not ref_str or ref_str == "is_blank":
        return set()

    # Try parsing as Python list
    try:
        parsed = ast.literal_eval(ref_str)
        if isinstance(parsed, list):
            return set(str(x).strip() for x in parsed if x)
    except (ValueError, SyntaxError):
        pass

    # Try comma-separated
    return set(x.strip() for x in ref_str.split(",") if x.strip())


def format_ref_ids(ref_set: set[str]) -> str:
    """Format ref_id set back to CSV format."""
    if not ref_set:
        return "is_blank"
    return str(sorted(list(ref_set)))


def normalize_value(value: str) -> str:
    """Normalize answer_value for comparison."""
    if not value:
        return "is_blank"
    value = str(value).strip()
    if not value:
        return "is_blank"
    return value


def load_csv(path: Path) -> dict[str, dict]:
    """Load a result CSV and return dict keyed by question id."""
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("id", "")
            if qid:
                rows[qid] = row
    return rows


def aggregate_results(
    csv_paths: list[Path],
    ref_mode: Literal["union", "intersection"] = "union",
    tiebreak_mode: Literal["blank", "first"] = "first",
) -> list[dict]:
    """
    Aggregate multiple result CSVs using majority voting.

    Args:
        csv_paths: List of paths to result CSVs
        ref_mode: How to aggregate ref_ids - "union" or "intersection"
        tiebreak_mode: What to do when all answers differ - "blank" or "first"

    Returns:
        List of aggregated result rows
    """
    # Load all CSVs
    all_data = []
    for path in csv_paths:
        data = load_csv(path)
        all_data.append(data)
        print(f"Loaded {len(data)} questions from {path.name}")

    if not all_data:
        return []

    # Get all question IDs
    all_qids = set()
    for data in all_data:
        all_qids.update(data.keys())

    results = []

    for qid in sorted(all_qids):
        # Collect all answers for this question
        answers = []
        for data in all_data:
            if qid in data:
                answers.append(data[qid])

        if not answers:
            continue

        # Count answer_value frequencies
        value_counter = Counter()
        value_to_rows = {}  # Map normalized value to list of rows with that value

        for row in answers:
            norm_val = normalize_value(row.get("answer_value", ""))
            value_counter[norm_val] += 1
            if norm_val not in value_to_rows:
                value_to_rows[norm_val] = []
            value_to_rows[norm_val].append(row)

        # Find most common answer
        most_common = value_counter.most_common()

        # Check for tie or all different
        if len(most_common) == len(answers) and len(answers) > 1:
            # All answers are different
            if tiebreak_mode == "blank":
                # Use blank answer
                result_row = answers[0].copy()
                result_row["answer"] = "is_blank"
                result_row["answer_value"] = "is_blank"
                result_row["ref_id"] = "is_blank"
                result_row["ref_url"] = "is_blank"
                result_row["supporting_materials"] = "is_blank"
                result_row["explanation"] = "is_blank"
                results.append(result_row)
                continue
            else:
                # Use first answer (tiebreak_mode == "first")
                selected_rows = [answers[0]]
        else:
            # Get rows with most common answer
            winning_value = most_common[0][0]
            selected_rows = value_to_rows[winning_value]

        # Use the first selected row as base
        result_row = selected_rows[0].copy()

        # Aggregate ref_ids from selected rows
        if ref_mode == "union":
            combined_refs = set()
            combined_urls = set()
            combined_materials = []

            for row in selected_rows:
                combined_refs.update(parse_ref_ids(row.get("ref_id", "")))
                # Also aggregate URLs if possible
                try:
                    urls = ast.literal_eval(row.get("ref_url", "[]"))
                    if isinstance(urls, list):
                        combined_urls.update(urls)
                except:
                    pass
                # Collect supporting materials
                mat = row.get("supporting_materials", "")
                if mat and mat != "is_blank":
                    combined_materials.append(mat)

            result_row["ref_id"] = format_ref_ids(combined_refs)
            if combined_urls:
                result_row["ref_url"] = str(sorted(list(combined_urls)))
            if combined_materials:
                # Deduplicate materials
                unique_materials = list(dict.fromkeys(combined_materials))
                result_row["supporting_materials"] = "|".join(unique_materials)

        else:  # intersection
            ref_sets = [parse_ref_ids(row.get("ref_id", "")) for row in selected_rows]
            if ref_sets:
                combined_refs = ref_sets[0]
                for ref_set in ref_sets[1:]:
                    combined_refs = combined_refs.intersection(ref_set)
                result_row["ref_id"] = format_ref_ids(combined_refs)

        results.append(result_row)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple result CSVs using majority voting"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input CSV files to aggregate",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--ref-mode",
        choices=["union", "intersection"],
        default="union",
        help="How to aggregate ref_ids from selected answers (default: union)",
    )
    parser.add_argument(
        "--tiebreak",
        choices=["blank", "first"],
        default="first",
        help="What to do when all answers differ: 'blank' uses is_blank, 'first' uses first CSV's answer (default: first)",
    )

    args = parser.parse_args()

    # Expand glob patterns (needed for Windows)
    expanded_inputs = []
    for pattern in args.inputs:
        matches = glob.glob(str(pattern))
        if matches:
            expanded_inputs.extend(Path(m) for m in matches)
        else:
            expanded_inputs.append(pattern)

    # Validate inputs exist
    for path in expanded_inputs:
        if not path.exists():
            print(f"Error: Input file not found: {path}", file=sys.stderr)
            sys.exit(1)

    if not expanded_inputs:
        print("Error: No input files found", file=sys.stderr)
        sys.exit(1)

    print(f"Aggregating {len(expanded_inputs)} result files...")
    print(f"  ref_id mode: {args.ref_mode}")
    print(f"  tiebreak mode: {args.tiebreak}")
    print()

    # Aggregate results
    results = aggregate_results(
        expanded_inputs,
        ref_mode=args.ref_mode,
        tiebreak_mode=args.tiebreak,
    )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in results:
            # Ensure all columns exist
            out_row = {col: row.get(col, "is_blank") for col in COLUMNS}
            writer.writerow(out_row)

    print(f"\nAggregated {len(results)} questions to {args.output}")


if __name__ == "__main__":
    main()
