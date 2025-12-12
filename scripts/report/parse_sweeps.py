#!/usr/bin/env python3
"""Parse sweep results and generate summary tables for the report.

Usage:
    python scripts/report/parse_sweeps.py
    python scripts/report/parse_sweeps.py --latex  # Output LaTeX tables
    python scripts/report/parse_sweeps.py --sweep llm_model_vs_top_k  # Single sweep
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SWEEPS_DIR = Path("outputs/sweeps")


def parse_sweep(sweep_path: Path) -> dict | None:
    """Parse a single sweep directory and return aggregated results."""
    results_path = sweep_path / "sweep_results.csv"
    metadata_path = sweep_path / "metadata.json"

    if not results_path.exists() or not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    with open(results_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    line_param = metadata["line_param"]
    x_param = metadata["x_param"]

    # Aggregate by (line_val, x_val)
    grouped = defaultdict(list)
    for row in rows:
        line_val = row[line_param]
        x_val = row[x_param]
        scores = {
            "final_score": float(row.get("final_score", 0)),
            "value_score": float(row.get("value_score", 0)),
            "ref_score": float(row.get("ref_score", 0)),
        }
        grouped[(line_val, x_val)].append(scores)

    # Compute statistics for each group
    results = {}
    for (line_val, x_val), score_list in grouped.items():
        final_scores = [s["final_score"] for s in score_list]
        value_scores = [s["value_score"] for s in score_list]
        ref_scores = [s["ref_score"] for s in score_list]

        results[(line_val, x_val)] = {
            "final": {
                "mean": np.mean(final_scores),
                "std": np.std(final_scores),
                "max": np.max(final_scores),
            },
            "value": {
                "mean": np.mean(value_scores),
                "std": np.std(value_scores),
                "max": np.max(value_scores),
            },
            "ref": {
                "mean": np.mean(ref_scores),
                "std": np.std(ref_scores),
                "max": np.max(ref_scores),
            },
            "n": len(score_list),
        }

    return {
        "name": sweep_path.name,
        "metadata": metadata,
        "results": results,
    }


def get_sorted_values(results: dict, key_idx: int) -> list:
    """Get sorted unique values from result keys."""
    vals = sorted(set(k[key_idx] for k in results.keys()), key=str)
    # Try numeric sort if possible
    try:
        return sorted(vals, key=lambda x: float(x))
    except (ValueError, TypeError):
        return vals


def compute_average_ranks(results: dict, line_vals: list, x_vals: list, metric: str = "final") -> dict[str, float]:
    """Compute average rank across x_vals for each line_val.

    For each x value (column), rank all line values by their score.
    Then compute the average rank for each line value across all columns.
    Lower rank = better (rank 1 is best).
    """
    # For each x_val, compute ranks of line_vals
    ranks_per_line = defaultdict(list)

    for x in x_vals:
        # Collect (line_val, score) pairs for this x
        scores_for_x = []
        for line in line_vals:
            if (line, x) in results:
                scores_for_x.append((line, results[(line, x)][metric]["mean"]))

        if not scores_for_x:
            continue

        # Sort by score descending (higher is better)
        scores_for_x.sort(key=lambda item: item[1], reverse=True)

        # Assign ranks (1-indexed)
        for rank, (line, _) in enumerate(scores_for_x, start=1):
            ranks_per_line[line].append(rank)

    # Compute average rank for each line_val
    avg_ranks = {}
    for line in line_vals:
        if line in ranks_per_line and ranks_per_line[line]:
            avg_ranks[line] = np.mean(ranks_per_line[line])
        else:
            avg_ranks[line] = float("inf")

    return avg_ranks


def print_sweep_summary(data: dict, metric: str = "final") -> None:
    """Print summary table for a sweep."""
    meta = data["metadata"]
    results = data["results"]
    line_param = meta["line_param"]
    x_param = meta["x_param"]

    print(f"\n{'=' * 70}")
    print(f"SWEEP: {data['name']}")
    print(f"Line: {line_param}, X: {x_param}")
    print(f"{'=' * 70}")

    # Find best result
    best_key = max(results.keys(), key=lambda k: results[k][metric]["max"])
    best = results[best_key][metric]
    print(f"Best ({metric}): {line_param}={best_key[0]}, {x_param}={best_key[1]}")
    print(f"  Max: {best['max']:.4f}, Mean: {best['mean']:.4f} +/- {best['std']:.4f}")

    # Print table
    line_vals = get_sorted_values(results, 0)
    x_vals = get_sorted_values(results, 1)

    # Compute average ranks
    avg_ranks = compute_average_ranks(results, line_vals, x_vals, metric)
    print(f"\nAverage ranks across {x_param}:")
    for line in sorted(line_vals, key=lambda l: avg_ranks.get(l, float("inf"))):
        print(f"  {line}: {avg_ranks.get(line, 'N/A'):.2f}")

    print(f"\nMean {metric} scores:")
    col_width = max(10, max(len(str(x)) for x in x_vals) + 2)
    header = f"{line_param[:20]:<22}"
    for x in x_vals:
        header += f" {str(x):>{col_width}}"
    print(header)
    print("-" * len(header))

    for line in line_vals:
        row = f"{str(line)[:20]:<22}"
        for x in x_vals:
            if (line, x) in results:
                r = results[(line, x)][metric]
                row += f" {r['mean']:.4f}"
            else:
                row += f" {'N/A':>{col_width}}"
        print(row)


def generate_latex_table(
    data: dict,
    metric: str = "final",
    mark_second_best: bool = True,
    show_avg_rank: bool = True,
) -> str:
    """Generate LaTeX table for a sweep.

    Args:
        data: Parsed sweep data
        metric: Which metric to display (final, value, ref)
        mark_second_best: If True, underline second best in each column
        show_avg_rank: If True, add average rank column when >=3 x values
    """
    meta = data["metadata"]
    results = data["results"]
    line_param = meta["line_param"]
    x_param = meta["x_param"]

    line_vals = get_sorted_values(results, 0)
    x_vals = get_sorted_values(results, 1)

    # Compute average ranks if we have enough x values
    avg_ranks = None
    if show_avg_rank and len(x_vals) >= 3:
        avg_ranks = compute_average_ranks(results, line_vals, x_vals, metric)

    # Find best and second best per column
    best_per_col = {}
    second_best_per_col = {}
    global_best = -1
    global_best_key = None

    for x in x_vals:
        col_scores = []
        for line in line_vals:
            if (line, x) in results:
                val = results[(line, x)][metric]["mean"]
                col_scores.append((line, val))
                if val > global_best:
                    global_best = val
                    global_best_key = (line, x)

        # Sort by score descending
        col_scores.sort(key=lambda item: item[1], reverse=True)
        if len(col_scores) >= 1:
            best_per_col[x] = col_scores[0][0]
        if len(col_scores) >= 2:
            second_best_per_col[x] = col_scores[1][0]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{" + f"{data['name'].replace('_', ' ').title()}" + "}")
    lines.append("\\label{tab:" + data["name"] + "}")

    # Column spec
    n_cols = len(x_vals)
    if avg_ranks:
        n_cols += 1  # Add avg rank column
    col_spec = "l" + "c" * n_cols
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")

    # Header
    header = f"\\textbf{{{line_param}}}"
    for x in x_vals:
        header += f" & \\textbf{{{x}}}"
    if avg_ranks:
        header += " & \\textbf{Avg Rank}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows
    for line in line_vals:
        row = str(line).replace("_", "\\_")
        for x in x_vals:
            if (line, x) in results:
                r = results[(line, x)][metric]
                val_str = f"{r['mean']:.3f}"
                # Bold if best in column
                if best_per_col.get(x) == line:
                    val_str = f"\\textbf{{{val_str}}}"
                # Underline if second best in column (and mark_second_best enabled)
                elif mark_second_best and second_best_per_col.get(x) == line:
                    val_str = f"\\underline{{{val_str}}}"
                row += f" & {val_str}"
            else:
                row += " & --"

        # Add average rank if available
        if avg_ranks:
            rank = avg_ranks.get(line, float("inf"))
            if rank != float("inf"):
                row += f" & {rank:.2f}"
            else:
                row += " & --"

        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parse sweep results")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX tables")
    parser.add_argument("--sweep", type=str, help="Specific sweep to parse")
    parser.add_argument(
        "--metric",
        default="final",
        choices=["final", "value", "ref"],
        help="Metric to display",
    )
    parser.add_argument("--output", type=str, help="Output file for LaTeX tables")
    parser.add_argument(
        "--no-second-best",
        action="store_true",
        help="Don't mark second best with underline",
    )
    parser.add_argument(
        "--no-avg-rank",
        action="store_true",
        help="Don't show average rank column",
    )
    args = parser.parse_args()

    # Find sweeps to parse
    if args.sweep:
        sweep_dirs = [SWEEPS_DIR / args.sweep]
    else:
        sweep_dirs = sorted(SWEEPS_DIR.iterdir())

    # Parse all sweeps
    all_sweeps = []
    for sweep_dir in sweep_dirs:
        if sweep_dir.is_dir():
            data = parse_sweep(sweep_dir)
            if data:
                all_sweeps.append(data)

    if not all_sweeps:
        print("No sweep results found")
        return

    # Output
    if args.latex:
        latex_output = []
        for data in all_sweeps:
            latex_output.append(
                generate_latex_table(
                    data,
                    args.metric,
                    mark_second_best=not args.no_second_best,
                    show_avg_rank=not args.no_avg_rank,
                )
            )
            latex_output.append("")

        output_text = "\n".join(latex_output)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_text)
            print(f"LaTeX tables written to {args.output}")
        else:
            print(output_text)
    else:
        for data in all_sweeps:
            print_sweep_summary(data, args.metric)


if __name__ == "__main__":
    main()
