"""Sweep: Ensemble Size vs Tiebreak Mode

Aggregates raw inference runs using different tiebreak strategies
when there's a tie in majority voting.

Line parameter (categorical): tiebreak_mode
X-axis parameter (numerical): ensemble_size

Tiebreak modes:
- "first": Pick first occurrence (deterministic)
- "random": Pick randomly among tied values

Usage:
    # First run inference (only once)
    python workflows/sweeps/ensemble_inference.py

    # Then run aggregation sweep
    python workflows/sweeps/ensemble_vs_tiebreak.py
"""

import argparse
import csv
import itertools
import json
import random
from collections import Counter
from pathlib import Path

from kohakuengine import Config, Script

# ============================================================================
# SWEEP PARAMETERS
# ============================================================================

# Ensemble sizes to test (odd numbers)
ENSEMBLE_SIZES = [1, 3, 5, 7, 9, 11, 13, 15]

# Max combinations per ensemble size (evenly sampled if exceeded)
DEFAULT_MAX_COMBINATIONS = 32

# Line parameter: tiebreak modes
LINE_PARAM = "tiebreak_mode"
LINE_VALUES = ["first", "random"]

# X-axis parameter: ensemble size
X_PARAM = "ensemble_size"
X_VALUES = ENSEMBLE_SIZES

# ============================================================================
# PATHS
# ============================================================================

INPUT_DIR = Path("outputs/sweeps/ensemble/raw_runs")
OUTPUT_DIR = Path("outputs/sweeps/ensemble_vs_tiebreak")
QUESTIONS = "data/train_QA.csv"


# ============================================================================
# AGGREGATION FUNCTIONS
# ============================================================================


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    """Load predictions CSV into dict keyed by question id."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["id"]: row for row in reader}


def majority_vote(values: list[str], tiebreak: str = "first") -> str:
    """Return most common value with tiebreak strategy."""
    if not values:
        return "is_blank"

    counter = Counter(values)
    max_count = counter.most_common(1)[0][1]

    # Get all values with max count
    tied = [v for v, c in counter.items() if c == max_count]

    if len(tied) == 1:
        return tied[0]

    # Tiebreak
    if tiebreak == "first":
        # Return first occurrence among tied values
        for v in values:
            if v in tied:
                return v
    elif tiebreak == "random":
        return random.choice(tied)

    return tied[0]


def aggregate_ensemble(
    pred_paths: list[Path],
    tiebreak: str,
) -> dict[str, dict[str, str]]:
    """Aggregate multiple prediction files using majority vote with tiebreak."""
    all_preds = [load_predictions(p) for p in pred_paths]

    all_ids = set()
    for preds in all_preds:
        all_ids.update(preds.keys())

    result = {}
    for qid in all_ids:
        rows = [p.get(qid, {}) for p in all_preds if qid in p]

        answer_values = [r.get("answer_value", "is_blank") or "is_blank" for r in rows]
        ref_ids = [r.get("ref_id", "is_blank") or "is_blank" for r in rows]

        best_val = majority_vote(answer_values, tiebreak)
        best_ref = majority_vote(ref_ids, tiebreak)

        result[qid] = {
            "id": qid,
            "answer_value": best_val,
            "ref_id": best_ref,
        }

    return result


def save_predictions(preds: dict[str, dict[str, str]], path: Path) -> None:
    """Save predictions to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["id", "answer_value", "ref_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qid in sorted(preds.keys()):
            writer.writerow(preds[qid])


def get_total_runs() -> int:
    """Detect total runs from raw_runs directory."""
    if not INPUT_DIR.exists():
        return 0
    return len(list(INPUT_DIR.glob("run*_preds.csv")))


def sample_combinations_evenly(all_combos: list, max_count: int) -> list:
    """Sample combinations evenly spread across the list.

    If len(all_combos) <= max_count, return all.
    Otherwise, pick max_count items with maximum spread.
    """
    n = len(all_combos)
    if n <= max_count:
        return all_combos

    # Evenly spaced indices
    indices = [int(i * (n - 1) / (max_count - 1)) for i in range(max_count)]
    return [all_combos[i] for i in indices]


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep: Ensemble Size vs Tiebreak Mode"
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=DEFAULT_MAX_COMBINATIONS,
        help=f"Max combinations per ensemble size (default: {DEFAULT_MAX_COMBINATIONS})",
    )
    args = parser.parse_args()

    max_combinations = args.max_combinations
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_runs = get_total_runs()
    if total_runs == 0:
        print("Error: No raw runs found. Run ensemble_inference.py first.")
        exit(1)

    print("=" * 70)
    print("Sweep: Ensemble Size vs Tiebreak Mode")
    print("=" * 70)
    print(f"Raw runs available: {total_runs}")
    print(f"Ensemble sizes: {ENSEMBLE_SIZES}")
    print(f"Tiebreak modes: {LINE_VALUES}")
    print(f"Max combinations per size: {max_combinations}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    # Check max ensemble size
    max_size = max(ENSEMBLE_SIZES)
    if max_size > total_runs:
        print(f"Warning: max ensemble size ({max_size}) > total runs ({total_runs})")
        ENSEMBLE_SIZES[:] = [s for s in ENSEMBLE_SIZES if s <= total_runs]
        print(f"Adjusted ensemble sizes: {ENSEMBLE_SIZES}")

    for ensemble_size in ENSEMBLE_SIZES:
        all_combos = list(itertools.combinations(range(total_runs), ensemble_size))
        combos = sample_combinations_evenly(all_combos, max_combinations)
        print(
            f"\nEnsemble size {ensemble_size}: C({total_runs},{ensemble_size}) = {len(all_combos)}, using {len(combos)}"
        )

        for mode in LINE_VALUES:
            for combo_idx, combo in enumerate(combos):
                pred_paths = [INPUT_DIR / f"run{i}_preds.csv" for i in combo]

                missing = [p for p in pred_paths if not p.exists()]
                if missing:
                    print(f"  Skipping combo {combo_idx}: missing files")
                    continue

                aggregated = aggregate_ensemble(pred_paths, mode)

                out_filename = f"{LINE_PARAM}={mode}_{X_PARAM}={ensemble_size}_run{combo_idx}_preds.csv"
                out_path = OUTPUT_DIR / out_filename
                save_predictions(aggregated, out_path)

            print(f"  Created {len(combos)} files for mode={mode}")

    # Save metadata
    sweep_metadata = {
        "line_param": LINE_PARAM,
        "line_values": LINE_VALUES,
        "x_param": X_PARAM,
        "x_values": ENSEMBLE_SIZES,
        "num_runs": max_combinations,
        "questions": QUESTIONS,
    }
    with (OUTPUT_DIR / "metadata.json").open("w") as f:
        json.dump(sweep_metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("Aggregation complete!")
    print("=" * 70)
    print(f"Run 'python workflows/sweeps/sweep_plot.py {OUTPUT_DIR}' to plot results")
