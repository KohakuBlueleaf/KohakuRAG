"""Analyze prediction errors across sweep results for report.

Precise error categorization with clear criteria:

1. **Unit conversion errors**: Numeric ratio is within ±5% of a power of 10 (10x, 100x, 0.1x, etc.)
   - Example: pred=1438, true=1.438 → ratio=1000x (likely lbs vs kg or similar)

2. **Rounding/calculation errors**: Numeric error within ±10% but outside 0.1% tolerance
   - Example: pred=14.0, true=14.4 → 2.8% error (minor calculation difference)

3. **Reference ID mismatch**: Value is correct (within 0.1% for numeric), but ref_id differs
   - Example: correct answer but cited wrong paper

4. **False negative (unnecessary abstention)**: Ground truth has value, prediction is blank
   - Model abstained when it shouldn't have

5. **False positive (hallucination)**: Ground truth is blank, prediction has value
   - Model answered when it should have abstained

6. **Value error (other)**: Numeric errors >10% that aren't unit conversion
   - Picked completely wrong value from context

7. **Categorical mismatch**: Non-numeric answer mismatch

Usage:
    python scripts/report/analyze_errors.py
"""

import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


TRUTH_PATH = Path("data/train_QA.csv")
SWEEP_DIR = Path("outputs/sweeps/llm_model_vs_top_k")

BLANK_TOKEN = "is_blank"


def is_blank(value: str | None) -> bool:
    """Check if a value should be treated as blank/NA."""
    if value is None:
        return True
    stripped = value.strip()
    return not stripped or stripped.lower() == BLANK_TOKEN


def parse_numeric(value: str) -> float | None:
    """Try to parse value as a number."""
    if is_blank(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_range(value: str) -> tuple[float, float] | None:
    """Parse [lower,upper] JSON range."""
    if is_blank(value):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if (
        isinstance(parsed, list)
        and len(parsed) == 2
        and all(isinstance(item, (int, float)) for item in parsed)
    ):
        return (float(parsed[0]), float(parsed[1]))
    return None


def parse_ref_ids(value: str) -> set[str]:
    """Parse ref_id field into set of document IDs."""
    if is_blank(value):
        return set()
    try:
        data = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        data = value
    if isinstance(data, (list, tuple, set)):
        tokens = (str(item) for item in data)
    else:
        cleaned = str(data).strip().strip("[]")
        tokens = (token.strip() for token in cleaned.split(","))
    return {token.lower() for token in tokens if token}


def is_power_of_10_ratio(ratio: float, tolerance: float = 0.05) -> bool:
    """Check if ratio is within ±tolerance of a power of 10.

    Examples:
        ratio=10.0 → True (10^1)
        ratio=100.0 → True (10^2)
        ratio=0.1 → True (10^-1)
        ratio=9.8 → True (within 5% of 10)
        ratio=105 → True (within 5% of 100)
        ratio=5.0 → False (not near any power of 10)
    """
    if ratio <= 0:
        return False

    # Get the exponent (log base 10)
    log_ratio = math.log10(ratio)
    nearest_int = round(log_ratio)

    # Skip 10^0 = 1 (that's not a unit conversion)
    if nearest_int == 0:
        return False

    # Check if within tolerance of nearest power of 10
    power_of_10 = 10**nearest_int
    relative_diff = abs(ratio - power_of_10) / power_of_10

    return relative_diff <= tolerance


def check_numeric_match(
    true_val: float, pred_val: float
) -> tuple[bool, float, str | None]:
    """Check numeric match and return (is_match, percent_error, error_type).

    Returns:
        - is_match: True if within 0.1% tolerance
        - percent_error: Absolute percentage error
        - error_type: None if match, or specific error type
    """
    if true_val == 0:
        if pred_val == 0:
            return True, 0.0, None
        else:
            return False, float("inf"), "value_error"

    percent_error = abs(true_val - pred_val) / abs(true_val) * 100

    # Within 0.1% tolerance - correct
    if percent_error <= 0.1:
        return True, percent_error, None

    # Check for unit conversion error (ratio near power of 10)
    ratio = pred_val / true_val if true_val != 0 else float("inf")
    if is_power_of_10_ratio(ratio):
        return False, percent_error, "unit_conversion"

    # Check for rounding/calculation error (within 10%)
    if percent_error <= 10:
        return False, percent_error, "rounding_error"

    # Other numeric error
    return False, percent_error, "value_error"


def categorize_error(
    true_value: str,
    pred_value: str,
    true_refs: str,
    pred_refs: str,
) -> tuple[str | None, dict]:
    """Categorize the type of error with detailed info.

    Returns:
        (error_type, details_dict) or (None, {}) if no error
    """
    true_blank = is_blank(true_value)
    pred_blank = is_blank(pred_value)

    details = {
        "true_value": true_value,
        "pred_value": pred_value,
        "true_refs": true_refs,
        "pred_refs": pred_refs,
    }

    # Abstention errors - most clear-cut
    if true_blank and not pred_blank:
        return "false_positive", details
    if not true_blank and pred_blank:
        return "false_negative", details

    # Both blank - correct abstention
    if true_blank and pred_blank:
        return None, {}

    # Check reference correctness first
    true_ref_set = parse_ref_ids(true_refs)
    pred_ref_set = parse_ref_ids(pred_refs)
    refs_match = true_ref_set == pred_ref_set

    # Handle range values
    true_range = parse_range(true_value)
    pred_range = parse_range(pred_value)

    if true_range is not None:
        if pred_range is not None:
            # Both are ranges - check both bounds
            lower_match, lower_err, lower_type = check_numeric_match(
                true_range[0], pred_range[0]
            )
            upper_match, upper_err, upper_type = check_numeric_match(
                true_range[1], pred_range[1]
            )

            value_correct = lower_match and upper_match

            if value_correct:
                if refs_match:
                    return None, {}
                else:
                    return "ref_mismatch", details
            else:
                # Return the more severe error type
                error_type = lower_type or upper_type or "value_error"
                details["percent_error"] = max(lower_err, upper_err)
                return error_type, details
        else:
            # Expected range, got non-range
            details["reason"] = "expected_range"
            return "type_mismatch", details

    # Handle numeric values
    true_num = parse_numeric(true_value)
    pred_num = parse_numeric(pred_value)

    if true_num is not None and pred_num is not None:
        value_correct, percent_error, error_type = check_numeric_match(
            true_num, pred_num
        )
        details["percent_error"] = percent_error
        details["ratio"] = pred_num / true_num if true_num != 0 else None

        if value_correct:
            if refs_match:
                return None, {}
            else:
                return "ref_mismatch", details
        else:
            return error_type, details

    # Type mismatches
    if true_num is not None and pred_num is None:
        details["reason"] = "expected_numeric_got_text"
        return "type_mismatch", details

    if true_num is None and pred_num is not None:
        details["reason"] = "expected_text_got_numeric"
        return "type_mismatch", details

    # Categorical comparison (both non-numeric text)
    if true_value.strip().lower() == pred_value.strip().lower():
        if refs_match:
            return None, {}
        else:
            return "ref_mismatch", details

    return "categorical_mismatch", details


def load_truth() -> dict[str, dict[str, str]]:
    """Load ground truth CSV."""
    with TRUTH_PATH.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["id"]: row for row in reader}


def load_predictions(pred_path: Path) -> dict[str, dict[str, str]]:
    """Load prediction CSV."""
    with pred_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["id"]: row for row in reader}


def analyze_file(pred_path: Path, truth: dict) -> tuple[dict[str, int], list[dict]]:
    """Analyze errors in a single prediction file."""
    preds = load_predictions(pred_path)
    errors = defaultdict(int)
    error_details = []

    for qid, true_row in truth.items():
        pred_row = preds.get(qid, {})

        error_type, details = categorize_error(
            true_row.get("answer_value", ""),
            pred_row.get("answer_value", ""),
            true_row.get("ref_id", ""),
            pred_row.get("ref_id", ""),
        )

        if error_type:
            errors[error_type] += 1
            details["qid"] = qid
            details["question"] = true_row.get("question", "")
            details["error_type"] = error_type
            details["file"] = pred_path.name
            error_details.append(details)

    return dict(errors), error_details


def main():
    """Run error analysis across sweep files."""
    truth = load_truth()
    total_questions = len(truth)

    print(f"Loaded {total_questions} ground truth questions")
    print(f"Analyzing predictions in: {SWEEP_DIR}")
    print()

    # Aggregate errors across all files
    all_errors = defaultdict(int)
    all_details = []
    file_count = 0

    for pred_file in sorted(SWEEP_DIR.glob("*.csv")):
        if "sweep_results" in pred_file.name:
            continue
        errors, details = analyze_file(pred_file, truth)
        for error_type, count in errors.items():
            all_errors[error_type] += count
        all_details.extend(details)
        file_count += 1

    print(f"Analyzed {file_count} prediction files")
    print()

    # Compute percentages
    total_predictions = file_count * total_questions
    total_errors = sum(all_errors.values())
    correct_predictions = total_predictions - total_errors

    print("=" * 70)
    print("ERROR CATEGORIZATION CRITERIA")
    print("=" * 70)
    print(
        """
1. Unit conversion: Numeric ratio within ±5% of power of 10 (10x, 100x, etc.)
2. Rounding/calculation: Numeric error within ±10% but outside 0.1% tolerance
3. Reference mismatch: Correct value, incorrect ref_id
4. False negative: Ground truth has value, prediction is blank
5. False positive: Ground truth is blank, prediction has value
6. Value error: Numeric error >10% not matching unit conversion pattern
7. Type mismatch: Expected numeric got text, or vice versa
8. Categorical mismatch: Text answer mismatch
"""
    )

    print("=" * 70)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Total predictions: {total_predictions}")
    print(
        f"Correct predictions: {correct_predictions} ({correct_predictions/total_predictions*100:.1f}%)"
    )
    print(f"Total errors: {total_errors} ({total_errors/total_predictions*100:.1f}%)")
    print()

    # Map internal names to report-friendly names
    error_names = {
        "unit_conversion": "Unit conversion errors",
        "rounding_error": "Rounding/calculation errors",
        "ref_mismatch": "Reference ID mismatch",
        "false_positive": "False positive (hallucination)",
        "false_negative": "False negative (unnecessary abstention)",
        "value_error": "Value selection errors",
        "type_mismatch": "Type mismatch",
        "categorical_mismatch": "Categorical mismatch",
    }

    # Sort by count descending
    sorted_errors = sorted(all_errors.items(), key=lambda x: -x[1])

    print("Error breakdown:")
    print("-" * 70)
    for error_type, count in sorted_errors:
        name = error_names.get(error_type, error_type)
        pct = count / total_errors * 100 if total_errors > 0 else 0
        pct_of_all = count / total_predictions * 100
        print(f"  {name}: {count} ({pct:.1f}% of errors, {pct_of_all:.1f}% of all)")

    print()

    # Show examples of each error type
    print("=" * 70)
    print("EXAMPLE ERRORS BY CATEGORY")
    print("=" * 70)

    for error_type, _ in sorted_errors:
        examples = [d for d in all_details if d["error_type"] == error_type][:2]
        if examples:
            name = error_names.get(error_type, error_type)
            print(f"\n{name}:")
            for ex in examples:
                print(f"  Q: {ex.get('qid')} - {ex.get('question', '')[:60]}...")
                print(
                    f"     True: {ex.get('true_value')} | Pred: {ex.get('pred_value')}"
                )
                if "percent_error" in ex:
                    print(f"     Error: {ex['percent_error']:.2f}%", end="")
                    if "ratio" in ex and ex["ratio"]:
                        print(f" | Ratio: {ex['ratio']:.4f}")
                    else:
                        print()

    print()
    print("=" * 70)

    # Output LaTeX-friendly format for appendix
    print("\n" + "=" * 70)
    print("LATEX OUTPUT FOR APPENDIX")
    print("=" * 70)
    print(
        """
\\paragraph{Error Categorization Criteria.}
We classify prediction errors using the following precise criteria:
\\begin{itemize}[noitemsep]
    \\item \\textbf{Unit conversion}: The ratio between predicted and true numeric values is within $\\pm 5\\%$ of a power of 10 (e.g., $10\\times$, $100\\times$, $0.01\\times$), indicating likely unit confusion (kW vs W, MWh vs kWh).
    \\item \\textbf{Rounding/calculation}: Numeric error within $\\pm 10\\%$ but exceeding the $0.1\\%$ tolerance, suggesting minor calculation mistakes or rounding differences.
    \\item \\textbf{Reference mismatch}: Answer value is correct (within tolerance) but the cited reference IDs differ from ground truth.
    \\item \\textbf{False negative}: Model outputs \\texttt{is\\_blank} when ground truth contains a valid answer.
    \\item \\textbf{False positive}: Model provides an answer when ground truth is \\texttt{is\\_blank}.
    \\item \\textbf{Value error}: Numeric error exceeds $10\\%$ and does not match unit conversion pattern---model selected wrong value from context.
    \\item \\textbf{Type mismatch}: Expected numeric value but received text, or vice versa.
    \\item \\textbf{Categorical mismatch}: Non-numeric text answer does not match ground truth.
\\end{itemize}
"""
    )

    print("\n\\paragraph{Error Distribution.}")
    print("\\begin{itemize}[noitemsep]")
    for error_type, count in sorted_errors:
        name = error_names.get(error_type, error_type)
        pct = count / total_errors * 100 if total_errors > 0 else 0
        print(f"    \\item \\textbf{{{name}}}: {pct:.1f}\\% of errors")
    print("\\end{itemize}")


if __name__ == "__main__":
    main()
