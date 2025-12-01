"""Validate WattBot predictions against the labeled train_QA.csv file.

Enhanced validation with detailed error reporting and robust numerical comparison.

Usage (CLI):
    python scripts/wattbot_validate.py --pred artifacts/predictions.csv --verbose

Usage (KohakuEngine):
    kogine run scripts/wattbot_validate.py --config configs/validate_config.py
"""

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BLANK_TOKEN = "is_blank"
VALUE_WEIGHT = 0.75
REF_WEIGHT = 0.15
NA_WEIGHT = 0.10

# ============================================================================
# GLOBAL CONFIGURATION
# These defaults can be overridden by KohakuEngine config injection or CLI args
# ============================================================================

truth = "data/train_QA.csv"
pred = ""  # Required, no default
show_errors = 0
verbose = False


@dataclass
class QuestionScore:
    question_id: str
    question_text: str
    value_score: float
    ref_score: float
    na_score: float
    true_value: str
    pred_value: str
    true_refs: str
    pred_refs: str
    failure_reason: str | None = None

    @property
    def weighted(self) -> float:
        """Combined score using official WattBot weights."""
        return (
            VALUE_WEIGHT * self.value_score
            + REF_WEIGHT * self.ref_score
            + NA_WEIGHT * self.na_score
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def _is_blank(value: str | None) -> bool:
    """Check if a value should be treated as blank/NA."""
    if value is None:
        return True
    stripped = value.strip()
    return not stripped or stripped.lower() == BLANK_TOKEN


def _load_rows(path: Path) -> dict[str, dict[str, str]]:
    """Load CSV and index by question ID."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows: dict[str, dict[str, str]] = {}

        for row in reader:
            qid = row.get("id")
            if not qid:
                raise ValueError(f"Row missing 'id' column: {row}")
            rows[qid] = row

        if not rows:
            raise ValueError(f"No rows found in {path}")

        return rows


def _parse_numeric(value: str) -> float | None:
    """Try to parse value as a number.

    Handles: 13, 13.0, 13.00, 1.3e1, etc.
    Returns None if not a valid number.
    """
    try:
        num = float(value)
        return num
    except (TypeError, ValueError):
        return None


def _parse_range(value: str) -> tuple[float, float] | None:
    """Parse [lower,upper] JSON range.

    Handles: [10,20], [10.0,20.0], etc.
    """
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


def _normalize_text(value: str) -> str:
    """Normalize text for categorical comparison (lowercase, whitespace collapsed)."""
    return " ".join(value.strip().lower().split())


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================


def _match_numeric(target: float, candidate: float) -> tuple[float, str | None]:
    """Check if numbers match within 0.1% tolerance.

    Returns:
        (score, error_reason)
        score: 1.0 if match, 0.0 if not
        error_reason: None if match, explanation if not
    """
    tolerance = max(abs(target) * 0.001, 1e-9)  # 0.1% tolerance
    diff = abs(target - candidate)

    if diff <= tolerance:
        return (1.0, None)
    else:
        percent_error = (diff / abs(target)) * 100 if target != 0 else float("inf")
        return (
            0.0,
            f"Numeric mismatch: {candidate} vs {target} (error: {percent_error:.2f}%, tolerance: 0.1%)",
        )


def _score_answer_value(true_value: str, pred_value: str) -> tuple[float, str | None]:
    """Score answer_value with numeric tolerance or exact text match.

    Returns:
        (score, error_reason)
    """
    # Both blank
    if _is_blank(true_value):
        if _is_blank(pred_value):
            return (1.0, None)
        else:
            return (0.0, f"Expected blank but got: {pred_value}")

    # Predicted blank when should be non-blank
    if _is_blank(pred_value):
        return (0.0, f"Predicted blank but expected: {true_value}")

    # Range matching: [lower,upper]
    true_range = _parse_range(true_value)
    pred_range = _parse_range(pred_value)

    if true_range is not None:
        if pred_range is None:
            return (0.0, f"Expected range {true_value} but got non-range: {pred_value}")

        # Check both bounds
        lower_score, lower_err = _match_numeric(true_range[0], pred_range[0])
        upper_score, upper_err = _match_numeric(true_range[1], pred_range[1])

        if lower_score == 1 and upper_score == 1:
            return (1.0, None)
        else:
            errors = []
            if lower_score == 0:
                errors.append(f"Lower: {lower_err}")
            if upper_score == 0:
                errors.append(f"Upper: {upper_err}")
            return (lower_score * upper_score, f"Range mismatch - {'; '.join(errors)}")

    # Numeric matching with 0.1% tolerance
    true_num = _parse_numeric(true_value)
    pred_num = _parse_numeric(pred_value)

    if true_num is not None and pred_num is not None:
        return _match_numeric(true_num, pred_num)

    # Type mismatch: expected number, got text
    if true_num is not None and pred_num is None:
        return (0.0, f"Expected numeric {true_value} but got text: {pred_value}")

    # Type mismatch: expected text, got number
    if true_num is None and pred_num is not None:
        return (
            0.0,
            f"Expected categorical '{true_value}' but got numeric: {pred_value}",
        )

    # Categorical matching (exact text, case-insensitive)
    true_norm = _normalize_text(true_value)
    pred_norm = _normalize_text(pred_value)

    if true_norm == pred_norm:
        return (1.0, None)
    else:
        return (0.0, f"Categorical mismatch: '{pred_value}' vs '{true_value}'")


def _parse_ref_ids(value: str) -> set[str]:
    """Parse ref_id field into set of document IDs.

    Handles formats: ['doc1','doc2'] or [doc1,doc2] or doc1,doc2
    """
    if _is_blank(value):
        return set()
    try:
        data = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        data = value
    tokens: Iterable[str]
    if isinstance(data, (list, tuple, set)):
        tokens = (str(item) for item in data)
    else:
        cleaned = str(data).strip().strip("[]")
        tokens = (token.strip() for token in cleaned.split(","))
    return {token.lower() for token in tokens if token}


def _score_ref_ids(true_value: str, pred_value: str) -> tuple[float, str | None]:
    """Compute Jaccard similarity between reference ID sets.

    Returns:
        (score, error_reason)
    """
    truth = _parse_ref_ids(true_value)
    pred = _parse_ref_ids(pred_value)

    if not truth and not pred:
        return (1.0, None)

    union = truth | pred
    if not union:
        return (0.0, "Both ref_id sets empty")

    intersection = truth & pred
    jaccard = len(intersection) / len(union)

    if jaccard == 1:
        return (1.0, None)
    else:
        missing = truth - pred
        extra = pred - truth
        errors = []
        if missing:
            errors.append(f"Missing: {sorted(missing)}")
        if extra:
            errors.append(f"Extra: {sorted(extra)}")
        return (jaccard, f"Ref mismatch (Jaccard: {jaccard:.2f}) - {'; '.join(errors)}")


def _score_na_row(
    true_row: dict[str, str], pred_row: dict[str, str]
) -> tuple[float, str | None]:
    """Check if NA questions are handled correctly (all blank or all non-blank).

    Returns:
        (score, error_reason)
    """
    if not _is_blank(true_row.get("answer_value")):
        return (1.0, None)  # Not an NA question

    # NA question: prediction should be all blank
    required = (
        pred_row.get("answer_value"),
        pred_row.get("ref_id"),
    )

    all_blank = all(_is_blank(value) for value in required)

    if all_blank:
        return (1.0, None)
    else:
        non_blank = [
            f for f, v in zip(["answer_value", "ref_id"], required) if not _is_blank(v)
        ]
        return (0.0, f"NA question but predicted non-blank fields: {non_blank}")


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate(
    truth_path: Path, pred_path: Path
) -> tuple[list[QuestionScore], set[str], set[str]]:
    """Compute per-question scores and identify missing/extra predictions."""
    truth_rows = _load_rows(truth_path)
    pred_rows = _load_rows(pred_path)

    missing_predictions = set(truth_rows) - set(pred_rows)
    extra_predictions = set(pred_rows) - set(truth_rows)

    # Score each question
    scores: list[QuestionScore] = []
    for qid, truth in truth_rows.items():
        pred = pred_rows.get(qid, {})

        # Score each component
        value_score, value_err = _score_answer_value(
            truth.get("answer_value", ""), pred.get("answer_value", "")
        )
        ref_score, ref_err = _score_ref_ids(
            truth.get("ref_id", ""), pred.get("ref_id", "")
        )
        na_score, na_err = (
            _score_na_row(truth, pred) if pred else (0.0, "Missing prediction")
        )

        # Determine overall failure reason
        failure_reason = None
        if value_score < 1.0 or ref_score < 1.0 or na_score < 1.0:
            reasons = []
            if value_score < 1.0 and value_err:
                reasons.append(f"VALUE: {value_err}")
            if ref_score < 1.0 and ref_err:
                reasons.append(f"REF: {ref_err}")
            if na_score < 1.0 and na_err:
                reasons.append(f"NA: {na_err}")
            failure_reason = " | ".join(reasons) if reasons else "Unknown"

        scores.append(
            QuestionScore(
                question_id=qid,
                question_text=truth.get("question", ""),
                value_score=value_score,
                ref_score=ref_score,
                na_score=na_score,
                true_value=truth.get("answer_value", ""),
                pred_value=pred.get("answer_value", ""),
                true_refs=truth.get("ref_id", ""),
                pred_refs=pred.get("ref_id", ""),
                failure_reason=failure_reason,
            )
        )

    return scores, missing_predictions, extra_predictions


# ============================================================================
# MAIN VALIDATION
# ============================================================================


def main() -> None:
    """Run validation and print summary stats."""
    if not pred:
        raise ValueError("pred must be set in config")

    # Evaluate predictions
    scores, missing, extra = evaluate(Path(truth), Path(pred))

    # Compute aggregate metrics
    total = len(scores)
    avg_value = sum(item.value_score for item in scores) / total
    avg_ref = sum(item.ref_score for item in scores) / total
    avg_na = sum(item.na_score for item in scores) / total
    overall = VALUE_WEIGHT * avg_value + REF_WEIGHT * avg_ref + NA_WEIGHT * avg_na

    # Count perfect and failed questions
    perfect = [s for s in scores if s.weighted == 1]
    failed = [s for s in scores if s.weighted < 1.0]

    # Always show per-question scores (compact format)
    print("=" * 70)
    print("Per-Question Scores")
    print("=" * 70)
    for entry in sorted(scores, key=lambda x: x.question_id):
        status = "✓" if entry.weighted == 1 else "✗"
        print(
            f"{status} [{entry.question_id}] "
            f"val:{entry.value_score:.3f} ref:{entry.ref_score:.3f} na:{entry.na_score:.3f} "
            f"→ final:{entry.weighted:.3f}"
        )

    # Show detailed error analysis
    if show_errors > 0:
        print("\n" + "=" * 70)
        print(f"Lowest-Scoring {show_errors} Questions (Detailed)")
        print("=" * 70)

        ranked = sorted(scores, key=lambda item: item.weighted)
        for entry in ranked[:show_errors]:
            print(f"\n{'─' * 70}")
            print(f"ID: {entry.question_id} | Weighted Score: {entry.weighted:.3f}")
            print(f"Question: {entry.question_text}")
            print(
                f"\nScores: value={entry.value_score:.2f}, ref={entry.ref_score:.2f}, NA={entry.na_score:.2f}"
            )
            print(f"\nCorrect answer_value: {entry.true_value}")
            print(f"Our answer_value:     {entry.pred_value}")
            print(f"\nCorrect ref_id: {entry.true_refs}")
            print(f"Our ref_id:     {entry.pred_refs}")

            if entry.failure_reason:
                print(f"\nFailure Reason: {entry.failure_reason}")

    # Verbose mode: show ALL failed questions
    if verbose and failed:
        print("\n" + "=" * 70)
        print(f"All Failed Questions ({len(failed)} total)")
        print("=" * 70)

        for entry in sorted(failed, key=lambda x: x.weighted):
            print(f"\n{'─' * 70}")
            print(f"[{entry.question_id}]")
            print(f"Q: {entry.question_text}")
            print()
            print(f"gt ref:     {entry.true_refs}")
            print(f"gt val:     {entry.true_value}")
            print(f"ref:        {entry.pred_refs}")
            print(f"answer:     {entry.pred_value}")
            print(f"ref score:  {entry.ref_score:.3f}")
            print(f"ans score:  {entry.value_score:.3f}")
            if entry.value_score < 1.0 and entry.failure_reason:
                # Extract only VALUE reason, not REF or NA
                value_reason = None
                for part in entry.failure_reason.split(" | "):
                    if part.startswith("VALUE:"):
                        value_reason = part.replace("VALUE: ", "")
                        break
                if value_reason:
                    print(f"reason:     {value_reason}")
            print(f"\nFinal score: {entry.weighted:.3f}")

    # Print aggregate summary at the end
    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Questions evaluated: {total}")

    if missing:
        preview = ", ".join(sorted(missing)[:5])
        suffix = "..." if len(missing) > 5 else ""
        print(f"Missing predictions: {len(missing)} - {preview}{suffix}")
    if extra:
        preview = ", ".join(sorted(extra)[:5])
        suffix = "..." if len(extra) > 5 else ""
        print(f"Extra predictions:   {len(extra)} - {preview}{suffix}")

    print(f"\nPerfect answers: {len(perfect)}/{total} ({len(perfect)/total*100:.1f}%)")
    print(f"Failed answers:  {len(failed)}/{total} ({len(failed)/total*100:.1f}%)")

    print(
        f"\nComponent scores: "
        f"value={avg_value:.4f}, ref={avg_ref:.4f}, is_NA={avg_na:.4f}"
    )
    print(f"\n🎯 FINAL WATTBOT SCORE: {overall:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
