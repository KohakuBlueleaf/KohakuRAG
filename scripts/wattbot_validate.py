"""Validate WattBot predictions against the labeled train_QA.csv file."""

import argparse
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


@dataclass
class QuestionScore:
    question_id: str
    value_score: float
    ref_score: float
    na_score: float

    @property
    def weighted(self) -> float:
        return (
            VALUE_WEIGHT * self.value_score
            + REF_WEIGHT * self.ref_score
            + NA_WEIGHT * self.na_score
        )


def _is_blank(value: str | None) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    return not stripped or stripped.lower() == BLANK_TOKEN


def _load_rows(path: Path) -> dict[str, dict[str, str]]:
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
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_range(value: str) -> tuple[float, float] | None:
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
    return " ".join(value.strip().lower().split())


def _score_answer_value(true_value: str, pred_value: str) -> float:
    print(true_value, pred_value)
    if _is_blank(true_value):
        return 1.0 if _is_blank(pred_value) else 0.0
    if _is_blank(pred_value):
        return 0.0

    true_range = _parse_range(true_value)
    pred_range = _parse_range(pred_value)
    if true_range is not None:
        if pred_range is None:
            return 0.0
        return _match_numeric(true_range[0], pred_range[0]) * _match_numeric(
            true_range[1], pred_range[1]
        )

    true_num = _parse_numeric(true_value)
    pred_num = _parse_numeric(pred_value)
    if true_num is not None and pred_num is not None:
        return _match_numeric(true_num, pred_num)

    if true_num is None and pred_num is not None:
        return 0.0

    return 1.0 if _normalize_text(true_value) == _normalize_text(pred_value) else 0.0


def _match_numeric(target: float, candidate: float) -> float:
    tolerance = max(abs(target) * 0.001, 1e-9)
    return 1.0 if abs(target - candidate) <= tolerance else 0.0


def _parse_ref_ids(value: str) -> set[str]:
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


def _score_ref_ids(true_value: str, pred_value: str) -> float:
    truth = _parse_ref_ids(true_value)
    pred = _parse_ref_ids(pred_value)
    if not truth and not pred:
        return 1.0
    union = truth | pred
    if not union:
        return 0.0
    return len(truth & pred) / len(union)


def _score_na_row(true_row: dict[str, str], pred_row: dict[str, str]) -> float:
    if not _is_blank(true_row.get("answer_value")):
        return 1.0
    required = (
        pred_row.get("answer_value"),
        pred_row.get("ref_id"),
    )
    return 1.0 if all(_is_blank(value) for value in required) else 0.0


def evaluate(
    truth_path: Path, pred_path: Path
) -> tuple[list[QuestionScore], set[str], set[str]]:
    truth_rows = _load_rows(truth_path)
    pred_rows = _load_rows(pred_path)

    missing_predictions = set(truth_rows) - set(pred_rows)
    extra_predictions = set(pred_rows) - set(truth_rows)
    scores: list[QuestionScore] = []
    for qid, truth in truth_rows.items():
        pred = pred_rows.get(qid, {})
        scores.append(
            QuestionScore(
                question_id=qid,
                value_score=_score_answer_value(
                    truth.get("answer_value", ""), pred.get("answer_value", "")
                ),
                ref_score=_score_ref_ids(
                    truth.get("ref_id", ""), pred.get("ref_id", "")
                ),
                na_score=_score_na_row(truth, pred) if pred else 0.0,
            )
        )
    return scores, missing_predictions, extra_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate WattBot predictions against train_QA.csv."
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path("data/train_QA.csv"),
        help="CSV containing the labeled WattBot answers.",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        required=True,
        help="CSV containing model predictions (same schema as train_QA.csv).",
    )
    parser.add_argument(
        "--show-errors",
        type=int,
        default=0,
        help="Print the lowest-scoring N questions for debugging.",
    )
    args = parser.parse_args()

    scores, missing, extra = evaluate(args.truth, args.pred)

    total = len(scores)
    avg_value = sum(item.value_score for item in scores) / total
    avg_ref = sum(item.ref_score for item in scores) / total
    avg_na = sum(item.na_score for item in scores) / total
    overall = VALUE_WEIGHT * avg_value + REF_WEIGHT * avg_ref + NA_WEIGHT * avg_na

    print(f"Questions evaluated: {total}")
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        suffix = "..." if len(missing) > 5 else ""
        print(f"Missing predictions for {len(missing)} id(s): {preview}{suffix}")
    if extra:
        preview = ", ".join(sorted(extra)[:5])
        suffix = "..." if len(extra) > 5 else ""
        print(f"Ignored {len(extra)} extra prediction id(s): {preview}{suffix}")
    print(
        "Component scores: "
        f"value={avg_value:.4f}, ref={avg_ref:.4f}, is_NA={avg_na:.4f}"
    )
    print(f"WattBot score: {overall:.4f}")

    if args.show_errors > 0:
        ranked = sorted(scores, key=lambda item: item.weighted)
        print("\nLowest-scoring questions:")
        for entry in ranked[: args.show_errors]:
            print(
                f"- {entry.question_id}: "
                f"value={entry.value_score:.2f}, "
                f"ref={entry.ref_score:.2f}, "
                f"is_NA={entry.na_score:.2f}, "
                f"weighted={entry.weighted:.3f}"
            )


if __name__ == "__main__":
    main()
