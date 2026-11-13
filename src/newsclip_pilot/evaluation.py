"""Metrics retain invalid responses and request errors in the denominator."""

from collections import Counter

LABELS = ("MATCH", "MISMATCH")


def evaluate(records, expected_count=None):
    columns = (*LABELS, "INVALID", "ERROR")
    matrix = {truth: {predicted: 0 for predicted in columns} for truth in LABELS}
    statuses = Counter()
    seen = set()
    for row in records:
        index = row["annotation_index"]
        if index in seen:
            raise ValueError(f"Duplicate result annotation_index={index}")
        seen.add(index)
        truth, status = row["true_label"], row["status"]
        if truth not in LABELS or status not in ("ok", "invalid_response", "request_error"):
            raise ValueError("Unrecognized truth or result status.")
        predicted = row["prediction"] if status == "ok" else "INVALID" if status == "invalid_response" else "ERROR"
        if status == "ok" and predicted not in LABELS:
            raise ValueError("Successful prediction must be MATCH or MISMATCH.")
        matrix[truth][predicted] += 1
        statuses[status] += 1
    n = len(records)
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label].values()) - tp
        per_class[label] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "support": sum(matrix[label].values()),
        }
    expected = n if expected_count is None else expected_count
    if n > expected:
        raise ValueError("More results than expected.")
    return {
        "n_evaluated": n, "n_expected": expected, "complete": n == expected,
        "accuracy": sum(matrix[label][label] for label in LABELS) / n if n else None,
        "macro_f1": sum(value["f1"] for value in per_class.values()) / 2 if n else None,
        "valid_response_rate": statuses["ok"] / n if n else None,
        "invalid_responses": statuses["invalid_response"], "request_errors": statuses["request_error"],
        "per_class": per_class,
        "confusion_matrix": {"rows_true": list(LABELS), "columns_predicted": list(LABELS),
                             "values": [[matrix[t][p] for p in LABELS] for t in LABELS]},
        "confusion_matrix_with_failures": {"rows_true": list(LABELS), "columns_predicted": list(columns),
                                           "values": [[matrix[t][p] for p in columns] for t in LABELS]},
        "policy": "All attempted records count. INVALID and ERROR are incorrect and false negatives for the true class. Macro-F1 averages MATCH and MISMATCH only. Interrupted runs report attempted-record metrics and complete=false; unattempted records are not scored.",
    }
