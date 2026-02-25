import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

def evaluate_at_threshold(y_true, y_proba, threshold: float) -> dict:
    """
    Compute metrics for a given threshold using predicted probabilities.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_rate": float(fp / (fp + tn + 1e-9)),
        "false_negative_rate": float(fn / (fn + tp + 1e-9)),
    }
    return metrics


def find_best_threshold_min_fp(y_true, y_proba, min_recall: float = 0.60):
    """
    Secondary objective: reduce false positives.
    Strategy:
    - Search thresholds
    - Keep only thresholds achieving recall >= min_recall (still catches churners)
    - Pick the threshold with minimum FP. If tie, pick higher precision, then higher F1.
    """
    thresholds = np.linspace(0.05, 0.95, 19)

    reports = [evaluate_at_threshold(y_true, y_proba, t) for t in thresholds]

    feasible = [r for r in reports if r["recall"] >= min_recall]
    if not feasible:
        # fallback: choose best F1 if recall constraint too strict
        feasible = reports

    feasible_sorted = sorted(
        feasible,
        key=lambda r: (r["fp"], -r["precision"], -r["f1"])
    )
    best = feasible_sorted[0]
    return best, reports