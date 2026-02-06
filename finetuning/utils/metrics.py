from typing import Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    metrics: Dict[str, Any]
    predictions: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    loss: float
    best_threshold: Optional[float] = None
    alternative_metrics: Optional[Dict[str, Any]] = None


def compute_metrics_default(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> Dict[str, Any]:
    """Compute metrics for given predictions"""
    metrics = classification_report(labels, predictions, output_dict=True, zero_division=0)

    # Add confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics["confusion_matrix"] = cm.tolist()

    # Binary classification specific metrics
    metrics["tp"] = int(cm[1, 1]) if cm.shape == (2, 2) else 0
    metrics["tn"] = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    metrics["fp"] = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    metrics["fn"] = int(cm[1, 0]) if cm.shape == (2, 2) else 0
    metrics["fpr"] = metrics["fp"] / (metrics["fp"] + metrics["tn"]) if (metrics["fp"] + metrics["tn"]) > 0 else 0

    probabilities = probabilities[:, 1]
    finite_mask = np.isfinite(probabilities)
    probabilities = probabilities[finite_mask]
    labels = labels[finite_mask]
    print(f"===probabilities contains {np.sum(~finite_mask)} non-finite values")

    if len(np.unique(labels)) > 1:  # Check if both classes are present
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)
        metrics["pr_auc"] = average_precision_score(labels, probabilities)

        # Alternative PR-AUC calculation
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        metrics["pr_auc_sklearn"] = auc(recall, precision)
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
        metrics["pr_auc_sklearn"] = 0.0

    return metrics


def find_best_threshold(probabilities: np.ndarray, labels: np.ndarray, num_thresholds: int = 100) -> float:
    """Find optimal decision threshold by maximizing F1 score"""
    thresholds = np.linspace(0, 1, num_thresholds)
    best_f1 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def calculate_test_metrics(true_labels: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    """Calculate test metrics: classification report + confusion matrix"""
    metrics = classification_report(true_labels, predictions, output_dict=True, zero_division=0)

    # Add confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    metrics["confusion_matrix"] = cm.tolist()

    # Binary classification specific metrics
    metrics["tp"] = int(cm[1, 1]) if cm.shape == (2, 2) else 0
    metrics["tn"] = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    metrics["fp"] = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    metrics["fn"] = int(cm[1, 0]) if cm.shape == (2, 2) else 0

    return metrics


def create_confusion_matrix_plot(true_labels: np.ndarray, predictions: np.ndarray) -> plt.Figure:
    """Create confusion matrix plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(true_labels, predictions, ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig

def create_roc_curve_plot(true_labels: np.ndarray, probabilities: np.ndarray) -> plt.Figure:
    """Create ROC curve plot"""
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    roc_auc = roc_auc_score(true_labels, probabilities)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random", alpha=0.7)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def create_pr_curve_plot(true_labels: np.ndarray, probabilities: np.ndarray) -> plt.Figure:
    """Create Precision-Recall curve plot with AUC"""
    precision, recall, _ = precision_recall_curve(true_labels, probabilities)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(true_labels, probabilities)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.3f}, AP = {avg_precision:.3f})", linewidth=2)

    # Add baseline (prevalence line)
    prevalence = np.mean(true_labels)
    ax.axhline(
        y=prevalence, color="gray", linestyle="--", label=f"Baseline (prevalence = {prevalence:.3f})", alpha=0.7
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    return fig