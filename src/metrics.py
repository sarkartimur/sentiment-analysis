import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    auc,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.calibration import CalibrationDisplay


def compute_metrics(y_test, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}")

    __calculate_confidence(y_pred, y_pred_proba, y_test)


def __calculate_confidence(y_pred, y_pred_proba, y_test):
    print("\n")

    # 1. Average confidence by predicted class
    def avg_class_confidence(pclass):
        mask = (y_pred == pclass)
        avg_conf_pred_class = np.mean(y_pred_proba[mask]) if pclass == 1 else np.mean(1 - y_pred_proba[mask])
        print(f"Avg. confidence for predictions of Class {pclass}: {avg_conf_pred_class:.4f}")

    avg_class_confidence(1)
    avg_class_confidence(0)

    # 2. Average confidence by result type
    def avg_pred_confidence(pclass, actual_class, result_type):
        mask = (y_pred == pclass) & (y_test == actual_class)
        avg_conf_correct_class = np.mean(y_pred_proba[mask]) if pclass == 1 else np.mean(1 - y_pred_proba[mask])
        print(f"Avg. confidence for Class {pclass} ({result_type}): {avg_conf_correct_class:.4f}")

    avg_pred_confidence(1, 1, 'True Positives')
    avg_pred_confidence(0, 0, 'True Negatives')
    avg_pred_confidence(1, 0, 'False Positives')
    avg_pred_confidence(0, 1, 'False Negatives')

def plot_graphs(y_test, y_pred, y_pred_proba):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    RocCurveDisplay.from_predictions(y_test, y_pred_proba)
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba)
    __plot_threshold_graph(y_test, y_pred_proba)
    CalibrationDisplay.from_predictions(y_test, y_pred_proba)
    __plot_class_overlap_graph(y_test, y_pred_proba)

def __plot_threshold_graph(y_test, y_pred_proba):
    thresholds = np.linspace(0, 1, 100)
    accuracies = [accuracy_score(y_test, (y_pred_proba >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, 'b-', linewidth=2)
    plt.axvline(x=0.5, color='r', linestyle='--', label=f'Default Threshold (0.5)')
    plt.axvline(x=best_threshold, color='g', linestyle='--', 
                label=f'Optimal Threshold ({best_threshold:.3f})')
    plt.scatter(best_threshold, best_accuracy, color='green', s=100, zorder=5)
    plt.xlabel('Classification Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Classification Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def __plot_class_overlap_graph(y_test, y_pred_proba):
    thresholds = np.linspace(0, 1, 100)
    accuracies = [accuracy_score(y_test, (y_pred_proba >= t).astype(int)) for t in thresholds]
    optimal_threshold = thresholds[np.argmax(accuracies)]

    plt.figure(figsize=(12, 5))
    plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, label='Class 0', density=True)
    plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, label='Class 1', density=True)
    plt.axvline(0.5, color='black', linestyle='--', label='Default threshold (0.5)')
    plt.axvline(optimal_threshold, color='red', linestyle='--', 
            label=f'Optimal threshold ({optimal_threshold:.3f})')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Class Overlap and Asymmetric Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
