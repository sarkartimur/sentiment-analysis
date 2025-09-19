import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    auc,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve


def compute_metrics(y_test, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f"ROC-AUC: {roc_auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}")

    __calculate_certainties(y_pred_proba, y_test)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='all')


def __calculate_certainties(y_pred_proba, y_test):
    # Get the predicted class by finding the column index with the max probability
    y_pred = np.argmax(y_pred_proba, axis=1)

    # 1. Overall average certainty
    certainty_list = np.max(y_pred_proba, axis=1)
    overall_avg_certainty = np.mean(certainty_list)
    print(f"\nOverall Avg. Certainty: {overall_avg_certainty:.4f}")

    # 2. Average certainty by predicted class
    def avg_class_certainty(pclass):
        mask = (y_pred == pclass)
        avg_cert_pred_class = np.mean(y_pred_proba[mask, pclass])
        print(f"Avg. Certainty for predictions of Class {pclass}: {avg_cert_pred_class:.4f}")

    avg_class_certainty(1)
    avg_class_certainty(0)

    # 3. Average certainty by result type
    def avg_pred_certainty(predicted_class, actual_class, result_type):
        mask = (y_pred == predicted_class) & (y_test == actual_class)
        avg_cert_correct_class = np.mean(y_pred_proba[mask, predicted_class])
        print(f"Avg. Certainty for Class {predicted_class} ({result_type}): {avg_cert_correct_class:.4f}")

    avg_pred_certainty(1, 1, 'True Positives')
    avg_pred_certainty(0, 0, 'True Negatives')
    avg_pred_certainty(1, 0, 'False Positives')
    avg_pred_certainty(0, 1, 'False Negatives')


def plot_roc_auc(y_test, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall/Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.text(0.6, 0.3, f'AUC = {roc_auc:.3f}', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.show()


def plot_threshold_graph(y_test, y_pred_proba):
    y_pred_proba = y_pred_proba[:, 1]
    thresholds = np.linspace(0, 1, 100)
    accuracies = [accuracy_score(y_test, (y_pred_proba >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, 'b-', linewidth=2)
    plt.axvline(x=0.5, color='r', linestyle='--', label=f'Default Threshold (0.5)')
    plt.axvline(x=best_threshold, color='g', linestyle='--', 
                label=f'Optimal Threshold ({best_threshold:.2f})')
    plt.scatter(best_threshold, best_accuracy, color='green', s=100, zorder=5)
    plt.xlabel('Classification Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Classification Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Default threshold (0.5) accuracy: {accuracies[50]:.3f}")
    print(f"Best threshold ({best_threshold:.3f}) accuracy: {best_accuracy:.3f}")

def plot_class_overlap_graph(y_test, y_pred_proba):
    y_pred_proba = y_pred_proba[:, 1]
    thresholds = np.linspace(0.3, 0.7, 100)
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

def plot_calibration_graph(y_test, y_pred_proba):
    y_pred_proba = y_pred_proba[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=5)

    plt.figure(figsize=(10, 4))
    plt.plot(prob_pred, prob_true, 's-', label='Model calibration')
    plt.plot([0, 1], [0, 1], '--', label='Perfectly calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Curve ABOVE Diagonal\n"Too cautious, too pessimistic"')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
