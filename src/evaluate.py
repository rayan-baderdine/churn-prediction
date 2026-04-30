import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score,
    confusion_matrix, classification_report
)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Print a full evaluation report for a trained classifier."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"=== {model_name} ===")
    print(f"AUC-ROC  : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print()
    print(classification_report(y_test, y_pred,
                                 target_names=['No churn', 'Churn']))

def plot_confusion_matrix(model, X_test, y_test, model_name="Model"):
    """Plot a styled confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No churn', 'Churn'],
                yticklabels=['No churn', 'Churn'], ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(f'Confusion matrix — {model_name}', fontsize=13)
    plt.tight_layout()
    plt.show()