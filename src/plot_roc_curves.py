import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from model import models, X_test_scaled, y_test
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# --- Your models, X_test_scaled, y_test must be defined or imported here ---
# For example, if you have a models.py that loads/trains models, import them
# from models import models, X_test_scaled, y_test

print("Starting ROC Curve and Metrics Summary script...")

# Check if models dict exists and is not empty
if 'models' not in globals() or not models:
    print("ERROR: 'models' dictionary not found or empty. Please define your models.")
    exit()

# Check if X_test_scaled and y_test exist
if 'X_test_scaled' not in globals() or 'y_test' not in globals():
    print("ERROR: Test data X_test_scaled or y_test not found. Please define them.")
    exit()

# For storing metrics summary
summary = []

plt.figure(figsize=(10, 8))

for name, model in models.items():
    print(f"Processing model: {name}")

    # Get predicted probabilities or decision function
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = model.decision_function(X_test_scaled)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    # Predict class labels for metrics
    y_pred = model.predict(X_test_scaled)

    # Calculate other metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    summary.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "AUC": roc_auc
    })

plt.plot([0, 1], [0, 1], "k--")  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison for All Models")
plt.legend(loc="lower right")
plt.grid(True)
print("About to show plot...")
plt.show()

# Print tabular summary
print("\nSummary of Metrics for all Models:")
print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'AUC':<10}")
print("-" * 75)
for row in summary:
    print(f"{row['Model']:<25} {row['Accuracy']:<10.3f} {row['Precision']:<10.3f} {row['Recall']:<10.3f} {row['F1-score']:<10.3f} {row['AUC']:<10.3f}")

print("\nScript finished.")

# 5. Confusion Matrices
for name, model in models.items():
    print(f"\nShowing Confusion Matrix for: {name}")
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

print("\nScript finished.")

