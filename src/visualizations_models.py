# visualize_models.py
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ========== SETTINGS ==========
plt.style.use('seaborn-v0_8-darkgrid')
sns.set(font_scale=1.2)
os.makedirs("visualizations", exist_ok=True)

# ========== 1. Load Data ==========
base_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_dir, "data", "raw", "wdbc.data")

columns = [
    "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
    "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
    "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

df = pd.read_csv(data_path, header=None, names=columns)
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})  # 0 = Benign, 1 = Malignant

X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

# ========== 2. Load Models ==========
models_dir = "src/models"
model_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]

models = {}
for file in model_files:
    name = file.replace(".joblib", "").replace("_", " ").title()
    models[name] = joblib.load(os.path.join(models_dir, file))

scaler = joblib.load("src/scaler.joblib")
X_scaled = scaler.transform(X)

print(f"✅ Loaded {len(models)} models for visualization.")

# ========== 3. Dataset Distribution Pie Chart ==========
plt.figure(figsize=(6, 6))
labels = ['Benign (0)', 'Malignant (1)']
sizes = df['diagnosis'].value_counts().sort_index()
colors = ['#66b3ff', '#ff6666']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title("Dataset Distribution (Benign vs Malignant)")
plt.savefig("visualizations/dataset_distribution.png")
plt.show()

# ========== 4. Accuracy Bar Chart ==========
accuracies = []
for name, model in models.items():
    acc = model.score(X_scaled, y)
    accuracies.append((name, acc))

accuracies = sorted(accuracies, key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 5))
sns.barplot(x=[a[0] for a in accuracies], y=[a[1]*100 for a in accuracies], palette='viridis')
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("visualizations/model_accuracy.png")
plt.show()

# ========== 5. Confusion Matrices ==========
for name, model in models.items():
    y_pred = model.predict(X_scaled)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"visualizations/confusion_matrix_{name.replace(' ', '_').lower()}.png")
    plt.show()

# ========== 6. ROC Curves ==========
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("visualizations/roc_curves.png")
plt.show()

print("\n🎉 All visualizations saved in the 'visualizations/' folder!")
