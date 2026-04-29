# model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import os

# Column names from UCI Breast Cancer dataset
columns = [
    "id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
    "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
    "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
    "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Load dataset
import os

base_dir = os.path.dirname(os.path.dirname(__file__))  # gets the parent of /src
data_path = os.path.join(base_dir, "data", "raw", "wdbc.data")

df = pd.read_csv(data_path, header=None, names=columns)


# Map diagnosis to 0 and 1
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Prepare features and labels
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine (SVM)": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Neural Network": MLPClassifier(max_iter=2000)
}

for model in models.values():
    model.fit(X_train_scaled, y_train)

# Save models and scaler
os.makedirs("src/models", exist_ok=True)

for name, model in models.items():
    joblib.dump(model, f"src/models/{name.replace(' ', '_').lower()}.joblib")

joblib.dump(scaler, "src/scaler.joblib")

# Optionally: Evaluate and select the best model based on accuracy
from sklearn.metrics import accuracy_score

best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    score = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"🔍 Accuracy of {name}: {score:.4f}")
    if score > best_score:
        best_model = model
        best_score = score
        best_name = name

# Save best model separately
joblib.dump(best_model, "src/best_model.joblib")
print(f"🏆 Best model: {best_name} (saved as best_model.joblib)")


print("✅ All models and scaler saved successfully.")
