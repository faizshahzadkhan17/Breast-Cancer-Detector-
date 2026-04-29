import pandas as pd
import joblib
import os
import time

from sklearn.preprocessing import StandardScaler

print("\n==============================")
print("📂  CSV Batch Prediction Mode")
print("==============================\n")

# Step 1: Ask for CSV path
csv_path = input("📁 Enter path to CSV file containing 30 features per patient: ")

if not os.path.exists(csv_path):
    print("❌ File not found. Make sure the path is correct.")
    exit()

# Step 2: Load CSV data
try:
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] != 30:
        print(f"❌ CSV must have exactly 30 features per row. Found: {df.shape[1]}")
        exit()
except Exception as e:
    print(f"❌ Failed to read CSV: {e}")
    exit()

print("📄 CSV loaded successfully! Preview:")
print(df.head(3))

# Step 3: Load scaler and model
try:
    scaler = joblib.load("src/scaler.joblib")
    model = joblib.load("src/best_model.joblib")
    print("✅ Model and Scaler loaded")
except:
    print("❌ Could not load model/scaler. Make sure model.py has been run.")
    exit()

# Step 4: Scale and predict
print("⚙️  Scaling input data...")
X_scaled = scaler.transform(df)

print("🔍 Running predictions...")
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)[:, 1]

# Step 5: Save predictions
result_df = df.copy()
result_df["Prediction"] = ["Malignant" if p == 1 else "Benign" for p in predictions]
result_df["Malignancy_Probability"] = probabilities

timestamp = time.strftime("%Y%m%d-%H%M%S")
output_path = f"predictions_{timestamp}.csv"
result_df.to_csv(output_path, index=False)

# Final Output
print("\n==============================")
print(f"📁 Results saved to: {output_path}")
print("📊 First few predictions:")
print(result_df[["Prediction", "Malignancy_Probability"]].head())
print("==============================\n")
