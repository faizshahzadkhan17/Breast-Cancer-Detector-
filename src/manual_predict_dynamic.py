import time
import numpy as np
from model import models, scaler  # uses trained models and scaler from models.py

print("=" * 40)
print("\U0001F916  Breast Cancer Prediction System")
print("=" * 40)

input("\nStep 1⃣: Press Enter to initialize the prediction system...")

print("\n\U0001F50D Step 2⃣: Preparing the data pipeline...")
print("Loading scaler and model...")
time.sleep(1.5)
print("... Done!")

print("\n\U0001F4CA Step 3⃣: Data Input Format")
print("You need to enter 30 feature values (tabular data from a breast cancer scan).")
print("\U0001F4A1 Format: comma-separated values (CSV)")
print("Example: 17.99,10.38,122.8,1001.0,... (30 total)")

print("\n✍️ Step 4⃣: Please enter the 30 feature values:")
user_input = input("\U0001F449 ")

# Step 5: Convert and scale input
print("\n⚙️ Step 5⃣: Processing input through scaler...")
try:
    input_values = [float(x.strip()) for x in user_input.split(",")]
    if len(input_values) != 30:
        raise ValueError("You must enter exactly 30 values!")

    input_array = np.array(input_values).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    print("✅ Data scaled successfully.")
except Exception as e:
    print("\n❌ Error in input:", e)
    exit()

# Step 6: Choose best model and predict
print("\n🚀 Final Step: Running prediction...")
time.sleep(1)

best_model = models["Neural Network"]  # Or use any other model like "Logistic Regression"
probability = best_model.predict_proba(scaled_input)[0][1]
prediction = best_model.predict(scaled_input)[0]

print("\n" + "=" * 30)
print("\u2705 Prediction result:")
if prediction == 1:
    print("\U0001F534 Likely Malignant (Potential Breast Cancer)")
else:
    print("\U0001F7E2 Likely Benign (No Breast Cancer Detected)")
print(f"\U0001F4CA Probability of being malignant: {probability:.2f}")
print("=" * 30)
