import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("1. Loading the processed Nagpur dataset...")
# Load the CSV you just generated
df = pd.read_csv("data/processed/edge_suitability.csv")

# Define the environmental features and the target score
features = ["lst", "ndvi", "pm", "population", "slope"]
X = df[features]
y = df["composite_suitability_score"]

print("2. Splitting data into training and testing sets...")
# 80% of streets for training, 20% to test accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Training the Random Forest Regressor...")
# Using 100 decision trees to learn the complex climate relationships
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("4. Evaluating Model Performance...")
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"   - Mean Squared Error (MSE): {mse:.4f}")
print(f"   - R-squared (Accuracy): {r2:.4f}")

# Display which environmental factors the model relied on most
importances = rf_model.feature_importances_
print("\n5. Feature Importances (What matters most to the model?):")
for feature, imp in zip(features, importances):
    print(f"   - {feature.upper()}: {imp * 100:.1f}%")

print("\n6. Saving the trained model for the routing engine...")
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/rf_suitability_model.pkl")

print("Success! Model saved to models/rf_suitability_model.pkl")