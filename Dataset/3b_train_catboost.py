import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor
import joblib

print("1. Loading the Dataset...")
df = pd.read_csv("data/processed/edge_suitability.csv")

features = ["lst", "ndvi", "pm", "population", "slope"]
X = df[features]
y = df["composite_suitability_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Training the CatBoost Model...")
cb_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_seed=42, verbose=0)
cb_model.fit(X_train, y_train)

print("3. Evaluating the Model...")
predictions = cb_model.predict(X_test)
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"   -> R² Score: {r2:.4f}")
print(f"   -> Mean Squared Error: {mse:.4f}")

joblib.dump(cb_model, "models/cb_suitability_model.pkl")
print("Success! CatBoost model saved to models/cb_suitability_model.pkl")