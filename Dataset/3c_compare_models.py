import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

print("Loading Dataset...")
df = pd.read_csv("data/processed/edge_suitability.csv")

features = ["lst", "ndvi", "pm", "population", "slope"]
X = df[features]
y = df["composite_suitability_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
    "CatBoost": CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_seed=42, verbose=0)
}

results = []

print("\nTraining and Evaluating Models (This may take a minute)...\n")
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    results.append({
        "Model": name,
        "R² Score": round(r2, 4),
        "MSE": round(mse, 6),
        "Training Time (sec)": round(train_time, 2)
    })

results_df = pd.DataFrame(results)

print("-" * 65)
print("MODEL PERFORMANCE COMPARISON")
print("-" * 65)
print(results_df.to_string(index=False))
print("-" * 65)