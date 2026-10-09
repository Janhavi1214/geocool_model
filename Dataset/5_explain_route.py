import pandas as pd
import numpy as np
import joblib
import shap
import networkx as nx
import osmnx as ox
import warnings

warnings.filterwarnings('ignore')

print("1. Loading Map, AI Model, and Environmental Data...")
G = ox.graph_from_point((21.1458, 79.0882), dist=2000, network_type='all')
rf_model = joblib.load("models/rf_suitability_model.pkl")
df = pd.read_csv("data/processed/edge_suitability.csv")

features = ["lst", "ndvi", "pm", "population", "slope"]
df["predicted_score"] = rf_model.predict(df[features])
score_map = dict(zip(df['edge_id'].astype(str), df['predicted_score']))

# Apply graph weights
for u, v, key, data in G.edges(keys=True, data=True):
    edge_osmid = str(data.get('osmid', ''))
    if isinstance(data.get('osmid'), list):
        edge_osmid = str(data['osmid'][0])
    score = score_map.get(edge_osmid, 0.5)
    length = data.get('length', 1.0)
    data['normal_weight'] = length
    data['cool_weight'] = length / (score + 0.01)

# Calculate Routes (VNIT to MIHAN)
start_node = ox.distance.nearest_nodes(G, X=79.0530, Y=21.1230)
end_node = ox.distance.nearest_nodes(G, X=79.0300, Y=21.0500)

fastest_route = nx.shortest_path(G, start_node, end_node, weight='normal_weight')
coolest_route = nx.shortest_path(G, start_node, end_node, weight='cool_weight')

# Extract Feature Data for Edges in Each Route
def get_route_edge_data(route):
    route_edges = []
    for u, v in zip(route[:-1], route[1:]):
        edge_data = G.get_edge_data(u, v)[0]
        edge_osmid = str(edge_data.get('osmid', ''))
        if isinstance(edge_data.get('osmid'), list):
            edge_osmid = str(edge_data['osmid'][0])
        route_edges.append(edge_osmid)
    
    route_df = df[df['edge_id'].isin(route_edges)]
    return route_df

df_fastest = get_route_edge_data(fastest_route)
df_coolest = get_route_edge_data(coolest_route)

print("2. Calculating SHAP (Explainable AI) Values...")
# TreeExplainer is specifically optimized for tree-based models like Random Forest
explainer = shap.TreeExplainer(rf_model)
shap_fastest = explainer.shap_values(df_fastest[features])
shap_coolest = explainer.shap_values(df_coolest[features])

# Compute average SHAP impact per feature across the routes
mean_shap_fastest = np.mean(shap_fastest, axis=0)
mean_shap_coolest = np.mean(shap_coolest, axis=0)

print("\n=======================================================")
print("             GEOCOOL EXPLAINABLE AI REPORT             ")
print("=======================================================\n")

print("--- AVERAGE ENVIRONMENTAL METRICS ---")
for f in features:
    v_fast = df_fastest[f].mean()
    v_cool = df_coolest[f].mean()
    print(f"  * {f.upper():<12} | Standard Route: {v_fast:6.2f} | GeoCool Route: {v_cool:6.2f}")

print("\n--- SHAP DECISION FACTORS (Why GeoCool is Better) ---")
print("Positive values mean the feature IMPROVED the route score:\n")

for idx, f in enumerate(features):
    diff_shap = mean_shap_coolest[idx] - mean_shap_fastest[idx]
    direction = "HIGHER" if diff_shap > 0 else "LOWER"
    print(f"  * {f.upper():<12} | SHAP Impact Gain: {diff_shap:+.4f} ({direction} suitability contribution)")

print("\n-------------------------------------------------------")
print("SUMMARY EXPLANATION FOR USER UI:")
top_feature_idx = np.argmax(np.abs(mean_shap_coolest - mean_shap_fastest))
top_feature = features[top_feature_idx].upper()

print(f"\"GeoCool rerouted you away from the standard path primarily due to better {top_feature}.")
print(f" The optimized route provides a overall predicted safety score of {df_coolest['predicted_score'].mean():.2f} ")
print(f" compared to {df_fastest['predicted_score'].mean():.2f} on the standard route.\"")
print("-------------------------------------------------------\n")

# --- PRACTICAL TRADE-OFF METRICS ---
print("\n" + "="*50)
print("PRACTICAL TIME & FUEL TRADE-OFF ANALYSIS")
print("="*50)

# FIX: Calculate the physical distance in meters for both routes first
standard_dist = nx.path_weight(G, fastest_route, weight='length')
geocool_dist = nx.path_weight(G, coolest_route, weight='length')

# 1. Time Calculation (at 60 km/hr)
# 60 km/hr is exactly 1 km per minute, or 1000 meters per minute
speed_kmh = 60
speed_m_per_min = (speed_kmh * 1000) / 60  

time_std_min = standard_dist / speed_m_per_min
time_geo_min = geocool_dist / speed_m_per_min
extra_time_min = time_geo_min - time_std_min

# 2. Fuel & Cost Calculation
# Assuming a standard car mileage of 15 km/liter and Nagpur petrol cost of ~₹105/liter
mileage_kml = 15.0
petrol_price = 105.0 
extra_dist_km = (geocool_dist - standard_dist) / 1000

extra_fuel_liters = extra_dist_km / mileage_kml
extra_cost_rupees = extra_fuel_liters * petrol_price

print(f"Driving Speed Assumed: {speed_kmh} km/hr")
print(f"Vehicle Efficiency Assumed: {mileage_kml} km/liter")
print("-" * 50)
print(f"Standard Route Time: {time_std_min:.1f} minutes")
print(f"GeoCool Route Time:  {time_geo_min:.1f} minutes")
print(f"--> Additional Travel Time: {extra_time_min:.1f} minutes")
print(f"--> Additional Fuel Used:   {extra_fuel_liters:.3f} Liters")
print(f"--> Additional Fuel Cost:   ₹{extra_cost_rupees:.2f}")
print("="*50 + "\n")