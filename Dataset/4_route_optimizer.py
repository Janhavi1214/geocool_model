import os
import requests 
import networkx as nx
import osmnx as ox
import pandas as pd
import joblib
import folium
import warnings

warnings.filterwarnings('ignore')

print("1. Loading the Map of Nagpur...")
# UPDATED: Changed dist=5000 to 20000 to load the full district and highways
G = ox.graph_from_point((21.1458, 79.0882), dist=20000, network_type='all')

print("2. Loading the AI Model and Dataset...")
rf_model = joblib.load("models/rf_suitability_model.pkl")
df = pd.read_csv("data/processed/edge_suitability.csv")

print("   -> Fetching Live Air Quality (PM2.5) for Nagpur...")
try:
    # Call the free Open-Meteo API for Nagpur's coordinates
    url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=21.1458&longitude=79.0882&current=pm2_5"
    response = requests.get(url, timeout=5)
    live_pm25 = response.json()['current']['pm2_5']
    print(f"   -> Success! Current Nagpur PM2.5 is: {live_pm25} µg/m³")
    
    # Overwrite the fake 65.0 data with the live reading
    df['pm'] = live_pm25 
except Exception as e:
    print(f"   -> API failed, falling back to historical average. Error: {e}")
    df['pm'] = 65.0 # Fallback if offline

# Let the AI predict the environmental safety score using the LIVE data
features = ["lst", "ndvi", "pm", "population", "slope"]
df["predicted_score"] = rf_model.predict(df[features])

# Create a lookup dictionary for fast matching
score_map = dict(zip(df['edge_id'].astype(str), df['predicted_score']))

print("3. Applying Climate Data to the Streets...")
for u, v, key, data in G.edges(keys=True, data=True):
    # Match the street ID from the map to your CSV
    edge_osmid = str(data.get('osmid', ''))
    if isinstance(data.get('osmid'), list):
        edge_osmid = str(data['osmid'][0])
        
    score = score_map.get(edge_osmid, 0.5) # Default score if street is new
    length = data.get('length', 1.0)
    
    # FASTEST ROUTE: Only cares about physical distance
    data['normal_weight'] = length
    
    # GEOCOOL ROUTE: Penalizes distance by bad climate, rewards good climate
    data['cool_weight'] = length / (score + 0.01)

print("4. Calculating the Best Routes...")
# UPDATED: Start point (VNIT Campus) and End point (MIHAN / Outer Ring Road)
start_point = (21.1230, 79.0530) 
end_point = (21.0500, 79.0300)

# Find the closest physical intersections on the map
start_node = ox.distance.nearest_nodes(G, X=start_point[1], Y=start_point[0])
end_node = ox.distance.nearest_nodes(G, X=end_point[1], Y=end_point[0])

# Run Dijkstra's Algorithm using the two different weights
fastest_route = nx.shortest_path(G, start_node, end_node, weight='normal_weight')
coolest_route = nx.shortest_path(G, start_node, end_node, weight='cool_weight')

print("5. Calculating the Distance Trade-off...")
# Calculate physical distance of both routes in meters using the correct variables
standard_dist = nx.path_weight(G, fastest_route, weight='length')
geocool_dist = nx.path_weight(G, coolest_route, weight='length')

# Calculate the percentage increase in distance
percent_increase = ((geocool_dist - standard_dist) / standard_dist) * 100

print(f"Standard Route Distance: {standard_dist:.2f} meters")
print(f"GeoCool Route Distance: {geocool_dist:.2f} meters")
print(f"Trade-off Metric: The climate-safe route is {percent_increase:.1f}% longer.")

print("6. Generating the Interactive Map...")
# Extract the GPS coordinates for both routes to draw them
def get_coords(route):
    return [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]

m = folium.Map(
    location=[21.1400, 79.0800], 
    zoom_start=11, # Zoomed out slightly to show the longer highway route
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', 
    attr='Esri'
)

# Draw the standard shortest path in RED
folium.PolyLine(
    get_coords(fastest_route), 
    color="red", weight=5, opacity=0.7, 
    tooltip="Standard Route (Hotter/More Exposed)"
).add_to(m)

# Draw the GeoCool optimized path in BLUE
folium.PolyLine(
    get_coords(coolest_route), 
    color="blue", weight=6, opacity=1.0, 
    tooltip="GeoCool Route (Cooler/More Greenery)"
).add_to(m)

m.save("geocool_map.html")
print("Success! Open 'geocool_map.html' in your browser to see your AI in action!")