"""
WII Dehradun: raster-based cumulative road travel-time analysis.

Outputs:
- Road_Speed_50m.tif
- WII_Cumulative_Road_Travel_Time.tif
- Hospitals_Inside_Dehradun.gpkg
- WII_Cumulative_Road_Travel_Time_Map.png

This is an approximate road-only model, not turn-by-turn navigation.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import osmnx as ox
import matplotlib.pyplot as plt

from shapely.geometry import Point
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.plot import plotting_extent
from scipy.ndimage import label
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric


# ============================================================
# 1. USER SETTINGS: CHANGE THESE INPUT PATHS AS REQUIRED
# ============================================================

BOUNDARY_PATH = r"D:\\gisday\\ddn.shp"
ROADS_PATH = r"D:\\gisday\\uk_rd.shp"
HOSPITALS_PATH = r"D:\\gisday\\uk_clinic.shp"
OUTPUT_FOLDER = r"D:\\gisday\\WII_CGHS_Accessibility"

TARGET_CRS = "EPSG:32644"
RESOLUTION = 50  # metres

# Set True to use the fixed coordinates below instead of online geocoding.
USE_FIXED_WII_COORDINATES = False
WII_LATITUDE = 30.2820163
WII_LONGITUDE = 77.9709713

# Assumed speeds in km/h, used when maxspeed is missing or invalid.
SPEED_ASSUMPTIONS = {
    "trunk": 50,
    "trunk_link": 40,
    "primary": 40,
    "primary_link": 30,
    "secondary": 35,
    "secondary_link": 30,
    "tertiary": 30,
    "tertiary_link": 25,
    "residential": 25,
    "unclassified": 20,
    "service": 15,
    "living_street": 10,
}
DRIVABLE_CLASSES = list(SPEED_ASSUMPTIONS.keys())


# ============================================================
# 2. CREATE OUTPUT FOLDER AND LOAD DATA
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

boundary = gpd.read_file(BOUNDARY_PATH)
roads = gpd.read_file(ROADS_PATH)
hospitals = gpd.read_file(HOSPITALS_PATH)

for name, layer in {
    "boundary": boundary,
    "roads": roads,
    "hospitals": hospitals,
}.items():
    if layer.crs is None:
        raise ValueError(f"The {name} layer has no CRS.")
    if layer.empty and name != "hospitals":
        raise ValueError(f"The {name} layer is empty.")

print("Boundary features:", len(boundary))
print("Road features:", len(roads))
print("Hospital/clinic points before clipping:", len(hospitals))


# ============================================================
# 3. CREATE WII ORIGIN
# ============================================================

if USE_FIXED_WII_COORDINATES:
    wii_lat, wii_lon = WII_LATITUDE, WII_LONGITUDE
else:
    # Requires an internet connection for OpenStreetMap geocoding.
    wii_lat, wii_lon = ox.geocode(
        "Wildlife Institute of India, Dehradun, India"
    )

wii_point = gpd.GeoDataFrame(
    {"name": ["Wildlife Institute of India"]},
    geometry=[Point(wii_lon, wii_lat)],
    crs="EPSG:4326",
)


# ============================================================
# 4. PROJECT ALL LAYERS AND CLIP HOSPITAL POINTS
# ============================================================

boundary = boundary.to_crs(TARGET_CRS)
roads = roads.to_crs(TARGET_CRS)
hospitals = hospitals.to_crs(TARGET_CRS)
wii_point = wii_point.to_crs(TARGET_CRS)

# Retain only hospital/clinic points inside the Dehradun boundary.
hospitals = gpd.clip(hospitals, boundary).reset_index(drop=True)

print("Hospital/clinic points inside Dehradun:", len(hospitals))

hospital_output = os.path.join(
    OUTPUT_FOLDER, "Hospitals_Inside_Dehradun.gpkg"
)
if os.path.exists(hospital_output):
    os.remove(hospital_output)

hospitals.to_file(
    hospital_output,
    layer="Hospitals_Inside_Dehradun",
    driver="GPKG",
)


# ============================================================
# 5. SELECT DRIVABLE ROADS AND ASSIGN SPEEDS
# ============================================================

if "fclass" not in roads.columns:
    raise ValueError("Road layer must contain an 'fclass' field.")

roads["fclass"] = roads["fclass"].astype(str).str.lower()

roads_drive = roads[
    roads["fclass"].isin(DRIVABLE_CLASSES)
].copy()

roads_drive = roads_drive.explode(
    index_parts=False
).reset_index(drop=True)

roads_drive = roads_drive[
    roads_drive.geometry.notna()
    & (~roads_drive.geometry.is_empty)
].copy()

roads_drive["assumed_speed"] = roads_drive["fclass"].map(
    SPEED_ASSUMPTIONS
)

if "maxspeed" in roads_drive.columns:
    roads_drive["maxspeed_numeric"] = pd.to_numeric(
        roads_drive["maxspeed"], errors="coerce"
    )
else:
    roads_drive["maxspeed_numeric"] = np.nan

roads_drive["speed_kmh"] = np.where(
    roads_drive["maxspeed_numeric"] > 0,
    roads_drive["maxspeed_numeric"],
    roads_drive["assumed_speed"],
)

roads_drive = roads_drive[
    roads_drive["speed_kmh"].notna()
    & (roads_drive["speed_kmh"] > 0)
].copy()

if roads_drive.empty:
    raise ValueError("No drivable roads with valid speeds were found.")

print("Drivable road features:", len(roads_drive))


# ============================================================
# 6. CREATE RASTER GRID AND RASTERIZE BOUNDARY
# ============================================================

xmin, ymin, xmax, ymax = boundary.total_bounds

cols = int(np.ceil((xmax - xmin) / RESOLUTION))
rows = int(np.ceil((ymax - ymin) / RESOLUTION))

transform = from_origin(xmin, ymax, RESOLUTION, RESOLUTION)

boundary_shapes = [
    (geom, 1)
    for geom in boundary.geometry
    if geom is not None and not geom.is_empty
]

boundary_raster = rasterize(
    boundary_shapes,
    out_shape=(rows, cols),
    transform=transform,
    fill=0,
    dtype="uint8",
)

print("Raster dimensions:", rows, "rows x", cols, "columns")


# ============================================================
# 7. RASTERIZE ROAD SPEEDS
# ============================================================

road_shapes = [
    (geom, float(speed))
    for geom, speed in zip(
        roads_drive.geometry, roads_drive["speed_kmh"]
    )
    if geom is not None and not geom.is_empty
]

road_speed_raster = rasterize(
    road_shapes,
    out_shape=(rows, cols),
    transform=transform,
    fill=0,
    dtype="float32",
)

# Remove road pixels outside the study boundary.
road_speed_raster[boundary_raster == 0] = 0
road_mask = road_speed_raster > 0

if not np.any(road_mask):
    raise ValueError("No road pixels were rasterized inside the boundary.")


# ============================================================
# 8. IDENTIFY CONNECTED ROAD PIXELS
# ============================================================

connected_roads, number_of_components = label(
    road_mask,
    structure=np.ones((3, 3), dtype=int),
)

print("Connected road components:", number_of_components)


# ============================================================
# 9. FIND WII STARTING PIXEL; SNAP TO ROAD IF NECESSARY
# ============================================================

wii_x = wii_point.geometry.iloc[0].x
wii_y = wii_point.geometry.iloc[0].y

wii_col_float, wii_row_float = ~transform * (wii_x, wii_y)
wii_row = int(np.floor(wii_row_float))
wii_col = int(np.floor(wii_col_float))

if not (0 <= wii_row < rows and 0 <= wii_col < cols):
    raise ValueError("WII location falls outside the raster grid.")

if road_mask[wii_row, wii_col]:
    start_row, start_col = wii_row, wii_col
    print("WII location is on a road pixel.")
else:
    road_rows, road_cols = np.where(road_mask)
    road_x = xmin + (road_cols + 0.5) * RESOLUTION
    road_y = ymax - (road_rows + 0.5) * RESOLUTION

    road_tree = cKDTree(np.column_stack((road_x, road_y)))
    snap_distance, nearest_index = road_tree.query([wii_x, wii_y])

    start_row = int(road_rows[nearest_index])
    start_col = int(road_cols[nearest_index])

    print("WII snapped to nearest road pixel.")
    print("Snap distance:", round(float(snap_distance), 2), "metres")

wii_component = connected_roads[start_row, start_col]

if wii_component == 0:
    raise ValueError("Could not identify a road component at WII.")

wii_road_mask = connected_roads == wii_component
print("Road pixels connected to WII:", int(wii_road_mask.sum()))


# ============================================================
# 10. CALCULATE CUMULATIVE ROAD TRAVEL TIME
# ============================================================

cost_surface = np.full(
    (rows, cols), np.inf, dtype=np.float64
)

# Cost is minutes per metre; MCP applies the raster step distance.
cost_surface[wii_road_mask] = (
    60.0 / (road_speed_raster[wii_road_mask] * 1000.0)
)

mcp = MCP_Geometric(
    cost_surface,
    fully_connected=True,
    sampling=(RESOLUTION, RESOLUTION),
)

cumulative_time, traceback = mcp.find_costs(
    [(start_row, start_col)]
)

cumulative_time[boundary_raster == 0] = np.inf

valid_times = cumulative_time[np.isfinite(cumulative_time)]
if valid_times.size == 0:
    raise ValueError("No finite travel-time values were calculated.")


# ============================================================
# 11. EXPORT ROAD SPEED RASTER
# ============================================================

road_speed_path = os.path.join(
    OUTPUT_FOLDER, "Road_Speed_50m.tif"
)

road_speed_output = np.where(
    road_speed_raster > 0,
    road_speed_raster,
    -9999,
).astype("float32")

with rasterio.open(
    road_speed_path,
    "w",
    driver="GTiff",
    height=rows,
    width=cols,
    count=1,
    dtype="float32",
    crs=TARGET_CRS,
    transform=transform,
    nodata=-9999,
    compress="lzw",
) as dst:
    dst.write(road_speed_output, 1)


# ============================================================
# 12. EXPORT CUMULATIVE TRAVEL-TIME RASTER
# ============================================================

time_path = os.path.join(
    OUTPUT_FOLDER, "WII_Cumulative_Road_Travel_Time.tif"
)

time_output = np.where(
    np.isfinite(cumulative_time),
    cumulative_time,
    -9999,
).astype("float32")

with rasterio.open(
    time_path,
    "w",
    driver="GTiff",
    height=rows,
    width=cols,
    count=1,
    dtype="float32",
    crs=TARGET_CRS,
    transform=transform,
    nodata=-9999,
    compress="lzw",
) as dst:
    dst.write(time_output, 1)


# ============================================================
# 13. PLOT THE MAP
# ============================================================

with rasterio.open(time_path) as src:
    time_data = src.read(1)
    extent = plotting_extent(src)
    nodata_value = src.nodata

time_data = np.where(time_data == nodata_value, np.nan, time_data)

fig, ax = plt.subplots(figsize=(12, 10))

image = ax.imshow(
    time_data,
    extent=extent,
    cmap="RdYlGn_r",
    origin="upper",
)

boundary.boundary.plot(
    ax=ax,
    color="black",
    linewidth=1.2,
    label="Dehradun Boundary",
    zorder=3,
)

# Clip road features for map display.
roads_for_plot = gpd.clip(roads_drive, boundary)
roads_for_plot.plot(
    ax=ax,
    color="grey",
    linewidth=0.25,
    alpha=0.5,
    label="Road Network",
    zorder=2,
)

wii_point.plot(
    ax=ax,
    color="blue",
    marker="*",
    markersize=180,
    edgecolor="white",
    linewidth=0.7,
    label="WII",
    zorder=5,
)

if not hospitals.empty:
    hospitals.plot(
        ax=ax,
        color="red",
        marker="^",
        markersize=45,
        edgecolor="black",
        linewidth=0.5,
        label="Hospitals/Clinics",
        zorder=4,
    )

colorbar = plt.colorbar(image, ax=ax, shrink=0.75)
colorbar.set_label("Cumulative Road Travel Time (minutes)")

ax.set_title(
    "Cumulative Road Travel Time from WII, Dehradun",
    fontsize=15,
    fontweight="bold",
)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()

map_path = os.path.join(
    OUTPUT_FOLDER, "WII_Cumulative_Road_Travel_Time_Map.png"
)
plt.savefig(map_path, dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 14. PRINT OUTPUT SUMMARY
# ============================================================

print("\n========================================")
print("ANALYSIS COMPLETED")
print("========================================")
print("CRS:", TARGET_CRS)
print("Raster resolution:", RESOLUTION, "metres")
print("Connected road components:", number_of_components)
print("Road pixels connected to WII:", int(wii_road_mask.sum()))
print("Accessible road pixels:", int(valid_times.size))
print("Minimum travel time:", round(float(valid_times.min()), 3), "minutes")
print("Maximum travel time:", round(float(valid_times.max()), 3), "minutes")

print("\nOutput files:")
print(road_speed_path)
print(time_path)
print(hospital_output)
print(map_path)
