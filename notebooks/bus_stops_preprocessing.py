# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: jose310
#     language: python
#     name: python3
# ---

# %%
import geopandas as gpd

# %% [markdown]
# ## Read input data

# %%
roads_gdf = gpd.read_file('../data/2_Estructura_via/estructura_via_completa/MVI.geojson')
roads_gdf = roads_gdf.to_crs(epsg = 4326)
roads_gdf["id_mvi"] = roads_gdf["MVICIV"].astype(str) + "_" + roads_gdf["MVISVIA"].str.replace(" ", "")
roads_gdf['contador'] = roads_gdf.groupby('id_mvi').cumcount() + 1
roads_gdf['id_mvi'] = roads_gdf['id_mvi'] + '.' + roads_gdf['contador'].astype(str)
roads_gdf = (
    roads_gdf.explode(ignore_index=True)          # if shapefile has split segments
       .dissolve(by='id_mvi')               # merge all segments of same road
       .reset_index()
)
roads_gdf['road_id'] = roads_gdf['id_mvi']
roads_gdf['buffer'] = roads_gdf.geometry.buffer(0.01)

# %%
stops_gdf = gpd.read_file('../data/15_Paraderos_transp_publico_Zonales_SITP/Paraderos_Zonales_del_SITP.shp')
stops_gdf = stops_gdf[["objectid", "latitud", "longitud", "tipo_m_s"]]
# Create a GeoDataFrame with the stops
stops_gdf = gpd.GeoDataFrame(
    stops_gdf,
    geometry=gpd.points_from_xy(stops_gdf.longitud, stops_gdf.latitud),
    crs="EPSG:4326"
)
stops_gdf = stops_gdf.to_crs(epsg=4326)
stops_gdf['buffer'] = stops_gdf.geometry.buffer(0.01)


# %% [markdown]
# ## Compute the assignment

# %%
def assign_stops_to_streets(bus_stops_gdf_4326, streets_gdf_4326):
    """
    Assigns bus stops to the nearest street segment using a fast join and then
    accurately calculates the distance in meters.

    Args:
        bus_stops_df (pd.DataFrame): A pandas DataFrame with bus stop information,
                                     including 'latitude' and 'longitude' columns.
        street_segments_gdf (gpd.GeoDataFrame): A GeoPandas GeoDataFrame with street
                                                segments, including a 'geometry'
                                                column of LINESTRINGs.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with the bus stops assigned to the
                          nearest street segment, including an accurate
                          'distance_meters' column.
    """
    # We use this join only to get the 'index_right' which maps stops to streets.
    # The distance calculated here is in degrees and inaccurate, so we don't use it.
    joined_gdf = gpd.sjoin_nearest(
        bus_stops_gdf_4326,
        streets_gdf_4326,
        how="left",
        max_distance=0.005  # Approx 500 meters in degrees, just to limit the search area
    )
    # We only need the index of the nearest street, which is named 'index_right' by the join
    joined_gdf = joined_gdf.drop(columns='geometry')

    # Now we project original GeoDataFrames for accurate calculation
    # We'll use a projected CRS suitable for Colombia, such as EPSG:21899
    projected_crs = "EPSG:21899"
    bus_stops_gdf_proj = bus_stops_gdf_4326.to_crs(projected_crs)
    streets_gdf_proj = streets_gdf_4326.to_crs(projected_crs)

    # Now we calculate the accurate distance in meters
    # Get the geometries from the projected dataframes using the original indices
    stop_geoms = bus_stops_gdf_proj.loc[joined_gdf.index].geometry
    # Use 'index_right' to get the corresponding street geometries
    street_geoms = streets_gdf_proj.loc[joined_gdf['index_right']].geometry
    joined_gdf['distance_meters'] = stop_geoms.distance(street_geoms, align=False)

    return joined_gdf

# Assign bus stops to the nearest street segment
assigned_stops = assign_stops_to_streets(
    stops_gdf[["objectid", "tipo_m_s", "geometry"]], roads_gdf[["road_id", "geometry"]]
)

# %%
# This is the final dataset
assigned_stops["is_stop_m"] = assigned_stops["tipo_m_s"] == "M"
assigned_stops["is_stop_s"] = assigned_stops["tipo_m_s"] == "S"
final_dataset = assigned_stops.groupby("road_id").agg(
    {
        "objectid": "count",
        "is_stop_m": "sum",
        "is_stop_s": "sum",
        "distance_meters": ["mean", "median"],
    }
)
final_dataset.columns = [
    "num_stops",
    "num_stops_m",
    "num_stops_s",
    "mean_distance_meters",
    "median_distance_meters",
]
final_dataset = final_dataset.reset_index()
final_dataset.to_parquet("../data/datos_procesados/bus_stops/bus_stops_per_road.parquet")

# %% [markdown]
# ## Visualizations

# %%
df_to_plot = assigned_stops.merge(
    roads_gdf[["road_id", "geometry"]],
    on='road_id',
    how='left'
).merge(
    stops_gdf[["objectid", "tipo_m_s", "geometry"]],
    on='objectid',
    how='left'
).rename(columns={
    "geometry_x": "road_geometry",
    "geometry_y": "stop_geometry"
})[["objectid", "road_id", "distance_meters", "road_geometry", "stop_geometry"]]

# %%
# Download the world map data for visualization
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)

# %%
road_geom = gpd.GeoSeries(df_to_plot.road_geometry)
stop_geom = gpd.GeoSeries(df_to_plot.stop_geometry)
# folium
m = road_geom.explore(style_kwds={"weight":3}, height=900, width=1500)
stop_geom.explore(m=m, color="red", marker_kwds={"radius":5})

m

# %%
# Show only those with the largest distances
df_to_plot_large_dist = df_to_plot[df_to_plot.distance_meters > 100].reset_index(drop=True)
road_geom = gpd.GeoSeries(df_to_plot_large_dist.road_geometry)
stop_geom = gpd.GeoSeries(df_to_plot_large_dist.stop_geometry)
# folium
m = road_geom.explore(style_kwds={"weight":3}, height=900, width=1500)
stop_geom.explore(m=m, color="red", marker_kwds={"radius":5})

m

# %%
