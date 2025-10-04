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
#     display_name: congestion-bogota
#     language: python
#     name: python3
# ---

# %%
# cd ..

# %%
import geopandas as gpd
import pandas as pd


# %%
def process_weather_df(df_path, target_col):
    print(f"Loading {target_col} data...")
    df = pd.read_parquet(df_path)
    df = df[["FECHALECTURA", "VALORLECTURA", "LATITUD", "LONGITUD", "ESTACION"]].rename(
        columns={
            "VALORLECTURA": target_col,
            "FECHALECTURA": "timestamp",
            "LONGITUD": f"long",
            "LATITUD": f"lat",
            "ESTACION": f"estacion",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")
    time_filter = df.timestamp.dt.year == 2024
    df = df[time_filter]
    # Aggregate to 15-minute intervals
    df = (
        df.set_index("timestamp")
        .groupby([pd.Grouper(freq="15min"), "estacion", "lat", "long"])
        .mean()
        .reset_index()
        .set_index("timestamp")
    )
    df = df.dropna(subset=[target_col])
    df = df.rename(columns={"estacion": f"estacion_{target_col}"})
    return df


def load_road_df(path):
    mvi = gpd.read_file(path)
    mvi = mvi.to_crs(epsg=4326)
    mvi["id_mvi"] = (
        mvi["MVICIV"].astype(str) + "_" + mvi["MVISVIA"].str.replace(" ", "")
    )
    mvi["contador"] = mvi.groupby("id_mvi").cumcount() + 1
    mvi["id_mvi"] = mvi["id_mvi"] + "." + mvi["contador"].astype(str)
    mvi = (
        mvi.explode(ignore_index=True)  # if shapefile has split segments
        .dissolve(by="id_mvi")  # merge all segments of same road
        .reset_index()
    )
    mvi["road_id"] = mvi["id_mvi"]
    buffer_distance = 15
    mvi_buffered = mvi.to_crs("EPSG:3116").copy()
    mvi_buffered["mvi_length"] = mvi_buffered.geometry.length
    mvi_buffered["geometry"] = mvi_buffered.geometry.buffer(
        buffer_distance, cap_style="square"
    )
    # Keep in projected CRS to avoid warnings with spatial operations
    return mvi_buffered[["road_id", "geometry", "mvi_length"]]


def assign_stations_to_streets(weather_df, streets_gdf, station_prefix="estacion_"):
    """
    Assigns stations to the nearest street segment using a fast join and then
    accurately calculates the distance in meters.

    Args:
        weather_df (pd.DataFrame):
            A Pandas DataFrame with station information,
            including 'lat' and 'long' columns.
        streets_gdf (gpd.GeoDataFrame):
            A GeoPandas GeoDataFrame with street segments,
            including a 'geometry' column of buffered POLYGONs.
        station_prefix (str):
            The prefix of the station column in the weather_df.
            Default is 'estacion_'.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with the stations assigned to the
                          nearest street segment, including both road and station geometries.
    """
    # Get the station column name
    station_col = [col for col in weather_df.columns if col.startswith(station_prefix)][
        0
    ]

    # Create a GeoDataFrame with the unique stations
    stations_df = weather_df.reset_index()[
        [station_col, "lat", "long"]
    ].drop_duplicates()

    # Create stations GeoDataFrame and convert to projected CRS for accurate distance calculations
    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(stations_df.long, stations_df.lat),
        crs="EPSG:4326",
    )

    # Convert streets to projected CRS if not already
    if streets_gdf.crs.to_epsg() != 3116:
        streets_gdf_proj = streets_gdf.to_crs(epsg=3116)
    else:
        streets_gdf_proj = streets_gdf

    # Convert stations to same projected CRS
    stations_gdf_proj = stations_gdf.to_crs(epsg=3116)

    # Perform spatial join in projected CRS
    joined_gdf = gpd.sjoin_nearest(
        streets_gdf_proj,
        stations_gdf_proj,
        how="left",
        lsuffix="_road",
        rsuffix="_station",
    )

    # Rename geometry columns for clarity
    result = joined_gdf[["road_id", station_col]].copy()
    result["road_geometry"] = joined_gdf["geometry"]

    # Add station geometry by merging with stations
    result = result.merge(
        stations_gdf_proj[[station_col, "geometry"]].rename(
            columns={"geometry": "station_geometry"}
        ),
        on=station_col,
        how="left",
    )

    return result


# %%
paths = [
    ("data/3_clima/tratados/direccion_viento.parquet", "direccion_viento"),
    ("data/3_clima/tratados/humedad.parquet", "humedad"),
    ("data/3_clima/tratados/precipitacion.parquet", "precipitacion"),
    ("data/3_clima/tratados/velocidad_viento.parquet", "velocidad_viento"),
    # ("/Users/martinsevilla/Downloads/congestion-bogota/data/3_clima/tratados/temperatura.parquet", "temperatura"),
]

weather_dfs = [process_weather_df(path, col) for path, col in paths]
roads_df = load_road_df("../data/2_Estructura_via/estructura_via_completa/MVI.geojson")[
    ["road_id", "geometry"]
]
assigned_stations_dfs = [assign_stations_to_streets(df, roads_df) for df in weather_dfs]

# %%
assigned_stations_df = (
    assigned_stations_dfs[0]
    .set_index("road_id")
    .drop(columns=["station_geometry", "road_geometry"])
)
for df in assigned_stations_dfs[1:]:
    assigned_stations_df = assigned_stations_df.join(
        df.set_index("road_id").drop(columns=["station_geometry", "road_geometry"]),
        how="outer",
    )

assigned_stations_df.to_parquet(
    "data/datos_procesados/weather/stations.parquet", index=True
)

# %%
var_names = [p[1] for p in paths]
for var, df in zip(var_names, weather_dfs):
    df.drop(columns=["long", "lat"]).to_parquet(
        f"data/datos_procesados/weather/{var}.parquet", index=True
    )

# %% [markdown]
# ## Visualize stations
#

# %%
# Download the world map data for visualization
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)

# %%
# Take a look at 2 random stations
vitelma_roads = assigned_stations_dfs[0][
    assigned_stations_dfs[0]["estacion_direccion_viento"] == "Vitelma"
]
uan_roads = assigned_stations_dfs[0][
    assigned_stations_dfs[0]["estacion_direccion_viento"] == "UAN Circunvalar"
]

# %%
road_geom = gpd.GeoSeries(vitelma_roads.road_geometry)
station_geom = gpd.GeoSeries(vitelma_roads.station_geometry)
# folium
m = road_geom.explore(style_kwds={"weight": 3}, height=900, width=1500)
station_geom.explore(m=m, color="red", marker_kwds={"radius": 5})

# Do the same with UAN_roads
road_geom = gpd.GeoSeries(uan_roads.road_geometry)
station_geom = gpd.GeoSeries(uan_roads.station_geometry)
# folium
road_geom.explore(m=m, color="green", style_kwds={"weight": 3}, height=900, width=1500)
station_geom.explore(m=m, color="yellow", marker_kwds={"radius": 5})
