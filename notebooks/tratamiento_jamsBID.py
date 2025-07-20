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
import pandas as pd
from tqdm.auto import tqdm
import geopandas as gpd
from datetime import datetime as dt
import numpy as np
import math
import glob
import re
import os
import shapely.wkb
from shapely import wkt, wkb
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
from math import ceil
pd.options.display.max_columns = None
import gc

# %%
tqdm.pandas()

# %%
folder_path = "../data/4_velocidades_waze/jams/jams_originales"
feather_files = glob.glob(os.path.join(folder_path, "*.parquet"))

# %%
# Cargar archivos
# Parámetros
n_chunks = 10
chunk_size = ceil(len(feather_files) / n_chunks)
filtrar = False

# Leer malla vial y preparar buffers
mvi = gpd.read_file("../data/2_Estructura_via/estructura_via_completa/MVI.geojson")
mvi.to_crs(4326, inplace=True)
mvi.columns = mvi.columns.str.lower()
mvi["id_mvi"] = mvi["mviciv"].astype(str) + "_" + mvi["mvisvia"].str.replace(" ", "")
mvi['contador'] = mvi.groupby('id_mvi').cumcount() + 1
mvi['road_id'] = mvi['id_mvi'] + '.' + mvi['contador'].astype(str)

buffer_distance = 4
mvi_buffered = mvi.to_crs("EPSG:3116").copy()
mvi_buffered['mvi_length'] = mvi_buffered.geometry.length
mvi_buffered['buffer_geom'] = mvi_buffered.geometry.buffer(buffer_distance, cap_style="square")
mvi_buffered_buffer = mvi_buffered.set_geometry('buffer_geom')
mvi_buffered_buffer = mvi_buffered_buffer.to_crs("EPSG:3116")

# Crear carpeta de salida si no existe
os.makedirs("outputs", exist_ok=True)

# %%
del mvi
gc.collect()

# %%
# feather_files = list(filter(lambda x: x>'../data/4_velocidades_waze/jams/jams_originales/jams_data_2024-05-24.parquet', feather_files))

# %%
file_output_name = "../data/datos_procesados/jams_procesados/"
# Crear carpeta de salida si no existe
os.makedirs(file_output_name, exist_ok=True)

# %%
n_chunks = 5
chunk_size = len(feather_files)// n_chunks+1
filtrar = False

# %%
# Proceso por chunks
for i in tqdm(range(chunk_size)):
    files_group = feather_files[i*n_chunks:(i+1)*n_chunks]
    min_file = min(files_group)
    max_file = max(files_group)
    start_date = min_file.split('/')[5].split('.')[0].split('_')[2]
    end_date = max_file.split('/')[5].split('.')[0].split('_')[2]
    output_file_name = f'{file_output_name}jams_proc_{start_date}_{end_date}.parquet'

    jams_gdf= []
    for archivo in tqdm(feather_files[i*n_chunks:(i+1)*n_chunks]):
        df = pd.read_parquet(archivo)
        df['geometry'] = df['geom_wkt'].apply(
            lambda geom: wkb.loads(geom) if isinstance(geom, bytes)
            else wkt.loads(geom) if isinstance(geom, str)
            else geom
        )
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)
        gdf_filtrado = gdf.copy()

        gdf_filtrado['timestamp_pubMillis'] = pd.to_datetime(gdf_filtrado['pubMillis_ts_cot'], unit='ms')
        gdf_filtrado['timestamp_startTime'] = pd.to_datetime(gdf_filtrado['startTime_ts_cot'], unit='ms')
        gdf_filtrado['timestamp_endTime'] = pd.to_datetime(gdf_filtrado['endTime_ts_cot'], unit='ms')

        gdf_filtrado = gdf_filtrado[['uuid', 'timestamp_pubMillis', 'timestamp_startTime', 'timestamp_endTime',
                                     'speedKMH', 'level', 'length', 'delay', 'geometry']]
        jams_gdf.append(gdf_filtrado)

    if not jams_gdf:
        continue

    jams_gdf = gpd.GeoDataFrame(pd.concat(jams_gdf, ignore_index=True), crs=4326)
    jams_gdf["jam_id"] = jams_gdf.index
    grouped_jams = jams_gdf.groupby("uuid").agg({
        'timestamp_startTime': 'min',
        'timestamp_endTime': 'max',
        'jam_id': 'count'
    }).reset_index()
    grouped_jams.columns = ['uuid', 'general_timestamp_startTime', 'general_timestamp_endTime', 'jam_id_counts']

    jams_gdf = jams_gdf[jams_gdf["delay"] != -1]  ## Esto de puede cambiar si no interesa que se eliminen los jams en vias cerradas.
    jams_gdf = jams_gdf.merge(grouped_jams, how="left", on='uuid')

    jams_gdf = jams_gdf.to_crs("EPSG:3116")

    # Aplicar overlay pesado dentro del for y guardar
    jams_gdf = gpd.overlay(jams_gdf, mvi_buffered_buffer[['road_id', 'mvi_length', 'buffer_geom']], how='intersection')
    jams_gdf['intersection_length'] = jams_gdf.geometry.length
    jams_gdf['overlap_ratio'] = jams_gdf['intersection_length'] / jams_gdf['mvi_length']
    jams_gdf = jams_gdf[jams_gdf['overlap_ratio'] > 0.2]

    jams_gdf["timeframe"] = jams_gdf["timestamp_pubMillis"].dt.floor("15min")
    jams_gdf['delay'] = jams_gdf['delay'] * (jams_gdf['intersection_length'] / jams_gdf['length'])

    aggregated = jams_gdf.groupby(["road_id", "timeframe"]).agg({
        "speedKMH": ["min", "mean", "max"],
        "level": ["min", "mean", "max"],
        "delay": ["min", "sum", "max"],
        "jam_id": "count"
    }).reset_index()
    del jams_gdf
    gc.collect()
    aggregated.columns = [
        "road_id", "timeframe",
        "vel_min", "vel_mean", "vel_max",
        "level_min", "level_mean", "level_max",
        "delay_min", "delay_sum", "delay_max",
        "count_jams"
    ]
    aggregated.sort_values(by=["road_id", "timeframe"], inplace=True)
    # # Guardar resultado del overlay intermedio para cada chunk
    # overlay_output_path = f"outputs/jams_intersections_chunk_{i+1}.parquet"
    # jams_intersections.to_parquet(overlay_output_path, index=False)
    # print(f"Overlay del chunk {i+1} guardado en {overlay_output_path}")
    aggregated.to_parquet(output_file_name, index=False)

    del aggregated
    gc.collect()

# %%
