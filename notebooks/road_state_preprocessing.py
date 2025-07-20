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
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, Point
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import Counter
import os
import datetime as dt

# %%
mvi = gpd.read_file('../data/2_Estructura_via/estructura_via_completa/MVI.geojson')
mvi = mvi.to_crs(epsg = 4326)
mvi["id_mvi"] = mvi["MVICIV"].astype(str) + "_" + mvi["MVISVIA"].str.replace(" ", "")
mvi['contador'] = mvi.groupby('id_mvi').cumcount() + 1
mvi['id_mvi'] = mvi['id_mvi'] + '.' + mvi['contador'].astype(str)
mvi = (
    mvi.explode(ignore_index=True)          # if shapefile has split segments
       .dissolve(by='id_mvi')               # merge all segments of same road
       .reset_index()
)
mvi['road_id'] = mvi['id_mvi']
buffer_distance = 15
mvi_buffered = mvi.to_crs("EPSG:3116").copy()
mvi_buffered['mvi_length'] = mvi_buffered.geometry.length
mvi_buffered['geometry'] = mvi_buffered.geometry.buffer(buffer_distance, cap_style="square")
mvi_buffered_buffer = mvi_buffered.set_geometry('geometry')
mvi_buffered_buffer = mvi_buffered_buffer.to_crs("EPSG:4326")

# %%
road_state_gdf = gpd.read_file('../data/14_Estado_malla_vial/EstadoMallaVialOD_IDU_Sem12024/EstadoMallaVialUrbanaConsolidado.shp').to_crs(epsg = 4326)
road_state_dict = {'SATISFACTORIO':3,'BUENO':2,'ACEPTABLE':1,'SIN DATO':0,'POBRE':-1,'MUY POBRE':-2,'FALLADO':-3,'GRAVE':-4}
road_state_gdf['numeric_road_state'] = list(tqdm(map(lambda x: road_state_dict[x], road_state_gdf['ESTAASHTO'])))
road_state_gdf['numeric_state_index'] = list(tqdm(map(lambda x: int(x), road_state_gdf['INDEST'])))
road_state_gdf = road_state_gdf.to_crs("EPSG:4326")
road_state_gdf_intersections = gpd.overlay(road_state_gdf, mvi_buffered_buffer[['road_id', 'mvi_length', 'geometry']], how='intersection')
road_state_gdf_intersections['intersection_length'] = road_state_gdf_intersections.geometry.length
road_state_gdf_intersections['overlap_ratio'] = road_state_gdf_intersections['intersection_length'] / road_state_gdf_intersections['mvi_length']

# %%
road_state_gdf_intersections['overlap_rank'] = (
    road_state_gdf_intersections
    .groupby('road_id')['overlap_ratio']
    .rank(method='dense', ascending=False)
    .astype(int)
)

# %%
road_state_gdf_intersections = road_state_gdf_intersections.loc[road_state_gdf_intersections.overlap_rank==1]
road_state_gdf_intersections['timeframe_start'] = '2024-01-01 00:00:00'
road_state_gdf_intersections['timeframe_end'] = '2024-06-30 23:45:00'
road_state_dateset = road_state_gdf_intersections[['road_id','timeframe_start','timeframe_end','numeric_road_state','INDEST','LOCEST', 'SUPERFEST', 'KMCAINDEST']]

# %%
file_output_name = "../data/datos_procesados/road_state/"
os.makedirs(file_output_name, exist_ok=True)
road_state_dateset.to_parquet(f'{file_output_name}road_state_dateset.parquet', index=False)
