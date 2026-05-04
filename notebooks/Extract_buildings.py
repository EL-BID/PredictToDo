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
import os
from collections import Counter

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon
from tqdm.auto import tqdm

# %%
roads_gdf = gpd.read_file('data/2_Estructura_via/estructura_via_completa/MVI.geojson')
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

minx, miny, maxx, maxy = roads_gdf['buffer'].total_bounds
north, south = maxy, miny
east, west   = maxx, minx

bbox = (west, south, east, north)

tags = {'building': True}
buildings = ox.features.features_from_bbox(
    bbox=bbox, tags=tags  # Pass bbox as a tuple
)#.to_crs(epsg=3116)


# %%
roads_gdf = roads_gdf.to_crs(epsg=3116)
roads_gdf['buffer'] = roads_gdf.geometry.buffer(100)

# %%
buildings = buildings.loc[
    buildings.geometry.type.isin(['Polygon','MultiPolygon'])
    & buildings['building'].notna(),
    ['building','geometry']
].rename(columns={'building':'bld_type'})

# %%
buildings = buildings.to_crs(epsg=3116)

# %%
roads_buf = roads_gdf.set_geometry('buffer')[['road_id','buffer']]
joined = gpd.sjoin(
    roads_buf,
    buildings,
    how='left',
    predicate='intersects'
)

# %%
joined

# %%
counts = (
    joined
      .dropna(subset=['bld_type'])
      .groupby(['road_id','bld_type'], sort=False)
      .size()
      .reset_index(name='count')
)
df_counts = (
    counts
      .pivot(index='road_id', columns='bld_type', values='count')
      .fillna(0)
      .astype(int)
      .reset_index()
)

# %%
df_counts

# %%
file_output_name = "datos_procesados/buildings/"
os.makedirs(file_output_name, exist_ok=True)
df_counts.to_csv(f'{file_output_name}roads_buildings.csv', index=False)

# %%

# %%

# %%

# %%

# %%

# %%

# %%
tags = {'building': True}
gdf = ox.features.features_from_place('Bogota', tags)

# %%
gdf.loc[gdf.building=='yes']

# %%
Counter(gdf.building)

# %%
gdf.plot()

# %%

# %%
bbox

# %%
minx, miny, maxx, maxy = roads_gdf['buffer'].total_bounds
pad = 200  # extra margin
north, south = maxy+pad, miny-pad
east, west   = maxx+pad, minx-pad

# %%
bbox = (west, south, east, north)

# %%
tags = {'building': True}
buildings = ox.features_from_bbox(
    bbox=bbox, tags=tags  # Pass bbox as a tuple
)#.to_crs(epsg=3116)

# %%
ox.__version__

# %%
# !pip install --upgrade osmnx

# %%

# %%

# %%
road_df['centroid'] = road_df.geometry.centroid

# 4. Extract latitude and longitude
road_df['latitude'] = road_df.centroid.y
road_df['longitude'] = road_df.centroid.x

# 5. (Optional) Create a new GeoDataFrame of centroids
road_df_centroids = gpd.GeoDataFrame(
    road_df.drop(columns='geometry'),
    geometry=road_df['centroid'],
    crs='EPSG:4326'
)

# %%
road_df.shape

# %%
import osmnx as ox
from osmnx import geometries

# %%
road_df_m = road_df.to_crs(epsg=3116)
road_df_m['buffer'] = road_df_m.geometry.buffer(100)

# 3. Fetch all buildings in the overall bbox extended by 100 m
minx, miny, maxx, maxy = road_df_m['buffer'].total_bounds
# pad bbox slightly
pad = 200
north, south, east, west = maxy+pad, miny-pad, maxx+pad, minx-pad
# pull OSM building footprints
tags = {'building': True}
gdf_bld = ox.geometries.geometries_from_bbox(
    north, south, east, west, tags=tags
).to_crs(epsg=3116)

# keep only polygons and a 'bld_type' column
gdf_bld = gdf_bld.loc[
    gdf_bld.geometry.type.isin(['Polygon','MultiPolygon'])
    & gdf_bld['building'].notna(),
    ['building','geometry']
].rename(columns={'building':'bld_type'})

# 4. Spatial‐join: which buildings intersect which point‐buffers?
road_df_m = road_df_m.set_geometry('buffer')
joined = gpd.sjoin(
    road_df_m[['pt_id','buffer']],
    gdf_bld,
    how='left',
    predicate='intersects'
)

# 5. Count by (pt_id, bld_type)
counts = (
    joined
      .dropna(subset=['bld_type'])       # remove points with no building match
      .groupby(['pt_id','bld_type'])
      .size()
      .reset_index(name='count')
)

# 6. Pivot to wide format: one row per pt_id, one column per building type
df_counts = counts.pivot(
    index='pt_id',
    columns='bld_type',
    values='count'
).fillna(0).astype(int).reset_index()

