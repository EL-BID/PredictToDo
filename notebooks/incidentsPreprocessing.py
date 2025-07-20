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
roads_gdf['buffer'] = roads_gdf.geometry.buffer(0.001)
roads_buf = roads_gdf.set_geometry('buffer')[['road_id','buffer']]

# %%
incidents = pd.read_csv('data/7_incidentes/incidents_dataset_prev.csv', encoding = 'latin-1')
incidents['geometry'] = list(map(lambda x,y:  Point(x,y), incidents['Coordenada X'], incidents['Coordenada Y']))
incidents = gpd.GeoDataFrame(incidents).set_crs(epsg = 4326)
incidents = incidents.rename({'Tipo Incidente':'incident_type','Especificación del Caso':'case_specifics'},axis=1)

# %%
joined = gpd.sjoin(
    roads_buf,
    incidents,
    how='left',
    predicate='intersects'
)

# %%
joined = joined.loc[pd.notnull(joined.index_right)].reset_index()
joined['totals'] =1
joined['closing_date'] = list(tqdm(map(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), joined['Fecha Cierre'])))
joined['timeframe'] = joined['closing_date'].dt.floor("15min")

# %%
incident_type_df = joined.groupby(['road_id','timeframe','incident_type']).agg({'totals':'sum'}).reset_index().pivot_table(index = ['road_id','timeframe'], columns = 'incident_type', values= 'totals').fillna(0).reset_index()
incident_type_df.columns  = [f'incident_type_{col}' if col not in ['road_id','timeframe'] else col for col in incident_type_df.columns]

# %%
case_specifics_df = joined.groupby(['road_id','timeframe','case_specifics']).agg({'totals':'sum'}).reset_index().pivot_table(index = ['road_id','timeframe'], columns = 'case_specifics', values= 'totals').fillna(0).reset_index()
case_specifics_df.columns  = [f'incident_case_specifics_{col}' if col not in ['road_id','timeframe'] else col for col in case_specifics_df.columns]

# %%
incidents_data = pd.merge(incident_type_df, case_specifics_df, on = ['road_id','timeframe'], how = 'left')

# %%
file_output_name = "datos_procesados/incidents/"
os.makedirs(file_output_name, exist_ok=True)
incidents_data.to_csv(f'{file_output_name}incidents_dataset.csv', index=False)
