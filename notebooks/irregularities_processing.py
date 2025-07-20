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
import glob
import os
from tqdm.auto import tqdm
import geopandas as gpd
from shapely import wkt
from math import ceil
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


# %%
def quarter_slices(start, end):
    """
    Given two pandas Timestamps (start < end), return a list of (slice_start, slice_end)
    pairs, each of which lies wholly within one 15-minute bucket (minute marks 00,15,30,45).
    """
    out = []
    current = start

    while current < end:
        # Find “floor” of current to the nearest 15 minutes, then add 15 min
        floor_q = current.floor("15T")
        next_bound = floor_q + pd.Timedelta(minutes=15)

        # If current is exactly on a quarter, jump to the next one
        if current == floor_q:
            next_bound = current + pd.Timedelta(minutes=15)

        # End of this chunk is either the next boundary or the original end
        chunk_end = min(next_bound, end)

        out.append((current, chunk_end))
        current = chunk_end
    return out


# %%
folder_path = "data/4_velocidades_waze/irregularidades"
feather_files = glob.glob(os.path.join(folder_path, "*.parquet"))

# %%
mvi = gpd.read_file("data/2_Estructura_via/estructura_via_completa/MVI.geojson")
mvi = mvi.to_crs("EPSG:4326")

# %%
mvi.columns = mvi.columns.str.lower()
mvi["id_mvi"] = mvi["mviciv"].astype(str) + "_" + mvi["mvisvia"].str.replace(" ", "")
mvi['contador'] = mvi.groupby('id_mvi').cumcount() + 1
mvi['id_mvi'] = mvi['id_mvi'] + '.' + mvi['contador'].astype(str)

buffer_distance = 4
mvi_buffered = mvi.to_crs("EPSG:3116").copy()
mvi_buffered['mvi_length'] = mvi_buffered.geometry.length
mvi_buffered['geometry'] = mvi_buffered.geometry.buffer(buffer_distance, cap_style="square")
mvi_buffered_buffer = mvi_buffered.set_geometry('geometry')
mvi_buffered_buffer = mvi_buffered_buffer.to_crs("EPSG:4326")


# %%
file_output_name = "datos_procesados/irregularidades_procesadas/"
# Crear carpeta de salida si no existe
os.makedirs(file_output_name, exist_ok=True)

# %%
n_chunks = 10
chunk_size = len(feather_files)// n_chunks+1
filtrar = False

# %%
# %%time
for i in tqdm(range(chunk_size)):
    files_group = feather_files[i*n_chunks:(i+1)*n_chunks]
    min_file = min(files_group)
    max_file = max(files_group)
    start_date = min_file.split('/')[3].split('.')[0].split('_')[2]
    end_date = max_file.split('/')[3].split('.')[0].split('_')[2]
    output_file_name = f'{file_output_name}irregs_proc_{start_date}_{end_date}.parquet'
    irreg_gdf= []
    for archivo in tqdm(feather_files[i*n_chunks:(i+1)*n_chunks]):
        df = pd.read_parquet(archivo)
        df['geometry'] = df['geom_wkt'].apply(
            lambda geom: wkb.loads(geom) if isinstance(geom, bytes)
            else wkt.loads(geom) if isinstance(geom, str)
            else geom
        )
        df_exploded = df[["id", "alerts"]].explode("alerts")
        df_exploded = df_exploded.dropna(subset=["alerts"])
        alerts_normalized = pd.json_normalize(df_exploded["alerts"])
        alerts_normalized = alerts_normalized.drop(columns=['id'], errors='ignore')
        df_exploded = df_exploded.reset_index(drop=True)
        alerts_normalized = alerts_normalized.reset_index(drop=True)
        alerts_details = pd.concat(
            [ df_exploded[["id"]],   # shape: (total_dicts, 1)
            alerts_normalized      # shape: (total_dicts, N_keys)
            ],
            axis=1
        )
        if 'nThumbsUp' not in alerts_details.columns:
            alerts_details['nThumbsUp'] = 0
        alerts_details['count'] =1
        alerts_details = alerts_details.groupby(['id','subtype']).aggregate({'count':'sum','nThumbsUp':'sum'}).reset_index().pivot_table(index = 'id', columns = 'subtype', values = ['count','nThumbsUp']).fillna(0)
        alerts_details.columns = [f'{col[1]}_{col[0]}' if col[1]!='' else f'NOT_DEFINED_{col[0]}' for col in alerts_details.columns]
        alerts_details =  alerts_details.reset_index()
        df['timestamp_detectionDateMillis'] = pd.to_datetime(df['detectionDateMillis'], unit='ms')
        df['timestamp_updateDateMillis'] = pd.to_datetime(df['updateDateMillis'], unit='ms')
        df['timestamp_detectionDateMillis_ts_cot'] = pd.to_datetime(df['detectionDateMillis_ts_cot'], unit='ms')
        df['timestamp_updateDateMillis_ts_cot'] = pd.to_datetime(df['updateDateMillis_ts_cot'], unit='ms')
        df['timestamp_startTime_ts_cot'] = pd.to_datetime(df['startTime_ts_cot'], unit='ms')
        df['timestamp_endTime_ts_cot'] = pd.to_datetime(df['endTime_ts_cot'], unit='ms')
        df = pd.merge(df, alerts_details, on =['id'], how='left')
        df = df.drop(['country','updateDate','city','detectionDateMillis','endNode','street','highway','length','updateDateMillis','detectionDate','causeAlert', 'startNode','detectionDateMillis_ts_cot',
        'updateDateMillis_ts_cot', 'geom_wkt', 'startTime_ts_cot','type','alerts',
        'endTime_ts_cot', 'fecha_lectura', 'hora_lectura_dato'],axis=1)
        # break
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)
        irreg_gdf.append(gdf)
    irreg_gdf = gpd.GeoDataFrame(pd.concat(irreg_gdf, ignore_index=True).fillna(0), crs=4326)
    irreg_gdf["irreg_id"] = irreg_gdf.index
    irreg_gdf = irreg_gdf.sort_values(["id", "timestamp_startTime_ts_cot"]).reset_index(drop=True)
    irreg_gdf["next_start"] = irreg_gdf.groupby("id")["timestamp_startTime_ts_cot"].shift(-1)
    mask = irreg_gdf["next_start"].notna()
    irreg_gdf.loc[mask, "timestamp_endTime_ts_cot"] = irreg_gdf.loc[mask, "next_start"]
    irreg_gdf = irreg_gdf.drop(columns="next_start")
    irreg_gdf["orig_duration"] = irreg_gdf["timestamp_endTime_ts_cot"] - irreg_gdf["timestamp_startTime_ts_cot"]
    irreg_gdf["split_slices"] = irreg_gdf.apply(
        lambda r: quarter_slices(r["timestamp_startTime_ts_cot"], r["timestamp_endTime_ts_cot"]),
        axis=1
    )
    irreg_gdf_exploded = irreg_gdf.explode("split_slices").reset_index(drop=True)
    irreg_gdf_exploded[["new_start", "new_end"]] = pd.DataFrame(
        irreg_gdf_exploded["split_slices"].tolist(), index=irreg_gdf_exploded.index
    )
    irreg_gdf_exploded["piece_duration"] = irreg_gdf_exploded["new_end"] - irreg_gdf_exploded["new_start"]
    irreg_gdf_exploded["pct_of_interval"] = (
        irreg_gdf_exploded["piece_duration"] / irreg_gdf_exploded["orig_duration"]
    )
    irreg_gdf_exploded['piece_duration_mins'] = (irreg_gdf_exploded["piece_duration"]
        .dt.total_seconds()
        .div(60))
    irreg_gdf = irreg_gdf_exploded.drop(
        columns=[
            "timestamp_startTime_ts_cot",
            "timestamp_endTime_ts_cot",
            "split_slices",
            "orig_duration",
            "piece_duration"
        ]
    )
    irreg_gdf = irreg_gdf.rename(
        columns={"new_start": "timestamp_startTime_ts_cot", "new_end": "timestamp_endTime_ts_cot"}
    )
    irreg_gdf = irreg_gdf.reset_index(drop=True)
    if irreg_gdf.crs != mvi_buffered_buffer.crs:
        irreg_gdf = irreg_gdf.to_crs(mvi_buffered_buffer.crs)

    irregs_intersections = gpd.overlay(irreg_gdf, mvi_buffered_buffer[['id_mvi', 'mvi_length', 'geometry']], how='intersection')
    irregs_intersections = irregs_intersections.to_crs("EPSG:3116").copy()
    irregs_intersections['intersection_length'] = irregs_intersections.geometry.length
    irregs_intersections['overlap_ratio'] = irregs_intersections['intersection_length'] / irregs_intersections['mvi_length']
    irregs_intersections = irregs_intersections[irregs_intersections['overlap_ratio'] > 0.2]
    irregs_intersections = irregs_intersections.to_crs("EPSG:4326").copy()
    irregs_intersections["timeframe"] = irregs_intersections["timestamp_startTime_ts_cot"].dt.floor("15min")

    non_agg_variables = ['type', 'timestamp_updateDateMillis_ts_cot', 'id', 'timeframe', 'timestamp_detectionDateMillis', 'geometry', 'overlap_ratio', 'intersection_length', 'irreg_id', 'pct_of_interval', 'timestamp_updateDateMillis', 'timestamp_startTime_ts_cot', 'alerts', 'timestamp_endTime_ts_cot', 'mvi_length', 'id_mvi', 'timestamp_detectionDateMillis_ts_cot']
    non_transformable_variables = ['trend','speed','jamLevel','severity','regularSpeed']
    countable_variables = ['causeType']
    transformable_variables = sorted(list(set(irregs_intersections.columns)-set(non_transformable_variables)-set(countable_variables)-set(non_agg_variables)))
    agg_dict = {}

    for col in transformable_variables:
        if col in irregs_intersections.columns:
            irregs_intersections[col] = irregs_intersections[col]*irregs_intersections["pct_of_interval"].clip(upper=1)*irregs_intersections["overlap_ratio"].clip(upper=1)
            agg_dict[col] = 'sum'

    for col in non_transformable_variables:
        if col in irregs_intersections.columns:
            irregs_intersections[f'{col}_weighted'] = irregs_intersections[col]*irregs_intersections["piece_duration_mins"]
            agg_dict[f'{col}_weighted'] = 'sum'
    agg_dict['piece_duration_mins'] = 'sum'

    dummies = pd.get_dummies(
        irregs_intersections[countable_variables].astype(str),
        prefix=countable_variables
    ).astype('int')
    for col in dummies.columns:
        agg_dict[col] = 'sum'
    irregs_intersections = pd.concat([irregs_intersections,dummies],axis=1)

    irregs_intersections = irregs_intersections.sort_values(['id_mvi','timeframe']).reset_index(drop=True)

    agg_irregs_intersections = irregs_intersections.groupby(['id_mvi','timeframe'], sort=False).agg(agg_dict)
    agg_irregs_intersections = agg_irregs_intersections.reset_index()


    for col in non_transformable_variables:
        agg_irregs_intersections[col] = agg_irregs_intersections[f'{col}_weighted']/agg_irregs_intersections['piece_duration_mins']
        agg_irregs_intersections = agg_irregs_intersections.drop([f'{col}_weighted'],axis=1)
    agg_irregs_intersections = agg_irregs_intersections.drop(['piece_duration_mins'],axis=1)
    agg_irregs_intersections.to_parquet(output_file_name, index=False)

# %%

# %%

# %%

# %%

# %%
