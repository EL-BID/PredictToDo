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
from datetime import timedelta, time
import datetime as dt
import datetime
import re
tqdm.pandas()


# %%
def generate_timeframes_vectorized(row):
    # Generate full range for the entire interval
    all_times = pd.date_range(start=row['FINI'], end=row['FFIN'], freq='15T')

    # Convert to array of times for filtering
    times_of_day = all_times.time

    # Handle normal vs overnight HCIE
    if row['hcie_start'] <= row['hcie_end']:
        mask = (times_of_day >= row['hcie_start']) & (times_of_day <= row['hcie_end'])
    else:
        mask = (times_of_day >= row['hcie_start']) | (times_of_day <= row['hcie_end'])

    # Apply mask and return filtered slots
    return all_times[mask]


# %%
mvi = gpd.read_file("../data/2_Estructura_via/estructura_via_completa/MVI.geojson")
mvi = mvi.to_crs("EPSG:4326")
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
file_output_name = "../data/datos_procesados/pmts_processed/"
# Crear carpeta de salida si no existe
os.makedirs(file_output_name, exist_ok=True)

# %% [markdown]
# #### Puntos Dataset

# %%
### 1️⃣ Constants

WEEKDAY_MAP = {
    "LUNES": 0, "MARTES": 1, "MIERCOLES": 2, "MIÉRCOLES": 2,
    "JUEVES": 3, "VIERNES": 4, "SABADO": 5, "SÁBADO": 5, "DOMINGO": 6
}

### 2️⃣ Parsing Functions

def normalize_hcie_text(text):
    text = str(text).upper()
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('L. A V.', 'LUNES A VIERNES')
    text = text.replace('L A V', 'LUNES A VIERNES')
    text = text.replace('L-VIERNES', 'LUNES A VIERNES')
    text = text.replace('L-V', 'LUNES A VIERNES')
    text = text.replace('L-VI', 'LUNES A VIERNES')
    text = text.replace('SAB', 'SABADO')
    text = text.replace('DOM', 'DOMINGO')
    text = text.replace('FESTIVOS', 'DOMINGO')
    text = text.replace('Á', 'A').replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U')
    text = re.sub(r'\([^)]*\)', '', text)
    text = text.strip()
    return text

def extract_weekday_ranges(text):
    weekdays = set()
    for match in re.finditer(r'(LUNES|MARTES|MIERCOLES|JUEVES|VIERNES|SABADO|DOMINGO)\s+A\s+(LUNES|MARTES|MIERCOLES|JUEVES|VIERNES|SABADO|DOMINGO)', text):
        start, end = WEEKDAY_MAP[match.group(1)], WEEKDAY_MAP[match.group(2)]
        if start <= end:
            weekdays.update(range(start, end + 1))
        else:
            weekdays.update(list(range(start, 7)) + list(range(0, end + 1)))
    for word in WEEKDAY_MAP.keys():
        if word in text:
            weekdays.add(WEEKDAY_MAP[word])
    if not weekdays:
        weekdays.update(range(7))
    return weekdays

def extract_timeframes(text):
    timeframes = []
    for match in re.finditer(r'(\d{1,2}[:.]\d{2})\s*[-–A]\s*(\d{1,2}[:.]\d{2})', text):
        start_str = match.group(1).replace('.', ':')
        end_str = match.group(2).replace('.', ':')

        if start_str == '24:00':
            start_str = '23:59'
        if end_str == '24:00':
            end_str = '23:59'

        start = datetime.datetime.strptime(start_str, "%H:%M").time()
        end = datetime.datetime.strptime(end_str, "%H:%M").time()
        timeframes.append((start, end))
    return timeframes

def parse_schedule_row(row):
    text = normalize_hcie_text(row)
    if '24 HORAS' in text:
        return {day: (datetime.time(0, 0), datetime.time(23, 59)) for day in range(7)}

    schedules = {}
    blocks = re.split(r'(LUNES|MARTES|MIERCOLES|JUEVES|VIERNES|SABADO|DOMINGO)', text)
    if len(blocks) > 1:
        reconstructed = []
        i = 0
        while i < len(blocks):
            if blocks[i] in WEEKDAY_MAP:
                reconstructed.append(blocks[i] + ' ' + blocks[i+1])
                i += 2
            else:
                reconstructed.append(blocks[i])
                i += 1
    else:
        reconstructed = [text]

    for block in reconstructed:
        days = extract_weekday_ranges(block)
        timeframes = extract_timeframes(block)
        for day in days:
            for tf in timeframes:
                schedules[day] = (tf[0], tf[1])
    return schedules

def generate_timeframes_vectorized_fully(row):
    if pd.isna(row['FINI']) or pd.isna(row['FFIN']):
        return pd.DatetimeIndex([])

    all_times = pd.date_range(start=row['FINI'], end=row['FFIN'], freq='15T')
    if all_times.empty:
        return pd.DatetimeIndex([])
    weekdays = all_times.weekday
    times_of_day = np.array(all_times.time)

    # Use numpy vectorize to retrieve schedules
    def get_schedule(day):
        return row['parsed_schedule'].get(day, (None, None))

    vfunc = np.vectorize(get_schedule)
    hcie_start, hcie_end = list(vfunc(weekdays))
    # print(schedules)
    # print(map(list,zip(*schedules)))
    # # schedules is a tuple of two arrays (start_times, end_times)
    # hcie_start, hcie_end = map(list,zip(*schedules))
    hcie_start = np.array(hcie_start)
    hcie_end = np.array(hcie_end)

    # Build valid_mask directly
    valid_mask = hcie_start != None

    # Convert None to dummy time for comparison
    dummy_time = datetime.time(0, 0)
    hcie_start = np.array([t if t is not None else dummy_time for t in hcie_start])
    hcie_end = np.array([t if t is not None else dummy_time for t in hcie_end])

    # Calculate masks
    normal_mask = (hcie_start <= hcie_end) & (times_of_day >= hcie_start) & (times_of_day <= hcie_end)
    overnight_mask = (hcie_start > hcie_end) & ((times_of_day >= hcie_start) | (times_of_day <= hcie_end))

    final_mask = (normal_mask | overnight_mask) & valid_mask

    return all_times[final_mask]


# %% [markdown]
# #### Tramos Dataset

# %%
def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second

def generate_timeframes_vectorized_per_row(row):
    if pd.isna(row['FINI']) or pd.isna(row['FFIN']):
        return pd.DatetimeIndex([])

    # Generate all 15-min timestamps for this row
    all_times = pd.date_range(start=row['FINI'], end=row['FFIN'], freq='15T')
    if all_times.empty:
        return pd.DatetimeIndex([])

    weekdays = all_times.weekday
    times_of_day = np.array([time_to_seconds(t) for t in all_times.time])

    # Build schedule arrays for all weekdays
    schedule = row['parsed_schedule']
    start_seconds = np.array([time_to_seconds(schedule.get(d, (None, None))[0]) if schedule.get(d, (None, None))[0] else None for d in weekdays])
    end_seconds = np.array([time_to_seconds(schedule.get(d, (None, None))[1]) if schedule.get(d, (None, None))[1] else None for d in weekdays])

    valid_mask = ~pd.isna(start_seconds)

    # Replace None with dummy values for comparison
    start_seconds = np.where(valid_mask, start_seconds, 0)
    end_seconds = np.where(valid_mask, end_seconds, 0)

    normal_mask = (start_seconds <= end_seconds) & (times_of_day >= start_seconds) & (times_of_day <= end_seconds)
    overnight_mask = (start_seconds > end_seconds) & ((times_of_day >= start_seconds) | (times_of_day <= end_seconds))

    final_mask = valid_mask & (normal_mask | overnight_mask)

    return all_times[final_mask]


# %%
def generate_monthly_splits(df):
    # Step 1: build list of month start dates per row
    def get_month_starts(row):
        start_month = row['FINI'].replace(day=1, hour=0, minute=0, second=0)
        end_month = row['FFIN'].replace(day=1, hour=0, minute=0, second=0)
        return pd.date_range(start=start_month, end=end_month, freq='MS').to_list()

    df['month_starts'] = df.progress_apply(get_month_starts, axis=1)

    # Step 2: explode
    df_exploded = df.explode('month_starts').reset_index(drop=True)

    # Step 3: calculate chunk_start
    df_exploded['chunk_start'] = df_exploded['month_starts']
    df_exploded.loc[df_exploded['chunk_start'] < df_exploded['FINI'], 'chunk_start'] = df_exploded['FINI']

    # Step 4: calculate chunk_end
    df_exploded['chunk_end'] = df_exploded['month_starts'] + pd.offsets.MonthEnd(1)
    df_exploded['chunk_end'] = df_exploded[['chunk_end', 'FFIN']].min(axis=1)
    df_exploded = df_exploded.drop(columns=['month_starts','FINI','FFIN'])
    df_exploded = df_exploded.rename({'chunk_start':'FINI','chunk_end':'FFIN'},axis=1)
    return df_exploded


# %%
coi_tramo = gpd.read_file('../data/9_pmts/COI_TRAMO_202504071919.geojson')
coi_tramo = coi_tramo.to_crs("EPSG:4326")
coi_tramo['id_coi_tramo'] = coi_tramo.index
coi_tramo_intersections = gpd.overlay(coi_tramo, mvi_buffered_buffer[['id_mvi', 'mvi_length', 'geometry']], how='intersection')
coi_tramo_intersections = coi_tramo_intersections.to_crs("EPSG:3116").copy()
coi_tramo_intersections['intersection_length'] = coi_tramo_intersections.geometry.length
coi_tramo_intersections['overlap_ratio'] = coi_tramo_intersections['intersection_length'] / coi_tramo_intersections['mvi_length']
coi_tramo_intersections = coi_tramo_intersections[coi_tramo_intersections['overlap_ratio'] > 0.15]
coi_tramo_intersections = coi_tramo_intersections.to_crs("EPSG:4326").copy()
coi_tramo_intersections['road_id'] = coi_tramo_intersections['id_mvi']
coi_tramo_intersections['coi_tramo'] = 1
coi_tramo_work = coi_tramo_intersections[['id_coi_tramo','road_id','FINI', 'FFIN','HCIE','coi_tramo']].copy()

# %%
coi_tramo_work = generate_monthly_splits(coi_tramo_work)

# %%
min_date = coi_tramo_work['FINI'].min().normalize()  # normalized to 00:00
max_date = coi_tramo_work['FFIN'].max().normalize()

# Add 1 month buffer on each side
start_date = (min_date).replace(day=1)
end_date = (max_date + pd.DateOffset(months=1)).replace(day=1)

# Generate full list of months
monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # MS = Month Start

# If you just want plain dates (not Timestamps)
monthly_dates_list = [d.date() for d in monthly_dates]

# %%
monthly_dates_list = [
    pd.Timestamp(d, tz='UTC') for d in monthly_dates_list
]

# %%
for i in tqdm(range(len(monthly_dates_list)-1)):
    date_file = dt.datetime.strftime(monthly_dates_list[i],'%Y%m%d')
    date_file_end = dt.datetime.strftime(monthly_dates_list[i+1],'%Y%m%d')
    mask = (coi_tramo_work['FFIN'] >= monthly_dates_list[i]) & (coi_tramo_work['FFIN'] < monthly_dates_list[i+1])
    coi_tramo_work_current = coi_tramo_work.loc[mask].copy()
    coi_tramo_work_current['parsed_schedule'] = coi_tramo_work_current['HCIE'].apply(parse_schedule_row)
    coi_tramo_work_current['time_slots'] = coi_tramo_work_current.progress_apply(generate_timeframes_vectorized_per_row, axis=1)
    coi_tramo_work_current = coi_tramo_work_current.explode('time_slots').reset_index(drop=True)
    coi_tramo_work_current['timeframe'] = coi_tramo_work_current['time_slots'].dt.strftime('%Y-%m-%d %H:%M:%S')
    coi_tramo_dataset = coi_tramo_work_current.groupby(['road_id','timeframe']).aggregate({'coi_tramo':'max'}).reset_index()
    coi_tramo_dataset.to_parquet(f'{file_output_name}coi_tramos/coi_tramo_dataset_{date_file}_{date_file_end}.parquet', index=False)

# %%

# %% [markdown]
# #### COOS Puntos

# %%
coos_punto = gpd.read_file('../data/9_pmts/COOS_PUNTO_202504071922.geojson')
coos_punto = coos_punto.to_crs("EPSG:4326")
coos_punto['id_coos_puntos'] = coos_punto.index
coos_punto_intersections = gpd.overlay(coos_punto, mvi_buffered_buffer[['id_mvi', 'mvi_length', 'geometry']], how='intersection')
coos_punto_intersections = coos_punto_intersections.to_crs("EPSG:4326").copy()
coos_punto_intersections['road_id'] = coos_punto_intersections['id_mvi']
coos_punto_intersections['coos_punto'] = 1

# %%
coos_punto_work = coos_punto_intersections[['id_coos_puntos','road_id','FINI', 'FFIN','HCIE','coos_punto']].copy()
coos_punto_work['parsed_schedule'] = coos_punto_work['HCIE'].apply(parse_schedule_row)
coos_punto_work['time_slots'] = coos_punto_work.progress_apply(generate_timeframes_vectorized_fully, axis=1)
coos_punto_work = coos_punto_work.explode('time_slots').reset_index(drop=True)
coos_punto_work['timeframe'] = coos_punto_work['time_slots'].dt.strftime('%Y-%m-%d %H:%M:%S')
coos_punto_dataset = coos_punto_work.groupby(['road_id','timeframe']).aggregate({'coos_punto':'max'}).reset_index()
coos_punto_dataset.to_parquet(f'{file_output_name}coos_punto_dataset.parquet', index=False)
