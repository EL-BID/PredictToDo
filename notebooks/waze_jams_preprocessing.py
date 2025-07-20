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
jams_df = pd.read_parquet(
    "../data/4_velocidades_waze/jams/jams_tratados_malla_completa/jams_intersections_aggregated.parquet"
)

# %%
jams_df
