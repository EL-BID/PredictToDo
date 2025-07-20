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
import networkx as nx
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from shapely.geometry import Point, LineString, MultiLineString
import numpy as np
import os
import pickle
import json


# %%
def calcular_sentidos_cardinales(tipo, sentido):
    if sentido == "B":  # bidirectional
        return "bidirectional"
    if tipo in ["AC", "CL"]:  # Calles (East-West)
        return "left-right" if sentido == "TF" else "right-left"
    elif tipo in ["KR", "AK"]:  # Carreras (North-South)
        return "south-north" if sentido == "FT" else "north-south"
    return "bidirectional"


# %%
def endpoints(geom):
    # Build one flat list of coords
    if isinstance(geom, MultiLineString):
        coords = []
        for part in geom.geoms:
            coords.extend(part.coords)
    elif isinstance(geom, LineString):
        coords = list(geom.coords)
    else:
        raise ValueError(f"Unsupported geometry type: {type(geom)}")
    return Point(coords[0]), Point(coords[-1])

def flat_coords(geom):
    if geom.geom_type=='MultiLineString':
        c = []
        for part in geom.geoms: c.extend(part.coords)
        return c
    return geom.coords


# %%
def compute_legal_vec(df, tol=1e-6):
    """
    Vectorized mask: can we legally go from left→right?
    Expects cols:
      - cardinal_left, sx_left, sy_left, ex_left, ey_left
      - cardinal_right, sx_right, sy_right, ex_right, ey_right
    """
    cl   = df['cardinal_left']
    cr   = df['cardinal_right']
    sxl, syl = df['sx_left'], df['sy_left']
    exl, eyl = df['ex_left'], df['ey_left']
    sxr, syr = df['sx_right'], df['sy_right']
    exr, eyr = df['ex_right'], df['ey_right']

    # 1) Exit‐allowed on left segment
    exit_allowed = (
        (cl == 'bidirectional') |
        ((cl == 'left-right')    & np.isclose(exl, exl, atol=tol))        |  # trivial true
        ((cl == 'right-left')    & np.isclose(sxl, exl, atol=tol))        |
        ((cl == 'south-north')   & np.isclose(exl, exl, atol=tol))        |
        ((cl == 'north-south')   & np.isclose(sxl, exl, atol=tol))
    )

    # 2) Entry‐allowed on right segment
    entry_allowed = (
        (cr == 'bidirectional') |
        ((cr == 'left-right')    & np.isclose(sxr, exl, atol=tol) & np.isclose(syr, eyl, atol=tol)) |
        ((cr == 'right-left')    & np.isclose(exr, exl, atol=tol) & np.isclose(eyr, eyl, atol=tol)) |
        ((cr == 'south-north')   & np.isclose(sxr, exl, atol=tol) & np.isclose(syr, eyl, atol=tol)) |
        ((cr == 'north-south')   & np.isclose(exr, exl, atol=tol) & np.isclose(eyr, eyl, atol=tol))
    )

    # 3) Only legal if both exit and entry are allowed
    return exit_allowed & entry_allowed


# %%
buildings = pd.read_csv('../data/datos_procesados/buildings/roads_buildings.csv')
stops_df = pd.read_parquet('../data/datos_procesados/bus_stops/bus_stops_per_road.parquet')

# %%
buildings.columns =[f'buildings_{col}' if col !='road_id' else col for col in buildings.columns]
stops_df.columns =[f'bus_stop_{col}' if col !='road_id' else col for col in stops_df.columns]

# %%
road_df = gpd.read_file('../data/2_Estructura_via/estructura_via_completa/MVI.geojson')
road_df["id_mvi"] = road_df["MVICIV"].astype(str) + "_" + road_df["MVISVIA"].str.replace(" ", "")
road_df['contador'] = road_df.groupby('id_mvi').cumcount() + 1
road_df['id_mvi'] = road_df['id_mvi'] + '.' + road_df['contador'].astype(str)
road_df = (
    road_df.explode(ignore_index=True)          # if shapefile has split segments
       .dissolve(by='id_mvi')               # merge all segments of same road
       .reset_index()
)
road_df['road_id'] = road_df['id_mvi']

# %%
road_df = gpd.GeoDataFrame(pd.merge(road_df, buildings, on = 'road_id', how='left')).to_crs(epsg=3116)
road_df = gpd.GeoDataFrame(pd.merge(road_df, stops_df, on = 'road_id', how='left')).to_crs(epsg=3116)

# %%
road_df['cardinal'] = road_df.apply(
    lambda r: calcular_sentidos_cardinales(r['MVITIPO'], r['MVISVIA']),
    axis=1
)
road_df[['start_pt','end_pt']] = road_df.geometry.apply(
    lambda g: pd.Series([
        Point(flat_coords(g)[0]),
        Point(flat_coords(g)[-1])
    ])
)

# %%
road_df[['sx','sy']] = pd.DataFrame(
    [(p.x,p.y) for p in road_df.start_pt], index=road_df.index
)
road_df[['ex','ey']] = pd.DataFrame(
    [(p.x,p.y) for p in road_df.end_pt], index=road_df.index
)

# %%
road_df  = road_df.to_crs(epsg=3116)
file_path = '../data/10_uso_del_suelo/Zonificacion_2023_GDB/ZAT_2023.gdb'
zats = gpd.read_file(file_path, layer=0)
pop_land = pd.read_csv('../data/13_poblacion_ZAT/zat_population_land_use.csv', encoding = 'latin-1' )
zat_pop_land = gpd.GeoDataFrame(pd.merge(pop_land, zats, on = 'ZAT'))
zat_pop_land  = zat_pop_land.to_crs(epsg=3116)

# %%
road_zats_data = gpd.overlay(
    road_df[['road_id','geometry']],
    zat_pop_land,
    how='intersection'
)

road_zats_data['seg_len'] = road_zats_data.geometry.length

# 4) Aggregate per road & LU class
agg = (
    road_zats_data
      .groupby(['road_id','ZAT'])['seg_len']
      .sum()
      .reset_index()
)

total = agg.groupby('road_id')['seg_len'].sum().rename('total_len')
agg = agg.join(total, on='road_id')
agg['pct'] = agg['seg_len'] / agg['total_len']

dominant = agg.loc[agg.groupby('road_id')['seg_len'].idxmax()]
dominant_reformed = pd.merge(dominant,dominant.groupby(['ZAT']).aggregate({'total_len':'sum'}).reset_index(), on = 'ZAT', how='left', suffixes = ('','_zat'))
dominant_reformed['zat_perc'] = dominant_reformed.total_len/dominant_reformed.total_len_zat
cols = ['land_use_km2_comercial', 'land_use_km2_industrial',
       'land_use_km2_equipamientos', 'land_use_km2_educacion',
       'land_use_km2_comser', 'land_use_km2_recreacional',
       'land_use_km2_cuposedu', 'land_use_km2_residencial',
       'population_5_24 años', 'population_25_64 años',
       'population_more_65 años', 'population_Total']
cols_work = ['land_use_km2_comercial', 'land_use_km2_industrial','land_use_km2_equipamientos', 'land_use_km2_educacion','land_use_km2_comser', 'land_use_km2_recreacional','land_use_km2_cuposedu', 'land_use_km2_residencial','population_5_24 años', 'population_25_64 años','population_more_65 años', 'population_Total']
id_road_pop_land  = pd.merge(dominant_reformed[['road_id','ZAT','zat_perc']], zat_pop_land[['ZAT',*cols_work]], on = 'ZAT', how = 'left')
for col in cols_work:
    id_road_pop_land[col] = id_road_pop_land[col]*id_road_pop_land.zat_perc
zat_road_vars_dict = {item['road_id']:{'land_use_km2_comercial':item['land_use_km2_comercial'],  'land_use_km2_industrial':item['land_use_km2_industrial'],  'land_use_km2_equipamientos':item['land_use_km2_equipamientos'],  'land_use_km2_educacion':item['land_use_km2_educacion'],  'land_use_km2_comser':item['land_use_km2_comser'],  'land_use_km2_recreacional':item['land_use_km2_recreacional'],  'land_use_km2_cuposedu':item['land_use_km2_cuposedu'],  'land_use_km2_residencial':item['land_use_km2_residencial'],  'population_5_24 años':item['population_5_24 años'],  'population_25_64 años':item['population_25_64 años'],  'population_more_65 años':item['population_more_65 años'],  'population_Total':item['population_Total']} for item in id_road_pop_land.to_dict('records')}

# %%
# 2. Initialize the road-graph R with one node per road
R = nx.DiGraph()
R.add_nodes_from(road_df['road_id'])

# 2. Reproject to metric CRS (e.g. EPSG:3116 for Bogotá)
orig_crs    = road_df.crs
road_df  = road_df.to_crs(epsg=3116)
road_df = road_df.fillna('Null')
road_df['length_m'] = road_df.geometry.length
# 3. Bulk-attach all your static attributes from road_df to R (excluding geometry and indexing cols)
static_cols = [c for c in road_df.columns if c not in ('road_id','geometry','start_pt','end_pt')]
attr_dict_prev = {
    row.road_id: {col: row[col] for col in static_cols} for _, row in road_df.iterrows()
}
attr_dict = {}
for k,dict_to_updt in tqdm(attr_dict_prev.items()):
    dict_to_updt.update(zat_road_vars_dict.get(k,{
        'land_use_km2_comercial':0,
        'land_use_km2_industrial':0,
        'land_use_km2_equipamientos':0,
        'land_use_km2_educacion':0,
        'land_use_km2_comser':0,
        'land_use_km2_recreacional':0,
        'land_use_km2_cuposedu':0,
        'land_use_km2_residencial':0,
        'population_5_24 años':0,
        'population_25_64 años':0,
        'population_more_65 años':0,
        'population_Total':0,
    }))
    attr_dict[k] = dict_to_updt

nx.set_node_attributes(R, attr_dict)

# 4. Build edges by finding intersecting roads via vectorized spatial join with tolerance
tol = 1 # spatial tolerance


# %%
road_df_buf = road_df[['road_id','geometry']].copy()
road_df_buf['geometry'] = road_df_buf.geometry.buffer(tol)
pairs = gpd.sjoin(
    road_df_buf[['road_id','geometry']],
    road_df[['road_id','geometry']],
    how='inner',
    predicate='intersects'
)[['road_id_left','road_id_right']]

pairs = pairs[pairs.road_id_left != pairs.road_id_right]
pairs = pairs.sort_values(['road_id_left','road_id_right'])
pairs = pairs.drop_duplicates(subset=['road_id_left','road_id_right'])

# %%
pairs_data = pd.merge(pairs, road_df[['road_id','cardinal','sx','sy','ex','ey']], how = 'left', left_on = 'road_id_left' , right_on = 'road_id').drop(['road_id'],axis=1)
pairs_data = pd.merge(pairs_data, road_df[['road_id','cardinal','sx','sy','ex','ey']], how = 'left', left_on = 'road_id_right' , right_on = 'road_id', suffixes=('_left','_right')).drop(['road_id'],axis=1)
pairs_data['legal'] = compute_legal_vec(pairs_data)
pairs_data['influence'] = np.where(pairs_data['legal'],1,0.3)
pairs_data['legal'] = np.where(pairs_data['legal'],1,0)

# %%
R.add_edges_from(list(zip(pairs_data.loc[pairs_data.legal==1,'road_id_left'],pairs_data.loc[pairs_data.legal==1,'road_id_right'])), legal=1,  influence_weight=1.0)
R.add_edges_from(list(zip(pairs_data.loc[pairs_data.legal==0,'road_id_left'],pairs_data.loc[pairs_data.legal==0,'road_id_right'])),  legal=0, influence_weight=0.3)

print("Road‐nodes:",     R.number_of_nodes())
print("Intersection‐edges:", R.number_of_edges())

example = next(iter(R.nodes()))
print("Attrs for", example, "→", R.nodes[example])

# %% [markdown]
# Distribuir poblacion y uso de suelo zatr entrer num calles, incluir relacion tamanio zat con numero de segmentos de calle

# %%
file_output_name = "../data/datos_procesados/conectivity_graph/"
os.makedirs(file_output_name, exist_ok=True)
with open(f'{file_output_name}road_graph.pickle', 'wb') as f:
    pickle.dump(R, f, pickle.HIGHEST_PROTOCOL)

# %%
nx.write_graphml(R, f'{file_output_name}road_graph.graphml')

# %%
from networkx.readwrite import json_graph
json_graph = json_graph.node_link_data(R)
json.dump(json_graph, open(f'{file_output_name}road_graph.json','w'))


# %%
def get_roads_within_hops(R, seed_road, k):
    # returns dict {road: distance_in_hops}
    dists = nx.single_source_shortest_path_length(R, seed_road, cutoff=k)
    return dists

seed = str(sorted(road_df.id_mvi.unique())[200])
roads_2hops = get_roads_within_hops(R, seed, k=5)

# filter your GeoDataFrame
to_plot = roads_2hops  # e.g. ['CIV123', 'CIV456', ...]
# Filter the original GeoDataFrame
road_df_sel = road_df[road_df['road_id'].isin(to_plot)]

# Plot just those roads
ax = road_df_sel.plot(figsize=(8, 8),
                  linewidth=3,
                  edgecolor='red',
                  facecolor='none')

# (Optional) Highlight the seed road in a different color
road_df_sel[road_df_sel['road_id'] == seed]\
    .plot(ax=ax, linewidth=4, edgecolor='blue')

ax.set_axis_off()
ax.set_title(f"Roads touching {seed}")
plt.show()


# %%
n_nodes = R.number_of_nodes()
n_edges = R.number_of_edges()
print(f"Graph has {n_nodes} nodes and {n_edges} edges.")

# Peek at a few nodes and edges
print("First 10 nodes:", list(R.nodes())[:10])
print("First 10 edges:", list(R.edges())[:10])

# Degree information (how many connections each node has)
degrees = dict(R.degree())
print("Sample degree distribution:", {k: degrees[k] for k in list(degrees)[:10]})


# %%
# pos = nx.spring_layout(R, seed=42)
# plt.figure(figsize=(6,6))
# nx.draw(
#     R, pos,
#     node_size=20,
#     node_color='skyblue',
#     edge_color='gray',
#     with_labels=False
# )
# plt.title("Graph topology (spring layout)")
# plt.show()

# %%

def get_roads_within_distance(R, seed_road, max_distance_m):
    """
    Returns all roads reachable from seed_road within max_distance_m meters.

    R                : networkx.Graph where nodes have attribute 'length_m'
    seed_road        : node ID (road_id) from which to start
    max_distance_m   : maximum cumulative distance in meters

    Returns:
      distances : dict {road_id: cumulative_distance}
      df_edges  : pandas.DataFrame with edges between reachable roads and their attributes
    """
    # 1. Ensure each edge has a 'weight' attribute: avg of node lengths
    for u, v, data in R.edges(data=True):
        if 'weight' not in data:
            data['weight'] = (R.nodes[u]['length_m'] + R.nodes[v]['length_m']) / 2.0

    # 2. Run Dijkstra up to cutoff
    distances = nx.single_source_dijkstra_path_length(
        R, seed_road, cutoff=max_distance_m, weight='weight'
    )
    reachable = set(distances.keys())

    # 3. Collect edges among reachable nodes
    rows = []
    for u, v, attrs in R.subgraph(reachable).edges(data=True):
        row = {'u': u, 'v': v, **attrs}
        rows.append(row)
    df_edges = pd.DataFrame(rows)

    return distances


# %%
seed = str(sorted(road_df.id_mvi.unique())[4003])
max_dist = 1000  # meters
distances = get_roads_within_distance(R, seed, max_dist)
print("Reached roads:", distances.keys())


to_plot = list(distances.keys())
# Filter the original GeoDataFrame
road_df_sel = road_df[road_df['road_id'].isin(to_plot)]

# Plot just those roads
ax = road_df_sel.plot(figsize=(8, 8),
                  linewidth=3,
                  edgecolor='red',
                  facecolor='none')

# (Optional) Highlight the seed road in a different color
road_df_sel[road_df_sel['road_id'] == seed]\
    .plot(ax=ax, linewidth=4, edgecolor='blue')

ax.set_axis_off()
ax.set_title(f"Roads touching {seed}")
plt.show()

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# file_path = '9_pmts/COI_PARQUES_202504071925.geojson'
# pmts_parques = gpd.read_file(file_path)

# %%
# file_path = '9_pmts/COI_PUNTO_202504071914.geojson'
# pmts_punto= gpd.read_file(file_path)

# %%
# file_path = '10_uso_del_suelo/Zonificacion_2023_GDB/ZAT_2023.gdb'
# zats = gpd.read_file(file_path, layer=0)
# pop_land = pd.read_csv('zat_population_land_use.csv', encoding = 'latin-1' )
# zat_pop_land = gpd.GeoDataFrame(pd.merge(pop_land, zats, on = 'ZAT'))
# zat_pop_land  = zat_pop_land.to_crs(epsg=3116)

# %%
# X[N,F,T]
# A[N,N]-------
# E[N*N, F2] -------------

# legal = 0,1
# influence = 1- 0.3
