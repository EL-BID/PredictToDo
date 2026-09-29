"""Extrae una muestra conexa de los recortes locales de ciudad; no modifica fuentes."""
import argparse
import json
import pickle
from itertools import islice
from pathlib import Path
import h5py
import networkx as nx
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path, help="Carpeta con los HDF5 de ciudad ya recortados a los 15 días y graph.pkl")
parser.add_argument("output", type=Path, help="Carpeta nueva para la muestra")
args = parser.parse_args()
source, out = args.source, args.output
seed = '17000085_N.1'  # CL 12A, nodo más cercano al punto (4.600, -74.075).
with (source / 'graph.pkl').open('rb') as f:
    graph = pickle.load(f)
nodes = [seed] + [v for _, v in islice(nx.bfs_edges(graph, seed), 4999)]
assert len(nodes) == 5000
subgraph = nx.DiGraph()
subgraph.graph.update(graph.graph)
subgraph.add_nodes_from((n, graph.nodes[n]) for n in nodes)
subgraph.add_edges_from(graph.subgraph(nodes).edges(data=True))
assert nx.is_strongly_connected(subgraph)
with (source / 'jams_dataset_metadata.pkl').open('rb') as f:
    base = pickle.load(f)
expected_times = {t: i for i, t in enumerate(pd.date_range("2024-10-01", periods=1440, freq="15min"))}
assert base['time_idx_map'] == expected_times, "Se requieren los 15 días ya recortados y reindexados"
node_map = {n: i for i, n in enumerate(nodes)}
lookup = np.full(len(base['node_idx_map']), -1, dtype=np.int64)
for n, i in node_map.items():
    lookup[base['node_idx_map'][n]] = i
out.mkdir(exist_ok=False)
with (out / 'graph.pkl').open('wb') as f:
    pickle.dump(subgraph, f, protocol=pickle.HIGHEST_PROTOCOL)
report = {'seed': seed, 'nodes': len(nodes), 'edges': subgraph.number_of_edges(),
          'strongly_connected': True, 'start': '2024-10-01 00:00:00',
          'end_exclusive': '2024-10-16 00:00:00', 'datasets': {}}
for path in sorted(source.glob('*.h5')):
    meta_name = path.stem + '_metadata.pkl'
    with (source / meta_name).open('rb') as f:
        meta = pickle.load(f)
    assert meta['node_idx_map'] == base['node_idx_map']
    temporal = 'time_idx_map' in meta
    if temporal:
        assert meta['time_idx_map'] == base['time_idx_map']
    coverage = np.zeros(1440, dtype=np.int64)
    previous = -1
    with h5py.File(path, 'r') as src, h5py.File(out / path.name, 'w') as dst:
        data = src['data']
        result = dst.create_dataset('data', shape=(0, data.shape[1]),
            maxshape=(None, data.shape[1]), dtype=data.dtype,
            chunks=(8192, data.shape[1]), compression='gzip', compression_opts=4, shuffle=True)
        for offset in range(0, len(data), 262144):
            block = data[offset:offset + 262144]
            mapped = lookup[block[:, 0].astype(np.int64)]
            keep = mapped >= 0
            block = block[keep]
            block[:, 0] = mapped[keep]
            assert np.isfinite(block).all()
            if temporal and len(block):
                times = block[:, 1].astype(np.int64)
                assert times.min() >= 0 and times.max() < 1440
                assert times[0] >= previous and np.all(np.diff(times) >= 0)
                previous = times[-1]
                coverage += np.bincount(times, minlength=1440)
            count = len(result)
            result.resize(count + len(block), axis=0)
            result[count:] = block
        record = {'rows': len(result)}
    meta['node_idx_map'] = node_map
    with (out / meta_name).open('wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    record['h5_bytes'] = (out / path.name).stat().st_size
    record['metadata_bytes'] = (out / meta_name).stat().st_size
    if temporal:
        record['populated_intervals'] = int((coverage > 0).sum())
    report['datasets'][path.name] = record
    print(path.name, record, flush=True)
report['package_bytes'] = sum(p.stat().st_size for p in out.iterdir())
(out.parent / (out.name + '_report.json')).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
