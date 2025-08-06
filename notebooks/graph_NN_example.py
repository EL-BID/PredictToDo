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
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool
from spektral.data import DisjointLoader, Dataset
from spektral.datasets import TUDataset

# from torch_geometric.datasets import TUDataset
from collections import Counter

# from spektral.utils import to_networkx
import networkx as nx
from networkx.convert_matrix import from_scipy_sparse_matrix

# %%

from torch_geometric.utils import to_networkx

# %%
dataset = TUDataset("PROTEINS")

# %%
# 2) Grab the first graph
g = dataset[0]

# %%
# 3) Inspect its raw tensors
print("• g.x (node features) shape    :", g.x.shape)  # (N, F)
print(
    "• g.a (adjacency) shape       :",
    g.a.shape,  # (N, N)
    f"nonzero entries = {g.a.nnz}",
)
print(
    "• g.e (edge features) shape   :", g.e.shape if g.e is not None else None
)  # (E, D) or None
print("• g.y (labels) shape          :", g.y.shape)  # (N,) or (1,)


# %%
A = g.a.tocoo()

# %%
N = A.shape[0]
G_nx = nx.Graph()
G_nx.add_nodes_from(range(N))

# %%
edges = [(i, j) for i, j in zip(A.row, A.col) if i < j]
G_nx.add_edges_from(edges)

# %%
n_nodes = G_nx.number_of_nodes()
n_edges = G_nx.number_of_edges()
print(f"Graph has {n_nodes} nodes and {n_edges} edges.")

# Peek at a few nodes and edges
print("First 10 nodes:", list(G_nx.nodes())[:10])
print("First 10 edges:", list(G_nx.edges())[:10])

# Degree information (how many connections each node has)
degrees = dict(G_nx.degree())
print("Sample degree distribution:", {k: degrees[k] for k in list(degrees)[:10]})

# %%
import matplotlib.pyplot as plt

pos = nx.spring_layout(G_nx, seed=42)
plt.figure(figsize=(6, 6))
nx.draw(
    G_nx, pos, node_size=20, node_color="skyblue", edge_color="gray", with_labels=False
)
plt.title("Graph topology (spring layout)")
plt.show()

# %%
print("Number of graphs:", len(dataset))
print("Number of classes:", dataset.num_classes)

# %%
loader = DisjointLoader(dataset, batch_size=32, epochs=10)

# %%
loader.load()

# %%
data = dataset[0]
print(data)

# %%
# 3. Print tensor shapes
print("Node features x shape:", data.x.shape)  # (num_nodes, num_node_features)
print("Edge index shape:", data.edge_index.shape)  # (2, num_edges)
print(
    "Edge attributes shape:",
    data.edge_attr.shape if data.edge_attr is not None else "None",
)

# %%
G = to_networkx(data, to_undirected=True)
pos = nx.spring_layout(G, seed=42)

# %%
