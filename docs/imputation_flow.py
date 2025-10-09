"""
Visual representation of the imputation logic flow.
"""

FLOW_DIAGRAM = """
┌─────────────────────────────────────────────────────────────────────────┐
│                      TrafficDataset Initialization                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌────────────────────┐         ┌────────────────────────┐
        │  Load Graph &      │         │  Load Neighbor         │
        │  Static Features   │         │  Features (parquet)    │
        └────────────────────┘         └────────────────────────┘
                    │                               │
                    │                               ▼
                    │              ┌─────────────────────────────────┐
                    │              │  _identify_imputation_features  │
                    │              │  Scan metadata files to find:   │
                    │              │  - 'trend' → index 72           │
                    │              │  - 'speed' → index 70           │
                    │              │  - 'regularSpeed' → index 67    │
                    │              └─────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Dataset Ready for __getitem__                        │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                     __getitem__(idx) - Data Loading                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌────────────────────┐         ┌────────────────────────┐
        │  Load Static       │         │  Load Dynamic Past     │
        │  Features for      │         │  Features from HDF5    │
        │  Node Batch        │         │  (sparse → dense)      │
        └────────────────────┘         └────────────────────────┘
                                                    │
                                    ┌───────────────┴───────────────┐
                                    │                               │
                                    ▼                               ▼
                        ┌────────────────────┐         ┌────────────────────┐
                        │  Populate dense    │         │  Check if current  │
                        │  tensor with       │         │  HDF5 file has     │
                        │  actual values     │         │  imputation        │
                        │  from COO data     │         │  features          │
                        └────────────────────┘         └────────────────────┘
                                                                    │
                                                                    ▼
                                                        ┌────────────────────┐
                                                        │  YES: Apply        │
                                                        │  Imputation        │
                                                        └────────────────────┘
                                                                    │
                            ┌───────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          IMPUTATION LOGIC                                │
│                                                                           │
│  For each imputation feature in current file:                            │
│    For each node in batch:                                               │
│      1. Get node_id from node_idx                                        │
│      2. Look up imputation value in neighbor_features DataFrame          │
│         - trend → mean_trend                                             │
│         - speed → mean_regularSpeed                                      │
│         - regularSpeed → mean_regularSpeed                               │
│      3. Find all timesteps where feature == 0 (missing)                  │
│      4. Fill those timesteps with imputation value                       │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Return PyG Data Object                              │
│  - x_static: [num_nodes, num_static_features]                           │
│  - x_dynamic_past: [num_nodes, in_timesteps, num_dynamic_features]      │
│    ↑ WITH IMPUTED VALUES FOR trend, speed, regularSpeed                 │
│  - x_dynamic_future: [num_nodes, out_timesteps, num_future_features]    │
│  - y: [num_nodes, out_timesteps]                                        │
│  - edge_index: [2, num_edges]                                           │
└─────────────────────────────────────────────────────────────────────────┘
"""

EXAMPLE_IMPUTATION = """
Example: Node "0_B.1" has these neighbor features:
    mean_trend = -0.0356
    mean_regularSpeed = 23.48

Before imputation (x_dynamic_past for node "0_B.1"):
    timestep 0: trend=0.0,    speed=0.0,     regularSpeed=0.0
    timestep 1: trend=0.5,    speed=25.0,    regularSpeed=30.0  ← Has actual data
    timestep 2: trend=0.0,    speed=0.0,     regularSpeed=0.0
    timestep 3: trend=0.0,    speed=0.0,     regularSpeed=0.0

After imputation:
    timestep 0: trend=-0.0356, speed=23.48,  regularSpeed=23.48  ← IMPUTED
    timestep 1: trend=0.5,     speed=25.0,   regularSpeed=30.0   ← Unchanged (had data)
    timestep 2: trend=-0.0356, speed=23.48,  regularSpeed=23.48  ← IMPUTED
    timestep 3: trend=-0.0356, speed=23.48,  regularSpeed=23.48  ← IMPUTED
"""

if __name__ == "__main__":
    print(FLOW_DIAGRAM)
    print("\n" + "=" * 80 + "\n")
    print(EXAMPLE_IMPUTATION)
