# 0824_saving

This folder contains synthetic data generation for variable-size detector node sets and several model variants for set-to-graph/centre prediction tasks.

## Data generation
- **File**: `synthetic_data_dynamic_nodes.py`
  - Generates annular stacked-detector node geometries and multi-cluster Gaussian energy depositions.
  - Saves datasets per total-points configuration as `synthetic_detector_data_{TP}pts.npz` under split directories (train/val/test), with keys such as:
    - `X_all_mod` [N, T, 8]: per-node features `[node_distance_sum, node_distance_mean, sum_energy, mean_energy, std_energy, x, y, z]`.
    - `y_all` [N, Kmax, 9]: per-event cluster summaries `[center(3), upper-tri covariance(6)]` (padded to max k).
    - `k_all` [N]: number of clusters per event.
    - `active_flags` [N, T]: binary node activity mask.
    - `node_centres` [N, T, 3]: per-node ground-truth assigned cluster center.
    - `per_node_inv_cov_upper` [N, T, 6]: per-node inverse covariance (upper triangular) labels.
    - Optional adjacency chunks saved and combined to `adj_matrix_{...}_{TP}pts.npy`.
  - Entrypoint `generate_dataset(config, dataset_name, save_dir)`; when run as a script, it produces train/val/test splits per the configs in the file.

## Models
- **`offset_transformer.py`**
  - A GPT-style encoder over per-node features with 3D positional MLP embedding and residual fusion.
  - Predicts per-node outputs:
    - centers: absolute center coordinates via residual head (xyz + Δ).
    - k logits: event-level cluster count prediction.
    - node indicators: active/inactive probability per node.
    - per-node covariance: `inv_cov_upper` (6 values) per node.

- **`k_set_transformer.py`**
  - Set-transformer variant with learnable `k` query vectors and cross-attention over encoded nodes.
  - Outputs per-center predictions (for each of the k queries):
    - center coordinates (3).
    - covariance upper-tri (6).
  - Includes a head to predict the number of clusters `k`.

- **`sumformer_pairwisemlp/`**
  - Components to predict adjacency (set-to-graph) using Sumformer blocks and a pairwise MLP head.
  - Key files:
    - `sumformer.py`: Batched Sumformer blocks (permutation-equivariant MLP-based aggregation over sets).
    - `sumformer_assignment.py`: `SumformerAdjacencyModel` that embeds features + xyz, runs Sumformer, then scores all node pairs with an MLP to produce adjacency logits.
    - `run_dynamic_set2graph.py`: Data pipeline for dynamic node-count configs, loading `X_all_mod` and using saved/packed adjacency or on-the-fly adjacency from labels.
    - `torch_heterogeneous_batching/`: batching utilities and transformer components used by Sumformer.

## Notes
- All models expect the 8-dim per-node feature layout produced by the generator: `[node_distance_sum, node_distance_mean, sum_energy, mean_energy, std_energy, x, y, z]`.
- Data and adjacency chunks are saved per total-points configuration, enabling training across multiple node counts via multi-config loaders in the model scripts.
