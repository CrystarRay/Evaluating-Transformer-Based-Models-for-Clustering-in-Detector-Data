import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import numpy as np
# If you only need the differentiable Sinkhorn distance, POT / SciPy are no longer required
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os
# For optional k-means visualisation
from sklearn.cluster import KMeans
import itertools
import argparse


# Debug mode - set to True to run on only 100 samples
DEBUG_MODE = False
DEBUG_SAMPLES = 10

# ===========================
# Dataset Definition
# ===========================
class SyntheticSet2GraphDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        # Per-node features: [node_distance_sum, node_distance_mean, sum_energy, mean_energy, std_energy, x, y, z] ⇒ 8 dims
        raw = data["X_all_mod"].astype(np.float32)
        # Use all 8 features
        self.input_data = raw
        
        # ground-truth centres for every node (N,E,3)
        # saved as key 'node_centres' in the NPZ file produced by synthetic_data_generation.py
        self.target_centres = data["node_centres"].astype(np.float32)
        # active flags (N,E) where 1 = active, 0 = inactive
        self.active_flags = data["active_flags"].astype(np.float32)
        # Store per-event k value (number of clusters) for stratified splits
        self.k_values = data["k_all"].astype(np.int64)
        # NEW: Store per-node covariance labels
        self.inv_cov_upper = data["per_node_inv_cov_upper"].astype(np.float32)  # [N, E, 6]

    def __len__(self):
        if DEBUG_MODE:
            return min(DEBUG_SAMPLES, len(self.input_data))
        return len(self.input_data)

    def __getitem__(self, idx):
        if DEBUG_MODE:
            idx = idx % min(DEBUG_SAMPLES, len(self.input_data))
        input_tensor = torch.tensor(self.input_data[idx], dtype=torch.float32)
        target_centres = torch.tensor(self.target_centres[idx], dtype=torch.float32)
        active_flags = torch.tensor(self.active_flags[idx], dtype=torch.float32)
        k_value = torch.tensor(self.k_values[idx], dtype=torch.long)  # k value for this event
        inv_cov_upper = torch.tensor(self.inv_cov_upper[idx], dtype=torch.float32)  # per-node covariance for this event [E, 6]
        return input_tensor, target_centres, active_flags, k_value, inv_cov_upper

# ===========================
# Dynamic datasets & loader
# ===========================
class DynamicCentresDataset(Dataset):
    """Dataset wrapper for a single total_points configuration without padding.

    Expects NPZ with keys created by synthetic_data_dynamic_nodes.py:
      - X_all_mod: [N, T, 8]
      - node_centres: [N, T, 3]
      - active_flags: [N, T]
      - k_all: [N]
      - per_node_inv_cov_upper: [N, T, 6]
    """
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        if 'X_all_mod' in data:
            self.X_all = data['X_all_mod'].astype(np.float32)
        elif 'X_all' in data:
            self.X_all = data['X_all'].astype(np.float32)
        else:
            raise KeyError('Neither X_all_mod nor X_all found in dataset.')

        self.node_centres = data['node_centres'].astype(np.float32)
        self.active_flags = data['active_flags'].astype(np.float32)
        self.k_values = data['k_all'].astype(np.int64)
        self.inv_cov_upper = data['per_node_inv_cov_upper'].astype(np.float32)

        # Convenience attributes used by the multi-config loader
        self.num_events = self.X_all.shape[0]
        self.num_nodes = self.X_all.shape[1]
        self.total_points = self.num_nodes

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        x = torch.tensor(self.X_all[idx], dtype=torch.float32)
        centres = torch.tensor(self.node_centres[idx], dtype=torch.float32)
        flags = torch.tensor(self.active_flags[idx], dtype=torch.float32)
        kval = torch.tensor(self.k_values[idx], dtype=torch.long)
        inv_cov = torch.tensor(self.inv_cov_upper[idx], dtype=torch.float32)
        return x, centres, flags, kval, inv_cov

class MultiConfigDataLoader:
    """Iterates multiple DynamicCentresDataset objects without padding.
    
    Each __next__ returns a dict: { total_points: (inputs, centres, flags, k, inv_cov) }
    where every batch inside the dict has a uniform number of nodes.
    """
    def __init__(self, datasets, batch_size, shuffle=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.loaders = {}
        for dataset in datasets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            self.loaders[dataset.total_points] = loader
        
        # total iterations = max number of batches across configs
        self.total_iterations = 0
        for dataset in datasets:
            # Subset doesn't expose __len__ as attribute; len() works for both
            ds_len = len(dataset)
            num_batches = ds_len // batch_size
            if ds_len % batch_size != 0:
                num_batches += 1
            self.total_iterations = max(self.total_iterations, num_batches)
        
        self.iterators = {}
        self.current_iteration = 0
        self.reset_iterators()
    
    def reset_iterators(self):
        for total_points, loader in self.loaders.items():
            self.iterators[total_points] = iter(loader)
        self.current_iteration = 0
    
    def __iter__(self):
        self.reset_iterators()
        return self
    
    def __next__(self):
        if self.current_iteration >= self.total_iterations:
            raise StopIteration
        batches = {}
        for total_points, iterator in self.iterators.items():
            try:
                batch = next(iterator)
                batches[total_points] = batch
            except StopIteration:
                continue
        self.current_iteration += 1
        if not batches:
            raise StopIteration
        return batches

def load_dynamic_datasets(data_dir: str, split: str, total_points_list: list):
    datasets = []
    split_dir = os.path.join(data_dir, split)
    for total_points in total_points_list:
        npz_path = os.path.join(split_dir, f'synthetic_detector_data_{total_points}pts.npz')
        if os.path.exists(npz_path):
            ds = DynamicCentresDataset(npz_path)
            # Respect global DEBUG_MODE by capping samples per config
            if DEBUG_MODE:
                from torch.utils.data import Subset
                limit = min(DEBUG_SAMPLES, len(ds))
                ds_dbg = Subset(ds, list(range(limit)))
                # Preserve attributes used by the loader
                ds_dbg.total_points = ds.total_points
                ds_dbg.num_nodes = ds.num_nodes
                # Also expose k_values for downstream stats if needed
                ds_dbg.k_values = ds.k_values[:limit]
                datasets.append(ds_dbg)
                print(f"Loaded {split} dataset {total_points}pts: {limit}/{len(ds)} events (DEBUG), {ds.num_nodes} nodes")
            else:
                datasets.append(ds)
                print(f"Loaded {split} dataset {total_points}pts: {len(ds)} events, {ds.num_nodes} nodes")
        else:
            print(f"Warning: missing {npz_path}, skipping")
    return datasets

# ============================
# Stratified split helper
# ============================

def stratified_split_indices(labels: np.ndarray, train_ratio=0.7, val_ratio=0.15, seed: int = 42):
    """Return train/val/test indices with identical per-class (k) proportions.

    labels: 1-D array of k values (int) for each event.
    """
    rng = np.random.RandomState(seed)
    unique_k = np.unique(labels)
    train_idx, val_idx, test_idx = [], [], []

    for k_val in unique_k:
        idx_k = np.where(labels == k_val)[0]
        rng.shuffle(idx_k)
        n_k = len(idx_k)
        n_train = int(train_ratio * n_k)
        n_val = int(val_ratio * n_k)
        train_idx.extend(idx_k[:n_train])
        val_idx.extend(idx_k[n_train:n_train + n_val])
        test_idx.extend(idx_k[n_train + n_val:])

    return train_idx, val_idx, test_idx


# ============================
# 3D Positional Embedding MLP
# ============================
class PositionalEmbedding3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, pos_xyz):
        # pos_xyz: [B, T, 3]
        return self.mlp(pos_xyz)  # [B, T, dim]

# ===========================
# NanoGPT Encoder Components
# ===========================
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + 1e-5)
        if self.bias is not None:
            return self.weight * x + self.bias
        else:
            return self.weight * x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(attn_output)
        return self.dropout(out)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTConfig:
    def __init__(self, block_size, n_layer, n_head, n_embd, dropout):
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

# --------------------------------------------------
# New Offset head with residual connection
# --------------------------------------------------
class OffsetHead(nn.Module):
    """Predict per-node offset Δ so that centre = xyz + Δ.

    Uses two linear layers with a residual skip for stability.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_dim, 3)

    def forward(self, h):
        # h: (..., C)
        z = self.act(self.fc1(h))
        h = h + z                  # residual skip
        offset = self.fc2(h)       # (..., 3)
        return offset

class GPTEncoderModel(nn.Module):
    def __init__(self, config, input_dim, max_k=10):
        """input_dim is the number of *non-geometric* feature channels.

        The last three columns of the incoming tensor are xyz; model outputs
        centre = xyz + Δ where Δ is a learned offset.
        
        max_k: maximum number of clusters to predict (k will be in range [1, max_k])
        """
        super().__init__()
        self.config = config
        self.max_k = max_k
        # input_dim should be 5 for the non-geometric features (first 5 features)
        self.input_proj = nn.Linear(5, config.n_embd)  # Fixed: 5 features, not 8
        self.xyz_affine = nn.Linear(3, 3, bias=True)
        # initialise affine as identity
        with torch.no_grad():
            self.xyz_affine.weight.copy_(torch.eye(3))
            self.xyz_affine.bias.zero_()

        self.pos_emb_mlp = PositionalEmbedding3D(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)

        # Fusion layer to combine energy features and positional embeddings
        self.feature_fusion = nn.Linear(config.n_embd + config.n_embd, config.n_embd)

        # Center prediction head - predict absolute centers
        self.center_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd, 3)
        )
        
        # Energy attention head - learn which nodes are important
        self.energy_attention = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Linear(config.n_embd // 2, 1),
            nn.Sigmoid()
        )
        
        # NEW: Node indicator head - predict active/inactive nodes
        self.node_indicator_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd // 2, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )
        
        # K-prediction head - predict number of clusters
        self.k_prediction_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd // 2, max_k)  # Output logits for k=1 to max_k
        )
        
        # NEW: Per-node covariance prediction head - predict inv_cov_upper (6 values) for each node
        self.covariance_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.n_embd // 2, 6)  # Output 6 values for inv_cov_upper per node
        )

    def forward(self, x):
        # Process all 8 features together: [node_distance_sum, node_distance_mean, sum_energy, mean_energy, std_energy, x, y, z]
        # Extract xyz coordinates for positional embedding and residual connection
        pos_xyz = x[..., 5:8]  # xyz columns (indices 5,6,7)
        feats = x[..., :5]     # first 5 features (indices 0,1,2,3,4)
        
        # Project features to embedding dimension
        energy_emb = self.input_proj(feats)
        
        # Create positional embedding from xyz coordinates
        pos_emb = self.pos_emb_mlp(pos_xyz)
        
        # Concatenate feature embeddings and positional embeddings, then fuse
        combined = torch.cat([energy_emb, pos_emb], dim=-1)  # [B, T, 2*n_embd]
        x = self.feature_fusion(combined)  # [B, T, n_embd]
        
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        """
        # Energy attention
        energy_weights = self.energy_attention(x)  # [B, T]
        x = x * energy_weights  # [B, T, n_embd]
        """
        # Predict absolute center coordinates
        centres_pred = self.center_head(x)  # (B,T,3)
        centres_pred = centres_pred + pos_xyz  # Add residual jump with xyz

        # NEW: Predict node indicators (active/inactive)
        node_indicators = self.node_indicator_head(x)  # [B, T, 1]
        node_indicators = node_indicators.squeeze(-1)  # [B, T]
        
        # Predict k (number of clusters) - use global average pooling
        # Global average pooling over nodes to get event-level representation
        x_global = x.mean(dim=1)  # [B, n_embd] - average over all nodes
        k_logits = self.k_prediction_head(x_global)  # [B, max_k]
        
        # NEW: Predict per-node covariance (inv_cov_upper) - use node-level features
        inv_cov_upper_pred = self.covariance_head(x)  # [B, T, 6] - per-node predictions
        
        return centres_pred, k_logits, node_indicators, inv_cov_upper_pred  # Also return covariance prediction

# ===========================
# Training Function
# ===========================
def _compute_k_prediction_loss(k_logits, k_gt, max_k=10):
    """Compute cross-entropy loss for k-prediction.
    
    k_logits: [B, max_k] - model predictions for k=1 to max_k
    k_gt: [B] - ground truth k values (1-indexed, so subtract 1 for 0-indexed)
    max_k: maximum k value to predict
    """
    # Convert k_gt to 0-indexed (since k=1 should correspond to index 0)
    k_gt_0indexed = k_gt - 1
    
    # Ensure k_gt is within valid range [0, max_k-1]
    k_gt_0indexed = torch.clamp(k_gt_0indexed, 0, max_k - 1)
    
    # Cross-entropy loss
    k_loss = F.cross_entropy(k_logits, k_gt_0indexed, reduction='mean')
    return k_loss

def _compute_node_indicator_loss(node_indicators, active_flags):
    """Compute binary cross-entropy loss for node indicator prediction.
    
    node_indicators: [B, T] - predicted probabilities for each node being active
    active_flags: [B, T] - ground truth binary flags (1 for active, 0 for inactive)
    """
    # Binary cross-entropy loss - compute for ALL nodes (both active and inactive)
    bce_loss = F.binary_cross_entropy(node_indicators, active_flags, reduction='mean')
    
    return bce_loss

def _compute_covariance_loss(inv_cov_upper_pred, inv_cov_upper_gt, active_flags):
    """Compute masked MSE loss for per-node covariance prediction.
    
    inv_cov_upper_pred: [B, T, 6] - predicted inv_cov_upper values per node
    inv_cov_upper_gt: [B, T, 6] - ground truth inv_cov_upper values per node
    active_flags: [B, T] - binary flags indicating active (1) or inactive (0) nodes
    """
    # MSE loss for per-node covariance prediction - MASKED: compute only for active nodes
    sq_err = (inv_cov_upper_pred - inv_cov_upper_gt) ** 2  # [B, T, 6]
    node_cov_mse_per_node = sq_err.mean(dim=-1)  # [B, T] - average over 6 covariance values
    masked_cov_mse = node_cov_mse_per_node * active_flags  # [B, T]
    if active_flags.sum() > 0:
        cov_loss = masked_cov_mse.sum() / active_flags.sum()
    else:
        cov_loss = torch.tensor(0.0, device=inv_cov_upper_pred.device)
    
    return cov_loss

def train_model(model, dataset, device='cuda:0', n_epochs=4, batch_size=4,
               mae_weight: float = 1.0,
               k_loss_weight: float = 0.1,
               node_indicator_weight: float = 1.0,
               covariance_weight: float = 0.1,
               checkpoint_dir: str = 'checkpoints'):  # NEW: checkpoint directory parameter
    """
    Training function with k-prediction loss, node indicator loss, and covariance loss.
    
    k_loss_weight: weight for k-prediction loss
    node_indicator_weight: weight for node indicator loss
    covariance_weight: weight for covariance loss
    """
    """Simple training loop printing per-epoch MAE without extra plotting/logging."""
    device = torch.device(device)
    model = model.to(device)

    total_len = len(dataset)
    # ------------------------------------------------------------------
    # In DEBUG mode, skip the (relatively expensive) stratified split that
    # requires scanning every event.  A simple random split is enough.
    # ------------------------------------------------------------------
    if DEBUG_MODE:
        len_train = int(total_len * 0.7)
        len_val   = int(total_len * 0.15)
        len_test  = total_len - len_train - len_val
        torch.manual_seed(42)  # reproducibility
        train_set, val_set, test_set = random_split(dataset, [len_train, len_val, len_test])
    else:
        # Keep the original behaviour (stratified split by k) in full runs
        if hasattr(dataset, 'k_values'):
            # Stratified split to keep k distribution identical across splits
            train_idx, val_idx, test_idx = stratified_split_indices(dataset.k_values, seed=42)
            train_set = Subset(dataset, train_idx)
            val_set   = Subset(dataset, val_idx)
            test_set  = Subset(dataset, test_idx)
        else:
            len_train = int(total_len * 0.7)
            len_val   = int(total_len * 0.15)
            len_test  = total_len - len_train - len_val
            torch.manual_seed(42)
            train_set, val_set, test_set = random_split(dataset, [len_train, len_val, len_test])
    
    # Return the test set for later visualization
    model._test_set = test_set

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Verify k-distribution within the training subset
    # -------------------------------------------------------------
    if hasattr(dataset, 'k_values'):
        if isinstance(train_set, Subset):
            train_indices = train_set.indices
        else:
            # random_split also returns Subset, but just in case
            train_indices = list(range(len(train_set)))
        train_k_vals = dataset.k_values[train_indices]
        uniq_k_train, cnt_k_train = np.unique(train_k_vals, return_counts=True)
        print("Train-set k distribution (after split):")
        for k_val, cnt in zip(uniq_k_train, cnt_k_train):
            print(f"  k={int(k_val)}  →  {cnt} events")
 
    # =========================
    # Subsample train set: max 1000 events per k
    # =========================
    if hasattr(dataset, 'k_values'):
        if isinstance(train_set, Subset):
            train_indices = np.array(train_set.indices)
        else:
            train_indices = np.arange(len(train_set))
        train_k_vals = dataset.k_values[train_indices]
        unique_k = np.unique(train_k_vals)
        rng = np.random.RandomState(42)
        selected_indices = []
        for k in unique_k:
            k_idx = np.where(train_k_vals == k)[0]
            rng.shuffle(k_idx)
            selected_indices.extend(train_indices[k_idx[:3500]])
        # Replace train_set with the subsampled version
        train_set = Subset(dataset, selected_indices)
        print("After subsampling, train set size:", len(train_set))
        # Print new k distribution
        train_k_vals = dataset.k_values[selected_indices]
        uniq_k_train, cnt_k_train = np.unique(train_k_vals, return_counts=True)
        print("Train-set k distribution (after subsample):")
        for k_val, cnt in zip(uniq_k_train, cnt_k_train):
            print(f"  k={int(k_val)}  →  {cnt} events")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # Higher LR to escape mode collapse
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    """
    # warmup scheduler for first few epochs
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=2
    )
    """
    train_loss_curve = []
    val_loss_curve = []
    train_l2_curve = []  # Track L2 loss for training
    val_l2_curve = []    # Track L2 loss for validation
    
    # NEW: Track interval L2 losses and event counts for plotting
    train_interval_l2_losses = []
    train_interval_event_counts = []
    val_interval_l2_losses = []
    val_interval_event_counts = []
    total_events_processed = 0
    
    # NEW: Synchronized interval tracking
    interval_event_counts = []
    interval_train_l2_losses = []
    interval_val_l2_losses = []
    interval_train_mae_losses = []
    interval_val_mae_losses = []
    interval_train_cov_cos_losses = []
    interval_val_cov_cos_losses = []


    
    # =========================
    # Training loop
    # =========================
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        running_cos = 0.0

        running_k_loss = 0.0
        running_k_acc = 0.0
        running_node_indicator_loss = 0.0  # NEW: Track node indicator loss
        running_node_indicator_acc = 0.0   # NEW: Track node indicator accuracy
        running_covariance_loss = 0.0      # NEW: Track covariance loss
        running_train_mae = 0.0  # <-- Add this line to track train MAE
        running_train_l2 = 0.0   # Track train L2 loss
        n_batches = 0
        
        # Add event counter and interval loss for printing every 3000 events
        event_counter = 0
        interval_loss = 0.0
        interval_l2_loss = 0.0  # Track L2 loss for intervals
        interval_mae_loss = 0.0 # Track MAE loss for intervals
        interval_cov_cos_loss = 0.0 # Track covariance cosine loss for intervals
        interval_events = 0
        
        # Store all interval losses for this epoch
        epoch_interval_losses = []
        epoch_interval_l2_losses = []  # Store L2 losses for intervals
        
        epoch_train_intervals = 0  # Count training intervals for this epoch
        val_loader_cycle = itertools.cycle(val_loader)
        for inputs, centres_gt, active_flags, k_value, inv_cov_upper_gt in train_loader:
            # print("Input shape to model:", inputs.shape)  # Should be [batch, nodes, 5]
            inputs = inputs.to(device)  # Move full input to device
            # inputs_model = inputs[..., :2]  # Only use sum and mean for the model
            centres_gt = centres_gt.to(device)
            active_flags = active_flags.to(device)  # Move active_flags to device
            k_value = k_value.to(device)  # Move k_value to device
            inv_cov_upper_gt = inv_cov_upper_gt.to(device)  # Move inv_cov_upper_gt to device

            optimizer.zero_grad()
            centres_pred, k_logits, node_indicators, inv_cov_upper_pred = model(inputs)  # inv_cov_upper_pred: [B, T, 6]

            # K-prediction loss
            k_loss = _compute_k_prediction_loss(k_logits, k_value, model.max_k)

            # Node indicator loss
            node_indicator_loss = _compute_node_indicator_loss(node_indicators, active_flags)

            # NEW: Covariance loss - use GT active flags for training
            covariance_loss = _compute_covariance_loss(inv_cov_upper_pred, inv_cov_upper_gt, active_flags)

            # NEW: Per-node covariance cosine similarity - MASKED: compute only for active nodes
            cov_cos_sim = F.cosine_similarity(inv_cov_upper_pred, inv_cov_upper_gt, dim=-1)  # [B, T]
            masked_cov_cos = cov_cos_sim * active_flags  # [B, T]
            if active_flags.sum() > 0:
                cov_cos_sim_avg = masked_cov_cos.sum() / active_flags.sum()
            else:
                cov_cos_sim_avg = torch.tensor(0.0, device=inv_cov_upper_pred.device)
            # Use cosine similarity directly (higher is better) instead of 1 - cos_sim
            cov_cos_loss = -cov_cos_sim_avg  # Negative because we want to maximize cosine similarity

            # NEW: Calculate node indicator accuracy for training
            node_pred_binary = (node_indicators >= 0.5).float()
            node_indicator_acc = (node_pred_binary == active_flags).float().mean().item()
            running_node_indicator_acc += node_indicator_acc

            # Debug: Print k predictions for first few batches
            """
            if n_batches < 3:  # Only for first 3 batches
                k_pred = k_logits.argmax(dim=1) + 1
                print(f"\nDEBUG Batch {n_batches} (batch_size={len(k_value)}):")
                for i in range(len(k_value)):
                    print(f"  Event {i}: GT k={k_value[i].item()}, Pred k={k_pred[i].item()}")
                print(f"  K loss: {k_loss.item():.4f}")
            """

            # Compute k-prediction accuracy for training
            k_pred = k_logits.argmax(dim=1) + 1  # Convert back to 1-indexed
            k_acc = (k_pred == k_value).float().mean().item()
            running_k_acc += k_acc

            # -------- losses --------
            # node-level MAE (distance in Cartesian space) - MASKED: compute only for active nodes
            abs_err = torch.abs(centres_pred - centres_gt)  # [B, T, 3]
            node_mae_per_node = abs_err.mean(dim=-1)  # [B, T]
            masked_mae = node_mae_per_node * active_flags  # [B, T]
            if active_flags.sum() > 0:
                node_mae = masked_mae.sum() / active_flags.sum()
            else:
                node_mae = torch.tensor(0.0, device=centres_pred.device)

            node_l2_loss = F.mse_loss(centres_pred, centres_gt, reduction='mean')

            running_train_mae += node_mae * inputs.size(0)  # <-- accumulate train MAE
            running_train_l2 += node_l2_loss * inputs.size(0)  # accumulate train L2

            # Masked center cosine with predicted active nodes
            center_cos = F.cosine_similarity(centres_pred, centres_gt, dim=-1)  # [B, T]
            masked_center_cos = center_cos * node_pred_binary  # [B, T]
            if node_pred_binary.sum() > 0:
                avg_center_cos = masked_center_cos.sum() / node_pred_binary.sum()
            else:
                avg_center_cos = torch.tensor(0.0, device=centres_pred.device)
            cos_loss = 1.0 - avg_center_cos
            avg_cos_sim = avg_center_cos.item()  # Track masked similarity for logging

            # NEW: Accumulate covariance cosine loss
            interval_cov_cos_loss += cov_cos_loss * inputs.size(0)

            # Simple loss - direct supervision with clustering enabled
            loss = mae_weight * node_mae + k_loss_weight * k_loss + node_indicator_weight * node_indicator_loss + covariance_weight * covariance_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            interval_loss += loss.item() * inputs.size(0)
            interval_l2_loss += node_l2_loss * inputs.size(0)  # Track L2 loss for intervals
            interval_mae_loss += node_mae * inputs.size(0)  # Track MAE loss for intervals
            event_counter += inputs.size(0)
            interval_events += inputs.size(0)
            total_events_processed += inputs.size(0)

            # Print every 3000 events
            if event_counter >= 3000:
                avg_interval_loss = interval_loss / interval_events
                avg_interval_l2_loss = interval_l2_loss / interval_events
                avg_interval_mae_loss = interval_mae_loss / interval_events
                # Compute validation L2 loss for 3000 events - NO masking, same as training
                val_l2_sum = 0.0
                val_mae_sum = 0.0
                val_cov_cos_sum = 0.0
                val_events_processed = 0
                val_events_needed = 3000  # Define the missing variable

                # Use a fresh validation loader instead of cycling
                val_loader_iter = iter(val_loader)
                while val_events_processed < val_events_needed:
                    try:
                        val_inputs, val_centres_gt, val_active_flags, val_k_value, val_inv_cov_upper_gt = next(val_loader_iter)
                    except StopIteration:
                        # If we run out of validation data, restart
                        val_loader_iter = iter(val_loader)
                        val_inputs, val_centres_gt, val_active_flags, val_k_value, val_inv_cov_upper_gt = next(val_loader_iter)

                    val_inputs = val_inputs.to(device)
                    val_centres_gt = val_centres_gt.to(device)
                    val_active_flags = val_active_flags.to(device)
                    val_k_value = val_k_value.to(device)
                    val_inv_cov_upper_gt = val_inv_cov_upper_gt.to(device)  # Move inv_cov_upper_gt to device

                    with torch.no_grad():
                        val_centres_pred, _, val_node_indicators, val_inv_cov_upper_pred = model(val_inputs)

                    # Masked MAE (GT) and L2 using model prediction
                    abs_err = torch.abs(val_centres_pred - val_centres_gt)  # [B, T, 3]
                    node_mae_per_node = abs_err.mean(dim=-1)  # [B, T]
                    node_pred_binary = (val_node_indicators >= 0.5).float()  # [B, T]
                    masked_mae = node_mae_per_node * val_active_flags  # [B, T]
                    if val_active_flags.sum() > 0:
                        batch_val_mae = masked_mae.sum().item() / val_active_flags.sum().item()
                    else:
                        batch_val_mae = 0.0

                    sq_err = (val_centres_pred - val_centres_gt) ** 2  # [B, T, 3]
                    node_l2_per_node = sq_err.mean(dim=-1)  # [B, T]
                    masked_l2 = node_l2_per_node * node_pred_binary  # [B, T]
                    if node_pred_binary.sum() > 0:
                        batch_val_l2 = masked_l2.sum().item() / node_pred_binary.sum().item()
                    else:
                        batch_val_l2 = 0.0

                    # NEW: Per-node covariance cosine similarity for validation - use predicted active flags
                    val_cov_cos_sim = F.cosine_similarity(val_inv_cov_upper_pred, val_inv_cov_upper_gt, dim=-1)  # [B, T]
                    node_pred_binary = (val_node_indicators >= 0.5).float()  # [B, T]
                    masked_val_cov_cos = val_cov_cos_sim * node_pred_binary  # [B, T]
                    if node_pred_binary.sum() > 0:
                        val_cov_cos_sim_avg = masked_val_cov_cos.sum().item() / node_pred_binary.sum().item()
                    else:
                        val_cov_cos_sim_avg = 0.0
                    # Use cosine similarity directly (higher is better) instead of 1 - cos_sim
                    batch_val_cov_cos_loss = -val_cov_cos_sim_avg  # Negative because we want to maximize cosine similarity

                    batch_size = val_inputs.size(0)
                    take = min(batch_size, val_events_needed - val_events_processed)
                    val_l2_sum += batch_val_l2 * take
                    val_mae_sum += batch_val_mae * take
                    val_cov_cos_sum += batch_val_cov_cos_loss * take
                    val_events_processed += take
                avg_val_l2_loss = val_l2_sum / val_events_needed
                avg_val_mae_loss = val_mae_sum / val_events_needed
                avg_val_cov_cos_loss = val_cov_cos_sum / val_events_needed
                avg_interval_cov_cos_loss = interval_cov_cos_loss / interval_events if interval_events > 0 else 0.0
                print(f"[Epoch {epoch:03d}] Processed {total_events_processed} events | Train Avg Loss (last 3000 events): {avg_interval_loss:.6f} | Train Avg L2 Loss: {avg_interval_l2_loss:.6f} | Val Avg L2 Loss: {avg_val_l2_loss:.6f} | Train Avg MAE: {avg_interval_mae_loss:.6f} | Val Avg MAE: {avg_val_mae_loss:.6f} | Train Avg Cov Cos: {-avg_interval_cov_cos_loss:.4f} | Val Avg Cov Cos: {-avg_val_cov_cos_loss:.4f}")
                # Synchronized storage for plotting
                interval_event_counts.append(total_events_processed)
                interval_train_l2_losses.append(avg_interval_l2_loss)
                interval_val_l2_losses.append(avg_val_l2_loss)
                interval_train_mae_losses.append(avg_interval_mae_loss)
                interval_val_mae_losses.append(avg_val_mae_loss)
                interval_train_cov_cos_losses.append(avg_interval_cov_cos_loss)
                interval_val_cov_cos_losses.append(avg_val_cov_cos_loss)
                epoch_train_intervals += 1
                interval_loss = 0.0
                interval_l2_loss = 0.0
                interval_mae_loss = 0.0
                interval_cov_cos_loss = 0.0
                interval_events = 0
                event_counter = 0


            # track cosine similarity for logging (higher is better)
            running_cos += avg_cos_sim  # Use avg_cos_sim instead of cos_loss
            
            running_k_loss += k_loss.item()
            running_node_indicator_loss += node_indicator_loss.item() # Track node indicator loss
            running_covariance_loss += covariance_loss.item() # Track covariance loss
            n_batches += 1

        # If there are leftover events in the last interval, average and store them too
        if interval_events > 0:
            avg_interval_loss = interval_loss / interval_events
            avg_interval_l2_loss = interval_l2_loss / interval_events
            avg_interval_mae_loss = interval_mae_loss / interval_events
            avg_interval_cov_cos_loss = interval_cov_cos_loss / interval_events
            # Synchronized storage for plotting
            interval_event_counts.append(total_events_processed)
            interval_train_l2_losses.append(avg_interval_l2_loss)
            interval_val_l2_losses.append(avg_interval_l2_loss) # For consistency, use train L2 for leftover
            interval_train_mae_losses.append(avg_interval_mae_loss)
            interval_val_mae_losses.append(avg_interval_mae_loss) # For consistency, use train MAE for leftover
            interval_train_cov_cos_losses.append(avg_interval_cov_cos_loss)
            interval_val_cov_cos_losses.append(avg_interval_cov_cos_loss) # For consistency, use train cov cos for leftover
        # Store the mean of all interval losses for this epoch
        train_loss_curve.append(np.mean(epoch_interval_losses))
        train_l2_curve.append(np.mean(epoch_interval_l2_losses))
        
        avg_train_loss = running_loss / len(train_loader.dataset)
        avg_train_mae = running_train_mae / len(train_loader.dataset)  # <-- compute average train MAE
        avg_train_l2 = running_train_l2 / len(train_loader.dataset)   # compute average train L2
        avg_train_cos = running_cos / max(n_batches,1)

        avg_k_loss = running_k_loss / max(n_batches,1)
        avg_train_k_acc = running_k_acc / max(n_batches,1)
        avg_node_indicator_loss = running_node_indicator_loss / max(n_batches,1)  # NEW: Average node indicator loss
        avg_train_node_indicator_acc = running_node_indicator_acc / max(n_batches,1)  # NEW: Average train node indicator accuracy
        avg_covariance_loss = running_covariance_loss / max(n_batches,1) # NEW: Average covariance loss
        val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step(val_metrics['centre_mae'])
        
        # Store validation loss for plotting
        val_loss_curve.append(val_metrics['centre_mae'])
        val_l2_curve.append(val_metrics['centre_l2'])  # Store validation L2 loss
        
        print(
            f"[Epoch {epoch:03d}] Train Loss: {avg_train_loss:.6f} | Train MAE: {avg_train_mae:.6f} | Train L2: {avg_train_l2:.6f} | Train CosSim: {avg_train_cos:.4f} | "
            f"K Loss: {avg_k_loss:.4f} | Train K Acc: {avg_train_k_acc:.4f} | Node Indicator Loss: {avg_node_indicator_loss:.4f} | Train Node Indicator Acc: {avg_train_node_indicator_acc:.4f} | "
            f"Covariance Loss: {avg_covariance_loss:.4f} | "
            f"Val MAE: {val_metrics['centre_mae']:.6f} | Val L2: {val_metrics['centre_l2']:.6f} | "
            f"Val CosSim: {val_metrics['centre_cos']:.4f} | Val K Acc: {val_metrics['k_accuracy']:.4f} | "
            f"Val Node Indicator Acc: {val_metrics['node_indicator_accuracy']:.4f} | "
            f"Val Cov MSE: {val_metrics['covariance_mse']:.6f} | Val Cov Cos: {val_metrics['covariance_cos']:.4f}"
        )

        # Save model weights after each epoch (simple state dict only)
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch:03d}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model weights to {checkpoint_path}")

    # Final test evaluation
    test_metrics = evaluate_model(model, test_loader, device)
    
    print("\nFinal Test MAE: {centre_mae:.6f} | Test L2: {centre_l2:.6f} | Test CosSim: {centre_cos:.4f} | Test K Acc: {k_accuracy:.4f} | Test Node Indicator Acc: {node_indicator_accuracy:.4f} | Test Cov MSE: {covariance_mse:.6f} | Test Cov Cos: {covariance_cos:.4f}".format(
          **test_metrics))

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")


    # =========================
    # Plot synchronized training and val L2 loss (interval-based)
    # =========================
    plt.figure(figsize=(12,7))
    # Convert to CPU if needed
    train_l2_cpu = [x.cpu().item() if torch.is_tensor(x) else x for x in interval_train_l2_losses]
    val_l2_cpu = [x.cpu().item() if torch.is_tensor(x) else x for x in interval_val_l2_losses]
    plt.plot(interval_event_counts, train_l2_cpu, label='Train L2 Loss (per 3000 events)', marker='o')
    plt.plot(interval_event_counts, val_l2_cpu, label='Val L2 Loss (per 3000 events)', marker='x')
    plt.xlabel('Number of Events Processed')
    plt.ylabel('Loss (L2)')
    plt.title('Training and Validation L2 Loss Curves (per 3000 events)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('train_val_l2_loss_curve_intervals.png')
    plt.close()
    # =========================
    # Plot synchronized training and val MAE loss (interval-based)
    # =========================
    plt.figure(figsize=(12,7))
    # Convert to CPU if needed
    train_mae_cpu = [x.cpu().item() if torch.is_tensor(x) else x for x in interval_train_mae_losses]
    val_mae_cpu = [x.cpu().item() if torch.is_tensor(x) else x for x in interval_val_mae_losses]
    plt.plot(interval_event_counts, train_mae_cpu, label='Train MAE (per 3000 events)', marker='o')  
    plt.plot(interval_event_counts, val_mae_cpu, label='Val MAE (per 3000 events)', marker='x')
    plt.xlabel('Number of Events Processed')
    plt.ylabel('Loss (MAE)')
    plt.title('Training and Validation MAE Curves (per 3000 events)')    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('train_val_mae_loss_curve_intervals.png')
    plt.close()

# ===========================
# Model Loading Function
# ===========================
def load_model_weights(checkpoint_path, model):
    """Load model weights from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load weights into
    
    Returns:
        model: The model with loaded weights
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {checkpoint_path}")
    return model

def resume_training(model, dataset, device='cuda:0', start_epoch=1, n_epochs=12, batch_size=4,
                   mae_weight: float = 1.0, k_loss_weight: float = 0.1, 
                   node_indicator_weight: float = 1.0, covariance_weight: float = 0.1,
                   checkpoint_dir: str = 'checkpoints'):
    """Resume training from a specific epoch checkpoint.
    
    Args:
        model: The model to train
        dataset: The dataset to train on
        device: Device to train on
        start_epoch: Epoch to start from (0-indexed)
        n_epochs: Total number of epochs to train
        batch_size: Batch size (reduced to avoid memory issues)
        mae_weight: Weight for MAE loss
        k_loss_weight: Weight for k-prediction loss
        node_indicator_weight: Weight for node indicator loss
        covariance_weight: Weight for covariance loss
        checkpoint_dir: Directory containing checkpoints
    """
    # Load the checkpoint from the specified epoch
    checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{start_epoch:03d}.pth')
    if os.path.exists(checkpoint_path):
        model = load_model_weights(checkpoint_path, model)
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
        start_epoch = 0
    
    # Continue training from the next epoch
    remaining_epochs = n_epochs - start_epoch
    if remaining_epochs > 0:
        print(f"Training for {remaining_epochs} more epochs (from epoch {start_epoch} to {n_epochs-1})")
        train_model(model, dataset, device=device, n_epochs=remaining_epochs, batch_size=batch_size,
                   mae_weight=mae_weight, k_loss_weight=k_loss_weight,
                   node_indicator_weight=node_indicator_weight, covariance_weight=covariance_weight,
                   checkpoint_dir=checkpoint_dir)
    else:
        print("No remaining epochs to train.")

# ===========================
# Evaluation Function
# ===========================
def evaluate_model(model, loader, device):
    # Return average MAE, L2 loss, cosine similarity, k-prediction accuracy, node indicator accuracy
    
    model.eval()
    mae_list, l2_list, cos_list, k_acc_list, node_indicator_acc_list, cov_mse_list, cov_cos_list = [], [], [], [], [], [], []
    k_predictions = []  # Store predictions for debugging
    k_ground_truth = []  # Store ground truth for debugging

    with torch.no_grad():
        for inputs, centres_gt, active_flags, k_value, inv_cov_upper_gt in loader:
            inputs = inputs.to(device)
            centres_gt = centres_gt.to(device)
            active_flags = active_flags.to(device)
            k_value = k_value.to(device)
            inv_cov_upper_gt = inv_cov_upper_gt.to(device) # Move inv_cov_upper_gt to device

            centres_pred, k_logits, node_indicators, inv_cov_upper_pred = model(inputs)  # inv_cov_upper_pred: [B, T, 6]

            # -------- per-batch MAE, L2 & CosSim (MASKED: compute only for nodes predicted as active) --------
            abs_err = torch.abs(centres_pred - centres_gt)  # [B, T, 3]
            node_mae_per_node = abs_err.mean(dim=-1)        # [B, T]
            node_pred_binary = (node_indicators >= 0.5).float()  # [B, T]
            masked_mae = node_mae_per_node * node_pred_binary   # [B, T]
            if node_pred_binary.sum() > 0:
                batch_mae = masked_mae.sum().item() / node_pred_binary.sum().item()
            else:
                batch_mae = 0.0

            # Masked L2 loss
            sq_err = (centres_pred - centres_gt) ** 2  # [B, T, 3]
            node_l2_per_node = sq_err.mean(dim=-1)      # [B, T]
            masked_l2 = node_l2_per_node * node_pred_binary # [B, T]
            if node_pred_binary.sum() > 0:
                batch_l2 = masked_l2.sum().item() / node_pred_binary.sum().item()
            else:
                batch_l2 = 0.0

            # Masked cosine similarity
            cos_sim = F.cosine_similarity(centres_pred, centres_gt, dim=-1)  # [B, T]
            masked_cos = cos_sim * node_pred_binary  # [B, T]
            if node_pred_binary.sum() > 0:
                batch_cos = masked_cos.sum().item() / node_pred_binary.sum().item()
            else:
                batch_cos = 0.0
            
            mae_list.append(batch_mae)
            l2_list.append(batch_l2)
            cos_list.append(batch_cos)
            
            # -------- per-batch covariance metrics (MASKED: compute only for nodes predicted as active) --------
            # Covariance MSE
            cov_sq_err = (inv_cov_upper_pred - inv_cov_upper_gt) ** 2  # [B, T, 6]
            node_cov_mse_per_node = cov_sq_err.mean(dim=-1)  # [B, T]
            masked_cov_mse = node_cov_mse_per_node * node_pred_binary  # [B, T]
            if node_pred_binary.sum() > 0:
                batch_cov_mse = masked_cov_mse.sum().item() / node_pred_binary.sum().item()
            else:
                batch_cov_mse = 0.0
            
            # Covariance cosine similarity
            cov_cos_sim = F.cosine_similarity(inv_cov_upper_pred, inv_cov_upper_gt, dim=-1)  # [B, T]
            masked_cov_cos = cov_cos_sim * node_pred_binary  # [B, T]
            if node_pred_binary.sum() > 0:
                batch_cov_cos = masked_cov_cos.sum().item() / node_pred_binary.sum().item()
            else:
                batch_cov_cos = 0.0
            
            cov_mse_list.append(batch_cov_mse)
            cov_cos_list.append(batch_cov_cos)
            
            # -------- k-prediction accuracy --------
            k_pred = k_logits.argmax(dim=1) + 1  # Convert back to 1-indexed
            k_acc = (k_pred == k_value).float().mean().item()
            k_acc_list.append(k_acc)
            
            # -------- node indicator accuracy --------
            # Convert predictions to binary (threshold at 0.5)
            node_pred_binary = (node_indicators >= 0.5).float()
            # Calculate accuracy for ALL nodes (both active and inactive)
            node_acc = (node_pred_binary == active_flags).float().mean().item()
            node_indicator_acc_list.append(node_acc)
            
            # DEBUG: Print node indicator statistics for first batch
            if len(node_indicator_acc_list) == 1:  # Only for first batch
                print(f"\nDEBUG: Node indicator analysis (first batch):")
                print(f"  node_indicators shape: {node_indicators.shape}")
                print(f"  node_indicators range: [{node_indicators.min().item():.4f}, {node_indicators.max().item():.4f}]")
                print(f"  node_indicators mean: {node_indicators.mean().item():.4f}")
                print(f"  node_pred_binary mean: {node_pred_binary.mean().item():.4f}")
                print(f"  active_flags mean: {active_flags.mean().item():.4f}")
                print(f"  node_acc: {node_acc:.4f}")
                
                # Show distribution of predictions
                pred_0 = (node_pred_binary == 0).sum().item()
                pred_1 = (node_pred_binary == 1).sum().item()
                gt_0 = (active_flags == 0).sum().item()
                gt_1 = (active_flags == 1).sum().item()
                print(f"  Predictions: 0={pred_0}, 1={pred_1}")
                print(f"  Ground truth: 0={gt_0}, 1={gt_1}")
                
                # Show confusion matrix
                tp = ((node_pred_binary == 1) & (active_flags == 1)).sum().item()
                tn = ((node_pred_binary == 0) & (active_flags == 0)).sum().item()
                fp = ((node_pred_binary == 1) & (active_flags == 0)).sum().item()
                fn = ((node_pred_binary == 0) & (active_flags == 1)).sum().item()
                print(f"  Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
            
            # Store for debugging
            k_predictions.extend(k_pred.cpu().numpy())
            k_ground_truth.extend(k_value.cpu().numpy())

    # Print some debug info
    print(f"\nDEBUG: K-prediction analysis:")
    print(f"Total events evaluated: {len(k_predictions)}")
    print(f"Ground truth k values: {k_ground_truth[:10]}...")  # First 10
    print(f"Predicted k values: {k_predictions[:10]}...")  # First 10
    
    # Count correct predictions
    correct = sum(1 for pred, gt in zip(k_predictions, k_ground_truth) if pred == gt)
    total = len(k_predictions)
    print(f"Correct predictions: {correct}/{total} = {correct/total:.4f}")
    
    # Show distribution
    from collections import Counter
    gt_counter = Counter(k_ground_truth)
    pred_counter = Counter(k_predictions)
    print(f"GT k distribution: {dict(gt_counter)}")
    print(f"Pred k distribution: {dict(pred_counter)}")

    return {
        'centre_mae': float(np.mean(mae_list)) if mae_list else 0.0,
        'centre_l2': float(np.mean(l2_list)) if l2_list else 0.0,
        'centre_cos': float(np.mean(cos_list)) if cos_list else 0.0,
        'k_accuracy': float(np.mean(k_acc_list)) if k_acc_list else 0.0,
        'node_indicator_accuracy': float(np.mean(node_indicator_acc_list)) if node_indicator_acc_list else 0.0,
        'covariance_mse': float(np.mean(cov_mse_list)) if cov_mse_list else 0.0,
        'covariance_cos': float(np.mean(cov_cos_list)) if cov_cos_list else 0.0,
    }

# ===========================
# Dynamic train/eval (no padding)
# ===========================
def evaluate_model_dynamic(model, loader, device):
    model.eval()
    mae_list, l2_list, cos_list, k_acc_list, node_indicator_acc_list, cov_mse_list, cov_cos_list = [], [], [], [], [], [], []
    with torch.no_grad():
        for batches in loader:
            for _, batch in batches.items():
                inputs, centres_gt, active_flags, k_value, inv_cov_upper_gt = batch
                inputs = inputs.to(device)
                centres_gt = centres_gt.to(device)
                active_flags = active_flags.to(device)
                k_value = k_value.to(device)
                inv_cov_upper_gt = inv_cov_upper_gt.to(device)

                centres_pred, k_logits, node_indicators, inv_cov_upper_pred = model(inputs)

                abs_err = torch.abs(centres_pred - centres_gt)
                node_mae_per_node = abs_err.mean(dim=-1)
                node_pred_binary = (node_indicators >= 0.5).float()
                masked_mae = node_mae_per_node * active_flags
                if active_flags.sum() > 0:
                    batch_mae = masked_mae.sum().item() / active_flags.sum().item()
                else:
                    batch_mae = 0.0

                sq_err = (centres_pred - centres_gt) ** 2
                node_l2_per_node = sq_err.mean(dim=-1)
                masked_l2 = node_l2_per_node * node_pred_binary
                if node_pred_binary.sum() > 0:
                    batch_l2 = masked_l2.sum().item() / node_pred_binary.sum().item()
                else:
                    batch_l2 = 0.0

                cos_sim = F.cosine_similarity(centres_pred, centres_gt, dim=-1)
                masked_cos = cos_sim * node_pred_binary
                if node_pred_binary.sum() > 0:
                    batch_cos = masked_cos.sum().item() / node_pred_binary.sum().item()
                else:
                    batch_cos = 0.0
                
                mae_list.append(batch_mae)
                l2_list.append(batch_l2)
                cos_list.append(batch_cos)
                
                # Covariance metrics
                cov_sq_err = (inv_cov_upper_pred - inv_cov_upper_gt) ** 2
                node_cov_mse_per_node = cov_sq_err.mean(dim=-1)
                masked_cov_mse = node_cov_mse_per_node * node_pred_binary
                if node_pred_binary.sum() > 0:
                    batch_cov_mse = masked_cov_mse.sum().item() / node_pred_binary.sum().item()
                else:
                    batch_cov_mse = 0.0
                cov_cos_sim = F.cosine_similarity(inv_cov_upper_pred, inv_cov_upper_gt, dim=-1)
                masked_cov_cos = cov_cos_sim * node_pred_binary
                if node_pred_binary.sum() > 0:
                    batch_cov_cos = masked_cov_cos.sum().item() / node_pred_binary.sum().item()
                else:
                    batch_cov_cos = 0.0
                cov_mse_list.append(batch_cov_mse)
                cov_cos_list.append(batch_cov_cos)
                
                # k acc and node indicator acc
                k_pred = k_logits.argmax(dim=1) + 1
                k_acc = (k_pred == k_value).float().mean().item()
                k_acc_list.append(k_acc)
                node_acc = (node_pred_binary == active_flags).float().mean().item()
                node_indicator_acc_list.append(node_acc)

    return {
        'centre_mae': float(np.mean(mae_list)) if mae_list else 0.0,
        'centre_l2': float(np.mean(l2_list)) if l2_list else 0.0,
        'centre_cos': float(np.mean(cos_list)) if cos_list else 0.0,
        'k_accuracy': float(np.mean(k_acc_list)) if k_acc_list else 0.0,
        'node_indicator_accuracy': float(np.mean(node_indicator_acc_list)) if node_indicator_acc_list else 0.0,
        'covariance_mse': float(np.mean(cov_mse_list)) if cov_mse_list else 0.0,
        'covariance_cos': float(np.mean(cov_cos_list)) if cov_cos_list else 0.0,
    }

def print_val_metrics_per_config_dynamic(model, loader, device):
    """Print per-configuration validation metrics aggregated by total_points (dynamic mode).

    Reports centre MAE, centre L2, centre cosine, covariance MSE, covariance cosine,
    k-accuracy, and node-indicator accuracy per total_points.
    """
    model.eval()
    # Accumulate sums and counts per total_points
    sums = {}
    counts = {}
    with torch.no_grad():
        for batches in loader:
            for total_points, batch in batches.items():
                inputs, centres_gt, active_flags, k_value, inv_cov_upper_gt = batch
                inputs = inputs.to(device)
                centres_gt = centres_gt.to(device)
                active_flags = active_flags.to(device)
                k_value = k_value.to(device)
                inv_cov_upper_gt = inv_cov_upper_gt.to(device)

                centres_pred, k_logits, node_indicators, inv_cov_upper_pred = model(inputs)

                # Masked metrics using predicted active flags
                node_pred_binary = (node_indicators >= 0.5).float()

                # Centre MAE (mask with GT labels)
                abs_err = torch.abs(centres_pred - centres_gt)
                node_mae_per_node = abs_err.mean(dim=-1)
                masked_mae = node_mae_per_node * active_flags
                if active_flags.sum() > 0:
                    batch_mae = masked_mae.sum().item() / active_flags.sum().item()
                else:
                    batch_mae = 0.0

                # Centre L2
                sq_err = (centres_pred - centres_gt) ** 2
                node_l2_per_node = sq_err.mean(dim=-1)
                masked_l2 = node_l2_per_node * node_pred_binary
                if node_pred_binary.sum() > 0:
                    batch_l2 = masked_l2.sum().item() / node_pred_binary.sum().item()
                else:
                    batch_l2 = 0.0

                # Centre cosine
                cos_sim = F.cosine_similarity(centres_pred, centres_gt, dim=-1)
                masked_cos = cos_sim * node_pred_binary
                if node_pred_binary.sum() > 0:
                    batch_cos = masked_cos.sum().item() / node_pred_binary.sum().item()
                else:
                    batch_cos = 0.0

                # Covariance metrics (masked)
                cov_sq_err = (inv_cov_upper_pred - inv_cov_upper_gt) ** 2
                node_cov_mse_per_node = cov_sq_err.mean(dim=-1)
                masked_cov_mse = node_cov_mse_per_node * node_pred_binary
                if node_pred_binary.sum() > 0:
                    batch_cov_mse = masked_cov_mse.sum().item() / node_pred_binary.sum().item()
                else:
                    batch_cov_mse = 0.0

                cov_cos_sim = F.cosine_similarity(inv_cov_upper_pred, inv_cov_upper_gt, dim=-1)
                masked_cov_cos = cov_cos_sim * node_pred_binary
                if node_pred_binary.sum() > 0:
                    batch_cov_cos = masked_cov_cos.sum().item() / node_pred_binary.sum().item()
                else:
                    batch_cov_cos = 0.0

                # Event-level metrics
                k_pred = k_logits.argmax(dim=1) + 1
                batch_k_acc = (k_pred == k_value).float().mean().item()
                batch_node_acc = (node_pred_binary == active_flags).float().mean().item()

                # Initialize accumulator for this config
                if total_points not in sums:
                    sums[total_points] = {
                        'mae': 0.0,
                        'l2': 0.0,
                        'cos': 0.0,
                        'cov_mse': 0.0,
                        'cov_cos': 0.0,
                        'k_acc': 0.0,
                        'node_acc': 0.0,
                    }
                    counts[total_points] = 0

                bs = inputs.size(0)
                sums[total_points]['mae'] += batch_mae * bs
                sums[total_points]['l2'] += batch_l2 * bs
                sums[total_points]['cos'] += batch_cos * bs
                sums[total_points]['cov_mse'] += batch_cov_mse * bs
                sums[total_points]['cov_cos'] += batch_cov_cos * bs
                sums[total_points]['k_acc'] += batch_k_acc * bs
                sums[total_points]['node_acc'] += batch_node_acc * bs
                counts[total_points] += bs

    if sums:
        print("Validation per configuration (dynamic mode):")
        for tp in sorted(sums.keys()):
            c = max(counts[tp], 1)
            avg_mae = sums[tp]['mae'] / c
            avg_l2 = sums[tp]['l2'] / c
            avg_cos = sums[tp]['cos'] / c
            avg_cov_mse = sums[tp]['cov_mse'] / c
            avg_cov_cos = sums[tp]['cov_cos'] / c
            avg_k_acc = sums[tp]['k_acc'] / c
            avg_node_acc = sums[tp]['node_acc'] / c
            print(
                f"  {tp} points ({counts[tp]} events): "
                f"CtrMAE={avg_mae:.6f} | CtrL2={avg_l2:.6f} | CtrCos={avg_cos:.4f} | "
                f"CovMSE={avg_cov_mse:.6f} | CovCos={avg_cov_cos:.4f} | "
                f"KAcc={avg_k_acc:.4f} | NodeAcc={avg_node_acc:.4f}"
            )

def train_model_dynamic(model, train_loader, val_loader, device='cuda:0', n_epochs=4, batch_size=4,
                        mae_weight: float = 1.0,
                        k_loss_weight: float = 0.1,
                        node_indicator_weight: float = 1.0,
                        covariance_weight: float = 0.1,
                        checkpoint_dir: str = 'checkpoints_dynamic'):
    device = torch.device(device)
    model = model.to(device)
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        running_train_mae = 0.0
        running_train_l2 = 0.0
        running_cos = 0.0
        running_k_loss = 0.0
        running_k_acc = 0.0
        running_node_indicator_loss = 0.0
        running_node_indicator_acc = 0.0
        running_covariance_loss = 0.0
        n_batches = 0
        total_events = 0

        # Interval tracking (print every ~3000 events)
        event_counter = 0
        interval_events = 0
        interval_loss = 0.0
        interval_l2_loss = 0.0
        interval_mae_loss = 0.0
        interval_cov_cos_loss = 0.0
        interval_center_cos_sum = 0.0
        interval_center_cos_count = 0

        for batches in train_loader:
            for _, batch in batches.items():
                inputs, centres_gt, active_flags, k_value, inv_cov_upper_gt = batch
                inputs = inputs.to(device)
                centres_gt = centres_gt.to(device)
                active_flags = active_flags.to(device)
                k_value = k_value.to(device)
                inv_cov_upper_gt = inv_cov_upper_gt.to(device)

                optimizer.zero_grad()
                centres_pred, k_logits, node_indicators, inv_cov_upper_pred = model(inputs)

                k_loss = _compute_k_prediction_loss(k_logits, k_value, model.max_k)
                node_indicator_loss = _compute_node_indicator_loss(node_indicators, active_flags)
                covariance_loss = _compute_covariance_loss(inv_cov_upper_pred, inv_cov_upper_gt, active_flags)

                # masked MAE
                abs_err = torch.abs(centres_pred - centres_gt)
                node_mae_per_node = abs_err.mean(dim=-1)
                masked_mae = node_mae_per_node * active_flags
                if active_flags.sum() > 0:
                    node_mae = masked_mae.sum() / active_flags.sum()
                else:
                    node_mae = torch.tensor(0.0, device=centres_pred.device)

                node_l2_loss = F.mse_loss(centres_pred, centres_gt, reduction='mean')
                # Predicted-active masked center cosine
                node_pred_binary = (node_indicators >= 0.5).float()
                center_cos_map = F.cosine_similarity(centres_pred, centres_gt, dim=-1)
                masked_center_cos = center_cos_map * node_pred_binary
                if node_pred_binary.sum() > 0:
                    avg_center_cos = masked_center_cos.sum() / node_pred_binary.sum()
                else:
                    avg_center_cos = torch.tensor(0.0, device=centres_pred.device)
                avg_cos_sim = avg_center_cos.item()

                # Covariance cosine similarity (masked with GT active flags)
                cov_cos_sim = F.cosine_similarity(inv_cov_upper_pred, inv_cov_upper_gt, dim=-1)
                masked_cov_cos = cov_cos_sim * active_flags
                if active_flags.sum() > 0:
                    cov_cos_sim_avg = masked_cov_cos.sum() / active_flags.sum()
                else:
                    cov_cos_sim_avg = torch.tensor(0.0, device=inv_cov_upper_pred.device)
                cov_cos_loss = -cov_cos_sim_avg

                loss = mae_weight * node_mae + k_loss_weight * k_loss + node_indicator_weight * node_indicator_loss + covariance_weight * covariance_loss
                loss.backward()
                optimizer.step()

                batch_size_now = inputs.size(0)
                running_loss += loss.item() * batch_size_now
                running_train_mae += node_mae * batch_size_now
                running_train_l2 += node_l2_loss * batch_size_now
                running_cos += avg_cos_sim
                running_k_loss += k_loss.item()
                k_pred = k_logits.argmax(dim=1) + 1
                running_k_acc += (k_pred == k_value).float().mean().item()
                node_pred_binary = (node_indicators >= 0.5).float()
                running_node_indicator_acc += (node_pred_binary == active_flags).float().mean().item()
                running_node_indicator_loss += node_indicator_loss.item()
                running_covariance_loss += covariance_loss.item()
                n_batches += 1
                total_events += batch_size_now

                # Interval accumulators
                interval_events += batch_size_now
                event_counter += batch_size_now
                interval_loss += loss.item() * batch_size_now
                interval_l2_loss += node_l2_loss * batch_size_now
                interval_mae_loss += node_mae * batch_size_now
                interval_cov_cos_loss += cov_cos_loss * batch_size_now
                interval_center_cos_sum += avg_cos_sim * batch_size_now
                interval_center_cos_count += batch_size_now

                # Print every ~3000 events with quick validation sample
                if event_counter >= 3000:
                    avg_interval_loss = interval_loss / max(interval_events, 1)
                    avg_interval_l2 = interval_l2_loss / max(interval_events, 1)
                    avg_interval_mae = interval_mae_loss / max(interval_events, 1)
                    avg_interval_cov_cos = interval_cov_cos_loss / max(interval_events, 1)

                    # Quick validation over ~3000 events
                    val_events_needed = 3000
                    val_events_processed = 0
                    val_l2_sum = 0.0
                    val_mae_sum = 0.0
                    val_cov_cos_sum = 0.0
                    val_center_cos_sum = 0.0
                    val_center_cos_count = 0
                    val_loader_iter = iter(val_loader)
                    while val_events_processed < val_events_needed:
                        try:
                            val_batches = next(val_loader_iter)
                        except StopIteration:
                            val_loader_iter = iter(val_loader)
                            val_batches = next(val_loader_iter)
                        for _, vbatch in val_batches.items():
                            v_inputs, v_centres_gt, v_active_flags, v_k_value, v_inv_cov_upper_gt = vbatch
                            v_inputs = v_inputs.to(device)
                            v_centres_gt = v_centres_gt.to(device)
                            v_active_flags = v_active_flags.to(device)
                            v_inv_cov_upper_gt = v_inv_cov_upper_gt.to(device)
                            with torch.no_grad():
                                v_centres_pred, _, v_node_indicators, v_inv_cov_upper_pred = model(v_inputs)
                            # Predicted-active mask
                            v_node_pred_bin = (v_node_indicators >= 0.5).float()
                            # Centre cosine similarity (masked with predicted active nodes)
                            v_center_cos_map = F.cosine_similarity(v_centres_pred, v_centres_gt, dim=-1)
                            v_center_cos_masked = v_center_cos_map * v_node_pred_bin
                            if v_node_pred_bin.sum() > 0:
                                v_center_cos = (v_center_cos_masked.sum() / v_node_pred_bin.sum()).item()
                            else:
                                v_center_cos = 0.0
                            # Masked MAE/L2 using predicted active flags
                            v_abs_err = torch.abs(v_centres_pred - v_centres_gt)
                            v_mae_per_node = v_abs_err.mean(dim=-1)
                            v_masked_mae = v_mae_per_node * v_node_pred_bin
                            if v_node_pred_bin.sum() > 0:
                                v_batch_mae = (v_masked_mae.sum() / v_node_pred_bin.sum()).item()
                            else:
                                v_batch_mae = 0.0
                            v_sq_err = (v_centres_pred - v_centres_gt) ** 2
                            v_l2_per_node = v_sq_err.mean(dim=-1)
                            v_masked_l2 = v_l2_per_node * v_node_pred_bin
                            if v_node_pred_bin.sum() > 0:
                                v_batch_l2 = (v_masked_l2.sum() / v_node_pred_bin.sum()).item()
                            else:
                                v_batch_l2 = 0.0
                            # Covariance cosine
                            v_cov_cos_sim = F.cosine_similarity(v_inv_cov_upper_pred, v_inv_cov_upper_gt, dim=-1)
                            v_masked_cov_cos = v_cov_cos_sim * v_node_pred_bin
                            if v_node_pred_bin.sum() > 0:
                                v_batch_cov_cos_loss = -(v_masked_cov_cos.sum().item() / v_node_pred_bin.sum().item())
                            else:
                                v_batch_cov_cos_loss = 0.0
                            bsz = v_inputs.size(0)
                            take = min(bsz, val_events_needed - val_events_processed)
                            val_l2_sum += v_batch_l2 * take
                            val_mae_sum += v_batch_mae * take
                            val_cov_cos_sum += v_batch_cov_cos_loss * take
                            val_events_processed += take
                            val_center_cos_sum += v_center_cos * bsz
                            val_center_cos_count += bsz
                            if val_events_processed >= val_events_needed:
                                break

                    avg_val_l2 = val_l2_sum / val_events_needed
                    avg_val_mae = val_mae_sum / val_events_needed
                    avg_val_cov_cos = val_cov_cos_sum / val_events_needed
                    avg_interval_center_cos = interval_center_cos_sum / max(interval_center_cos_count, 1)
                    avg_val_center_cos = val_center_cos_sum / max(val_center_cos_count, 1)

                    print(f"[Epoch {epoch:03d}] Processed {total_events} events | Train Avg Loss (last 3000 events): {avg_interval_loss:.6f} | Train Avg L2: {avg_interval_l2:.6f} | Val Avg L2: {avg_val_l2:.6f} | Train Avg MAE: {avg_interval_mae:.6f} | Val Avg MAE: {avg_val_mae:.6f} | Train Avg Center Cos: {avg_interval_center_cos:.4f} | Val Avg Center Cos: {avg_val_center_cos:.4f} | Train Avg Cov Cos: {-avg_interval_cov_cos:.4f} | Val Avg Cov Cos: {-avg_val_cov_cos:.4f}")

                    # Reset interval accumulators
                    event_counter = 0
                    interval_events = 0
                    interval_loss = 0.0
                    interval_l2_loss = 0.0
                    interval_mae_loss = 0.0
                    interval_cov_cos_loss = 0.0
                    interval_center_cos_sum = 0.0
                    interval_center_cos_count = 0

        val_metrics = evaluate_model_dynamic(model, val_loader, device)
        scheduler.step(val_metrics['centre_mae'])

        # Per-configuration validation summary (dynamic mode)
        print_val_metrics_per_config_dynamic(model, val_loader, device)

        avg_train_loss = running_loss / max(total_events, 1)
        avg_train_mae = running_train_mae / max(total_events, 1)
        avg_train_l2 = running_train_l2 / max(total_events, 1)
        avg_train_cos = running_cos / max(n_batches, 1)
        avg_k_loss = running_k_loss / max(n_batches, 1)
        avg_train_k_acc = running_k_acc / max(n_batches, 1)
        avg_node_indicator_loss = running_node_indicator_loss / max(n_batches, 1)
        avg_train_node_indicator_acc = running_node_indicator_acc / max(n_batches, 1)
        avg_covariance_loss = running_covariance_loss / max(n_batches, 1)

        print(
            f"[Epoch {epoch:03d}] Train Loss: {avg_train_loss:.6f} | Train MAE: {avg_train_mae:.6f} | Train L2: {avg_train_l2:.6f} | Train CosSim: {avg_train_cos:.4f} | "
            f"K Loss: {avg_k_loss:.4f} | Train K Acc: {avg_train_k_acc:.4f} | Node Indicator Loss: {avg_node_indicator_loss:.4f} | Train Node Indicator Acc: {avg_train_node_indicator_acc:.4f} | "
            f"Covariance Loss: {avg_covariance_loss:.4f} | "
            f"Val MAE: {val_metrics['centre_mae']:.6f} | Val L2: {val_metrics['centre_l2']:.6f} | "
            f"Val CosSim: {val_metrics['centre_cos']:.4f} | Val K Acc: {val_metrics['k_accuracy']:.4f} | "
            f"Val Node Indicator Acc: {val_metrics['node_indicator_accuracy']:.4f} | "
            f"Val Cov MSE: {val_metrics['covariance_mse']:.6f} | Val Cov Cos: {val_metrics['covariance_cos']:.4f}"
        )

        # checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch:03d}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model weights to {checkpoint_path}")





# ==============================================================
# Visualisation helper (removed)
# ==============================================================
# The original visualization utilities (analyze_center_predictions, visualise_events,
# visualize_gt_labels_3d) have been removed to streamline the script and eliminate
# heavy matplotlib dependencies during training.

# ===========================
# Run Training
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train centre-regression model with optional dynamic variable-node mode')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic mode (variable nodes, no padding)')
    parser.add_argument('--data_dir', type=str, default='./synthetic_events', help='Base directory with train/val/test splits')
    parser.add_argument('--train_points', type=str, default='50,100,150,200,300,400,500', help='Comma-separated train total_points list')
    parser.add_argument('--val_points', type=str, default='600,700,800,900,1000', help='Comma-separated val total_points list')
    parser.add_argument('--test_points', type=str, default='600,700,800,900,1000', help='Comma-separated test total_points list')
    parser.add_argument('--epochs', type=int, default=12, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device')
    parser.add_argument('--resume_epoch', type=int, default=None, help='Resume from epoch (static mode only)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.dynamic:
        # Dynamic mode: train small, val/test large without padding
        train_points = [int(x) for x in args.train_points.split(',') if x]
        val_points = [int(x) for x in args.val_points.split(',') if x]
        test_points = [int(x) for x in args.test_points.split(',') if x]

        print("Dynamic mode enabled")
        print(f"Train points: {train_points}")
        print(f"Val   points: {val_points}")
        print(f"Test  points: {test_points}")

        train_datasets = load_dynamic_datasets(args.data_dir, 'train', train_points)
        val_datasets = load_dynamic_datasets(args.data_dir, 'val', val_points)
        test_datasets = load_dynamic_datasets(args.data_dir, 'test', test_points)
        if not train_datasets:
            raise ValueError('No training datasets found for dynamic mode')

        # Print k-to-nodes mapping per split to verify k coverage vs total_points
        def summarize_k_nodes(datasets, split_name):
            mapping = {}
            for ds in datasets:
                total_points_ds = getattr(ds, 'total_points', None)
                k_vals = getattr(ds, 'k_values', None)
                if total_points_ds is None or k_vals is None:
                    continue
                unique_k = np.unique(k_vals)
                for k_item in unique_k:
                    k_int = int(k_item)
                    if k_int not in mapping:
                        mapping[k_int] = set()
                    mapping[k_int].add(int(total_points_ds))
            print(f"{split_name} k→nodes mapping:")
            if not mapping:
                print("  (no data)")
                return
            for k_val in sorted(mapping.keys()):
                nodes_sorted = sorted(mapping[k_val])
                nodes_str = ", ".join(str(n) for n in nodes_sorted)
                print(f"  k={k_val}: {nodes_str}")

        summarize_k_nodes(train_datasets, "Train")
        summarize_k_nodes(val_datasets, "Val")
        summarize_k_nodes(test_datasets, "Test")

        # Build loaders
        train_loader = MultiConfigDataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)
        val_loader = MultiConfigDataLoader(val_datasets, batch_size=args.batch_size, shuffle=False)
        test_loader = MultiConfigDataLoader(test_datasets, batch_size=args.batch_size, shuffle=False)

        # Configure model
        config = GPTConfig(block_size=3333, n_layer=6, n_head=8, n_embd=256, dropout=0.1)
        # Determine max k across all training datasets
        all_k = np.concatenate([ds.k_values for ds in train_datasets])
        max_k_in_data = int(all_k.max())
        model = GPTEncoderModel(config, input_dim=5, max_k=max_k_in_data + 1)

        print("Training (dynamic) with MAE, k, node-indicator and covariance losses")
        train_model_dynamic(
            model,
            train_loader,
            val_loader,
            device=args.device,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            mae_weight=1.0,
            k_loss_weight=0.1,
            node_indicator_weight=1.0,
            covariance_weight=0.1,
            checkpoint_dir=args.checkpoint_dir,
        )

        # Final evaluation on test (dynamic)
        test_metrics = evaluate_model_dynamic(model, test_loader, args.device)
        print("\nFinal Test (dynamic) | "
              f"MAE: {test_metrics['centre_mae']:.6f} | L2: {test_metrics['centre_l2']:.6f} | Cos: {test_metrics['centre_cos']:.4f} | "
              f"K Acc: {test_metrics['k_accuracy']:.4f} | Node Acc: {test_metrics['node_indicator_accuracy']:.4f} | "
              f"Cov MSE: {test_metrics['covariance_mse']:.6f} | Cov Cos: {test_metrics['covariance_cos']:.4f}")

        # Per-configuration test summary (dynamic mode)
        print_val_metrics_per_config_dynamic(model, test_loader, args.device)

        # ------------------------------
        # Simple dynamic test visualization
        # ------------------------------
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – enable 3-D
        out_dir = "event_vis"
        os.makedirs(out_dir, exist_ok=True)
        model.eval()
        with torch.no_grad():
            target_points = (600, 700, 800, 900, 1000)
            for tp in target_points:
                if tp not in getattr(test_loader, 'loaders', {}):
                    print(f"Visualization: total_points={tp} not found in test loader; skipping")
                    continue
                loader_tp = test_loader.loaders[tp]
                ds_tp = loader_tp.dataset
                # Determine available k values in this config
                if hasattr(ds_tp, 'k_values'):
                    unique_k_vals = sorted(np.unique(ds_tp.k_values))
                else:
                    # Fallback: scan one pass
                    seen = set()
                    for batch in loader_tp:
                        _, _, _, k_value_b, _ = batch
                        seen.update(k_value_b.numpy().tolist())
                    unique_k_vals = sorted(seen)

                rng = np.random.RandomState()
                for k_target in unique_k_vals:
                    # Sample two random events with this k from dataset
                    if hasattr(ds_tp, 'k_values'):
                        idx_all = np.where(ds_tp.k_values == k_target)[0]
                        if idx_all.size == 0:
                            continue
                        rng.shuffle(idx_all)
                        idx_pick = idx_all[:2]
                        for pick_i, idx in enumerate(idx_pick):
                            # Fetch single sample and run model
                            s_inputs, s_centres_gt, s_active_flags, s_k_value, s_inv_cov_upper_gt = ds_tp[idx]
                            s_inputs = s_inputs.unsqueeze(0).to(args.device)
                            s_centres_gt = s_centres_gt.unsqueeze(0).to(args.device)
                            s_active_flags = s_active_flags.unsqueeze(0).to(args.device)
                            centres_pred, k_logits, node_indicators, inv_cov_upper_pred = model(s_inputs)

                            b0_inputs = s_inputs[0].cpu().numpy()
                            b0_centres_gt = s_centres_gt[0].cpu().numpy()
                            b0_node_ind = node_indicators[0].cpu().numpy()
                            b0_mask = b0_node_ind >= 0.5
                            b0_centres_pred = centres_pred[0].detach().cpu().numpy()
                            xyz = b0_inputs[:, 5:8]

                            # Figure 1: GT vs Pred centres (masked) + GT
                            fig = plt.figure(figsize=(12, 5))
                            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
                            for ax in (ax1, ax2):
                                ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='lightgrey', s=4, alpha=0.3)
                                ax.set_xlim(-120, 120)
                                ax.set_ylim(-120, 120)
                                ax.set_zlim(-100, 100)
                                ax.view_init(elev=25, azim=135)
                                ax.set_xlabel('X')
                                ax.set_ylabel('Y')
                                ax.set_zlabel('Z')
                            unique_gt = np.unique(b0_centres_gt, axis=0)
                            k_gt = unique_gt.shape[0]
                            ax1.scatter(unique_gt[:, 0], unique_gt[:, 1], unique_gt[:, 2], c='red', marker='x', s=60, linewidths=2)
                            ax1.set_title(f"GT centres (k={k_gt}) | T={tp}")
                            pred_masked = b0_centres_pred[b0_mask]
                            k_pred = (k_logits[0].argmax().item() + 1) if k_logits is not None else k_gt
                            k_to_use = max(1, min(k_pred, k_gt))
                            if pred_masked.shape[0] >= k_to_use:
                                labels_pred = KMeans(n_clusters=k_to_use, n_init=10, random_state=0).fit(pred_masked).labels_
                                cmap = plt.cm.get_cmap('tab10')
                                colors_pred = cmap(labels_pred % 10)
                                ax2.scatter(pred_masked[:, 0], pred_masked[:, 1], pred_masked[:, 2], c=colors_pred, s=4, alpha=0.85)
                            else:
                                ax2.scatter(pred_masked[:, 0], pred_masked[:, 1], pred_masked[:, 2], c='tab:blue', s=4, alpha=0.85)
                            ax2.scatter(unique_gt[:, 0], unique_gt[:, 1], unique_gt[:, 2], c='red', marker='x', s=60, linewidths=2)
                            ax2.set_title(f"Pred centres (KMeans, masked) + GT | k*={k_to_use}")
                            plt.tight_layout()
                            outfile = os.path.join(out_dir, f"dynamic_test_T{tp}_k{k_target}_idx{pick_i}.png")
                            plt.savefig(outfile, dpi=150, bbox_inches='tight')
                            plt.close(fig)
                            print(f"Saved dynamic test visualisation: {outfile}")

                            # Figure 2: Pool-style visualization (3 panels)
                            b0_active_flags = s_active_flags[0].cpu().numpy()
                            node_pred_binary = (b0_node_ind >= 0.5).astype(int)
                            if pred_masked.shape[0] >= k_to_use:
                                kmeans = KMeans(n_clusters=k_to_use, n_init=10, random_state=0)
                                labels_pred_masked = kmeans.fit_predict(pred_masked)
                                labels_pred_full = -1 * np.ones(b0_centres_pred.shape[0], dtype=int)
                                masked_idx = np.where(b0_mask)[0]
                                labels_pred_full[masked_idx] = labels_pred_masked
                            else:
                                labels_pred_full = -1 * np.ones(b0_centres_pred.shape[0], dtype=int)

                            cmap = plt.cm.get_cmap('tab10')
                            fig2 = plt.figure(figsize=(15, 6))
                            axp1 = fig2.add_subplot(1, 3, 1, projection='3d')
                            axp2 = fig2.add_subplot(1, 3, 2, projection='3d')
                            axp3 = fig2.add_subplot(1, 3, 3, projection='3d')
                            active_mask_gt = b0_active_flags == 1
                            inactive_mask_gt = b0_active_flags == 0
                            _, inv_gt = np.unique(b0_centres_gt, axis=0, return_inverse=True)
                            colors_gt = cmap(inv_gt % 10)
                            axp1.scatter(xyz[active_mask_gt, 0], xyz[active_mask_gt, 1], xyz[active_mask_gt, 2], c=colors_gt[active_mask_gt], s=4, alpha=0.85)
                            axp1.scatter(xyz[inactive_mask_gt, 0], xyz[inactive_mask_gt, 1], xyz[inactive_mask_gt, 2], c='lightgray', s=4, alpha=0.4)
                            axp1.set_title(f"GT clusters (k={k_gt}) | GT active: {active_mask_gt.sum()}/{len(b0_active_flags)}")

                            active_mask_pred = node_pred_binary == 1
                            inactive_mask_pred = node_pred_binary == 0
                            safe_labels = np.where(labels_pred_full >= 0, labels_pred_full % 10, 0)
                            colors_pred = cmap(safe_labels)
                            axp2.scatter(xyz[active_mask_pred, 0], xyz[active_mask_pred, 1], xyz[active_mask_pred, 2], c=colors_pred[active_mask_pred], s=4, alpha=0.85)
                            axp2.scatter(xyz[inactive_mask_pred, 0], xyz[inactive_mask_pred, 1], xyz[inactive_mask_pred, 2], c='lightgray', s=4, alpha=0.4)
                            axp2.set_title(f"Pred clusters (k*={k_to_use}) | Pred active: {active_mask_pred.sum()}/{len(node_pred_binary)}")

                            axp3.scatter(xyz[active_mask_gt, 0], xyz[active_mask_gt, 1], xyz[active_mask_gt, 2], c=colors_pred[active_mask_gt], s=4, alpha=0.85)
                            axp3.scatter(xyz[inactive_mask_gt, 0], xyz[inactive_mask_gt, 1], xyz[inactive_mask_gt, 2], c='lightgray', s=4, alpha=0.4)
                            axp3.set_title(f"Pred clusters (k*={k_to_use}) | GT active: {active_mask_gt.sum()}/{len(b0_active_flags)}")

                            for axp in (axp1, axp2, axp3):
                                axp.set_xlabel('X')
                                axp.set_ylabel('Y')
                                axp.set_zlabel('Z')
                                axp.set_xlim(-120, 120)
                                axp.set_ylim(-120, 120)
                                axp.set_zlim(-100, 100)
                                axp.view_init(elev=25, azim=135)

                            plt.tight_layout()
                            outfile2 = os.path.join(out_dir, f"dynamic_test_pool_T{tp}_k{k_target}_idx{pick_i}.png")
                            plt.savefig(outfile2, dpi=150, bbox_inches='tight')
                            plt.close(fig2)
                            print(f"Saved dynamic pool visualisation: {outfile2}")
    else:
        # Original static path
        if DEBUG_MODE:
            print(f"🐛 DEBUG MODE: Training on only {DEBUG_SAMPLES} samples")

        dataset = SyntheticSet2GraphDataset('synthetic_events/synthetic_detector_data.npz')
        print(f"Dataset size: {len(dataset)} events")

        unique_k, counts_k = np.unique(dataset.k_values[:len(dataset)], return_counts=True)
        print("k distribution in loaded dataset:")
        for k_val, cnt in zip(unique_k, counts_k):
            print(f"  k={int(k_val)}  →  {cnt} events")

        sample_event = dataset[0]
        inputs, centres_gt, flags_gt, k_value, inv_cov_upper_gt = sample_event
        np.set_printoptions(precision=10, suppress=False, linewidth=120, floatmode='maxprec_equal')
        print(f"\nDEBUG: Input features shape: {inputs.shape}")
        print(f"DEBUG: k_value for event 0: {k_value.item()}")
        print("First 10 nodes of event 0 ( [sum_norm, x, y, z] ):")
        for node_idx in range(10):
            features = inputs[node_idx].numpy()
            print(f"  Node {node_idx:2d}: {features}")
        
        print(f"\nDEBUG: Node energy statistics (feature 0 = normed sum):")
        for event_idx in [0, 1, 2]:
            node_sum = dataset[event_idx][0][:,0].numpy()
            k_val = dataset[event_idx][3].item()
            print(f"  Event {event_idx}: min={node_sum.min():.4f}, max={node_sum.max():.4f}, mean={node_sum.mean():.4f}, std={node_sum.std():.4f}, k={k_val}")
        
        config = GPTConfig(block_size=3333, n_layer=6, n_head=8, n_embd=256, dropout=0.1)
        unique_k, counts_k = np.unique(dataset.k_values, return_counts=True)
        max_k_in_data = int(unique_k.max())
        print(f"Dataset k range: {int(unique_k.min())} to {max_k_in_data}")
        print("k distribution in dataset:")
        for k_val, cnt in zip(unique_k, counts_k):
            print(f"  k={int(k_val)}  →  {cnt} events")

        model = GPTEncoderModel(config, input_dim=5, max_k=max_k_in_data)
        print("Training model with MAE loss, k-prediction loss, node indicator loss, and covariance loss")

        if args.resume_epoch is not None:
            resume_training(model, dataset, device=args.device, start_epoch=args.resume_epoch,
                           n_epochs=args.epochs, batch_size=args.batch_size, mae_weight=1.0, k_loss_weight=0.1,
                           node_indicator_weight=1.0, covariance_weight=0.1, checkpoint_dir=args.checkpoint_dir)
        else:
            train_model(model, dataset, device=args.device, n_epochs=args.epochs, batch_size=args.batch_size,
                        mae_weight=1.0, k_loss_weight=0.1, node_indicator_weight=1.0, covariance_weight=0.1,
                        checkpoint_dir=args.checkpoint_dir)
