import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.metrics import f1_score, adjusted_rand_score
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sumformer_assignment import SumformerAdjacencyModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


def find_adj_chunks(chunk_dir: str):
    pattern = re.compile(r"adj_chunk_(\d+)\.npy$")
    files = []
    if not os.path.isdir(chunk_dir):
        return []
    for name in os.listdir(chunk_dir):
        m = pattern.match(name)
        if m:
            files.append((int(m.group(1)), os.path.join(chunk_dir, name)))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def find_combined_adj(chunk_dir: str, total_points: int, packed: bool = True):
    if not os.path.isdir(chunk_dir):
        return None
    fname = f'adj_matrix_packed_{total_points}pts.npy' if packed else f'adj_matrix_uint8_{total_points}pts.npy'
    path = os.path.join(chunk_dir, fname)
    return path if os.path.exists(path) else None


def unpack_if_needed(adj_block: np.ndarray, num_nodes: int, packed: bool) -> np.ndarray:
    if not packed:
        adj = adj_block.astype(np.uint8)
    else:
        # adj_block shape: (num_nodes, ceil(num_nodes/8))
        adj = np.unpackbits(adj_block, axis=1)[:, :num_nodes]
    np.fill_diagonal(adj, 0)
    return adj


def compute_adj_from_labels(labels: np.ndarray, active_flags: np.ndarray) -> np.ndarray:
    # labels: (N,), active_flags: (N,)
    lab = labels.copy()
    lab[active_flags == 0] = -1
    adj = (lab[:, None] >= 0) & (lab[:, None] == lab[None, :])
    np.fill_diagonal(adj, False)
    return adj.astype(np.uint8)


class DynamicSet2GraphDataset(Dataset):
    def __init__(self, data_npz_path: str, chunk_dir: str = None, packed: bool = True, use_chunks: bool = True):
        self.data = np.load(data_npz_path, allow_pickle=True)

        # Features: prefer X_all_mod, fallback to X_all
        if 'X_all_mod' in self.data:
            self.X_all = self.data['X_all_mod']
        elif 'X_all' in self.data:
            self.X_all = self.data['X_all']
        else:
            raise KeyError('Neither X_all_mod nor X_all found in dataset.')

        # Labels for on-the-fly adjacency if needed
        self.node_labels = self.data['node_labels'] if 'node_labels' in self.data else None
        self.active_flags = self.data['active_flags'] if 'active_flags' in self.data else None

        # k distribution
        self.k_all = self.data['k_all'] if 'k_all' in self.data else None

        self.num_events = self.X_all.shape[0]
        self.num_nodes = self.X_all.shape[1]
        self.packed = packed

        # Extract total_points from filename
        filename = os.path.basename(data_npz_path)
        match = re.search(r'_(\d+)pts\.npz$', filename)
        if match:
            self.total_points = int(match.group(1))
        else:
            self.total_points = self.num_nodes

        # Chunked adjacency
        self.use_chunks = use_chunks
        self.combined_adj_path = find_combined_adj(chunk_dir, self.total_points, packed=packed) if (use_chunks and chunk_dir) else None
        self.combined_adj = np.load(self.combined_adj_path, mmap_mode='r') if self.combined_adj_path else None

        # Sanity check: mapping length must match events
        if self.use_chunks and self.combined_adj is not None:
            if self.combined_adj.shape[0] != self.num_events:
                raise ValueError(f"Combined adjacency events ({self.combined_adj.shape[0]}) != num_events in npz ({self.num_events})")

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        x = torch.tensor(self.X_all[idx], dtype=torch.float32)
        if self.use_chunks and self.combined_adj is not None:
            adj_block = self.combined_adj[idx]
            adj = unpack_if_needed(adj_block, self.num_nodes, self.packed)
        else:
            if self.node_labels is None or self.active_flags is None:
                raise KeyError('Adjacency chunks not provided and node_labels/active_flags missing to compute on-the-fly adjacency.')
            labels = self.node_labels[idx]
            active = self.active_flags[idx]
            adj = compute_adj_from_labels(labels, active)
        adj = torch.tensor(adj, dtype=torch.float32)
        return x, adj


class MultiConfigDataLoader:
    """Data loader that handles multiple total_points configurations with separate mini-batches"""
    
    def __init__(self, datasets, batch_size, shuffle=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create separate data loaders for each configuration
        self.loaders = {}
        for dataset in datasets:
            total_points = dataset.total_points
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            self.loaders[total_points] = loader
        
        # Calculate total iterations needed: max batches across configs
        # In normal mode: each config has 8000/bs batches → 1000, so total_iterations=1000
        # In debug mode: each config has 32/bs batches → 4 (bs=8), so total_iterations=4
        self.total_iterations = 0
        for dataset in datasets:
            num_batches = len(dataset) // batch_size
            if len(dataset) % batch_size != 0:
                num_batches += 1
            self.total_iterations = max(self.total_iterations, num_batches)
        
        # Create iterators for each loader
        self.iterators = {}
        self.current_iteration = 0
        self.reset_iterators()
        
        print(f"MultiConfigDataLoader: {len(datasets)} configs, {self.total_iterations} total iterations")
    
    def reset_iterators(self):
        """Reset all iterators"""
        for total_points, loader in self.loaders.items():
            self.iterators[total_points] = iter(loader)
        self.current_iteration = 0
    
    def __iter__(self):
        self.reset_iterators()
        return self
    
    def __next__(self):
        # Check if we've reached the end
        if self.current_iteration >= self.total_iterations:
            raise StopIteration
        
        # Get one batch from each available configuration
        batches = {}
        for total_points, iterator in self.iterators.items():
            try:
                batch = next(iterator)
                batches[total_points] = batch
            except StopIteration:
                # This configuration is exhausted, skip it
                continue
        
        self.current_iteration += 1
        
        if self.current_iteration % 100 == 0:
            print(f"Current iteration: {self.current_iteration}/{self.total_iterations}")
        
        if not batches:
            # All configurations are exhausted
            raise StopIteration
        
        return batches


def print_k_distribution(k_all: np.ndarray):
    if k_all is None:
        print('k_all not found in dataset. Skipping k distribution summary.')
        return
    unique, counts = np.unique(k_all, return_counts=True)
    mapping = dict(zip(unique.tolist(), counts.tolist()))
    print('K distribution:')
    for k in sorted(mapping.keys()):
        print(f'  k={k}: {mapping[k]} events')


def stratified_split_indices(k_all: np.ndarray, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed: int = 42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    indices_by_k = {}
    for k in np.unique(k_all):
        idxs = np.where(k_all == k)[0]
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_test = n - n_train - n_val
        train_idx = idxs[:n_train]
        val_idx = idxs[n_train:n_train + n_val]
        test_idx = idxs[n_train + n_val:]
        indices_by_k[int(k)] = (train_idx, val_idx, test_idx)

    train_all = np.concatenate([v[0] for v in indices_by_k.values()])
    val_all = np.concatenate([v[1] for v in indices_by_k.values()])
    test_all = np.concatenate([v[2] for v in indices_by_k.values()])
    rng.shuffle(train_all); rng.shuffle(val_all); rng.shuffle(test_all)
    return train_all.tolist(), val_all.tolist(), test_all.tolist(), indices_by_k


def load_dynamic_datasets(data_dir: str, split: str, total_points_list: list, packed: bool = True, use_chunks: bool = True):
    """
    Load multiple datasets for different total_points configurations
    
    Args:
        data_dir: Base directory containing train/val/test folders
        split: 'train', 'val', or 'test'
        total_points_list: List of total_points configurations to load
        packed: Whether adjacency matrices are packed
        use_chunks: Whether to use chunked adjacency files
    
    Returns:
        List of DynamicSet2GraphDataset objects
    """
    datasets = []
    split_dir = os.path.join(data_dir, split)
    
    for total_points in total_points_list:
        data_path = os.path.join(split_dir, f'synthetic_detector_data_{total_points}pts.npz')
        if os.path.exists(data_path):
            print(f"Loading {split} dataset with {total_points} points from {data_path}")
            dataset = DynamicSet2GraphDataset(
                data_npz_path=data_path,
                chunk_dir=split_dir,
                packed=packed,
                use_chunks=use_chunks
            )
            datasets.append(dataset)
            print(f"  Loaded {len(dataset)} events with {dataset.num_nodes} nodes")
        else:
            print(f"Warning: {data_path} not found, skipping {total_points} points configuration")
    
    return datasets


def build_secondary_splits_from_train(train_datasets, target_k: int = None, target_ks: list = None, total_events: int = 9000, seed: int = 42):
    """
    From the provided training datasets (multiple total_points configs), select a total of
    `total_events` samples with k == target_k across all configs, split them 50/50 into
    secondary validation and secondary test sets, and remove them from training.

    Returns:
        new_train_datasets, sec_val_datasets, sec_test_datasets
    """
    rng = np.random.default_rng(seed)

    # Build matcher for k values
    if target_ks is not None and len(target_ks) > 0:
        target_set = set(int(k) for k in target_ks)
        def match_fn(arr):
            return np.where(np.isin(arr, list(target_set)))[0]
    elif target_k is not None:
        def match_fn(arr):
            return np.where(arr == target_k)[0]
    else:
        return train_datasets, [], []

    # Collect all candidate indices across datasets
    candidates = []  # list of (ds_idx, idx_in_dataset_space)
    for ds_idx, dataset in enumerate(train_datasets):
        if isinstance(dataset, Subset):
            base_k_all = dataset.dataset.k_all
            if base_k_all is None:
                continue
            subset_base_indices = np.array(dataset.indices)
            if subset_base_indices.size == 0:
                continue
            base_matches = set(match_fn(base_k_all).tolist())
            mask = np.array([idx in base_matches for idx in subset_base_indices])
            subset_positions = np.nonzero(mask)[0]
            for pos in subset_positions.tolist():
                candidates.append((ds_idx, pos))
        else:
            k_all = dataset.k_all
            if k_all is None:
                continue
            idxs = match_fn(k_all)
            for idx in idxs.tolist():
                candidates.append((ds_idx, idx))

    if not candidates:
        # No candidates; return originals and empty secondary splits
        return train_datasets, [], []

    rng.shuffle(candidates)
    selected = candidates[:min(total_events, len(candidates))]

    # Split 50/50 into secondary val and test
    mid = len(selected) // 2
    selected_val = selected[:mid]
    selected_test = selected[mid:]

    # Organize by dataset index
    val_by_ds = {}
    test_by_ds = {}
    for ds_idx, idx in selected_val:
        val_by_ds.setdefault(ds_idx, []).append(idx)
    for ds_idx, idx in selected_test:
        test_by_ds.setdefault(ds_idx, []).append(idx)

    # Build new training subsets (with selected indices removed), and secondary subsets
    new_train_datasets = []
    sec_val_datasets = []
    sec_test_datasets = []

    for ds_idx, dataset in enumerate(train_datasets):
        total_points = dataset.total_points
        num_nodes = dataset.num_nodes

        ds_len = len(dataset)
        all_indices = np.arange(ds_len)

        # Selected indices for this dataset
        selected_val_indices = sorted(val_by_ds.get(ds_idx, []))
        selected_test_indices = sorted(test_by_ds.get(ds_idx, []))
        selected_all = set(selected_val_indices) | set(selected_test_indices)

        # Remaining for training
        remaining_indices = sorted([i for i in all_indices if i not in selected_all])

        # Create Subsets and carry over attributes used by MultiConfigDataLoader
        train_subset = Subset(dataset, remaining_indices)
        train_subset.total_points = total_points
        train_subset.num_nodes = num_nodes
        new_train_datasets.append(train_subset)

        val_subset = Subset(dataset, selected_val_indices)
        val_subset.total_points = total_points
        val_subset.num_nodes = num_nodes
        sec_val_datasets.append(val_subset)

        test_subset = Subset(dataset, selected_test_indices)
        test_subset.total_points = total_points
        test_subset.num_nodes = num_nodes
        sec_test_datasets.append(test_subset)

    return new_train_datasets, sec_val_datasets, sec_test_datasets


def main():
    parser = argparse.ArgumentParser(description='Train Set2Graph model on dynamic total points datasets')
    parser.add_argument('--data_dir', type=str, default='./synthetic_events', 
                       help='Directory containing train/val/test datasets')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=19, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='model_checkpoints_dynamic', 
                       help='Directory to save model checkpoints')
    parser.add_argument('--resume', type=int, help='Resume from epoch checkpoint')
    
    args = parser.parse_args()
    
    # Debug mode - set to True for quick testing
    debug = False

    # Model saving directory
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Fixed configurations
    train_points = [50, 100, 150, 200, 300, 400, 500]
    val_points = [600, 700, 800, 900, 1000]
    test_points = [600, 700, 800, 900, 1000]
    
    print("Loading dynamic total points datasets...")
    print(f"Training points: {train_points}")
    print(f"Validation points: {val_points}")
    print(f"Test points: {test_points}")

    # Load datasets
    train_datasets = load_dynamic_datasets(args.data_dir, 'train', train_points)
    val_datasets = load_dynamic_datasets(args.data_dir, 'val', val_points)
    test_datasets = load_dynamic_datasets(args.data_dir, 'test', test_points)
    
    # Quick check: print k distribution for each configuration to ensure expected k values (e.g., 3, 4, 5)
    print("K distribution check per split/configuration:")
    for ds_list, split_name in [(train_datasets, 'Train'), (val_datasets, 'Val'), (test_datasets, 'Test')]:
        for ds in ds_list:
            if isinstance(ds, Subset):
                if hasattr(ds, 'indices'):
                    k_all = ds.dataset.k_all[ds.indices] if ds.dataset.k_all is not None else None
                else:
                    k_all = ds.dataset.k_all
                total_points = getattr(ds, 'total_points', getattr(ds.dataset, 'total_points', 'unknown'))
            else:
                k_all = ds.k_all
                total_points = ds.total_points
            print(f"  {split_name} {total_points}pts:")
            print_k_distribution(k_all)

    if not train_datasets:
        raise ValueError("No training datasets found!")

    if debug:
        print('DEBUG MODE: Using only 16 events per configuration.')
        # Create debug datasets with only 16 events each
        debug_train_datasets = []
        for dataset in train_datasets:
            debug_dataset = Subset(dataset, range(min(32, len(dataset))))
            debug_dataset.total_points = dataset.total_points
            debug_dataset.num_nodes = dataset.num_nodes
            debug_train_datasets.append(debug_dataset)
        
        debug_val_datasets = []
        for dataset in val_datasets:
            debug_dataset = Subset(dataset, range(min(32, len(dataset))))
            debug_dataset.total_points = dataset.total_points
            debug_dataset.num_nodes = dataset.num_nodes
            debug_val_datasets.append(debug_dataset)
        
        debug_test_datasets = []
        for dataset in test_datasets:
            debug_dataset = Subset(dataset, range(min(32, len(dataset))))
            debug_dataset.total_points = dataset.total_points
            debug_dataset.num_nodes = dataset.num_nodes
            debug_test_datasets.append(debug_dataset)
        
        # Replace with debug datasets
        train_datasets = debug_train_datasets
        val_datasets = debug_val_datasets
        test_datasets = debug_test_datasets

    # Build secondary validation/test from training data (k in {3,4,5} across 50..500), and remove from training
    target_ks = [3, 4, 5]
    total_events_for_sec = 9000
    print(f"Creating secondary validation/test from training data (k in {target_ks}, total {total_events_for_sec} across 50..500)...")
    train_datasets, sec_val_datasets, sec_test_datasets = build_secondary_splits_from_train(
        train_datasets,
        target_ks=target_ks,
        total_events=total_events_for_sec,
        seed=42
    )

    # Create multi-config data loaders (after debug mode + secondary split modifications)
    train_loader = MultiConfigDataLoader(train_datasets, batch_size=args.batch_size, shuffle=True)
    val_loader = MultiConfigDataLoader(val_datasets, batch_size=args.batch_size, shuffle=False)
    test_loader = MultiConfigDataLoader(test_datasets, batch_size=args.batch_size, shuffle=False)
    sec_val_loader = MultiConfigDataLoader(sec_val_datasets, batch_size=args.batch_size, shuffle=False)
    sec_test_loader = MultiConfigDataLoader(sec_test_datasets, batch_size=args.batch_size, shuffle=False)

    # Print dataset sizes
    total_train_events = sum(len(dataset) for dataset in train_datasets)
    total_val_events = sum(len(dataset) for dataset in val_datasets)
    total_test_events = sum(len(dataset) for dataset in test_datasets)
    total_sec_val_events = sum(len(dataset) for dataset in sec_val_datasets)
    total_sec_test_events = sum(len(dataset) for dataset in sec_test_datasets)
    
    print(f"Training dataset: {total_train_events} events")
    print(f"Validation dataset: {total_val_events} events")
    print(f"Test dataset: {total_test_events} events")
    print(f"Secondary Val (from train k in {target_ks}, 50..500): {total_sec_val_events} events")
    print(f"Secondary Test(from train k in {target_ks}, 50..500): {total_sec_test_events} events")

    # Get input dimension from first training dataset
    if isinstance(train_datasets[0], Subset):
        input_dim = train_datasets[0].dataset.X_all.shape[2]
    else:
        input_dim = train_datasets[0].X_all.shape[2]
    print(f"Input dimension: {input_dim}")

    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create one model that can handle variable input sizes
    # The model will handle different graph sizes dynamically
    model = SumformerAdjacencyModel(
        input_dim=input_dim, 
        hidden_dim=256, 
        num_blocks=3, 
        num_nodes=1000  # Dummy value, will be overridden dynamically
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Add learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        min_lr=1e-6
    )
    
    # Add gradient clipping for stability
    max_grad_norm = 0.5
    
    # Add early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume is not None:
        checkpoint_path = os.path.join(SAVE_DIR, f'epoch_{args.resume}.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = args.resume
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")

    def adjacency_metrics(pred_adj, true_adj):
        pred_flat = pred_adj.flatten()
        true_flat = true_adj.flatten()
        f1 = f1_score(true_flat, pred_flat)
        pred_graph = csr_matrix(pred_adj)
        true_graph = csr_matrix(true_adj)
        _, pred_labels = connected_components(pred_graph, directed=False)
        _, true_labels = connected_components(true_graph, directed=False)
        ari = adjusted_rand_score(true_labels, pred_labels)
        ri = (ari + 1) / 2
        return f1, ri, ari

    def run_epoch(loader, train_mode: bool, interval_events: int = 200):
        if train_mode:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_f1 = 0.0
        total_ri = 0.0
        total_ari = 0.0
        n_graphs = 0
        interval_loss = 0.0
        interval_f1 = 0.0
        interval_ri = 0.0
        interval_ari = 0.0
        interval_count = 0
        
        # Track metrics per configuration for validation
        if not train_mode:
            config_metrics = {}
            config_counts = {}
        else:
            config_metrics = {}
            config_counts = {}

        # Iterate through batches from each configuration
        for batches in loader:
            # Process each configuration's batch
            # if not train_mode:
            #     print(f"Validation batch configs: {list(batches.keys())}")
            for total_points, (x, adj_true) in batches.items():
                x = x.to(device)
                adj_true = adj_true.to(device)
                
                batch_size, num_nodes, input_dim = x.shape
                
                # No padding needed! Each batch has graphs of the same size
                if train_mode:
                    optimizer.zero_grad()
                with torch.set_grad_enabled(train_mode):
                    adj_logits = model(x)
                    loss = criterion(adj_logits, adj_true)
                    if train_mode:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()

                # Move metrics accumulation INSIDE the loop for each configuration
                total_loss += loss.item() * batch_size
                interval_loss += loss.item() * batch_size

                with torch.no_grad():
                    preds_np = (torch.sigmoid(adj_logits) > 0.5).detach().cpu().numpy().astype(np.int32)
                    trues_np = adj_true.detach().cpu().numpy().astype(np.int32)
                    for pred_adj, true_adj in zip(preds_np, trues_np):
                        f1, ri, ari = adjacency_metrics(pred_adj, true_adj)
                        total_f1 += f1; total_ri += ri; total_ari += ari
                        interval_f1 += f1; interval_ri += ri; interval_ari += ari
                        n_graphs += 1; interval_count += 1
                        
                        # Track per-configuration metrics for validation
                        if not train_mode:
                            if total_points not in config_metrics:
                                config_metrics[total_points] = {'loss': 0.0, 'f1': 0.0, 'ri': 0.0, 'ari': 0.0}
                                config_counts[total_points] = 0
                            config_metrics[total_points]['loss'] += loss.item()
                            config_metrics[total_points]['f1'] += f1
                            config_metrics[total_points]['ri'] += ri
                            config_metrics[total_points]['ari'] += ari
                            config_counts[total_points] += 1

            if interval_count >= interval_events:
                avg_int_loss = interval_loss / interval_count
                avg_int_f1 = interval_f1 / interval_count
                avg_int_ri = interval_ri / interval_count
                avg_int_ari = interval_ari / interval_count
                split = 'Train' if train_mode else 'Val'
                print(f'{split} interval ({n_graphs} graphs): loss={avg_int_loss:.4f} | F1={avg_int_f1:.4f} | RI={avg_int_ri:.4f} | ARI={avg_int_ari:.4f}', flush=True)
                interval_loss = interval_f1 = interval_ri = interval_ari = 0.0
                interval_count = 0

        avg_loss = total_loss / max(n_graphs, 1)
        avg_f1 = total_f1 / max(n_graphs, 1)
        avg_ri = total_ri / max(n_graphs, 1)
        avg_ari = total_ari / max(n_graphs, 1)
        
        # Print per-configuration metrics for validation
        if not train_mode:
            print(f"DEBUG: train_mode={train_mode}, config_metrics keys={list(config_metrics.keys()) if 'config_metrics' in locals() else 'not defined'}")
            if 'config_metrics' in locals() and config_metrics:
                print("Validation per configuration:")
                for total_points in sorted(config_metrics.keys()):
                    count = config_counts[total_points]
                    metrics = config_metrics[total_points]
                    avg_config_loss = metrics['loss'] / count
                    avg_config_f1 = metrics['f1'] / count
                    avg_config_ri = metrics['ri'] / count
                    avg_config_ari = metrics['ari'] / count
                    print(f"  {total_points} points ({count} graphs): Loss={avg_config_loss:.4f} | F1={avg_config_f1:.4f} | RI={avg_config_ri:.4f} | ARI={avg_config_ari:.4f}")
            else:
                print("DEBUG: No config_metrics found")
        
        return avg_loss, avg_f1, avg_ri, avg_ari

    # Run training
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        tr_loss, tr_f1, tr_ri, tr_ari = run_epoch(train_loader, train_mode=True, interval_events=3000)
        va_loss, va_f1, va_ri, va_ari = run_epoch(val_loader, train_mode=False, interval_events=3000)
        sva_loss, sva_f1, sva_ri, sva_ari = run_epoch(sec_val_loader, train_mode=False, interval_events=3000)
        print(f'Epoch {epoch+1} | Train Loss: {tr_loss:.4f} | F1: {tr_f1:.4f} | RI: {tr_ri:.4f} | ARI: {tr_ari:.4f}', flush=True)
        print(f'           | Val   Loss: {va_loss:.4f} | F1: {va_f1:.4f} | RI: {va_ri:.4f} | ARI: {va_ari:.4f}', flush=True)
        print(f'           | SecVal Loss: {sva_loss:.4f} | F1: {sva_f1:.4f} | RI: {sva_ri:.4f} | ARI: {sva_ari:.4f}', flush=True)
        print(f'           | LR: {optimizer.param_groups[0]["lr"]:.2e}', flush=True)

        # Update scheduler
        scheduler.step(va_loss)

        # Save model after each epoch
        save_path = os.path.join(SAVE_DIR, f'epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': tr_loss,
            'val_loss': va_loss
        }, save_path)
        print(f"Saved model checkpoint to {save_path}", flush=True)

        # Save best model based on validation loss
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_model_path = os.path.join(SAVE_DIR, 'best_model.pt')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': tr_loss,
                'val_loss': va_loss,
                'best_val_loss': best_val_loss
            }, best_model_path)
            print(f"New best model saved! Val Loss: {va_loss:.4f}", flush=True)

    # Final test evaluation (standard test and secondary test from training k in {3,4,5})
    te_loss, te_f1, te_ri, te_ari = run_epoch(test_loader, train_mode=False, interval_events=10000)
    ste_loss, ste_f1, ste_ri, ste_ari = run_epoch(sec_test_loader, train_mode=False, interval_events=10000)
    print(f'Test    Loss: {te_loss:.4f} | F1: {te_f1:.4f} | RI: {te_ri:.4f} | ARI: {te_ari:.4f}', flush=True)
    print(f'SecTest Loss: {ste_loss:.4f} | F1: {ste_f1:.4f} | RI: {ste_ri:.4f} | ARI: {ste_ari:.4f}', flush=True)

    print("Training completed!")


if __name__ == '__main__':
    main()
