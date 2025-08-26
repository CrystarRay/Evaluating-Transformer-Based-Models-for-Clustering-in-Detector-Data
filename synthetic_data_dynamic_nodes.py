import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap
import os

# --- Generate Detector Geometry ---
def generate_annular_stack(total_pts=3333,r_inner=90, r_outer=100, radial_step=5, z_step=2, n_layers=11):
    points = []
    z_vals = np.linspace(-z_step*(n_layers//2), z_step*(n_layers//2), n_layers)

    for z in z_vals:
        r = r_inner
        while r <= r_outer:
            n_phi = max(1, int(np.ceil(2 * np.pi * r / radial_step)))
            for j in range(n_phi):
                theta = j * 2 * np.pi / n_phi
                x, y = r * np.cos(theta), r * np.sin(theta)
                points.append([x, y, z])
            r += radial_step

    points = np.array(points)
    if len(points) > total_pts:
        np.random.seed(0)
        points = points[np.random.choice(len(points), total_pts, replace=False)]
    return points

# Configuration for different datasets
R_INNER = 70.0
R_OUTER = 120.0
RADIAL_STEP = 5.0
Z_STEP = 3.0
N_LAYERS = 10

# Dataset configurations
TRAINING_CONFIG = {
    'total_points_list': [50, 100, 150, 200, 300, 400, 500],
    'events_per_config': 10000,
    'k_range': [3, 4, 5]
}

VALIDATION_CONFIG = {
    'total_points_list': [600, 700, 800, 900, 1000],
    'events_per_config': 1800,
    'k_range': [3, 4, 5]
}

TEST_CONFIG = {
    'total_points_list': [600, 700, 800, 900, 1000],
    'events_per_config': 1800,
    'k_range': [3, 4, 5]
}

def sample_vertical_centers(detector_xyz, k, min_distance=90.0, max_attempts=100000):
    """Pick *k* distinct detector nodes as Gaussian centres.

    A candidate is accepted only if it is at least *min_distance* away from
    every previously accepted centre.  This guarantees the returned centres
    all lie *on* the detector stack geometry.
    """
    centres = []
    attempts = 0
    while len(centres) < k and attempts < max_attempts:
        candidate = detector_xyz[np.random.randint(0, len(detector_xyz))]
        if all(np.linalg.norm(candidate - c) >= min_distance for c in centres):
            centres.append(candidate)
        attempts += 1
    if len(centres) < k:
        raise RuntimeError(f"Could not sample {k} well-separated centres after {max_attempts} attempts")
    return np.array(centres)

def covariance_to_six(cov):
    return np.array([
        cov[0, 0], cov[0, 1], cov[0, 2], cov[1, 1], cov[1, 2], cov[2, 2]
    ], dtype=np.float32)

def deterministic_rotation_matrix(center, scales, cluster_idx):
    """
    Generate a deterministic rotation matrix based on cluster properties.
    This makes covariance prediction more feasible while maintaining variability.
    
    Parameters:
    - center: cluster center coordinates (3,)
    - scales: diagonal scales for the covariance matrix (3,)
    - cluster_idx: index of the cluster (for additional variability)
    
    Returns:
    - Q: rotation matrix (3, 3)
    """
    # Use center coordinates and scales to create a deterministic but varied rotation
    # Normalize center coordinates to create rotation angles
    center_norm = center / (np.linalg.norm(center) + 1e-8)
    
    # Create rotation angles based on center position and cluster index
    theta_x = np.arctan2(center_norm[1], center_norm[2]) * 0.5  # Reduced amplitude
    theta_y = np.arctan2(center_norm[0], center_norm[2]) * 0.5
    theta_z = np.arctan2(center_norm[0], center_norm[1]) * 0.5
    
    # Add small cluster-specific variation
    cluster_factor = (cluster_idx + 1) * 0.1
    theta_x += cluster_factor * np.sin(cluster_idx)
    theta_y += cluster_factor * np.cos(cluster_idx)
    theta_z += cluster_factor * np.sin(cluster_idx * 2)
    
    # Create rotation matrices around each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations: R = Rz @ Ry @ Rx
    Q = Rz @ Ry @ Rx
    
    # Ensure Q is orthogonal (numerical stability)
    U, _, Vt = np.linalg.svd(Q)
    Q = U @ Vt
    
    return Q

def generate_gaussian_event_per_cluster(detector_xyz, k=None, event_idx=None):
    if k is None:
        k = np.random.randint(4, 8)
    num_points = detector_xyz.shape[0]
    centers = sample_vertical_centers(detector_xyz, k=k, min_distance=60.0)

    # Choose ONE set of scales for the whole event (same scales for all clusters)
    scales = np.array([12, 115, 3])

    if event_idx is not None and event_idx in [1000, 1250, 1500, 3000, 3250, 3500, 5000, 5250, 5500, 7000, 7250, 7500, 9000, 9250, 9500]:
        print(f"Event {event_idx}: axis scales used for covariance: {scales}")
        print(f"Event {event_idx}: Using deterministic rotation matrices for predictable covariance")
    
    # Create k unique rotation matrices for k unique covariances
    covariances = []
    cluster_summary_label = np.zeros((k, 9), dtype=np.float32)  # 9 = centre(3) + cov(6)
    
    for i in range(k):
        # Create unique rotation matrix for each cluster
        R_i = deterministic_rotation_matrix(centers[i], scales, i)
        cov_i = R_i @ np.diag(scales) @ R_i.T  # (3,3)
        covariances.append(cov_i)
        
        # Fill the summary label (centres first, unique covariance for each cluster)
        six_vec_cov = covariance_to_six(cov_i)
        cluster_summary_label[i, 0:3] = centers[i]
        cluster_summary_label[i, 3:9] = six_vec_cov

    # Generate energy depositions using unique covariances
    per_cluster_energy = np.zeros((k, num_points), dtype=np.float32)
    for i in range(k):
        inv_cov_i = np.linalg.inv(covariances[i])
        diff = detector_xyz - centers[i]
        exponent = np.einsum('ij,jk,ik->i', diff, inv_cov_i, diff)
        per_cluster_energy[i] = -0.5 * exponent
    
    # Create simple covariance-aware input features
    # Energy statistics per node
    sum_energy_per_node = np.sum(per_cluster_energy, axis=0)  # [num_points]
    mean_energy_per_node = np.mean(per_cluster_energy, axis=0)  # [num_points]
    std_energy_per_node = np.std(per_cluster_energy, axis=0)  # [num_points]
    
    # Combine simple features
    covariance_features = np.column_stack([
        sum_energy_per_node,           # [1] - sum energy per node
        mean_energy_per_node,          # [1] - mean energy per node  
        std_energy_per_node,           # [1] - energy std per node
    ])  # Shape: [num_points, 3]
    
    # Create final input features with simple covariance features
    # Original features: [node_distance_sum, node_distance_mean, x, y, z] = 5 features
    # Add simple covariance features: 3 features
    # Total: 8 features
    
    # Original features (from existing code)
    diffs = detector_xyz[None, :, :] - centers[:, None, :]  # shape: (k, num_points, 3)
    dists = np.linalg.norm(diffs, axis=2)  # shape: (k, num_points)
    node_distance_sum = np.sum(dists, axis=0)[:, None]    # shape: (num_points, 1)
    node_distance_mean = np.mean(dists, axis=0)[:, None]  # shape: (num_points, 1)
    
    # Combine original + simple covariance features
    input_features_mod = np.concatenate([
        node_distance_sum,                 # [1] - node total distance from all clusters
        node_distance_mean,                # [1] - node average distance per cluster  
        covariance_features,               # [3] - simple covariance features
        detector_xyz.astype(np.float32),   # [3] - 3D coordinates
    ], axis=1).astype(np.float32)  # Total: 8 features
    
    # Create per-node covariance labels (num_points, 6) - each node gets its cluster's covariance
    # Pre-compute cluster assignments once
    cluster_assignments = np.argmax(per_cluster_energy, axis=0)  # [num_points]
    
    # Pre-compute all inverse covariances
    inv_covariances = [np.linalg.inv(cov) for cov in covariances]
    
    # Initialize per-node covariance labels
    per_node_inv_cov_upper = np.zeros((num_points, 6), dtype=np.float32)
    
    # Vectorized assignment using advanced indexing
    for cluster_idx in range(k):
        # Find nodes assigned to this cluster
        cluster_mask = cluster_assignments == cluster_idx
        if np.any(cluster_mask):
            # Get the inverse covariance for this cluster
            inv_cov_cluster = inv_covariances[cluster_idx]
            
            # Extract upper triangular part of inv_cov_cluster (6 values)
            inv_cov_upper = np.array([
                inv_cov_cluster[0, 0], inv_cov_cluster[0, 1], inv_cov_cluster[0, 2],  # First row
                inv_cov_cluster[1, 1], inv_cov_cluster[1, 2], inv_cov_cluster[2, 2]   # Second and third rows
            ], dtype=np.float32)
            
            # Assign to all nodes in this cluster at once
            per_node_inv_cov_upper[cluster_mask] = inv_cov_upper
    
    # Also keep the event-level inv_cov_upper for backward compatibility (use first cluster's covariance)
    inv_cov_upper = per_node_inv_cov_upper[0]  # Use first node's covariance as event-level
    
    return per_cluster_energy, centers, covariances, cluster_summary_label, k, input_features_mod, inv_cov_upper, per_node_inv_cov_upper

def generate_balanced_k_distribution(k_range, events_per_k, shuffle=True, random_seed=42):
    """
    Generate a balanced distribution of k values for dataset generation.
    
    Parameters:
    - k_range: list of k values to include
    - events_per_k: number of events per k value
    - shuffle: whether to shuffle the order (default: True)
    - random_seed: random seed for reproducibility (default: 42)
    
    Returns:
    - list of k values with balanced distribution
    """
    k_values_balanced = []
    for k in k_range:
        k_values_balanced.extend([k] * events_per_k)
    
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(k_values_balanced)
    
    return k_values_balanced

def generate_dataset(config, dataset_name, save_dir):
    """
    Generate a complete dataset with dynamic total points.
    
    Parameters:
    - config: dictionary with 'total_points_list', 'events_per_config', 'k_range'
    - dataset_name: name of the dataset (train/val/test)
    - save_dir: directory to save the dataset
    """
    print(f"\n{'='*60}")
    print(f"GENERATING {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    total_points_list = config['total_points_list']
    events_per_config = config['events_per_config']
    k_range = config['k_range']
    
    print(f"Total points configurations: {total_points_list}")
    print(f"Events per configuration: {events_per_config}")
    print(f"K range: {k_range}")
    
    # Create dataset-specific save directory
    dataset_save_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_save_dir, exist_ok=True)
    
    # Generate balanced k distribution with exactly `events_per_config` items
    # Evenly distribute counts across k_range, then distribute any remainder, and shuffle
    num_k = len(k_range)
    base_per_k = events_per_config // num_k
    remainder = events_per_config % num_k
    k_values_balanced = []
    for i, k in enumerate(k_range):
        count = base_per_k + (1 if i < remainder else 0)
        k_values_balanced.extend([k] * count)
    # Mixed ordering
    np.random.shuffle(k_values_balanced)
    
    # Store all data for this dataset
    all_data = {}
    
    for total_points in total_points_list:
        print(f"\n--- Generating data for {total_points} total points ---")
        
        # Generate detector geometry for this total_points
        detector_xyz = generate_annular_stack(
            total_pts=total_points,
            r_inner=R_INNER,
            r_outer=R_OUTER,
            radial_step=RADIAL_STEP,
            z_step=Z_STEP,
            n_layers=N_LAYERS
        )
        print(f"Detector XYZ shape: {detector_xyz.shape}")
        
        # Initialize data containers for this configuration
        X_all_mod = []
        y_all = []
        m_all = []
        k_all = []
        node_centres_all = []
        active_flags_all = []
        node_labels_all = []
        cluster_sizes_all = []
        per_cluster_energy_all = []
        inv_cov_upper_all = []
        per_node_inv_cov_upper_all = []
        
        # Adjacency chunking config
        ADJ_CHUNK_SIZE = 1000
        adj_chunk_list = []
        adj_chunk_paths = []
        BUILD_SET2GRAPH = True
        USE_GPU_FOR_ADJ = True
        PACK_ADJACENCY = True
        
        # Generate events for this total_points configuration
        for event_idx in range(events_per_config):
            # Use pre-determined k value
            k_fixed = k_values_balanced[event_idx]
            
            per_cluster_energy, centers, covariances, cluster_summary_label, k, input_features_mod, inv_cov_upper, per_node_inv_cov_upper = generate_gaussian_event_per_cluster(
                detector_xyz, k=k_fixed, event_idx=event_idx
            )
            
            # Print progress every 1000 events
            if (event_idx + 1) % 1000 == 0:
                current_k_counts = dict(zip(*np.unique(k_all + [k], return_counts=True)))
                print(f"Progress: {event_idx + 1}/{events_per_config} events | Current k distribution: {current_k_counts}")
            
            # Process event data
            raw_cluster_ids = np.argmax(per_cluster_energy, axis=0)
            max_energy = np.max(per_cluster_energy, axis=0)
            energy_threshold = np.median(max_energy)
            active_flags = (max_energy >= energy_threshold).astype(np.int8)
            
            cluster_ids = raw_cluster_ids.copy()
            cluster_ids[active_flags == 0] = -1
            
            # Build adjacency matrix
            labels_for_edges = raw_cluster_ids.copy()
            labels_for_edges[active_flags == 0] = -1
            
            adj_uint8 = None
            if BUILD_SET2GRAPH and USE_GPU_FOR_ADJ:
                try:
                    import cupy as cp
                    lab = cp.asarray(labels_for_edges)
                    adj_cp = (lab[:, None] >= 0) & (lab[:, None] == lab[None, :])
                    cp.fill_diagonal(adj_cp, False)
                    adj_uint8 = cp.asarray(adj_cp, dtype=cp.uint8).get()
                except Exception:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            device = torch.device('cuda')
                            lab_t = torch.from_numpy(labels_for_edges).to(device)
                            adj_t = (lab_t[:, None] >= 0) & (lab_t[:, None] == lab_t[None, :])
                            idx = torch.arange(adj_t.size(0), device=device)
                            adj_t[idx, idx] = False
                            adj_uint8 = adj_t.to(torch.uint8).cpu().numpy()
                    except Exception:
                        adj_uint8 = None
            
            if adj_uint8 is None:
                adj_np = (labels_for_edges[:, None] >= 0) & (labels_for_edges[:, None] == labels_for_edges[None, :])
                np.fill_diagonal(adj_np, False)
                adj_uint8 = adj_np.astype(np.uint8)
            
            if PACK_ADJACENCY:
                adj_packed = np.packbits(adj_uint8, axis=1)
                adj_chunk_list.append(adj_packed)
            else:
                adj_chunk_list.append(adj_uint8)
            
            # Save chunk periodically
            if len(adj_chunk_list) == ADJ_CHUNK_SIZE or event_idx == events_per_config - 1:
                adj_chunk = np.stack(adj_chunk_list)
                chunk_id = len(adj_chunk_paths)
                chunk_path = os.path.join(dataset_save_dir, f"adj_chunk_{total_points}_{chunk_id}.npy")
                np.save(chunk_path, adj_chunk)
                adj_chunk_paths.append(chunk_path)
                adj_chunk_list = []
            
            # Calculate cluster statistics
            cluster_counts = np.zeros(k, dtype=np.int32)
            cl_total = np.zeros(k, dtype=np.float32)
            cl_mean = np.zeros(k, dtype=np.float32)
            cl_var = np.zeros(k, dtype=np.float32)
            
            for i in range(k):
                node_indices = np.where(raw_cluster_ids == i)[0]
                count_i = len(node_indices)
                cluster_counts[i] = count_i
                if count_i == 0:
                    continue
                energies = per_cluster_energy[i, node_indices]
                total_val = energies.sum()
                mean_val = energies.mean()
                var_val = energies.var()
                
                cl_total[i] = total_val
                cl_mean[i] = mean_val
                cl_var[i] = var_val
            
            # Store data
            X_all_mod.append(input_features_mod)
            y_all.append(cluster_summary_label)
            m_all.append(np.array(cluster_counts, dtype=np.float32) / total_points)
            k_all.append(k)
            node_centres_all.append(centers[raw_cluster_ids].astype(np.float32))
            active_flags_all.append(active_flags)
            node_labels_all.append(raw_cluster_ids.astype(np.int8))
            cluster_sizes_all.append(cluster_counts.copy())
            per_cluster_energy_all.append(per_cluster_energy)
            inv_cov_upper_all.append(inv_cov_upper)
            per_node_inv_cov_upper_all.append(per_node_inv_cov_upper)
        
        # Stack all data for this total_points configuration
        X_all_mod = np.stack(X_all_mod)
        node_centres_all = np.stack(node_centres_all)
        active_flags_all = np.stack(active_flags_all)
        node_labels_all = np.stack(node_labels_all)
        inv_cov_upper_all = np.stack(inv_cov_upper_all)
        per_node_inv_cov_upper_all = np.stack(per_node_inv_cov_upper_all)
        
        # Pad y_all and m_all to maximum k
        max_k = max(k_all)
        y_padded = np.zeros((events_per_config, max_k, 9), dtype=np.float32)
        m_padded = np.zeros((events_per_config, max_k), dtype=np.float32)
        for i, (y, mvec) in enumerate(zip(y_all, m_all)):
            y_padded[i, :y.shape[0], :] = y
            m_padded[i, :len(mvec)] = mvec
        
        # Pad cluster sizes
        cluster_sizes_padded = np.zeros((events_per_config, max_k), dtype=np.int32)
        for i, sizes in enumerate(cluster_sizes_all):
            cluster_sizes_padded[i, :len(sizes)] = sizes
        
        # Store data for this total_points configuration
        config_data = {
            'X_all_mod': X_all_mod,
            'y_all': y_padded,
            'm_all': m_padded,
            'k_all': np.array(k_all),
            'node_centres': node_centres_all,
            'node_labels': node_labels_all,
            'active_flags': active_flags_all,
            'energy_to_centers': np.array(per_cluster_energy_all, dtype=object),
            'inv_cov_upper': inv_cov_upper_all,
            'per_node_inv_cov_upper': per_node_inv_cov_upper_all,
            'cluster_sizes': cluster_sizes_padded,
            'adj_chunk_paths': np.array(adj_chunk_paths, dtype=object),
            'adj_is_packed': np.array(True) if PACK_ADJACENCY else np.array(False),
            'detector_xyz': detector_xyz
        }
        
        all_data[total_points] = config_data
        
        # Save individual configuration data
        config_filename = f'synthetic_detector_data_{total_points}pts.npz'
        config_path = os.path.join(dataset_save_dir, config_filename)
        np.savez(config_path, **config_data)
        print(f"Saved {total_points} points configuration to: {config_path}")
        
        # Combine adjacency chunks for this configuration
        if len(adj_chunk_paths) > 0:
            loaded_chunks = [np.load(p, mmap_mode='r') for p in adj_chunk_paths]
            adj_full = np.concatenate(loaded_chunks, axis=0)
            final_adj_path = os.path.join(dataset_save_dir, 
                                        f'adj_matrix_packed_{total_points}pts.npy' if PACK_ADJACENCY 
                                        else f'adj_matrix_uint8_{total_points}pts.npy')
            np.save(final_adj_path, adj_full)
            
            # Remove chunk files
            for p in adj_chunk_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
            print(f"Adjacency combined and saved to: {final_adj_path}")
        
        # Print summary for this configuration
        k_final_counts = dict(zip(*np.unique(k_all, return_counts=True)))
        print(f"Configuration summary for {total_points} points:")
        print(f"  K distribution: {k_final_counts}")
        print(f"  Total events: {len(k_all)}")
        print(f"  Input features shape: {X_all_mod.shape}")
        print(f"  Active flags shape: {active_flags_all.shape}")
    
    # Save combined dataset metadata
    metadata = {
        'total_points_list': total_points_list,
        'events_per_config': events_per_config,
        'k_range': k_range,
        'dataset_name': dataset_name
    }
    
    metadata_path = os.path.join(dataset_save_dir, 'dataset_metadata.npz')
    np.savez(metadata_path, **metadata)
    print(f"\nSaved dataset metadata to: {metadata_path}")
    
    return all_data

if __name__ == "__main__":
    # Ensure save directory exists for outputs
    save_dir = './synthetic_events'
    os.makedirs(save_dir, exist_ok=True)
    
    print("Starting dataset generation with dynamic total points...")
    print(f"Training config: {TRAINING_CONFIG}")
    print(f"Validation config: {VALIDATION_CONFIG}")
    print(f"Test config: {TEST_CONFIG}")
    
    # Generate training, validation, and test datasets
    print("\n" + "="*80)
    print("GENERATING TRAINING DATASET")
    print("="*80)
    training_data = generate_dataset(TRAINING_CONFIG, 'train', save_dir)
    
    print("\n" + "="*80)
    print("GENERATING VALIDATION DATASET")
    print("="*80)
    validation_data = generate_dataset(VALIDATION_CONFIG, 'val', save_dir)
    
    print("\n" + "="*80)
    print("GENERATING TEST DATASET")
    print("="*80)
    test_data = generate_dataset(TEST_CONFIG, 'test', save_dir)
    
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE")
    print("="*80)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print("-" * 40)
    
    # Training summary
    total_training_events = sum(len(data['k_all']) for data in training_data.values())
    print(f"Training: {total_training_events} total events")
    for total_points, data in training_data.items():
        print(f"  {total_points} points: {len(data['k_all'])} events")
    
    # Validation summary
    total_validation_events = sum(len(data['k_all']) for data in validation_data.values())
    print(f"Validation: {total_validation_events} total events")
    for total_points, data in validation_data.items():
        print(f"  {total_points} points: {len(data['k_all'])} events")
    
    # Test summary
    total_test_events = sum(len(data['k_all']) for data in test_data.values())
    print(f"Test: {total_test_events} total events")
    for total_points, data in test_data.items():
        print(f"  {total_points} points: {len(data['k_all'])} events")
    
    print(f"\nTotal events generated: {total_training_events + total_validation_events + total_test_events}")
    print(f"Data saved to: {save_dir}")
    
    # Create a simple visualization for the first event of each configuration
    print("\nCreating sample visualizations...")
    
    # Plot sample detector geometries
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, total_points in enumerate(TRAINING_CONFIG['total_points_list'][:8]):
        if idx >= len(axes):
            break
            
        detector_xyz = generate_annular_stack(
            total_pts=total_points,
            r_inner=R_INNER,
            r_outer=R_OUTER,
            radial_step=RADIAL_STEP,
            z_step=Z_STEP,
            n_layers=N_LAYERS
        )
        
        ax = axes[idx]
        ax.scatter(detector_xyz[:, 0], detector_xyz[:, 1], c=detector_xyz[:, 2], s=4, cmap='viridis')
        ax.set_title(f'{total_points} points')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-120, 120)
        ax.set_ylim(-120, 120)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_detector_geometries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Sample visualizations saved!")
    print(f"All data saved to: {save_dir}")
