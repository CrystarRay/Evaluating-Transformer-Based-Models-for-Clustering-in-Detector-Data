import torch
import torch.nn as nn
from sumformer import Sumformer

class SumformerAdjacencyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, num_nodes):
        super().__init__()
        # Separate heads: positional encoding and input projection (excl. xyz)
        if input_dim < 4:
            raise ValueError("Expected at least 4 input dims (>=1 feature + 3D coords)")
        self.num_raw_features = input_dim - 3

        # Project non-xyz features -> hidden_dim
        self.feature_projection = nn.Linear(self.num_raw_features, hidden_dim)

        # Positional encoding from xyz -> hidden_dim (small MLP)
        self.positional_encoding = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Sumformer now operates on combined representation (hidden_dim)
        self.sumformer = Sumformer(
            num_blocks=num_blocks,
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            key_dim=hidden_dim,  # Can be tuned
        )
        
        # Projection to map Sumformer output to hidden_dim (kept explicit)
        self.node_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Proper pairwise adjacency head (like Set2Graph) with regularization
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # [u_i, u_j]
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(hidden_dim // 2, 1)  # single edge score
        )
        self.num_nodes = num_nodes

    def forward(self, x):
        """Forward pass with proper pairwise edge prediction.

        Parameters
        ----------
        x : Tensor | Batch
            • If ``Tensor`` with shape ``[num_nodes, input_dim]``  — single event (back-compat).  
            • If ``Tensor`` with shape ``[batch_size, num_nodes, input_dim]`` — mini-batch of events.  
            • If a ``torch_heterogeneous_batching.batch.Batch``  — already batched (any batch size).

        Returns
        -------
        Tensor
            Adjacency logits with shape  
            • ``[num_nodes, num_nodes]`` for a single event  
            • ``[batch_size, num_nodes, num_nodes]`` for a mini-batch.
        """

        # ------------------------------------------------------------------
        # 1. Wrap the input into a Batch object that Sumformer can consume.
        # ------------------------------------------------------------------
        from torch_heterogeneous_batching.batch import Batch  # local import to avoid circular deps

        # Determine whether we already have a Batch or need to build one
        if isinstance(x, Batch):
            batch_obj = x
            batch_size = batch_obj.batch_size
        else:
            if x.ndim == 2:
                # Single event, split features and xyz, then combine
                feats = x[:, :self.num_raw_features]
                xyz = x[:, -3:]
                proj = self.feature_projection(feats) + self.positional_encoding(xyz)
                batch_obj = Batch.from_unbatched(proj)
                batch_size = 1
            elif x.ndim == 3:
                # Batched tensor, split features and xyz, then combine
                bsz, num_nodes, _ = x.shape
                feats = x[:, :, :self.num_raw_features]
                xyz = x[:, :, -3:]
                proj = self.feature_projection(feats) + self.positional_encoding(xyz)  # [B, N, hidden_dim]
                # Convert to list[T] -> Batch for variable future generality
                batch_list = [proj[i] for i in range(bsz)]
                batch_obj = Batch.from_list(batch_list, order=1)
                batch_size = bsz
            else:
                raise ValueError(
                    "Unsupported input shape for SumformerAdjacencyModel: "
                    f"{tuple(x.shape)}. Expected 2D or 3D tensor or Batch object."
                )

        # ------------------------------------------------------------------
        # 2. Run Sumformer to obtain node embeddings.
        # ------------------------------------------------------------------
        node_features_batch = self.sumformer(batch_obj)

        # Sumformer may return (out, witness) — only keep the first part
        if isinstance(node_features_batch, tuple):
            node_features_batch = node_features_batch[0]

        # Extract the features tensor
        if hasattr(node_features_batch, "data"):
            node_features_flat = node_features_batch.data  # [total_nodes, input_dim]
        else:
            node_features_flat = node_features_batch

        # ------------------------------------------------------------------
        # 3. Project to hidden_dim and reshape for pairwise operations
        # ------------------------------------------------------------------
        # Project from input_dim to hidden_dim
        node_embeddings_flat = self.node_projection(node_features_flat)  # [total_nodes, hidden_dim]
        
        if batch_size == 1:
            node_embeddings = node_embeddings_flat.unsqueeze(0)  # [1, num_nodes, hidden_dim]
        else:
            # Since our MultiConfigDataLoader ensures all graphs in a batch have the same size,
            # we can simply calculate the number of nodes per graph
            num_nodes_per_graph = node_embeddings_flat.shape[0] // batch_size
            node_embeddings = node_embeddings_flat.view(batch_size, num_nodes_per_graph, -1)

        # ------------------------------------------------------------------
        # 4. Create pairwise features [u_i, u_j] for all node pairs
        # ------------------------------------------------------------------
        B, N, D = node_embeddings.shape
        
        # Create pairwise features [u_i, u_j] for all pairs
        u_i = node_embeddings.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
        u_j = node_embeddings.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
        pairwise = torch.cat([u_i, u_j], dim=-1)  # [B, N, N, 2*D]
        
        # ------------------------------------------------------------------
        # 5. Predict edge scores for all pairs
        # ------------------------------------------------------------------
        edge_logits = self.edge_mlp(pairwise).squeeze(-1)  # [B, N, N]

        # Always return [B, N, N] to match training targets
        return edge_logits

# Example usage:
if __name__ == "__main__":
    num_nodes = 100  # Smaller for testing
    input_dim = 10  # e.g., condition_bits + cluster_feat + xyz, or as appropriate
    hidden_dim = 256
    num_blocks = 3

    model = SumformerAdjacencyModel(input_dim, hidden_dim, num_blocks, num_nodes)
    
    # Test single event
    x_single = torch.randn(num_nodes, input_dim)
    adj_logits_single = model(x_single)  # [num_nodes, num_nodes]
    print("Single event adjacency logits shape:", adj_logits_single.shape)
    
    # Test batch
    batch_size = 4
    x_batch = torch.randn(batch_size, num_nodes, input_dim)
    adj_logits_batch = model(x_batch)  # [batch_size, num_nodes, num_nodes]
    print("Batch adjacency logits shape:", adj_logits_batch.shape)
    
    # Test that predictions are reasonable
    adj_probs = torch.sigmoid(adj_logits_batch)
    print("Adjacency probabilities range:", adj_probs.min().item(), "to", adj_probs.max().item())