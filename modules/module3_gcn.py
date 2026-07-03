"""
Step 3.3: Module 3 - Spatiotemporal Graph Convolution (FIXED)
Goal: Propagate information between dynamically connected leads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

print("\n" + "="*60)
print("STEP 3.3: MODULE 3 - SPATIOTEMPORAL GRAPH CONVOLUTION (FIXED)")
print("="*60)

# ----------------------------------------------------------------------------
# SUB-MODULE 3.1: GRAPH CONVOLUTIONAL LAYER
# ----------------------------------------------------------------------------

class GraphConvolutionLayer(nn.Module):
    """
    Graph Convolutional Network layer
    Propagates information between connected leads
    """
    def __init__(self, in_features, out_features, use_bias=True):
        super(GraphConvolutionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Linear transformation
        self.W = nn.Linear(in_features, out_features, bias=use_bias)

        # Optional: Layer normalization for stability
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, H, A):
        """
        Args:
            H: node features, shape (B, 12, in_features)
            A: adjacency matrix, shape (B, 12, 12)
        Returns:
            H_next: updated node features, shape (B, 12, out_features)
        """
        # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
        D = torch.diag_embed(torch.sum(A, dim=-1) + 1e-8)  # (B, 12, 12)
        D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(torch.sum(A, dim=-1) + 1e-8))

        # Normalized adjacency
        A_norm = torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)  # (B, 12, 12)

        # Graph convolution: (B, 12, 12) @ (B, 12, in_features)
        H_agg = torch.bmm(A_norm, H)  # (B, 12, in_features)

        # Apply linear transformation
        H_next = self.W(H_agg)  # (B, 12, out_features)

        # Apply layer normalization
        H_next = self.layer_norm(H_next)

        # Apply activation
        H_next = F.relu(H_next)

        return H_next

    def get_adjacency_impact(self, H, A):
        """Compute how much each adjacency entry affects the output"""
        return torch.abs(A) * torch.norm(H, dim=-1, keepdim=True)

# ----------------------------------------------------------------------------
# SUB-MODULE 3.2: STACKED GCN (FIXED)
# ----------------------------------------------------------------------------

class SpatiotemporalGCN(nn.Module):
    """
    Stacked GCN with multiple layers
    Processes each time window separately with window-specific features
    """
    def __init__(self,
                 in_features=64,
                 hidden_dim=128,
                 out_features=32,
                 num_layers=3,
                 num_windows=5,
                 dropout=0.2):
        """
        Args:
            in_features: input feature dimension (from Module 1)
            hidden_dim: hidden dimension for GCN layers
            out_features: output feature dimension
            num_layers: number of GCN layers
            num_windows: number of time windows
            dropout: dropout rate
        """
        super(SpatiotemporalGCN, self).__init__()

        self.num_layers = num_layers
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.num_windows = num_windows

        # Create GCN layers
        self.gcn_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer: in_features -> hidden_dim
        self.gcn_layers.append(GraphConvolutionLayer(in_features, hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))

        # Hidden layers: hidden_dim -> hidden_dim
        for i in range(num_layers - 2):
            self.gcn_layers.append(GraphConvolutionLayer(hidden_dim, hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

        # Output layer: hidden_dim -> out_features
        if num_layers > 1:
            self.gcn_layers.append(GraphConvolutionLayer(hidden_dim, out_features))
            self.dropouts.append(nn.Dropout(dropout))
        else:
            self.gcn_layers.append(GraphConvolutionLayer(in_features, out_features))
            self.dropouts.append(nn.Dropout(dropout))

        # Residual connection if dimensions match
        self.use_residual = (in_features == out_features)
        if not self.use_residual:
            self.residual_proj = nn.Linear(in_features, out_features)

        # Store attention for interpretability
        self.layer_attention = []

        # NEW: Temporal feature projection for each window
        self.temporal_proj = nn.ModuleList([
            nn.Linear(in_features, in_features) for _ in range(num_windows)
        ])

    def forward(self, H, A_windows):
        """
        Args:
            H: node features, shape (B, 12, in_features)
               These are features from Module 1 (averaged across time)
            A_windows: list of adjacency matrices, each (B, 12, 12)
        Returns:
            H_final: updated features, shape (B, 12, out_features)
        """
        # Process each window separately with its own features
        window_outputs = []

        for t, (A, proj) in enumerate(zip(A_windows, self.temporal_proj)):
            # ✅ FIXED: Different features for each window
            # Project features to be window-specific
            H_t = proj(H)  # (B, 12, in_features) - different per window

            # Apply GCN layers
            for i, (gcn, dropout) in enumerate(zip(self.gcn_layers, self.dropouts)):
                # Graph convolution
                H_t = gcn(H_t, A)

                # Apply dropout
                H_t = dropout(H_t)

                # Store attention for interpretability (last layer)
                if i == len(self.gcn_layers) - 1:
                    self.layer_attention = gcn.get_adjacency_impact(H_t, A)

            # Residual connection
            if self.use_residual:
                H_t = H_t + H  # Add original features
            else:
                H_t = H_t + self.residual_proj(H)  # Project and add

            window_outputs.append(H_t)

        # Average across windows
        H_final = torch.mean(torch.stack(window_outputs, dim=0), dim=0)

        return H_final

    def get_layer_attention(self):
        """Return attention for interpretability"""
        return self.layer_attention

# ----------------------------------------------------------------------------
# COMPLETE MODULE 3 (FIXED)
# ----------------------------------------------------------------------------

class Module3_Complete(nn.Module):
    """
    Complete Module 3: Spatiotemporal Graph Convolution
    Wraps SpatiotemporalGCN for easier use
    """
    def __init__(self,
                 in_features=64,
                 hidden_dim=128,
                 out_features=32,
                 num_layers=3,
                 num_windows=5,
                 dropout=0.2):
        super(Module3_Complete, self).__init__()

        self.gcn = SpatiotemporalGCN(
            in_features=in_features,
            hidden_dim=hidden_dim,
            out_features=out_features,
            num_layers=num_layers,
            num_windows=num_windows,
            dropout=dropout
        )

    def forward(self, H, A_windows):
        """
        Args:
            H: node features, shape (B, 12, in_features)
            A_windows: list of adjacency matrices, each (B, 12, 12)
        Returns:
            H_final: updated features, shape (B, 12, out_features)
        """
        return self.gcn(H, A_windows)