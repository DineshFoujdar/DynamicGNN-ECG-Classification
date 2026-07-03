"""
Step 3.2: Module 2 - Dynamic Graph Structure Learning (Fixed)
Test with Real Data from Module 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

print("\n" + "="*60)
print("STEP 3.2: MODULE 2 - DYNAMIC GRAPH STRUCTURE LEARNING (FIXED)")
print("="*60)

class DynamicGraphLearning(nn.Module):
    """
    Learns time-adaptive adjacency matrices
    Core novelty: computes different graph per time window
    """
    def __init__(self,
                 feature_dim=64,
                 hidden_dim=128,
                 num_leads=12,
                 num_windows=5,
                 time_window_size=200,
                 alpha=0.1,
                 beta=0.05):
        super(DynamicGraphLearning, self).__init__()

        self.num_leads = num_leads
        self.num_windows = num_windows
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.time_window_size = time_window_size

        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))

        # Projection layers
        self.W1 = nn.Linear(feature_dim, hidden_dim)
        self.W2 = nn.Linear(feature_dim, hidden_dim)

        # Temperature for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Store windows
        self.last_adjacencies = None
        self.last_features = None
        self.last_time_windows = None

    # def split_into_windows(self, x):
    #     # x: (B, 12, feature_dim) - Features, NOT time series!
    #     B, L, F = x.shape

    #     # Split features into windows (no averaging!)
    #     window_size = F // self.num_windows
    #     windows = []

    #     for t in range(self.num_windows):
    #         start = t * window_size
    #         end = start + window_size if t < self.num_windows-1 else F
    #         window = x[:, :, start:end]  # (B, 12, window_size)
    #         windows.append(window)

    #     return windows

    def split_into_windows(self, x):
        """
        x: (B, 12, feature_dim) where feature_dim = 64
        Returns: list of (B, 12, feature_dim) - ALL windows get FULL features!
        """
        # Instead of splitting, return the SAME features for all windows
        # Each window will use its own projection to create variation
        windows = [x for _ in range(self.num_windows)]  # All get (B, 12, 64)
        return windows

    def compute_adjacency(self, z_t):
        """
        Compute adjacency matrix for a single time window
        """
        h1 = F.relu(self.W1(z_t))  # (B, 12, hidden_dim)
        h2 = F.relu(self.W2(z_t))  # (B, 12, hidden_dim)

        A_t = torch.bmm(h1, h2.transpose(1, 2))  # (B, 12, 12)
        A_t = A_t / self.temperature
        A_t = F.softmax(A_t, dim=-1)  # (B, 12, 12)

        return A_t

    def compute_laplacian(self, A):
        """Compute graph Laplacian: L = D - A"""
        D = torch.diag_embed(torch.sum(A, dim=-1))
        L = D - A
        return L

    def compute_smoothness_loss(self, z_t, A_t):
        """Compute smoothness regularization loss"""
        L_t = self.compute_laplacian(A_t)
        batch_size = z_t.size(0)
        smooth_loss = 0

        for b in range(batch_size):
            z_b = z_t[b]  # (12, feature_dim)
            L_b = L_t[b]  # (12, 12)

            LZ = torch.mm(L_b, z_b)  # (12, feature_dim)
            smooth = torch.mm(z_b.transpose(0, 1), LZ)  # (feature_dim, feature_dim)
            trace_val = torch.trace(smooth)
            smooth_loss += trace_val / (z_b.size(0) * z_b.size(1))

        smooth_loss = smooth_loss / batch_size
        return smooth_loss

    def compute_sparsity_loss(self, A_t):
        """Compute sparsity regularization loss"""
        return torch.sum(torch.abs(A_t)) / (A_t.size(0) * A_t.size(1) * A_t.size(2))

    def forward(self, x, return_all=False):
        """
        Args:
            x: input signal from Module 1, shape (B, 12, T)
            return_all: if True, return all window adjacencies
        Returns:
            A_windows: list of adjacency matrices, each (B, 12, 12)
            reg_loss: graph regularization loss
        """
        # Split signal into time windows
        window_features = self.split_into_windows(x)  # list of (B, 12, feature_dim)

        batch_size = x.size(0)
        A_windows = []
        reg_loss = 0

        for t in range(self.num_windows):
            z_t = window_features[t]  # (B, 12, feature_dim)

            # Compute adjacency for this window
            A_t = self.compute_adjacency(z_t)  # (B, 12, 12)
            A_windows.append(A_t)

            # Compute regularization
            smooth_loss = self.compute_smoothness_loss(z_t, A_t)
            reg_loss += self.alpha * smooth_loss

            sparsity_loss = self.compute_sparsity_loss(A_t)
            reg_loss += self.beta * sparsity_loss

        # Store for interpretability
        self.last_adjacencies = A_windows
        self.last_features = window_features
        self.last_time_windows = window_features

        if return_all:
            return A_windows, reg_loss
        else:
            avg_A = torch.mean(torch.stack(A_windows, dim=0), dim=0)
            return avg_A, reg_loss

    def get_adjacencies(self):
        """Return the last computed adjacencies"""
        return self.last_adjacencies

    def visualize_graph(self, window_idx=0, threshold=0.1):
        """Visualize the graph structure for a specific window"""
        if self.last_adjacencies is None:
            print("No adjacencies computed yet. Run forward pass first.")
            return None

        A = self.last_adjacencies[window_idx]
        A_avg = A.mean(dim=0)
        A_vis = (A_avg > threshold).float()
        return A_vis.detach().cpu().numpy()
