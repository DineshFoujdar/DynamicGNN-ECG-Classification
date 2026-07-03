"""
Step 3.5: Complete Model Integration
Goal: Combine all 4 modules into one complete model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.module1_features import Module1_Complete
from modules.module2_graph import DynamicGraphLearning
from modules.module3_gcn import Module3_Complete
from modules.module4_classifier import Module4_Complete

print("\n" + "="*60)
print("STEP 3.5: COMPLETE MODEL INTEGRATION")
print("="*60)

# Import all modules (assuming they are defined in previous cells)
# If not, they are already defined in the notebook

class DynamicECGModel(nn.Module):
    """
    Complete Dynamic Temporal Graph Network for ECG Diagnosis
    Combines all 4 modules end-to-end
    """
    def __init__(self,
                 # Module 1 parameters
                 in_channels=12,
                 conv_out_channels=64,
                 kernel_sizes=[3, 7, 15, 31],
                 wavelet='db4',
                 wavelet_level=5,

                 # Module 2 parameters
                 graph_hidden_dim=128,
                 num_windows=5,
                 alpha=0.1,
                 beta=0.05,

                 # Module 3 parameters
                 gcn_hidden_dim=128,
                 gcn_out_dim=32,
                 num_gcn_layers=3,
                 dropout_gcn=0.2,

                 # Module 4 parameters
                 classifier_hidden=[128, 64],
                 num_classes=5,
                 dropout_cls=0.2,
                 pooling_type='mean'):

        super(DynamicECGModel, self).__init__()

        # Module 1: Multi-Scale Temporal Feature Extraction
        self.module1 = Module1_Complete(
            in_channels=in_channels,
            out_channels=conv_out_channels,
            kernel_sizes=kernel_sizes,
            wavelet=wavelet,
            wavelet_level=wavelet_level
        )

        # Module 2: Dynamic Graph Structure Learning
        self.module2 = DynamicGraphLearning(
            feature_dim=conv_out_channels,
            hidden_dim=graph_hidden_dim,
            num_leads=in_channels,
            num_windows=num_windows,
            alpha=alpha,
            beta=beta
        )

        # Module 3: Spatiotemporal Graph Convolution
        self.module3 = Module3_Complete(
            in_features=conv_out_channels,
            hidden_dim=gcn_hidden_dim,
            out_features=gcn_out_dim,
            num_layers=num_gcn_layers,
            dropout=dropout_gcn
        )

        # Module 4: Classification and Interpretability
        self.module4 = Module4_Complete(
            input_dim=gcn_out_dim,
            hidden_dims=classifier_hidden,
            num_classes=num_classes,
            dropout=dropout_cls,
            pooling_type=pooling_type
        )

        # Store for interpretability
        self.last_attentions = None
        self.last_adjacencies = None

    def forward(self, x):
        """
        Args:
            x: ECG signal, shape (B, 12, T)
        Returns:
            logits: classification output, shape (B, num_classes)
            reg_loss: graph regularization loss
        """
        # Module 1: Multi-Scale Temporal Feature Extraction
        z, scale_attentions = self.module1(x)  # (B, 12, 64)
        self.last_attentions = scale_attentions

        # Module 2: Dynamic Graph Structure Learning
        A_windows, reg_loss = self.module2(z, return_all=True)  # list of (B, 12, 12)
        self.last_adjacencies = A_windows

        # Module 3: Spatiotemporal Graph Convolution
        H = self.module3(z, A_windows)  # (B, 12, 32)

        # Module 4: Classification
        logits = self.module4(H)  # (B, num_classes)

        return logits, reg_loss

    def get_interpretability(self):
        """Get interpretability information"""
        return {
            'scale_attentions': self.last_attentions,
            'adjacencies': self.last_adjacencies,
            'classifier_attention': self.module4.get_interpretability()
        }

    def get_adjacency(self, window_idx=0):
        """Get adjacency matrix for a specific window"""
        if self.last_adjacencies is not None:
            return self.last_adjacencies[window_idx]
        return None

    def get_scale_attentions(self):
        """Get attention weights for each scale"""
        return self.last_attentions
