"""
STEP 3.4: MODULE 4 - CLASSIFICATION AND INTERPRETABILITY
Complete Integration Test with Real Data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

print("\n" + "="*60)
print("STEP 3.4: MODULE 4 - CLASSIFICATION AND INTERPRETABILITY (COMPLETE)")
print("="*60)

# ----------------------------------------------------------------------------
# SUB-MODULE 4.1: READOUT (GRAPH-LEVEL POOLING)
# ----------------------------------------------------------------------------

class Readout(nn.Module):
    """
    Pooling to get graph-level representation
    """
    def __init__(self, pooling_type='mean'):
        super(Readout, self).__init__()
        self.pooling_type = pooling_type

        if pooling_type == 'attention':
            self.attention = nn.Linear(32, 1)

    def forward(self, H):
        """
        Args:
            H: node features, shape (B, 12, feature_dim)
        Returns:
            h_graph: graph-level representation, shape (B, feature_dim)
        """
        if self.pooling_type == 'mean':
            h_graph = torch.mean(H, dim=1)
        elif self.pooling_type == 'max':
            h_graph = torch.max(H, dim=1)[0]
        elif self.pooling_type == 'attention':
            attn_scores = self.attention(H)
            attn_weights = F.softmax(attn_scores, dim=1)
            h_graph = torch.sum(H * attn_weights, dim=1)
        else:
            h_graph = torch.mean(H, dim=1)

        return h_graph

    def get_attention_weights(self, H):
        if self.pooling_type == 'attention':
            attn_scores = self.attention(H)
            return F.softmax(attn_scores, dim=1)
        return None

# ----------------------------------------------------------------------------
# SUB-MODULE 4.2: CLASSIFIER (MLP)
# ----------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """
    MLP for final classification
    """
    def __init__(self,
                 input_dim=32,
                 hidden_dims=[128, 64],
                 num_classes=5,
                 dropout=0.2):
        super(MLPClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.mlp = nn.Sequential(*layers)
        self.last_gradients = None

    def forward(self, x):
        return self.mlp(x)

    def get_gradients(self):
        return self.last_gradients

# ----------------------------------------------------------------------------
# SUB-MODULE 4.3: INTERPRETABILITY
# ----------------------------------------------------------------------------

class Interpretability(nn.Module):
    """
    Interpretability module using gradient-based methods
    """
    def __init__(self):
        super(Interpretability, self).__init__()

    def compute_node_importance(self, model, H, A_windows, target_class=None):
        """
        Compute importance of each node using gradient-based method
        """
        model.eval()
        H.requires_grad = True

        logits = model(H, A_windows) if hasattr(model, 'forward') else model(H)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1)

        gradients = []
        for b in range(logits.size(0)):
            loss = logits[b, target_class[b]]
            loss.backward(retain_graph=True)
            grad = H.grad[b].detach().cpu().numpy()
            gradients.append(grad)
            H.grad.zero_()

        node_importance = np.array([np.linalg.norm(grad, axis=1) for grad in gradients])
        return node_importance

    def compute_edge_importance(self, A_windows):
        edge_importance = []
        for A in A_windows:
            imp = torch.abs(A)
            edge_importance.append(imp.detach().cpu().numpy())
        return edge_importance

# ----------------------------------------------------------------------------
# COMPLETE MODULE 4
# ----------------------------------------------------------------------------

class Module4_Complete(nn.Module):
    """
    Complete Module 4: Classification and Interpretability
    """
    def __init__(self,
                 input_dim=32,
                 hidden_dims=[128, 64],
                 num_classes=5,
                 dropout=0.2,
                 pooling_type='mean'):
        super(Module4_Complete, self).__init__()

        self.readout = Readout(pooling_type=pooling_type)
        self.classifier = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        )
        self.interpretability = Interpretability()
        self.last_attentions = None

    def forward(self, H):
        """
        Args:
            H: node features from Module 3, shape (B, 12, 32)
        Returns:
            logits: classification output, shape (B, num_classes)
        """
        h_graph = self.readout(H)
        self.last_attentions = self.readout.get_attention_weights(H)
        logits = self.classifier(h_graph)
        return logits

    def get_interpretability(self):
        return {
            'attention_weights': self.last_attentions,
        }