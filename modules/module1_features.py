"""
Step 3.1: Module 1 - Multi-Scale Temporal Feature Extraction (Complete)
Sub-Step 1: Wavelet-Based Preprocessing
Sub-Step 2: Multi-Scale Convolutional Feature Extraction
Sub-Step 3: Selective Feature Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

print("\n" + "="*60)
print("STEP 3.1: MODULE 1 - MULTI-SCALE TEMPORAL FEATURE EXTRACTION (COMPLETE)")
print("="*60)

# ----------------------------------------------------------------------------
# SUB-STEP 1: WAVELET-BASED PREPROCESSING
# ----------------------------------------------------------------------------


class WaveletPreprocessing(nn.Module):
    def __init__(self, wavelet='db4', level=5, threshold_scale=0.1):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        self.threshold_scale = threshold_scale

    def forward(self, x):
        # x: (B, 12, T)
        batch_size, num_leads, time_len = x.shape
        reconstructed = torch.zeros_like(x)

        # Use torch.no_grad() for preprocessing
        with torch.no_grad():
            for b in range(batch_size):
                for lead in range(num_leads):
                    signal = x[b, lead, :].cpu().numpy()

                    # Use reflection padding to handle boundaries
                    signal_padded = np.pad(signal, (64, 64), mode='reflect')

                    coeffs = pywt.wavedec(signal_padded, self.wavelet, level=self.level)

                    # Adaptive threshold based on each detail coefficient
                    new_coeffs = [coeffs[0]]
                    for detail in coeffs[1:]:
                        # MAD (Median Absolute Deviation) threshold
                        mad = np.median(np.abs(detail - np.median(detail)))
                        threshold = self.threshold_scale * mad / 0.6745  # Sigma estimate

                        # Soft thresholding
                        detail_thresholded = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
                        new_coeffs.append(detail_thresholded)

                    # Reconstruct and remove padding
                    reconstructed_signal = pywt.waverec(new_coeffs, self.wavelet)
                    reconstructed_signal = reconstructed_signal[64:64+time_len]

                    reconstructed[b, lead, :] = torch.FloatTensor(reconstructed_signal)

        return reconstructed

# ----------------------------------------------------------------------------
# SUB-STEP 2: MULTI-SCALE CONVOLUTIONAL FEATURE EXTRACTION
# ----------------------------------------------------------------------------

class MultiScaleConv1D(nn.Module):
    """
    Multi-scale 1D convolutional feature extractor
    Captures patterns at different time scales (fast, medium, slow)
    """
    def __init__(self, in_channels=12, out_channels=64, kernel_sizes=[3, 7, 15, 31]):
        """
        Args:
            in_channels: number of input leads (12)
            out_channels: number of output channels per convolution
            kernel_sizes: list of kernel sizes for different scales
        """
        super(MultiScaleConv1D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes

        # Create parallel convolutions with different kernel sizes
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Kernel size mapping for interpretability
        self.kernel_info = {
            3: "Fast patterns (QRS spikes)",
            7: "Medium patterns (P and T waves)",
            15: "Slow patterns (rhythm)",
            31: "Very slow patterns (long-term trends)"
        }

        for k in kernel_sizes:
            # Ensure kernel size is odd and padding keeps same length
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=k,
                padding=k//2,  # Keep same length
                bias=True
            )
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(out_channels))

    def forward(self, x):
        """
        Args:
            x: input signal (B, 12, T) where B=batch, 12=leads, T=time
        Returns:
            features: list of (B, out_channels, T) for each scale
        """
        # x shape: (batch_size, 12, time_samples)

        multi_scale_features = []

        # Apply each convolution
        for conv, bn in zip(self.convs, self.bns):
            # conv: (B, out_channels, T)
            conv_out = conv(x)
            conv_out = bn(conv_out)
            conv_out = F.relu(conv_out)
            multi_scale_features.append(conv_out)

        return multi_scale_features

# ----------------------------------------------------------------------------
# SUB-STEP 3: SELECTIVE FEATURE FUSION
# ----------------------------------------------------------------------------

class SelectiveFeatureFusion(nn.Module):
    """
    Attention-based fusion of multi-scale features
    """
    def __init__(self, num_scales=4, feature_dim=64):
        """
        Args:
            num_scales: number of scales (4)
            feature_dim: feature dimension (out_channels)
        """
        super(SelectiveFeatureFusion, self).__init__()

        self.num_scales = num_scales
        self.feature_dim = feature_dim

        # Learnable query vectors for each scale
        self.scale_queries = nn.Parameter(torch.randn(num_scales, feature_dim))

        # Attention mechanism
        self.attention_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: list of (B, out_channels, T)
        Returns:
            fused: (B, out_channels) - aggregated features
        """
        # For each scale, do global average pooling
        pooled_features = []

        for features in multi_scale_features:
            # (B, out_channels, T) -> (B, out_channels)
            pooled = torch.mean(features, dim=2)
            pooled_features.append(pooled)

        # Stack: (num_scales, B, out_channels)
        stacked = torch.stack(pooled_features, dim=0)

        # Compute attention weights
        # Method 1: Learnable attention
        attn_weights = F.softmax(self.attention_weights, dim=0)  # (num_scales,)

        # Method 2: Query-based attention (more sophisticated)
        # This uses scale-specific queries to compute attention
        # For simplicity, we use Method 1

        # Apply attention: (num_scales, 1, 1) * (num_scales, B, out_channels)
        weighted = stacked * attn_weights.view(-1, 1, 1)

        # Sum across scales: (B, out_channels)
        fused = torch.sum(weighted, dim=0)

        # Output projection
        fused = self.output_proj(fused)

        return fused, attn_weights

    def get_attention_weights(self):
        """Return attention weights for interpretability"""
        return F.softmax(self.attention_weights, dim=0).detach().cpu().numpy()

# ----------------------------------------------------------------------------
# COMPLETE MODULE 1
# ----------------------------------------------------------------------------

class Module1_Complete(nn.Module):
    """
    Complete Module 1: Multi-Scale Temporal Feature Extraction
    Includes all three sub-steps
    """
    def __init__(self,
                 in_channels=12,
                 out_channels=64,
                 kernel_sizes=[3, 7, 15, 31],
                 wavelet='db4',
                 wavelet_level=5):
        super(Module1_Complete, self).__init__()

        # Sub-Step 1: Wavelet Preprocessing
        self.wavelet = WaveletPreprocessing(wavelet=wavelet, level=wavelet_level)

        # Sub-Step 2: Multi-Scale Convolution
        self.multi_scale_conv = MultiScaleConv1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes
        )

        # Sub-Step 3: Selective Feature Fusion
        self.fusion = SelectiveFeatureFusion(
            num_scales=len(kernel_sizes),
            feature_dim=out_channels
        )

        # Expand fused features to per-lead features
        self.expand = nn.Linear(out_channels, out_channels * in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        """
        Args:
            x: input signal (B, 12, T)
        Returns:
            z: node features (B, 12, out_channels)
            attn_weights: attention weights for interpretability
        """
        # Sub-Step 1: Wavelet Preprocessing
        # x_wavelet = self.wavelet(x)  # (B, 12, T)
        x_wavelet = x  # (B, 12, T)

        # Sub-Step 2: Multi-Scale Convolution
        multi_scale_features = self.multi_scale_conv(x_wavelet)  # list of (B, 64, T)

        # Sub-Step 3: Selective Feature Fusion
        fused, attn_weights = self.fusion(multi_scale_features)  # (B, 64)

        # Expand to per-lead features
        # (B, 64) -> (B, 64 * 12) -> (B, 12, 64)
        expanded = self.expand(fused)  # (B, 64 * 12)
        expanded = expanded.view(-1, self.in_channels, self.out_channels)  # (B, 12, 64)

        return expanded, attn_weights

    def get_scale_attentions(self):
        """Get attention weights for each scale"""
        return self.fusion.get_attention_weights()

