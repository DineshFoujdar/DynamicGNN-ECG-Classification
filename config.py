"""
Configuration settings for TGLLNet ECG Classification
"""

import os

import torch

class Config:
    # Data paths
    DATA_PATH = "data/"
    TRAIN_PATH = os.path.join(DATA_PATH, "ptbxl_train.npz")
    VAL_PATH = os.path.join(DATA_PATH, "ptbxl_val.npz")
    TEST_PATH = os.path.join(DATA_PATH, "ptbxl_test.npz")
    
    # Model parameters
    IN_CHANNELS = 12
    CONV_OUT_CHANNELS = 64
    KERNEL_SIZES = [3, 7, 15, 31]
    WAVELET = 'db4'
    WAVELET_LEVEL = 5
    
    # Graph parameters
    GRAPH_HIDDEN_DIM = 128
    NUM_WINDOWS = 5
    ALPHA = 0.1
    BETA = 0.05
    
    # GCN parameters
    GCN_HIDDEN_DIM = 128
    GCN_OUT_DIM = 32
    NUM_GCN_LAYERS = 3
    DROPOUT_GCN = 0.2
    
    # Classifier parameters
    CLASSIFIER_HIDDEN = [128, 64]
    NUM_CLASSES = 5
    DROPOUT_CLS = 0.2
    POOLING_TYPE = 'mean'
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    GRADIENT_CLIP = 0.1
    EARLY_STOPPING_PATIENCE = 20
    THRESHOLD = 0.3

    # PATHS
    # ============================================================
    CHECKPOINT_DIR = "checkpoints/"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    RESULTS_DIR = "results/"
    
    # Device
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Checkpoint
    CHECKPOINT_PATH = "checkpoints/best_model.pth"

config = Config()