"""
evaluate.py - Evaluation script for Dynamic ECG Model
This script evaluates the trained model on the test set and computes metrics.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.dynamic_ecg_model import DynamicECGModel

# ----------------------------------------------------------------------------
# METRICS FUNCTION
# ----------------------------------------------------------------------------

def compute_metrics(predictions, labels, threshold=0.3):
    """
    Compute ALL metrics with per-class breakdown
    """
    binary_preds = (predictions > threshold).astype(int)
    
    per_class_accuracy = []
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    
    for class_idx in range(labels.shape[1]):
        true = labels[:, class_idx]
        pred = binary_preds[:, class_idx]
        
        acc = np.mean(true == pred)
        per_class_accuracy.append(acc)
        
        tp = np.sum((pred == 1) & (true == 1))
        fp = np.sum((pred == 1) & (true == 0))
        fn = np.sum((pred == 0) & (true == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)
    
    subset_acc = np.mean((binary_preds == labels).all(axis=1))
    hamming_acc = np.mean(binary_preds == labels)
    f1_macro = np.mean(per_class_f1)
    
    try:
        auc_macro = roc_auc_score(labels, predictions, average='macro')
    except:
        auc_macro = 0.0
    
    return {
        'per_class_accuracy': per_class_accuracy,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'subset_accuracy': subset_acc,
        'hamming_accuracy': hamming_acc,
        'f1_macro': f1_macro,
        'auc_macro': auc_macro
    }


def print_metrics(metrics, phase="Test"):
    """Pretty print all metrics"""
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    print(f"\n{phase} Results:")
    print("-" * 60)
    
    # Per-class accuracy
    print(f"📊 Per-Class Accuracy:")
    for i, (name, acc) in enumerate(zip(class_names, metrics['per_class_accuracy'])):
        bar = '█' * int(acc * 30) + '░' * (30 - int(acc * 30))
        print(f"  {name:12s}: {acc:.4f} ({acc*100:5.1f}%) {bar}")
    
    print(f"\n  Average: {np.mean(metrics['per_class_accuracy']):.4f}")
    
    # Per-class F1
    print(f"\n📊 Per-Class F1 Score:")
    for i, (name, f1) in enumerate(zip(class_names, metrics['per_class_f1'])):
        status = "✅" if f1 > 0.7 else "🟡" if f1 > 0.5 else "🔴"
        print(f"  {status} {name:12s}: {f1:.4f}")
    
    # Overall metrics
    print(f"\n📊 Overall Metrics:")
    print(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f} (ALL labels correct)")
    print(f"  Hamming Accuracy: {metrics['hamming_accuracy']:.4f} (per-label)")
    print(f"  Macro F1: {metrics['f1_macro']:.4f}")
    print(f"  ROC-AUC: {metrics['auc_macro']:.4f}")


# ----------------------------------------------------------------------------
# VALIDATION FUNCTION
# ----------------------------------------------------------------------------

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating")
        for signals, labels in pbar:
            signals = signals.to(device)
            labels = labels.to(device)

            logits, reg_loss = model(signals)
            class_loss = criterion(logits, labels)
            loss = class_loss + 0.1 * reg_loss

            total_loss += loss.item()

            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_preds, all_labels)

    return total_loss / len(loader), metrics


# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------

def load_data():
    """Load test data"""
    data_path = Config.DATA_PATH
    test_file = os.path.join(data_path, "ptbxl_test.npz")
    
    if not os.path.exists(test_file):
        print(f"❌ Test data not found at: {test_file}")
        print("   Please run preprocess_data.py first")
        return None, None
    
    data = np.load(test_file, allow_pickle=True)
    X_test = data['signals']
    y_test = data['labels']
    
    print(f"✅ Loaded test data: {X_test.shape[0]} samples")
    return X_test, y_test


# ----------------------------------------------------------------------------
# CREATE DATALOADER
# ----------------------------------------------------------------------------

def create_dataloader(signals, labels, batch_size=32, shuffle=False):
    """Create PyTorch DataLoader"""
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    class ECG_Dataset(Dataset):
        def __init__(self, signals, labels):
            self.signals = signals
            self.labels = labels
        
        def __len__(self):
            return len(self.signals)
        
        def __getitem__(self, idx):
            signal = torch.FloatTensor(self.signals[idx].T)
            label = torch.FloatTensor(self.labels[idx])
            return signal, label
    
    dataset = ECG_Dataset(signals, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ----------------------------------------------------------------------------
# MAIN EVALUATION
# ----------------------------------------------------------------------------

def main():
    """Main evaluation function"""
    print("\n" + "="*60)
    print("DYNAMIC ECG MODEL - EVALUATION")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1] Using device: {device}")
    
    # Load data
    print("\n[2] Loading test data...")
    X_test, y_test = load_data()
    if X_test is None:
        return
    
    test_loader = create_dataloader(X_test, y_test, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print("\n[3] Initializing model...")
    model = DynamicECGModel(
        in_channels=12,
        conv_out_channels=64,
        kernel_sizes=[3, 7, 15, 31],
        wavelet='db4',
        wavelet_level=5,
        graph_hidden_dim=128,
        num_windows=5,
        alpha=0.1,
        beta=0.05,
        gcn_hidden_dim=128,
        gcn_out_dim=32,
        num_gcn_layers=3,
        dropout_gcn=0.2,
        classifier_hidden=[128, 64],
        num_classes=5,
        dropout_cls=0.2,
        pooling_type='mean'
    )
    model = model.to(device)
    
    checkpoint_path = "checkpoints/best_model.pth"
    print(f"\n[3.1] Loading best model from checkpoint: {checkpoint_path}")
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at: {checkpoint_path}")
        print("   Please run train.py first")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Loaded best model from epoch {checkpoint['epoch']+1}")
    print(f"   Best validation F1: {checkpoint['val_f1']:.4f}")
    print(f"   Best validation AUC: {checkpoint['val_auc']:.4f}")
    
    # Loss function
    import torch.nn as nn
    criterion = nn.BCEWithLogitsLoss()
    
    # Evaluate
    print("\n[4] Evaluating on test set...")
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS")
    print("="*60)
    print_metrics(test_metrics, "Test")
    
    # Save results
    print("\n[5] Saving results...")
    results_dir = Config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    results_path = os.path.join(results_dir, "test_results.txt")
    with open(results_path, 'w') as f:
        f.write("TEST RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Macro F1: {test_metrics['f1_macro']:.4f}\n")
        f.write(f"ROC-AUC: {test_metrics['auc_macro']:.4f}\n")
        f.write(f"Hamming Accuracy: {test_metrics['hamming_accuracy']:.4f}\n")
        f.write(f"Subset Accuracy: {test_metrics['subset_accuracy']:.4f}\n\n")
        f.write("Per-Class F1:\n")
        class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        for i, name in enumerate(class_names):
            f.write(f"  {name}: {test_metrics['per_class_f1'][i]:.4f}\n")
    
    print(f"   ✅ Results saved to: {results_path}")
    
    return test_metrics


# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()