"""
Step 4: Training the Model (UPDATED - Based on Paper with Class Balance)
Goal: Train the complete model on the ECG dataset with paper's settings + class weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

#  Now import model components
from modules.module1_features import Module1_Complete
from modules.module2_graph import DynamicGraphLearning
from modules.module3_gcn import Module3_Complete
from modules.module4_classifier import Module4_Complete
from models.dynamic_ecg_model import DynamicECGModel

# Create model
import models.dynamic_ecg_model as dynamic_ecg_model

dynamic_ecg_model.Module1_Complete = Module1_Complete
dynamic_ecg_model.DynamicGraphLearning = DynamicGraphLearning
dynamic_ecg_model.Module3_Complete = Module3_Complete
dynamic_ecg_model.Module4_Complete = Module4_Complete

print("\n" + "="*60)
print("STEP 4: TRAINING THE MODEL (PAPER'S SETTINGS + CLASS BALANCE)")
print("="*60)

# ----------------------------------------------------------------------------
# 1. LOAD THE DATA
# ----------------------------------------------------------------------------

print("\n[1] Loading data...")

# Paths
DATA_PATH = "data/"

# Load the saved data
TRAIN_PATH = os.path.join(DATA_PATH, "ptbxl_train.npz")
VAL_PATH = os.path.join(DATA_PATH, "ptbxl_val.npz")
TEST_PATH = os.path.join(DATA_PATH, "ptbxl_test.npz")

# Load training data
train_data = np.load(TRAIN_PATH, allow_pickle=True)
X_train = train_data['signals']   # (N, 1000, 12)
y_train = train_data['labels']    # (N, 5)

# Load validation data
val_data = np.load(VAL_PATH, allow_pickle=True)
X_val = val_data['signals']       # (N, 1000, 12)
y_val = val_data['labels']        # (N, 5)

# Load test data
test_data = np.load(TEST_PATH, allow_pickle=True)
X_test = test_data['signals']     # (N, 1000, 12)
y_test = test_data['labels']      # (N, 5)

print(f"   Train: {X_train.shape[0]} samples")
print(f"   Val: {X_val.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")

# Check label distribution
print(f"\n   Label distribution (Train):")
class_names = ['Normal', 'Arrhythmia', 'Ischemia', 'Hypertrophy', 'Conduction']
for i, name in enumerate(class_names):
    count = y_train[:, i].sum()
    print(f"     {name}: {count:.0f} ({count/len(y_train)*100:.1f}%)")

# ----------------------------------------------------------------------------
# 2. CALCULATE CLASS WEIGHTS (FOR IMBALANCED DATA)
# ----------------------------------------------------------------------------

print("\n[2] Calculating class weights for balanced loss...")

# Method 1: Inverse frequency weighting
class_counts = y_train.sum(axis=0)
total = y_train.shape[0]
class_weights = []
for i, count in enumerate(class_counts):
    # weight = total / (num_classes * count)
    weight = total / (len(class_counts) * count)
    class_weights.append(weight)
    print(f"   {class_names[i]}: count={count:.0f}, weight={weight:.4f}")

class_weights = torch.FloatTensor(class_weights)

# Method 2: sklearn's compute_class_weight (alternative)
# from sklearn.utils.class_weight import compute_class_weight
# sklearn_weights = []
# for i in range(y_train.shape[1]):
#     weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train[:, i])
#     sklearn_weights.append(weights[1])  # weight for positive class
# class_weights = torch.FloatTensor(sklearn_weights)

print(f"\n   Final class weights: {class_weights.cpu().numpy()}")

# ----------------------------------------------------------------------------
# 3. CREATE DATALOADERS
# ----------------------------------------------------------------------------

print("\n[3] Creating dataloaders...")

class ECG_Dataset(torch.utils.data.Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]  # (1000, 12)
        signal = torch.FloatTensor(signal.T)  # (12, 1000)
        label = torch.FloatTensor(self.labels[idx])  # (5,)
        return signal, label

# Create datasets
train_dataset = ECG_Dataset(X_train, y_train)
val_dataset = ECG_Dataset(X_val, y_val)
test_dataset = ECG_Dataset(X_test, y_test)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"   Batch size: {batch_size}")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")

# ----------------------------------------------------------------------------
# 4. METRICS FUNCTION (PAPER'S METRICS)
# ----------------------------------------------------------------------------

def compute_metrics(predictions, labels, threshold=0.5):
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
    f1_micro = f1_score(labels, binary_preds, average='micro', zero_division=0)

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
        'f1_micro': f1_micro,
        'auc_macro': auc_macro
    }

def print_metrics(metrics, phase="Train"):
    """Pretty print all metrics"""
    print(f"\n{phase} Results:")
    print("-" * 60)

    print(f" Per-Class Accuracy:")
    for i, (name, acc) in enumerate(zip(class_names, metrics['per_class_accuracy'])):
        bar = '█' * int(acc * 30) + '░' * (30 - int(acc * 30))
        print(f"  {name:12s}: {acc:.4f} ({acc*100:5.1f}%) {bar}")

    print(f"\n  Average: {np.mean(metrics['per_class_accuracy']):.4f}")

    print(f"\n Per-Class F1 Score:")
    for i, (name, f1) in enumerate(zip(class_names, metrics['per_class_f1'])):
        print(f"  {name:12s}: {f1:.4f}")

    print(f"\n Overall Metrics:")
    print(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"  Hamming Accuracy: {metrics['hamming_accuracy']:.4f}")
    print(f"  Macro F1: {metrics['f1_macro']:.4f}")
    print(f"  Micro F1: {metrics['f1_micro']:.4f}")
    print(f"  ROC-AUC: {metrics['auc_macro']:.4f}")

# ----------------------------------------------------------------------------
# 5. INITIALIZE MODEL
# ----------------------------------------------------------------------------

print("\n[4] Initializing model...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

try:
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
    print(f"    Model created successfully!")
except NameError:
    print("    DynamicECGModel not defined!")
    exit()

total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,}")

# ----------------------------------------------------------------------------
# 6. SETUP OPTIMIZER AND LOSS (PAPER'S SETTINGS + CLASS WEIGHTS)
# ----------------------------------------------------------------------------

print("\n[5] Setting up optimizer and loss (PAPER'S SETTINGS + CLASS WEIGHTS)...")

#  PAPER'S SETTINGS + CLASS WEIGHTS
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

#  Loss with class weights for imbalanced data
class_weights = class_weights.to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)  # ← CLASS WEIGHTS ADDED!

# Learning rate scheduler (paper uses ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)

print(f"   Optimizer: Adam (lr=0.0001, weight_decay=1e-4)")
print(f"   Scheduler: ReduceLROnPlateau (mode='max', factor=0.5, patience=10)")
print(f"   Loss: BCEWithLogitsLoss with class weights")
print(f"   Class weights: {class_weights.cpu().numpy()}")

# ----------------------------------------------------------------------------
# 7. TRAINING FUNCTIONS (WITH PAPER'S AUGMENTATION)
# ----------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch with paper's data augmentation"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (signals, labels) in enumerate(pbar):
        signals = signals.to(device)
        labels = labels.to(device)

        #  PAPER'S DATA AUGMENTATION: Add noise
        signals = signals + torch.randn_like(signals) * 0.1

        # Forward pass
        logits, reg_loss = model(signals)

        # Clamp logits to prevent explosion
        logits = torch.clamp(logits, min=-50, max=50)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f" NaN/Inf in logits at batch {batch_idx}!")
            break

        # Classification loss (PAPER'S LOSS + CLASS WEIGHTS)
        class_loss = criterion(logits, labels)
        loss = class_loss + 0.1 * reg_loss

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f" Loss is NaN/Inf at batch {batch_idx}!")
            print(f"   logits min: {logits.min().item()}, max: {logits.max().item()}")
            print(f"   class_loss: {class_loss.item()}, reg_loss: {reg_loss.item()}")
            break

        if loss.item() < 0:
            print(f" Negative loss at batch {batch_idx}: {loss.item():.6f}")
            loss = torch.abs(loss) + 1e-8

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        total_loss += loss.item()

        # Store predictions
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_metrics(all_preds, all_labels)

    return total_loss / len(loader), metrics

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
# 8. RUN TRAINING (PAPER'S SELECTION CRITERIA)
# ----------------------------------------------------------------------------

print("\n[6] Starting training (PAPER'S SETTINGS + CLASS WEIGHTS)...")

num_epochs = 50
best_score = 0  # AUC + F1 (PAPER'S METHOD)
best_f1 = 0
best_auc = 0

train_history = {'loss': [], 'auc': [], 'f1': [], 'acc': []}
val_history = {'loss': [], 'auc': [], 'f1': [], 'acc': []}

for epoch in range(num_epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("="*60)

    # Train
    train_loss, train_metrics = train_epoch(
        model, train_loader, optimizer, criterion, device
    )
    train_history['loss'].append(train_loss)
    train_history['auc'].append(train_metrics['auc_macro'])
    train_history['f1'].append(train_metrics['f1_macro'])
    train_history['acc'].append(train_metrics['hamming_accuracy'])

    # Validate
    val_loss, val_metrics = validate(
        model, val_loader, criterion, device
    )
    val_history['loss'].append(val_loss)
    val_history['auc'].append(val_metrics['auc_macro'])
    val_history['f1'].append(val_metrics['f1_macro'])
    val_history['acc'].append(val_metrics['hamming_accuracy'])

    #  PAPER'S SELECTION CRITERIA: AUC + F1
    current_score = val_metrics['auc_macro'] + val_metrics['f1_macro']

    print(f"\n Summary:")
    print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"  Train AUC: {train_metrics['auc_macro']:.4f}, Val AUC: {val_metrics['auc_macro']:.4f}")
    print(f"  Train F1: {train_metrics['f1_macro']:.4f}, Val F1: {val_metrics['f1_macro']:.4f}")
    print(f"  Train Acc: {train_metrics['hamming_accuracy']:.4f}, Val Acc: {val_metrics['hamming_accuracy']:.4f}")

    # Print per-class accuracy
    print(f"\n  Per-Class F1 (Val):")
    for i, name in enumerate(class_names):
        f1 = val_metrics['per_class_f1'][i]
        status = "✅" if f1 > 0.7 else "🟡" if f1 > 0.5 else "🔴"
        print(f"    {status} {name:12s}: F1={f1:.4f}")

    #  PAPER'S SAVE CRITERION
    if current_score > best_score:
        best_score = current_score
        best_f1 = val_metrics['f1_macro']
        best_auc = val_metrics['auc_macro']
    
        #  Create checkpoints directory if it doesn't exist
        import os
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
    
        #  Save with full path
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_metrics['f1_macro'],
            'val_auc': val_metrics['auc_macro'],
            'val_metrics': val_metrics,
            'best_score': best_score
        }, checkpoint_path)
    
        print(f"\n    Best model saved! (AUC+F1={best_score:.4f})")
        print(f"      Path: {checkpoint_path}")

    # Update learning rate (paper uses ReduceLROnPlateau)
    scheduler.step(val_metrics['auc_macro'])

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

print(f"\n BEST RESULTS:")
print(f"  AUC: {best_auc:.4f}")
print(f"  F1: {best_f1:.4f}")
print(f"  AUC+F1: {best_score:.4f}")

