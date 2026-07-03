"""
Utility functions for metrics and evaluation
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def compute_metrics(predictions, labels, threshold=0.5):
    """Compute all metrics with per-class breakdown"""
    binary_preds = (predictions > threshold).astype(int)
    
    per_class_accuracy = []
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
        'per_class_f1': per_class_f1,
        'subset_accuracy': subset_acc,
        'hamming_accuracy': hamming_acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'auc_macro': auc_macro
    }


def print_metrics(metrics, class_names, phase="Train"):
    """Pretty print metrics"""
    print(f"\n{phase} Results:")
    print("-" * 60)
    
    print(f"📊 Per-Class Accuracy:")
    for i, (name, acc) in enumerate(zip(class_names, metrics['per_class_accuracy'])):
        bar = '█' * int(acc * 30) + '░' * (30 - int(acc * 30))
        print(f"  {name:12s}: {acc:.4f} ({acc*100:5.1f}%) {bar}")
    
    print(f"\n📊 Per-Class F1 Score:")
    for i, (name, f1) in enumerate(zip(class_names, metrics['per_class_f1'])):
        print(f"  {name:12s}: {f1:.4f}")
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"  Hamming Accuracy: {metrics['hamming_accuracy']:.4f}")
    print(f"  Macro F1: {metrics['f1_macro']:.4f}")
    print(f"  Micro F1: {metrics['f1_micro']:.4f}")
    print(f"  ROC-AUC: {metrics['auc_macro']:.4f}")