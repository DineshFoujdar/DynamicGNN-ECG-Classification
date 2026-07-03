"""
Data preprocessing script for PTB-XL ECG dataset
Converts raw PTB-XL data to processed npz files for training
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = r"C:/ECG_project/dynamic_ecg_paper/data/ptb_xl"
RECORDS_PATH = os.path.join(DATA_PATH, "records100/")
CSV_PATH = os.path.join(DATA_PATH, "ptbxl_database.csv")
SCP_PATH = os.path.join(DATA_PATH, "scp_statements.csv")
OUTPUT_PATH = "data/"

CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
SIGNAL_LENGTH = 1000
NUM_LEADS = 12
SAMPLING_RATE = 100


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_scp_codes(codes_str):
    """Parse SCP codes string to dictionary"""
    if isinstance(codes_str, dict):
        return codes_str
    if isinstance(codes_str, str):
        try:
            return ast.literal_eval(codes_str)
        except:
            return {}
    return {}


def get_record_path(ecg_id):
    """Get record path from ECG ID"""
    folder = f"{(ecg_id // 1000) * 1000:05d}"
    return f"{folder}/{ecg_id:05d}_lr"


def load_ecg_signal(record_path):
    """Load ECG signal from WFDB files"""
    full_path = os.path.join(RECORDS_PATH, record_path)
    try:
        record = wfdb.rdsamp(full_path)
        signal = record[0]
        
        # Ensure consistent length
        if signal.shape[0] >= SIGNAL_LENGTH:
            signal = signal[:SIGNAL_LENGTH, :]
        else:
            pad_size = SIGNAL_LENGTH - signal.shape[0]
            signal = np.pad(signal, ((0, pad_size), (0, 0)), mode='constant')
        
        return signal
    except Exception as e:
        print(f"   Error loading {record_path}: {e}")
        return np.zeros((SIGNAL_LENGTH, NUM_LEADS))


def scp_to_superdiag_vector(scp_dict, scp_to_superdiag):
    """Convert SCP codes to multi-label vector"""
    codes = [code for code, weight in scp_dict.items() if weight > 0]
    vector = [0, 0, 0, 0, 0]
    
    for code in codes:
        if code in scp_to_superdiag:
            super_class = scp_to_superdiag[code]
            if super_class in CLASS_NAMES:
                idx = CLASS_NAMES.index(super_class)
                vector[idx] = 1
    
    # If no classes detected, mark as NORM
    if sum(vector) == 0:
        vector[0] = 1
    
    return vector


def print_label_distribution(labels, name):
    """Print label distribution for a split"""
    total = len(labels)
    print(f"\n   {name} distribution:")
    for i, cls in enumerate(CLASS_NAMES):
        count = np.sum(labels[:, i])
        print(f"     {cls}: {count:.0f} ({count/total*100:.1f}%)")


def create_dataloader(signals, labels, batch_size=32, shuffle=True):
    """Create PyTorch DataLoader"""
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


# ============================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================

def load_ptbxl_data(data_path=None):
    """
    Load PTB-XL dataset and process for TGLLNet model
    
    Args:
        data_path: Path to PTB-XL data (optional)
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    print("\n" + "="*60)
    print("PREPROCESSING PTB-XL DATASET")
    print("="*60)
    
    # Use default path if not provided
    if data_path is not None:
        global DATA_PATH, RECORDS_PATH, CSV_PATH, SCP_PATH
        DATA_PATH = data_path
        RECORDS_PATH = os.path.join(DATA_PATH, "records100/")
        CSV_PATH = os.path.join(DATA_PATH, "ptbxl_database.csv")
        SCP_PATH = os.path.join(DATA_PATH, "scp_statements.csv")
    
    # ============================================================
    # STEP 1: LOAD DATABASE
    # ============================================================
    
    print("\n[1] Loading PTB-XL database...")
    
    # Check files exist
    for path in [CSV_PATH, SCP_PATH, RECORDS_PATH]:
        if os.path.exists(path):
            print(f"   Found: {path}")
        else:
            print(f"   NOT FOUND: {path}")
            raise FileNotFoundError(f"Required file not found: {path}")
    
    df_database = pd.read_csv(CSV_PATH)
    df_scp = pd.read_csv(SCP_PATH)
    
    print(f"   Database: {df_database.shape[0]} recordings")
    print(f"   SCP Statements: {df_scp.shape[0]} codes")
    
    # ============================================================
    # STEP 2: CREATE LABELS
    # ============================================================
    
    print("\n[2] Creating super-diagnostic labels...")
    
    # Create SCP to super-diag mapping
    SUPER_DIAG_MAPPING = {
        'NORM': 'NORM',
        'MI': 'MI',
        'STTC': 'STTC',
        'CD': 'CD',
        'HYP': 'HYP'
    }
    
    scp_to_superdiag = {}
    for idx, row in df_scp.iterrows():
        scp_code = row[df_scp.columns[0]]
        diagnostic_class = str(row.get('diagnostic_class', '')).strip()
        if diagnostic_class in SUPER_DIAG_MAPPING:
            scp_to_superdiag[scp_code] = SUPER_DIAG_MAPPING[diagnostic_class]
    
    print(f"   Created mapping with {len(scp_to_superdiag)} SCP codes")
    
    # Parse SCP codes and create labels
    df_database['scp_codes_dict'] = df_database['scp_codes'].apply(parse_scp_codes)
    df_database['super_diag_labels'] = df_database['scp_codes_dict'].apply(
        lambda x: scp_to_superdiag_vector(x, scp_to_superdiag)
    )
    
    # Add record path
    df_database['record_path'] = df_database['ecg_id'].apply(get_record_path)
    
    print(f"   Created labels for {len(df_database)} recordings")
    
    # ============================================================
    # STEP 3: LOAD SIGNALS
    # ============================================================
    
    print("\n[3] Loading ECG signals...")
    print(f"   Loading {len(df_database):,} recordings...")
    
    signals = []
    for idx, row in tqdm(df_database.iterrows(), total=len(df_database), desc="Loading ECGs"):
        signal = load_ecg_signal(row['record_path'])
        signals.append(signal)
    
    signals = np.array(signals)
    labels = np.array(df_database['super_diag_labels'].tolist())
    
    print(f"   Loaded {len(signals)} signals")
    print(f"   Signal shape: {signals.shape}")
    print(f"   Memory usage: {signals.nbytes / 1024 / 1024:.2f} MB")
    
    # ============================================================
    # STEP 4: SPLIT DATA
    # ============================================================
    
    print("\n[4] Splitting data at patient level...")
    
    patient_ids = df_database['patient_id'].values
    unique_patients = np.unique(patient_ids)
    print(f"   Unique patients: {len(unique_patients)}")
    
    # Group indices by patient
    patient_to_indices = {}
    for idx, patient_id in enumerate(patient_ids):
        if patient_id not in patient_to_indices:
            patient_to_indices[patient_id] = []
        patient_to_indices[patient_id].append(idx)
    
    # Split patients: 70% train, 15% val, 15% test
    train_patients, temp_patients = train_test_split(
        unique_patients, test_size=0.3, random_state=42
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.5, random_state=42
    )
    
    # Get indices
    train_indices = []
    val_indices = []
    test_indices = []
    
    for patient in train_patients:
        train_indices.extend(patient_to_indices[patient])
    for patient in val_patients:
        val_indices.extend(patient_to_indices[patient])
    for patient in test_patients:
        test_indices.extend(patient_to_indices[patient])
    
    # Create splits
    X_train, y_train = signals[train_indices], labels[train_indices]
    X_val, y_val = signals[val_indices], labels[val_indices]
    X_test, y_test = signals[test_indices], labels[test_indices]
    
    print(f"\n   Train: {len(X_train)} recordings ({len(train_patients)} patients)")
    print(f"   Val:   {len(X_val)} recordings ({len(val_patients)} patients)")
    print(f"   Test:  {len(X_test)} recordings ({len(test_patients)} patients)")
    
    # ============================================================
    # STEP 5: VERIFY DISTRIBUTION
    # ============================================================
    
    print("\n[5] Verifying label distribution...")
    print_label_distribution(y_train, "Train")
    print_label_distribution(y_val, "Val")
    print_label_distribution(y_test, "Test")
    
    # ============================================================
    # STEP 6: SAVE DATA
    # ============================================================
    
    print("\n[6] Saving processed data...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    np.savez_compressed(
        os.path.join(OUTPUT_PATH, "ptbxl_train.npz"),
        signals=X_train,
        labels=y_train
    )
    np.savez_compressed(
        os.path.join(OUTPUT_PATH, "ptbxl_val.npz"),
        signals=X_val,
        labels=y_val
    )
    np.savez_compressed(
        os.path.join(OUTPUT_PATH, "ptbxl_test.npz"),
        signals=X_test,
        labels=y_test
    )
    
    print(f"    Saved to: {OUTPUT_PATH}")
    print(f"     - ptbxl_train.npz ({X_train.shape})")
    print(f"     - ptbxl_val.npz ({X_val.shape})")
    print(f"     - ptbxl_test.npz ({X_test.shape})")
    
    # ============================================================
    # STEP 7: CREATE DATALOADERS
    # ============================================================
    
    print("\n[7] Creating DataLoaders...")
    batch_size = 32
    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)
    
    print(f"   Train loader: {len(train_loader)} batches")
    print(f"   Val loader:   {len(val_loader)} batches")
    print(f"   Test loader:  {len(test_loader)} batches")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"""
 SUMMARY:
├─ Total recordings: {len(signals)}
├─ Unique patients: {len(unique_patients)}
├─ Signal shape: {signals.shape}
├─ Train: {len(X_train)} recordings
├─ Val:   {len(X_val)} recordings
└─ Test:  {len(X_test)} recordings

 Output: {OUTPUT_PATH}
 Ready for training!
""")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_dataloaders(batch_size=32):
    """
    Load preprocessed data and create DataLoaders
    
    Args:
        batch_size: Batch size for DataLoader
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    print("\n[1] Loading preprocessed data...")
    
    # Load data
    train = np.load(os.path.join(OUTPUT_PATH, "ptbxl_train.npz"), allow_pickle=True)
    val = np.load(os.path.join(OUTPUT_PATH, "ptbxl_val.npz"), allow_pickle=True)
    test = np.load(os.path.join(OUTPUT_PATH, "ptbxl_test.npz"), allow_pickle=True)
    
    X_train, y_train = train['signals'], train['labels']
    X_val, y_val = val['signals'], val['labels']
    X_test, y_test = test['signals'], test['labels']
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    
    # Create dataloaders
    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, y_train.shape[1]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Preprocessing PTB-XL data...")
    
    # Run preprocessing
    X_train, y_train, X_val, y_val, X_test, y_test = load_ptbxl_data()
    
    # Print final shapes
    print("\n FINAL DATA SHAPES:")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"   X_test:  {X_test.shape}, y_test:  {y_test.shape}")