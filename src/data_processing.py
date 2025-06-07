import pandas as pd
import numpy as np
import wfdb
import os
from ast import literal_eval # To safely evaluate the string representation of a dictionary

def load_ptbxl_data(root_dir: str, sampling_frequency: int = 100):
    """
    Loads PTB-XL data, processes labels into multi-label format, and splits
    data according to the recommended 10-fold split.

    Args:
        root_dir: The root directory of the PTB-XL dataset.
        sampling_frequency: The sampling frequency to use (100 or 500). 100 is faster.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # 1. Load database metadata
    db_path = os.path.join(root_dir, "ptbxl_database.csv")
    data = pd.read_csv(db_path, index_col='ecg_id')

    # 2. Load SCP statements to map codes to diagnostic superclasses
    scp_path = os.path.join(root_dir, "scp_statements.csv")
    scp_statements = pd.read_csv(scp_path, index_col=0)
    scp_statements = scp_statements[scp_statements.diagnostic == 1]

    def aggregate_diagnostic(scp_codes_str):
        # The scp_codes are stored as a string dictionary, e.g., "{'NORM': 100,...}"
        # We use literal_eval to safely parse this into a Python dictionary.
        scp_codes = literal_eval(scp_codes_str)
        res = set()
        for code, confidence in scp_codes.items():
            if confidence >= 0: # Check if a diagnosis is present
                if code in scp_statements.index:
                    super_class = scp_statements.loc[code].diagnostic_class
                    res.add(super_class)
        return list(res)

    # Apply this function to the scp_codes column
    data['diagnostic_superclass'] = data.scp_codes.apply(aggregate_diagnostic)

    # 3. Create the multi-label (one-hot encoded) target y
    # Get all unique superclasses
    all_superclasses = sorted(list(set(item for sublist in data.diagnostic_superclass for item in sublist)))
    
    # Create a column for each superclass
    for sc in all_superclasses:
        data[sc] = data.diagnostic_superclass.apply(lambda x: 1 if sc in x else 0)
    
    # The columns NORM, MI, STTC, CD, HYP are now our labels
    label_columns = all_superclasses
    y = data[label_columns].values

    # 4. Load waveform data (X)
    if sampling_frequency == 100:
        file_col = 'filename_lr'
    elif sampling_frequency == 500:
        file_col = 'filename_hr'
    else:
        raise ValueError("Sampling frequency must be 100 or 500")

    # For simplicity, we'll start with a single lead (e.g., lead II, which is index 1)
    # An advanced model could use all 12 leads.
    X = np.zeros((len(data), sampling_frequency * 10, 1)) # N_records, 10 seconds, 1 lead
    
    for i, ecg_id in enumerate(data.index):
        file_path = os.path.join(root_dir, data.loc[ecg_id][file_col])
        record = wfdb.rdrecord(file_path)
        
        # Using lead II (index 1) as an example
        signal = record.p_signal[:, 1]
        
        # Z-score normalization
        signal = (signal - np.mean(signal)) / np.std(signal)
        
        X[i, :, 0] = signal

    # 5. Split data using the predefined folds
    train_fold = data[data.strat_fold <= 8]
    val_fold = data[data.strat_fold == 9]
    test_fold = data[data.strat_fold == 10]

    X_train = X[train_fold.index.map(data.index.get_loc)]
    y_train = y[train_fold.index.map(data.index.get_loc)]

    X_val = X[val_fold.index.map(data.index.get_loc)]
    y_val = y[val_fold.index.map(data.index.get_loc)]

    X_test = X[test_fold.index.map(data.index.get_loc)]
    y_test = y[test_fold.index.map(data.index.get_loc)]

    print("Data loading and splitting complete.")
    print(f"Label columns: {label_columns}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, label_columns