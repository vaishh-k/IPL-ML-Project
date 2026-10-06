# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: preprocessing.py
# Path: backend/services/preprocessing.py
# Description: Service layer handling data scaling, one-hot encoding, label encoding, and feature column generation.
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

def encode_categorical(df: pd.DataFrame, encoding_map: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """
    Encodes categorical features according to mapping:
    column -> method ('label', 'onehot', 'ordinal', 'none')
    Returns:
        tuple (encoded_df, encoding_metadata)
    """
    df_encoded = df.copy()
    metadata = {}
    
    for col, method in encoding_map.items():
        if col not in df_encoded.columns or method == "none":
            continue
            
        series_non_null = df_encoded[col].fillna("Missing")
        unique_vals = series_non_null.unique().tolist()
        
        if method in ("label", "ordinal"):
            # Use LabelEncoder
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(series_non_null.astype(str))
            
            # Save mapping in metadata
            mapping = {str(c): int(i) for i, c in enumerate(le.classes_)}
            metadata[col] = {
                "method": method,
                "original_columns": [col],
                "new_columns": [col],
                "mapping": mapping
            }
            
        elif method == "onehot":
            # Perform one-hot encoding using pd.get_dummies
            # We preserve missing values as a column if there are any, or fillna first
            dummies = pd.get_dummies(series_non_null, prefix=col, dtype=int)
            new_cols = dummies.columns.tolist()
            
            # Concatenate dummies and drop original column
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
            
            metadata[col] = {
                "method": "onehot",
                "original_columns": [col],
                "new_columns": new_cols,
                "mapping": {val: f"{col}_{val}" for val in unique_vals}
            }
            
    return df_encoded, metadata

def scale_numerical(df: pd.DataFrame, scaling_map: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    """
    Scales numerical features according to mapping:
    column -> scaler ('standard', 'minmax', 'none')
    Returns:
        tuple (scaled_df, scaling_metadata)
    """
    df_scaled = df.copy()
    metadata = {}
    
    for col, method in scaling_map.items():
        if col not in df_scaled.columns or method == "none":
            continue
            
        # Reshape to 2D for sklearn scalers
        series = df_scaled[col].fillna(df_scaled[col].median()) # Impute with median if any missing value leaked
        vals = series.values.reshape(-1, 1)
        
        if method == "standard":
            scaler = StandardScaler()
            scaled_vals = scaler.fit_transform(vals)
            df_scaled[col] = scaled_vals.flatten()
            metadata[col] = {
                "method": "standard",
                "mean": float(scaler.mean_[0]),
                "scale": float(scaler.scale_[0])
            }
        elif method == "minmax":
            scaler = MinMaxScaler()
            scaled_vals = scaler.fit_transform(vals)
            df_scaled[col] = scaled_vals.flatten()
            metadata[col] = {
                "method": "minmax",
                "min": float(scaler.data_min_[0]),
                "max": float(scaler.data_max_[0])
            }
        elif method == "robust":
            scaler = RobustScaler()
            scaled_vals = scaler.fit_transform(vals)
            df_scaled[col] = scaled_vals.flatten()
            metadata[col] = {
                "method": "robust",
                "center": float(scaler.center_[0]),
                "scale": float(scaler.scale_[0])
            }
            
    return df_scaled, metadata

def split_train_test(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    """
    Computes split dimensions and sample indexes.
    Returns metadata for UI rendering.
    """
    total_rows = len(df)
    test_count = int(total_rows * test_size)
    train_count = total_rows - test_count
    
    # Shuffle indices
    shuffled_indices = np.random.permutation(df.index)
    train_indices = shuffled_indices[:train_count].tolist()
    test_indices = shuffled_indices[train_count:].tolist()
    
    return {
        "total_rows": total_rows,
        "train_count": train_count,
        "test_count": test_count,
        "train_percentage": round((train_count / total_rows) * 100, 1),
        "test_percentage": round((test_count / total_rows) * 100, 1),
        "train_indices": train_indices[:100], # Keep a sample of indices
        "test_indices": test_indices[:100]
    }
