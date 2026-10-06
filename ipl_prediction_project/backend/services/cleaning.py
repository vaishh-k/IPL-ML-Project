# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: cleaning.py
# Path: backend/services/cleaning.py
# Description: Data cleaning service handling missing value imputation, duplicate rows detection, and variance analysis.
# ==============================================================================

import pandas as pd
import numpy as np

def get_missing_summary(df: pd.DataFrame) -> dict:
    """
    Returns a dictionary summarizing missing values in each column
    along with a recommended imputation strategy.
    """
    summary = []
    total_rows = len(df)
    
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = float(missing_count / total_rows * 100) if total_rows > 0 else 0.0
        
        # Determine recommended strategy
        if missing_count > 0:
            if missing_pct > 50.0:
                recommended = "drop_column"
            elif pd.api.types.is_numeric_dtype(df[col]):
                recommended = "median"
            else:
                recommended = "mode"
        else:
            recommended = "none"
            
        summary.append({
            "column": col,
            "missing_count": missing_count,
            "missing_percentage": round(missing_pct, 2),
            "recommended_strategy": recommended,
            "data_type": str(df[col].dtype)
        })
        
    return {"total_rows": total_rows, "columns": summary}

def impute_missing_values(df: pd.DataFrame, strategies: dict[str, str]) -> pd.DataFrame:
    """
    Imputes missing values based on a dictionary of column -> strategy mapping.
    Supported strategies: mean, median, mode, drop_rows, drop_column, none.
    """
    df_clean = df.copy()
    
    cols_to_drop = []
    
    for col, strategy in strategies.items():
        if col not in df_clean.columns:
            continue
        
        missing_count = df_clean[col].isna().sum()
        if missing_count == 0:
            continue
            
        if strategy == "mean":
            val = df_clean[col].mean()
            df_clean[col] = df_clean[col].fillna(val)
        elif strategy == "median":
            val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(val)
        elif strategy == "mode":
            mode_series = df_clean[col].mode()
            if not mode_series.empty:
                val = mode_series[0]
                df_clean[col] = df_clean[col].fillna(val)
        elif strategy == "drop_rows":
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == "drop_column":
            cols_to_drop.append(col)
            
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        
    return df_clean

def get_duplicate_summary(df: pd.DataFrame) -> dict:
    """Returns duplicate row counts."""
    dup_count = int(df.duplicated().sum())
    total_rows = len(df)
    return {
        "total_rows": total_rows,
        "duplicate_rows": dup_count,
        "unique_rows": total_rows - dup_count
    }

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes all duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)

def detect_outliers(df: pd.DataFrame, col: str, method: str = "iqr") -> dict:
    """
    Detects outliers in a specific numerical column using either IQR or Z-score method.
    """
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return {"error": f"Column {col} is not a numeric column."}
        
    series = df[col].dropna()
    total_count = len(df)
    non_null_count = len(series)
    
    if non_null_count == 0:
        return {"error": f"Column {col} has only null values."}
        
    if method == "iqr":
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_indices = outliers.index.tolist()
        outlier_count = len(outliers)
        
        return {
            "method": "iqr",
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
            "outlier_percentage": round(outlier_count / total_count * 100, 2) if total_count > 0 else 0.0,
            "outlier_values": outliers.head(50).tolist(),
            "outlier_indices": outlier_indices
        }
    else:  # Z-score
        mean = float(series.mean())
        std = float(series.std())
        if std == 0:
            std = 1e-9
            
        z_scores = (series - mean) / std
        outliers = series[np.abs(z_scores) > 3]
        outlier_indices = outliers.index.tolist()
        outlier_count = len(outliers)
        
        return {
            "method": "zscore",
            "mean": mean,
            "std": std,
            "lower_bound": mean - 3 * std,
            "upper_bound": mean + 3 * std,
            "outlier_count": outlier_count,
            "outlier_percentage": round(outlier_count / total_count * 100, 2) if total_count > 0 else 0.0,
            "outlier_values": outliers.head(50).tolist(),
            "outlier_indices": outlier_indices
        }

def handle_outliers(df: pd.DataFrame, col: str, strategy: str, method: str = "iqr") -> pd.DataFrame:
    """
    Handles outliers for a numerical column:
    - keep: do nothing
    - cap: replace values outside boundaries with boundary values
    - remove: drop rows containing outliers
    """
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return df
        
    df_handled = df.copy()
    metrics = detect_outliers(df_handled, col, method)
    if "error" in metrics:
        return df_handled
        
    lower_bound = metrics["lower_bound"]
    upper_bound = metrics["upper_bound"]
    
    if strategy == "cap":
        df_handled[col] = np.clip(df_handled[col], lower_bound, upper_bound)
    elif strategy == "remove":
        # Filter indices where the column is not an outlier or is null
        non_nulls = df_handled[col].notna()
        outlier_mask = (df_handled[col] < lower_bound) | (df_handled[col] > upper_bound)
        df_handled = df_handled[~(non_nulls & outlier_mask)].reset_index(drop=True)
        
    return df_handled
