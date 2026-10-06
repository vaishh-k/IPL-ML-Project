# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: pca_service.py
# Path: backend/services/pca_service.py
# Description: Service layer executing PCA fitting, variance projections, and PC coordinate calculations.
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def run_pca(df: pd.DataFrame, n_components: int = 2, target_col: str = None) -> dict:
    """
    Performs Principal Component Analysis on the dataset.
    Excludes the target column and non-numeric columns.
    Imputes missing values with medians to prevent PCA crashes.
    """
    # 1. Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target column if it's numeric
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
        
    # Exclude ID-like columns (match_id, index, etc.)
    for id_col in ["match_id", "id", "index"]:
        if id_col in numeric_cols:
            numeric_cols.remove(id_col)
            
    if not numeric_cols:
        return {"error": "No numerical features found for PCA."}
        
    # Ensure n_components is valid
    max_components = min(len(numeric_cols), len(df))
    n_comp = min(n_components, max_components)
    if n_comp < 1:
        n_comp = 2
        
    # 2. Extract PCA features and impute leaked NaNs
    X = df[numeric_cols].copy()
    X = X.fillna(X.median())
    
    # 3. Fit PCA
    pca = PCA(n_components=n_comp)
    X_projected = pca.fit_transform(X)
    
    # 4. Create projection dataframe
    proj_cols = [f"PC{i+1}" for i in range(n_comp)]
    proj_df = pd.DataFrame(X_projected, columns=proj_cols, index=df.index)
    
    # Merge target column back if it exists in original df
    if target_col and target_col in df.columns:
        proj_df[target_col] = df[target_col]
        
    # 5. Extract results
    explained_var = pca.explained_variance_ratio_.tolist()
    cumulative_var = np.cumsum(pca.explained_variance_ratio_).tolist()
    
    # Return projected dataframe (to be saved in session state) and stats
    return {
        "columns_used": numeric_cols,
        "n_components": n_comp,
        "explained_variance_ratio": explained_var,
        "cumulative_variance_ratio": cumulative_var,
        "projection_dataframe": proj_df
    }
