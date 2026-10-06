# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: data_loader.py
# Path: backend/services/data_loader.py
# Description: Utility service managing CSV data parsing, schema checks, and session caching on the server.
# ==============================================================================

import pandas as pd
import io

def load_uploaded_file(contents: bytes, filename: str) -> pd.DataFrame:
    """
    Loads raw bytes from a file (CSV or XLSX) into a Pandas DataFrame.
    Trims column names to prevent issues.
    """
    if filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    elif filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        raise ValueError("Unsupported file format. Please upload a .csv or .xlsx file.")
    
    # Standardize column names (strip whitespaces)
    df.columns = [str(col).strip() for col in df.columns]
    return df
