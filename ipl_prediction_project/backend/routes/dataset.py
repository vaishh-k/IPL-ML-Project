# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: dataset.py
# Path: backend/routes/dataset.py
# Description: FastAPI router handling dataset file loading, status check, previews, summary stats, and processing stages.
# ==============================================================================

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
import pandas as pd
import os
import sys

# Add project root to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.data_loader import load_uploaded_file
from utils.helpers import save_dataframe, load_dataframe, has_stage_dataframe, get_latest_dataframe, sanitize_nans

router = APIRouter(prefix="/dataset", tags=["Dataset"])

SAMPLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "ipl_sample.csv")

def get_preview_metadata(df: pd.DataFrame) -> dict:
    """Helper to generate metadata and row preview from a dataframe."""
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Calculate numerical and categorical counts
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Calculate memory usage
    mem_bytes = df.memory_usage(deep=True).sum()
    if mem_bytes < 1024:
        mem_str = f"{mem_bytes} B"
    elif mem_bytes < 1024 * 1024:
        mem_str = f"{mem_bytes / 1024:.1f} KB"
    else:
        mem_str = f"{mem_bytes / (1024 * 1024):.1f} MB"
        
    return {
        "rows": total_rows,
        "columns": total_cols,
        "numerical_cols_count": len(num_cols),
        "categorical_cols_count": len(cat_cols),
        "memory_usage": mem_str,
        "column_names": df.columns.tolist()
    }

@router.post("/upload")
async def upload_dataset(
    session_id: str = Query(..., description="Unique user session ID"),
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()
        df = load_uploaded_file(contents, file.filename)
        
        # Save as original and also copy to latest stages to initialize
        save_dataframe(df, session_id, "original")
        
        meta = get_preview_metadata(df)
        return {"status": "success", "message": "Dataset uploaded successfully.", "metadata": meta}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/select-sample")
async def select_sample(
    session_id: str = Query(..., description="Unique user session ID")
):
    if not os.path.exists(SAMPLE_PATH):
        raise HTTPException(status_code=404, detail="Sample dataset not found on server.")
    try:
        df = pd.read_csv(SAMPLE_PATH)
        save_dataframe(df, session_id, "original")
        
        meta = get_preview_metadata(df)
        return {"status": "success", "message": "Sample IPL dataset selected successfully.", "metadata": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/preview")
async def get_preview(
    session_id: str = Query(..., description="Unique user session ID"),
    stage: str = Query(None, description="Pipeline stage to preview: original, cleaned, encoded, scaled, pca"),
    rows_type: str = Query("first", description="Rows to slice: first or last"),
    limit: int = Query(10, description="Number of rows to return")
):
    try:
        # Load the requested stage, or fallback to the latest stage
        if stage:
            if not has_stage_dataframe(session_id, stage):
                raise HTTPException(status_code=404, detail=f"Stage '{stage}' data not found. Please execute earlier steps first.")
            df = load_dataframe(session_id, stage)
            stage_name = stage
        else:
            df, stage_name = get_latest_dataframe(session_id)
            
        metadata = get_preview_metadata(df)
        
        # Get head or tail
        if rows_type == "last":
            sample_df = df.tail(limit)
        else:
            sample_df = df.head(limit)
            
        # Convert values to native Python types and handle NaN/Inf
        records = sample_df.to_dict(orient="records")
        sanitized_records = sanitize_nans(records)
        
        # Also send datatypes for display
        dtypes = {col: str(df[col].dtype) for col in df.columns}
        
        return {
            "stage": stage_name,
            "metadata": metadata,
            "dtypes": dtypes,
            "records": sanitized_records
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list-local")
async def list_local():
    """Lists all CSV files in the project's data directory."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
    try:
        if not os.path.exists(data_dir):
            return {"files": []}
        # Filter files to only show CSV
        files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        return {"files": sorted(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/select-local")
async def select_local(
    filename: str = Query(..., description="Name of the CSV file to load"),
    session_id: str = Query(..., description="Unique user session ID")
):
    """Loads a specific CSV file from the local data directory into the current session."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
    file_path = os.path.join(data_dir, filename)
    
    # Path traversal security check
    if not os.path.abspath(file_path).startswith(os.path.abspath(data_dir)):
        raise HTTPException(status_code=400, detail="Access denied: invalid file path.")
        
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"CSV file '{filename}' not found on server.")
        
    try:
        df = pd.read_csv(file_path)
        save_dataframe(df, session_id, "original")
        meta = get_preview_metadata(df)
        return {"status": "success", "message": f"Local file '{filename}' loaded successfully.", "metadata": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save-processed")
async def save_processed(
    session_id: str = Query(..., description="Unique user session ID"),
    filename: str = Query(..., description="Original filename of the dataset")
):
    """Saves the latest stage dataframe of the session into the data folder as processed_<filename>."""
    try:
        df, stage_name = get_latest_dataframe(session_id)
        
        # Determine output filename
        out_filename = f"processed_{filename}"
        if not out_filename.endswith(".csv"):
            out_filename += ".csv"
            
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
        out_path = os.path.join(data_dir, out_filename)
        
        df.to_csv(out_path, index=False)
        return {"status": "success", "message": f"Processed dataset saved successfully as {out_filename} in the data folder."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
