# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: preprocessing.py
# Path: backend/routes/preprocessing.py
# Description: FastAPI router orchestrating missing values filling, duplicates dropping, scaling, and categorical encoding.
# ==============================================================================

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import (
    load_dataframe, save_dataframe, has_stage_dataframe, 
    get_latest_dataframe, sanitize_nans
)
import services.preprocessing as prep

router = APIRouter(prefix="/preprocessing", tags=["Data Preprocessing"])

class EncodeRequest(BaseModel):
    encoding_map: dict[str, str]

class ScaleRequest(BaseModel):
    scaling_map: dict[str, str]

class SplitRequest(BaseModel):
    test_size: float = 0.2

@router.post("/encode")
async def encode_categorical_endpoint(
    session_id: str = Query(...),
    payload: EncodeRequest = Body(...)
):
    try:
        # Load from cleaned (which is the output of cleaning steps), or original
        if has_stage_dataframe(session_id, "cleaned"):
            df = load_dataframe(session_id, "cleaned")
        else:
            df = load_dataframe(session_id, "original")
            
        df_encoded, metadata = prep.encode_categorical(df, payload.encoding_map)
        
        # Save as encoded stage
        save_dataframe(df_encoded, session_id, "encoded")
        
        return sanitize_nans({
            "status": "success",
            "message": "Categorical columns encoded successfully.",
            "shape": df_encoded.shape,
            "metadata": metadata
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scale")
async def scale_numerical_endpoint(
    session_id: str = Query(...),
    payload: ScaleRequest = Body(...)
):
    try:
        # Load from encoded if it exists, otherwise cleaned, otherwise original
        if has_stage_dataframe(session_id, "encoded"):
            df = load_dataframe(session_id, "encoded")
        elif has_stage_dataframe(session_id, "cleaned"):
            df = load_dataframe(session_id, "cleaned")
        else:
            df = load_dataframe(session_id, "original")
            
        df_scaled, metadata = prep.scale_numerical(df, payload.scaling_map)
        
        # Save as scaled stage
        save_dataframe(df_scaled, session_id, "scaled")
        
        return sanitize_nans({
            "status": "success",
            "message": "Numerical columns scaled successfully.",
            "shape": df_scaled.shape,
            "metadata": metadata
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/split")
async def split_dataset_endpoint(
    session_id: str = Query(...),
    payload: SplitRequest = Body(...)
):
    try:
        # Split the latest processed dataframe available
        df, stage = get_latest_dataframe(session_id)
        
        split_results = prep.split_train_test(df, payload.test_size)
        
        # We also generate a pie chart representing the train vs test split sizes
        import plotly.graph_objects as go
        from services.visualization import apply_dark_theme, ACCENT_GREEN, ACCENT_BLUE
        
        fig = go.Figure(data=[go.Pie(
            labels=["Training Set (Learn)", "Testing Set (Evaluate)"],
            values=[split_results["train_count"], split_results["test_count"]],
            hole=.4,
            marker=dict(colors=[ACCENT_BLUE, ACCENT_GREEN])
        )])
        fig.update_layout(title_text="Train-Test Split Distribution Ratio")
        apply_dark_theme(fig)
        
        return sanitize_nans({
            "stage": stage,
            "split_info": split_results,
            "chart": fig.to_dict()
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
