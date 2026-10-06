# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: pca.py
# Path: backend/routes/pca.py
# Description: FastAPI router managing PCA computation, variance analysis, and dimensionality reduction steps.
# ==============================================================================

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import (
    load_dataframe, save_dataframe, get_latest_dataframe, sanitize_nans
)
import services.pca_service as pca_service
import services.visualization as visualization

router = APIRouter(prefix="/pca", tags=["Dimensionality Reduction (PCA)"])

class PcaRequest(BaseModel):
    n_components: int = 2
    target_col: str = None

@router.post("/run")
async def run_pca_endpoint(
    session_id: str = Query(...),
    payload: PcaRequest = Body(...)
):
    try:
        # Load from scaled if it exists, otherwise encoded, otherwise cleaned, otherwise original
        # This aligns with the pipeline flow where PCA follows scaling.
        df, stage = get_latest_dataframe(session_id)
        
        results = pca_service.run_pca(df, payload.n_components, payload.target_col)
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
            
        projection_df = results["projection_dataframe"]
        
        # Save the projection dataframe as the 'pca' stage in the session state
        save_dataframe(projection_df, session_id, "pca")
        
        # Generate Plotly charts
        variance_chart = visualization.get_pca_variance_chart(results["explained_variance_ratio"])
        
        pca_2d_chart = visualization.get_pca_2d_chart(projection_df, payload.target_col)
        
        pca_3d_chart = {}
        if payload.n_components >= 3:
            pca_3d_chart = visualization.get_pca_3d_chart(projection_df, payload.target_col)
            
        return sanitize_nans({
            "status": "success",
            "message": f"PCA run successfully on {len(results['columns_used'])} features.",
            "stage_source": stage,
            "columns_used": results["columns_used"],
            "explained_variance_ratio": results["explained_variance_ratio"],
            "cumulative_variance_ratio": results["cumulative_variance_ratio"],
            "charts": {
                "variance": variance_chart,
                "pca_2d": pca_2d_chart,
                "pca_3d": pca_3d_chart
            }
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
