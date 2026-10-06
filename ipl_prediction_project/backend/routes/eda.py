# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: eda.py
# Path: backend/routes/eda.py
# Description: FastAPI router providing Exploratory Data Analysis metrics, outlier statistics, and missing value checks.
# ==============================================================================

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import (
    load_dataframe, save_dataframe, has_stage_dataframe, 
    get_latest_dataframe, sanitize_nans
)
import services.analysis as analysis
import services.cleaning as cleaning
import services.visualization as visualization

router = APIRouter(prefix="/eda", tags=["Exploratory Data Analysis"])

class ImputationRequest(BaseModel):
    strategies: dict[str, str]

class OutlierRequest(BaseModel):
    column: str
    strategy: str
    method: str = "iqr"

def get_current_df(session_id: str) -> tuple[pd.DataFrame, str]:
    """Helper to get the current dataframe state (cleaned if exists, else original)."""
    if has_stage_dataframe(session_id, "cleaned"):
        return load_dataframe(session_id, "cleaned"), "cleaned"
    return load_dataframe(session_id, "original"), "original"

@router.get("/overview")
async def get_overview(session_id: str = Query(..., description="Unique user session ID")):
    try:
        df, stage = get_current_df(session_id)
        types = analysis.classify_columns(df)
        
        # Build dataset description summary
        describe_dict = {}
        for col in df.columns:
            series = df[col].dropna()
            col_info = {
                "count": len(df),
                "non_null_count": len(series),
                "missing_count": int(df[col].isna().sum()),
                "unique_count": int(df[col].nunique())
            }
            if pd.api.types.is_numeric_dtype(df[col]) and not series.empty:
                col_info.update({
                    "mean": float(series.mean()),
                    "std": float(series.std()) if len(series) > 1 else 0.0,
                    "min": float(series.min()),
                    "max": float(series.max())
                })
            describe_dict[col] = col_info
            
        return sanitize_nans({
            "stage": stage,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "classified_columns": types,
            "classifications": {
                "numerical": len(types["numerical"]),
                "categorical": len(types["categorical"]),
                "date": len(types["date"]),
                "boolean": len(types["boolean"])
            },
            "summary": describe_dict
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality-check")
async def get_quality(session_id: str = Query(..., description="Unique user session ID")):
    try:
        df, stage = get_current_df(session_id)
        quality_data = analysis.get_data_quality(df)
        return sanitize_nans(quality_data)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/missing-values")
async def get_missing(session_id: str = Query(..., description="Unique user session ID")):
    try:
        df, stage = get_current_df(session_id)
        summary = cleaning.get_missing_summary(df)
        chart_dict = visualization.get_missing_values_chart(df)
        
        return sanitize_nans({
            "stage": stage,
            "summary": summary,
            "chart": chart_dict
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clean-missing")
async def clean_missing(
    session_id: str = Query(...),
    payload: ImputationRequest = Body(...)
):
    try:
        # Load from original (or cleaned if modifying an existing clean)
        df = load_dataframe(session_id, "original")
        df_clean = cleaning.impute_missing_values(df, payload.strategies)
        
        # Save as cleaned stage
        save_dataframe(df_clean, session_id, "cleaned")
        
        # Re-evaluate missing summary
        summary = cleaning.get_missing_summary(df_clean)
        return {"status": "success", "message": "Missing values imputed successfully.", "summary": summary}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/duplicates")
async def get_duplicates(session_id: str = Query(...)):
    try:
        df, stage = get_current_df(session_id)
        summary = cleaning.get_duplicate_summary(df)
        chart_dict = visualization.get_duplicates_chart(df)
        
        return sanitize_nans({
            "stage": stage,
            "summary": summary,
            "chart": chart_dict
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/remove-duplicates")
async def remove_dups(session_id: str = Query(...)):
    try:
        # Load latest dataframe (usually cleaned if missing values cleaned, or original)
        df, stage = get_current_df(session_id)
        df_clean = cleaning.remove_duplicates(df)
        
        # Save as cleaned
        save_dataframe(df_clean, session_id, "cleaned")
        
        summary = cleaning.get_duplicate_summary(df_clean)
        return {"status": "success", "message": "Duplicate rows removed successfully.", "summary": summary}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dtypes")
async def get_dtypes(session_id: str = Query(...)):
    try:
        df, stage = get_current_df(session_id)
        types = analysis.classify_columns(df)
        
        cols_summary = []
        for type_name, cols in types.items():
            for c in cols:
                cols_summary.append({
                    "column": c,
                    "type": type_name,
                    "dtype": str(df[c].dtype)
                })
                
        return {
            "stage": stage,
            "counts": {k: len(v) for k, v in types.items()},
            "details": cols_summary
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/univariate")
async def get_univariate(
    session_id: str = Query(...),
    column: str = Query(..., description="Feature to analyze")
):
    try:
        df, stage = get_current_df(session_id)
        stats = analysis.get_univariate_stats(df, column)
        chart_dict = visualization.get_univariate_chart(df, column)
        
        return sanitize_nans({
            "stage": stage,
            "column": column,
            "stats": stats,
            "chart": chart_dict
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/distribution")
async def get_distribution(session_id: str = Query(...)):
    try:
        df, stage = get_current_df(session_id)
        results = analysis.get_distribution_analysis(df)
        
        # Add distribution charts for all numerical columns
        for dist in results["distributions"]:
            col = dist["column"]
            dist["chart"] = visualization.get_distribution_chart(df, col)
            
        return sanitize_nans(results)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/outliers")
async def get_outliers(
    session_id: str = Query(...),
    column: str = Query(..., description="Numeric column to inspect"),
    method: str = Query("iqr", description="Outlier detection method: iqr or zscore")
):
    try:
        df, stage = get_current_df(session_id)
        metrics = cleaning.detect_outliers(df, column, method)
        if "error" in metrics:
            raise HTTPException(status_code=400, detail=metrics["error"])
            
        chart_dict = visualization.get_outliers_chart(
            df, column, metrics["lower_bound"], metrics["upper_bound"]
        )
        
        return sanitize_nans({
            "stage": stage,
            "column": column,
            "metrics": metrics,
            "chart": chart_dict
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/handle-outliers")
async def handle_outliers_endpoint(
    session_id: str = Query(...),
    payload: OutlierRequest = Body(...)
):
    try:
        df, stage = get_current_df(session_id)
        df_clean = cleaning.handle_outliers(df, payload.column, payload.strategy, payload.method)
        
        # Save as cleaned
        save_dataframe(df_clean, session_id, "cleaned")
        
        # Recalculate metrics
        metrics = cleaning.detect_outliers(df_clean, payload.column, payload.method)
        return {
            "status": "success", 
            "message": f"Outliers in '{payload.column}' handled using '{payload.strategy}' strategy.",
            "metrics": sanitize_nans(metrics)
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/bivariate")
async def get_bivariate(
    session_id: str = Query(...),
    feature_x: str = Query(...),
    feature_y: str = Query(...),
    show_trendline: bool = Query(True),
    show_points: bool = Query(True),
    agg_type: str = Query("mean"),
    normalize_stack: bool = Query(False),
    normalize_contingency: str = Query("none")
):
    try:
        df, stage = get_current_df(session_id)
        if feature_x not in df.columns or feature_y not in df.columns:
            raise HTTPException(status_code=400, detail=f"Invalid features: {feature_x} or {feature_y} not found.")
            
        is_x_num = pd.api.types.is_numeric_dtype(df[feature_x])
        is_y_num = pd.api.types.is_numeric_dtype(df[feature_y])
        
        # Case 1: Numerical vs Numerical
        if is_x_num and is_y_num:
            relationship_type = "num_vs_num"
            stats_data = analysis.get_num_vs_num_bivariate(df, feature_x, feature_y)
            scatter_chart = visualization.get_bivariate_scatter_chart(df, feature_x, feature_y, show_trendline=show_trendline)
            
            charts = {
                "scatter": scatter_chart
            }
            primary_chart = scatter_chart
            interpretation = stats_data.get("interpretation", "")
            
            return sanitize_nans({
                "stage": stage,
                "relationship_type": relationship_type,
                "feature_x": feature_x,
                "feature_y": feature_y,
                "chart": primary_chart,
                "charts": charts,
                "stats": stats_data,
                "interpretation": interpretation
            })
            
        # Case 2: Categorical vs Numerical
        elif (is_x_num and not is_y_num) or (not is_x_num and is_y_num):
            relationship_type = "cat_vs_num"
            num_col = feature_x if is_x_num else feature_y
            cat_col = feature_y if is_x_num else feature_x
            
            stats_data = analysis.get_cat_vs_num_bivariate(df, cat_col, num_col)
            if stats_data.get("too_many"):
                return sanitize_nans({
                    "stage": stage,
                    "relationship_type": relationship_type,
                    "feature_x": feature_x,
                    "feature_y": feature_y,
                    "cat_col": cat_col,
                    "num_col": num_col,
                    "too_many": True,
                    "message": stats_data["message"],
                    "stats": stats_data,
                    "chart": {}
                })
                
            box_chart = visualization.get_bivariate_box_chart(df, cat_col, num_col, show_points=show_points)
            violin_chart = visualization.get_bivariate_violin_chart(df, cat_col, num_col, show_points=show_points)
            bar_mean_chart = visualization.get_bivariate_bar_chart(df, cat_col, num_col, agg_type="mean")
            bar_median_chart = visualization.get_bivariate_bar_chart(df, cat_col, num_col, agg_type="median")
            
            charts = {
                "box": box_chart,
                "violin": violin_chart,
                "bar_mean": bar_mean_chart,
                "bar_median": bar_median_chart
            }
            primary_chart = box_chart
            interpretation = stats_data.get("interpretation", "")
            
            return sanitize_nans({
                "stage": stage,
                "relationship_type": relationship_type,
                "feature_x": feature_x,
                "feature_y": feature_y,
                "cat_col": cat_col,
                "num_col": num_col,
                "chart": primary_chart,
                "charts": charts,
                "stats": stats_data,
                "interpretation": interpretation
            })
            
        # Case 3: Categorical vs Categorical
        else:
            relationship_type = "cat_vs_cat"
            stats_data = analysis.get_cat_vs_cat_bivariate(df, feature_x, feature_y)
            if stats_data.get("too_many"):
                return sanitize_nans({
                    "stage": stage,
                    "relationship_type": relationship_type,
                    "feature_x": feature_x,
                    "feature_y": feature_y,
                    "too_many": True,
                    "message": stats_data["message"],
                    "stats": stats_data,
                    "chart": {}
                })
                
            heatmap_counts = visualization.get_bivariate_contingency_heatmap(df, feature_x, feature_y, normalize="none")
            heatmap_row_pct = visualization.get_bivariate_contingency_heatmap(df, feature_x, feature_y, normalize="row")
            heatmap_col_pct = visualization.get_bivariate_contingency_heatmap(df, feature_x, feature_y, normalize="column")
            heatmap_tot_pct = visualization.get_bivariate_contingency_heatmap(df, feature_x, feature_y, normalize="all")
            
            stacked_counts = visualization.get_bivariate_stacked_bar_chart(df, feature_x, feature_y, normalize=False)
            stacked_pct = visualization.get_bivariate_stacked_bar_chart(df, feature_x, feature_y, normalize=True)
            grouped_bar = visualization.get_bivariate_grouped_bar_chart(df, feature_x, feature_y)
            
            charts = {
                "heatmap_counts": heatmap_counts,
                "heatmap_row_pct": heatmap_row_pct,
                "heatmap_col_pct": heatmap_col_pct,
                "heatmap_tot_pct": heatmap_tot_pct,
                "stacked_counts": stacked_counts,
                "stacked_pct": stacked_pct,
                "grouped_bar": grouped_bar
            }
            primary_chart = stacked_counts
            interpretation = stats_data.get("interpretation", "")
            
            return sanitize_nans({
                "stage": stage,
                "relationship_type": relationship_type,
                "feature_x": feature_x,
                "feature_y": feature_y,
                "chart": primary_chart,
                "charts": charts,
                "stats": stats_data,
                "interpretation": interpretation
            })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/correlation")
async def get_correlation(session_id: str = Query(...)):
    try:
        df, stage = get_current_df(session_id)
        analysis_data = analysis.get_correlation_analysis(df)
        if "error" in analysis_data:
            return analysis_data
            
        chart_dict = visualization.get_correlation_heatmap(df, analysis_data["columns"])
        
        return sanitize_nans({
            "stage": stage,
            "correlation_data": analysis_data,
            "chart": chart_dict
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feature-selection")
async def get_feature_selection(
    session_id: str = Query(...),
    target: str = Query(None, description="Target column for mutual information")
):
    try:
        df, stage = get_current_df(session_id)
        recommendations = analysis.get_feature_selection_recommendations(df, target)
        
        # Separate selected and removed
        selected = [r for r in recommendations if r["status"] == "Select"]
        removed = [r for r in recommendations if r["status"] == "Remove"]
        
        return sanitize_nans({
            "stage": stage,
            "target": target,
            "recommendations": recommendations,
            "selected_features": [s["feature"] for s in selected],
            "removed_features": removed
        })
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_summary_report(
    session_id: str = Query(...),
    target: str = Query(None, description="Optional target feature")
):
    try:
        orig_df = load_dataframe(session_id, "original")
        df, latest_stage = get_latest_dataframe(session_id)
        
        # Basic Counts
        orig_rows, orig_cols = orig_df.shape
        curr_rows, curr_cols = df.shape
        
        types = analysis.classify_columns(orig_df)
        num_count = len(types["numerical"])
        cat_count = len(types["categorical"])
        
        # Missing & Dups before-after
        orig_missing = int(orig_df.isna().sum().sum())
        curr_missing = int(df.isna().sum().sum())
        
        orig_dups = int(orig_df.duplicated().sum())
        curr_dups = int(df.duplicated().sum())
        
        # Data Quality Score
        quality_data = analysis.get_data_quality(orig_df)
        quality_score = quality_data["quality_score"]
        
        # Correlation Multicollinearity
        corr_info = analysis.get_correlation_analysis(orig_df)
        multicol_pairs = len(corr_info.get("multicollinearity_warnings", [])) if "error" not in corr_info else 0
        
        # Transformations applied
        transformations = []
        if has_stage_dataframe(session_id, "cleaned"):
            transformations.append("Data Cleaning (Imputation & Duplicates)")
        if has_stage_dataframe(session_id, "encoded"):
            transformations.append("Categorical Encoding (One-Hot/Label)")
        if has_stage_dataframe(session_id, "scaled"):
            transformations.append("Numerical Feature Scaling")
        if has_stage_dataframe(session_id, "pca"):
            transformations.append("Principal Component Analysis (PCA)")
            
        return {
            "original_shape": [orig_rows, orig_cols],
            "current_shape": [curr_rows, curr_cols],
            "numerical_features_count": num_count,
            "categorical_features_count": cat_count,
            "original_missing_values": orig_missing,
            "current_missing_values": curr_missing,
            "original_duplicates": orig_dups,
            "current_duplicates": curr_dups,
            "data_quality_score": quality_score,
            "multicollinear_pairs_count": multicol_pairs,
            "transformations_applied": transformations,
            "latest_stage": latest_stage
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
