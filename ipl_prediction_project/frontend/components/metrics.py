# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: metrics.py
# Path: frontend/components/metrics.py
# Description: Streamlit grid components to display model scores and summary counts.
# ==============================================================================

import streamlit as st

def render_metric_grid(metrics: list[dict], cols_per_row: int = 3):
    """
    Renders a list of metric dictionaries in a responsive column grid.
    Each metric dict should have: label, value, delta (optional), help (optional).
    """
    if not metrics:
        return
        
    # Split metrics into chunks of cols_per_row size
    for i in range(0, len(metrics), cols_per_row):
        row_metrics = metrics[i : i + cols_per_row]
        cols = st.columns(len(row_metrics))
        
        for idx, metric in enumerate(row_metrics):
            with cols[idx]:
                st.metric(
                    label=metric.get("label", ""),
                    value=metric.get("value", ""),
                    delta=metric.get("delta", None),
                    help=metric.get("help", None)
                )
