# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: charts.py
# Path: frontend/components/charts.py
# Description: Plotly chart layouts for scatter plots, correlation heatmaps, box plots, and histograms.
# ==============================================================================

import streamlit as st
import plotly.graph_objects as go

def render_plotly_chart(figure_dict: dict, key: str = None):
    """
    Reconstructs a Plotly figure from a serialized dictionary and renders it in Streamlit.
    """
    if not figure_dict or "data" not in figure_dict:
        st.info("No chart data available.")
        return
        
    try:
        # Reconstruct the figure
        fig = go.Figure(figure_dict)
        # Render the chart
        st.plotly_chart(fig, use_container_width=True, key=key)
    except Exception as e:
        st.error(f"Error rendering chart: {str(e)}")
