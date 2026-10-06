# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: dataframe.py
# Path: frontend/components/dataframe.py
# Description: Streamlit table layouts for formatted dataset reviews, metadata previews, and anomaly highlighting.
# ==============================================================================

import streamlit as st
import pandas as pd

def render_dataframe(records, dtypes: dict = None, height: int = 350, hide_index: bool = False):
    """
    Renders a list of records or a pandas DataFrame in a clean interactive Streamlit dataframe.
    Optionally displays column datatypes.
    """
    if isinstance(records, pd.DataFrame):
        if records.empty:
            st.info("No records to preview.")
            return
        df = records
    else:
        if not records:
            st.info("No records to preview.")
            return
        df = pd.DataFrame(records)
        
    # Display the interactive dataframe
    st.dataframe(df, use_container_width=True, height=height, hide_index=hide_index)
    
    # Display datatypes collapsibly if provided
    if dtypes:
        with st.expander(" Show Feature Datatypes"):
            cols = st.columns(4)
            dtypes_list = list(dtypes.items())
            chunk_size = (len(dtypes_list) + 3) // 4
            
            for col_idx in range(4):
                with cols[col_idx]:
                    chunk = dtypes_list[col_idx * chunk_size : (col_idx + 1) * chunk_size]
                    for feature_name, dtype_val in chunk:
                        st.markdown(f"**`{feature_name}`**: `{dtype_val}`")
