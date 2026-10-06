# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: progress.py
# Path: frontend/components/progress.py
# Description: Streamlit navigation headers displaying the 21-step pipeline progress bar.
# ==============================================================================

import streamlit as st
import numpy as np

# Node mappings matching the user image
TIMELINE_NODES = [
    {"label": "Dataset Overview", "icon": "", "color": "#F97316"},
    {"label": "Data Understanding", "icon": "", "color": "#06B6D4"},
    {"label": "Data Quality Check", "icon": "", "color": "#0EA5E9"},
    {"label": "Data Cleaning", "icon": "", "color": "#14B8A6"},
    {"label": "Missing Value Analysis", "icon": "", "color": "#10B981"},
    {"label": "Duplicate Analysis", "icon": "", "color": "#84CC16"},
    {"label": "Univariate Analysis", "icon": "", "color": "#F59E0B"},
    {"label": "Distribution Analysis", "icon": "", "color": "#F97316"},
    {"label": "Outlier Detection", "icon": "", "color": "#EF4444"},
    {"label": "Bivariate Analysis", "icon": "", "color": "#EC4899"},
    {"label": "Multivariate Analysis", "icon": "", "color": "#D946EF"},
    {"label": "Correlation Analysis", "icon": "", "color": "#8B5CF6"},
    {"label": "Feature Selection", "icon": "", "color": "#6366F1"},
    {"label": "Encoding & Scaling", "icon": "", "color": "#F97316"},
    {"label": "Train/Test Split", "icon": "", "color": "#10B981"},
    {"label": "PCA", "icon": "", "color": "#EAB308"},
    {"label": "PCA Visualization", "icon": "", "color": "#EF4444"},
    {"label": "Dataset & Summary", "icon": "", "color": "#EC4899"}
]

def get_active_circle_idx(current_step: int) -> int:
    """Maps the 20 current_step indices to the 19 nodes on the horizontal timeline."""
    mapping = {
        0: 0,   # Overview
        1: 1,   # Data Understanding
        2: 2,   # Quality Check
        3: 3,   # Data Cleaning
        4: 4,   # Missing Value Analysis
        5: 5,   # Duplicate Analysis
        6: 6,   # Univariate Analysis
        7: 7,   # Distribution Analysis
        8: 8,   # Outlier Detection
        9: 9,   # Bivariate Analysis
        10: 10, # Multivariate Analysis
        11: 11, # Correlation Analysis
        12: 12, # Feature Selection
        13: 13, # Categorical Encoding
        14: 13, # Numerical Scaling (mapped to Encoding & Scaling)
        15: 14, # Train/Test Split
        16: 15, # PCA
        17: 16, # PCA Visualization
        18: 17  # Final Processed Dataset & Summary
    }
    return mapping.get(current_step, 0)

def render_step_navigation(current_idx: int, total_steps: int) -> tuple[bool, bool]:
    """Renders Previous/Next step buttons and step indicator row."""
    st.markdown("<hr style='margin-top: 2rem; margin-bottom: 1.5rem; border-color: #263244;' />", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    prev_clicked = False
    next_clicked = False
    with col1:
        if current_idx > 0:
            prev_clicked = st.button(" Previous", use_container_width=True)
    with col2:
        st.markdown(f"""
            <div style="text-align: center; font-size: 1rem; font-weight: 600; color: #9CA3AF; padding-top: 0.25rem;">
                Step {current_idx + 1} / {total_steps}
            </div>
        """, unsafe_allow_html=True)
    with col3:
        if current_idx < total_steps - 1:
            next_clicked = st.button("Next ", use_container_width=True)
    return prev_clicked, next_clicked

def render_horizontal_timeline(current_step: int):
    """
    Renders a premium custom styled horizontal timeline progress bar
    at the top of the main container.
    """
    active_idx = get_active_circle_idx(current_step)
    
    # Custom CSS and HTML generation for the nodes
    nodes_html = []
    
    for idx, node in enumerate(TIMELINE_NODES):
        color = node["color"]
        label = node["label"]
        icon = node["icon"]
        
        # Determine status
        if idx == active_idx:
            status_class = "active"
            node_style = f"border: 3px solid {color}; color: #FFFFFF; box-shadow: 0 0 16px {color}, inset 0 0 8px {color}; background-color: #0B0F14;"
            label_style = f"color: {color}; font-weight: 600; text-shadow: 0 0 8px rgba(249, 115, 22, 0.4);"
        elif idx < active_idx:
            status_class = "completed"
            node_style = f"border: 2px solid {color}; color: {color}; opacity: 0.85; background-color: #111827;"
            label_style = "color: #9CA3AF; font-weight: 400;"
        else:
            status_class = "pending"
            node_style = "border: 2px solid #374151; color: #4B5563; opacity: 0.4; background-color: #111827;"
            label_style = "color: #4B5563; font-weight: 400; opacity: 0.5;"
            
        node_html = (
            f"<div class='timeline-node {status_class}' style='display: flex; flex-direction: column; align-items: center; min-width: 58px; z-index: 2; position: relative;'>"
            f"<div class='node-circle' style='width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1rem; transition: all 0.3s ease; {node_style}'>"
            f"{icon}"
            f"</div>"
            f"<div class='node-label' style='font-size: 0.55rem; text-align: center; margin-top: 6px; line-height: 1.2; width: 58px; white-space: normal; height: 26px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; {label_style}'>"
            f"{label}"
            f"</div>"
            f"</div>"
        )
        nodes_html.append(node_html)

    # Render entire timeline with custom CSS overrides (static layout, no scrollbars)
    css_style = (
        "<style>"
        ".timeline-container::-webkit-scrollbar { display: none; }"
        "</style>"
    )
    
    container_html = (
        f"<div class='timeline-container' style='display: flex; justify-content: space-between; align-items: flex-start; position: relative; margin-bottom: 1.5rem; padding: 0.75rem 0.25rem; background-color: #111827; border: 1px solid #263244; border-radius: 10px; overflow: hidden; gap: 2px;'>"
        f"<div class='timeline-line' style='position: absolute; top: 25px; left: 30px; right: 30px; height: 3px; background: linear-gradient(90deg, #F97316 0%, #06B6D4 15%, #10B981 30%, #84CC16 45%, #EAB308 55%, #F97316 65%, #EF4444 75%, #EC4899 85%, #8B5CF6 100%); z-index: 1;'></div>"
        f"{''.join(nodes_html)}"
        f"</div>"
    )
    
    st.markdown(css_style + container_html, unsafe_allow_html=True)
