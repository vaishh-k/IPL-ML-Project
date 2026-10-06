# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: cards.py
# Path: frontend/components/cards.py
# Description: Streamlit layout widgets for info banners and automated data quality score gauges.
# ==============================================================================

import streamlit as st

def render_info_card(title: str, text: str, type: str = "info"):
    """
    Renders an info panel card.
    Types supported: info, warning, success, error.
    """
    colors = {
        "info": {"border": "#F97316", "bg": "#111827", "text": "#FB923C"},
        "warning": {"border": "#F59E0B", "bg": "#111827", "text": "#FBBF24"},
        "success": {"border": "#10B981", "bg": "#111827", "text": "#34D399"},
        "error": {"border": "#EF4444", "bg": "#111827", "text": "#F87171"}
    }
    cfg = colors.get(type, colors["info"])
    
    st.markdown(f"""
        <div style="border-left: 4px solid {cfg['border']}; background-color: {cfg['bg']}; padding: 1.25rem; border-radius: 0 8px 8px 0; border-top: 1px solid #263244; border-bottom: 1px solid #263244; border-right: 1px solid #263244; margin-bottom: 1rem;">
            <div style="font-weight: 600; color: {cfg['text']}; font-size: 1.05rem; margin-bottom: 0.25rem;">{title}</div>
            <div style="color: #D1D5DB; font-size: 0.95rem; line-height:1.5;">{text}</div>
        </div>
    """, unsafe_allow_html=True)

def render_quality_card(score: int):
    """
    Renders the data quality score block with an automated horizontal score bar.
    """
    # Color mapping based on score
    if score >= 80:
        bar_color = "#10B981" # Green
    elif score >= 50:
        bar_color = "#F59E0B" # Yellow
    else:
        bar_color = "#EF4444" # Red
        
    filled_blocks = int(score / 5) # Scale to 20 blocks
    bar_str = "█" * filled_blocks + "▒" * (20 - filled_blocks)
    
    st.markdown(f"""
        <div style="background-color: #172033; border: 1px solid #263244; border-radius: 12px; padding: 1.5rem; text-align: center; margin-bottom: 1.5rem;">
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.1rem; color: #9CA3AF; font-weight: 500; letter-spacing: 1px;">DATA QUALITY SCORE</div>
            <div style="font-size: 3.5rem; font-weight: 800; color: {bar_color}; margin-top: 0.25rem; margin-bottom: 0.5rem; line-height: 1;">{score}%</div>
            <div style="font-family: monospace; font-size: 1.4rem; color: {bar_color}; letter-spacing: 2px;">{bar_str}</div>
            <div style="font-size: 0.85rem; color: #9CA3AF; margin-top: 0.75rem; line-height: 1.4;">Calculated dynamically from dataset integrity checks: presence of missing values, duplicate rows, low-variance fields, and high-cardinality values.</div>
        </div>
    """, unsafe_allow_html=True)
