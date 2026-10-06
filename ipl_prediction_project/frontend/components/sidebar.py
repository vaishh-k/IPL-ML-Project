import streamlit as st

STEPS = [
    "Dataset Overview",
    "Data Understanding",
    "Data Quality Check",
    "Data Cleaning",
    "Missing Value Analysis",
    "Duplicate Analysis",
    "Univariate Analysis",
    "Distribution Analysis",
    "Outlier Detection",
    "Bivariate Analysis",
    "Multivariate Analysis",
    "Correlation Analysis",
    "Feature Selection",
    "Categorical Encoding",
    "Numerical Scaling",
    "Train/Test Preparation",
    "PCA",
    "PCA Visualization",
    "Final Processed Dataset & Summary"
]

_NAV_CSS = """
<style>
/* Active EDA nav button */
.nav-eda-active + div button {
    background: rgba(249,115,22,0.18) !important;
    border: 1.5px solid #F97316 !important;
    color: #F97316 !important;
    font-weight: 700 !important;
}
/* Active Regression nav button */
.nav-reg-active + div button {
    background: rgba(249,115,22,0.18) !important;
    border: 1.5px solid #F97316 !important;
    color: #F97316 !important;
    font-weight: 700 !important;
}
/* Inactive nav buttons  subtle secondary style */
.nav-eda-inactive + div button,
.nav-reg-inactive + div button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #9CA3AF !important;
}
</style>
"""

def render_sidebar(current_step: int, max_completed_step: int):
    """
    Renders the sidebar navigation with:
      - EDA VISUALIZER heading & ML Data Prep Pipeline button  (orange when active)
      - ALGORITHMS section with Regression button              (orange when active)
    Returns:
        tuple (clicked_step, restart_clicked, pipeline_start_clicked, regression_clicked)
    """
    reg_mode   = st.session_state.get("reg_mode", False)
    cls_mode   = st.session_state.get("cls_mode", False)
    unsup_mode = st.session_state.get("unsup_mode", False)
    ens_mode   = st.session_state.get("ens_mode", False)
    eda_active = not reg_mode and not cls_mode and not unsup_mode and not ens_mode  # EDA is active only if neither is active

    #  Inject nav highlight CSS 
    st.sidebar.markdown(_NAV_CSS, unsafe_allow_html=True)

    #  Brand Header 
    st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 1rem; border-bottom: 1px solid #263244; padding-bottom: 0.5rem;">
            <h2 style="margin:0; font-size:1.8rem; background: linear-gradient(135deg, #FB923C 0%, #F97316 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">IPL ML LAB</h2>
            <div style="font-size:1rem; color:#9CA3AF; letter-spacing: 1px; margin-top:0.25rem;">EDA VISUALIZER</div>
        </div>
    """, unsafe_allow_html=True)

    #  EDA Pipeline button 
    st.sidebar.markdown("<div style='margin-bottom: 0.3rem;'></div>", unsafe_allow_html=True)

    # Marker div  CSS sibling selector targets the button immediately after this
    eda_class = "nav-eda-active" if eda_active else "nav-eda-inactive"
    st.sidebar.markdown(f"<div class='{eda_class}'></div>", unsafe_allow_html=True)

    is_dataset_staged = st.session_state.get("temp_loaded", False) or st.session_state.get("dataset_loaded", False)

    pipeline_start_clicked = st.sidebar.button(
        "Machine Learning Data Preparation Pipeline",
        use_container_width=True,
        key="ml_data_prep_pipeline_btn"
    )

    if pipeline_start_clicked and not is_dataset_staged:
        st.sidebar.warning("Please load a dataset first!")
        pipeline_start_clicked = False

    #  ALGORITHMS section 
    st.sidebar.markdown("""
        <div style="margin-top:1.2rem; margin-bottom:0.5rem; border-top:1px solid #263244; padding-top:1rem;">
            <div style="font-size:0.72rem; color:#F97316; font-weight:700; letter-spacing:2px; text-transform:uppercase;">
                Algorithms
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Marker div for Regression button
    reg_class = "nav-reg-active" if reg_mode else "nav-reg-inactive"
    st.sidebar.markdown(f"<div class='{reg_class}'></div>", unsafe_allow_html=True)

    regression_clicked = st.sidebar.button(
        "Regression",
        use_container_width=True,
        key="regression_nav_btn"
    )

    # Marker div for Classification button
    cls_class = "nav-reg-active" if cls_mode else "nav-reg-inactive"
    st.sidebar.markdown(f"<div class='{cls_class}'></div>", unsafe_allow_html=True)

    classification_clicked = st.sidebar.button(
        "Classification",
        use_container_width=True,
        key="classification_nav_btn"
    )

    # Marker div for Unsupervised button
    unsup_class = "nav-reg-active" if unsup_mode else "nav-reg-inactive"
    st.sidebar.markdown(f"<div class='{unsup_class}'></div>", unsafe_allow_html=True)

    unsupervised_clicked = st.sidebar.button(
        "Unsupervised",
        use_container_width=True,
        key="unsupervised_nav_btn"
    )

    # Marker div for Ensemble button
    ens_class = "nav-reg-active" if ens_mode else "nav-reg-inactive"
    st.sidebar.markdown(f"<div class='{ens_class}'></div>", unsafe_allow_html=True)

    ensemble_clicked = st.sidebar.button(
        "Ensemble Learning",
        use_container_width=True,
        key="ensemble_nav_btn"
    )

    return None, False, pipeline_start_clicked, regression_clicked, classification_clicked, unsupervised_clicked, ensemble_clicked
