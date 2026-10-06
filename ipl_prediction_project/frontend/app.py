
import streamlit as st
st.set_page_config(
    page_title="IPL Project",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

import uuid
import time
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.api_client import ApiClient
from ipl_utils.helpers import inject_custom_css, render_explanation, render_card, save_current_dataset_state, load_dataset_state
from components.sidebar import render_sidebar, STEPS
from components.progress import render_step_navigation, render_horizontal_timeline
from components.metrics import render_metric_grid
from components.charts import render_plotly_chart
from components.dataframe import render_dataframe
from components.cards import render_info_card, render_quality_card
from components.regression_page import render_regression_page
from components.classification_page import render_classification_page
from components.unsupervised_page import render_unsupervised_page
from components.ensemble_page import render_ensemble_page
from components.bivariate_page import render_bivariate_page

# Initialize Page Settings


# Inject Dark Theme CSS
inject_custom_css()

# Initialize session state variables
if "base_session_id" not in st.session_state:
    st.session_state["base_session_id"] = str(uuid.uuid4())
    st.session_state["dataset_states"] = {}

# Temp client to fetch initial dataset list
temp_api = ApiClient("temp_init")
res = temp_api.list_local_datasets()
if not res.get("error") and "files" in res:
    st.session_state["available_datasets"] = res["files"]
else:
    st.session_state["available_datasets"] = []

if "active_dataset" not in st.session_state:
    # Prioritize the smaller matches dataset for instantaneous initial loading
    if "ipl_matches.csv" in st.session_state["available_datasets"]:
        st.session_state["active_dataset"] = "ipl_matches.csv"
    elif st.session_state["available_datasets"]:
        st.session_state["active_dataset"] = st.session_state["available_datasets"][0]
    else:
        st.session_state["active_dataset"] = "ipl_matches.csv"

# State helpers are now imported from utils.helpers

# Init regression mode flag
if "reg_mode" not in st.session_state:
    st.session_state["reg_mode"] = False

# Load active dataset's state on first run to initialize session state keys
if "dataset_loaded" not in st.session_state:
    load_dataset_state(st.session_state["active_dataset"])

# Save the current state to the active dataset dictionary so that it is always synchronized on every run
save_current_dataset_state()

# Render active dataset selector at the top of the sidebar
st.sidebar.markdown("###  Active Dataset")
files = st.session_state.get("available_datasets", [])
if files:
    try:
        active_idx = files.index(st.session_state["active_dataset"])
    except ValueError:
        active_idx = 0
        st.session_state["active_dataset"] = files[0]
        
    selected_dataset = st.sidebar.selectbox(
        "Switch Dataset:",
        options=files,
        index=active_idx,
        key="active_dataset_selectbox_sidebar"
    )
    
    if selected_dataset != st.session_state["active_dataset"]:
        save_current_dataset_state()
        st.session_state["active_dataset"] = selected_dataset
        load_dataset_state(selected_dataset)
        st.rerun()

# Instantiate API Client using the subsession ID
st.session_state["session_id"] = f"{st.session_state['base_session_id']}_{st.session_state['active_dataset']}"
api = ApiClient(st.session_state["session_id"])

#  Render sidebar ONCE (avoids duplicate-key errors across routing branches) 
_sb_step  = st.session_state.get("current_step", 0)
_sb_max   = st.session_state.get("max_completed_step", -1)
_clicked_step, _restart_clicked, _pipeline_start_clicked, _regression_clicked, _classification_clicked, _unsupervised_clicked, _ensemble_clicked = render_sidebar(_sb_step, _sb_max)

#  URL query-param sync 
# Read URL params ONLY ONCE on first page load (for bookmark/sharing restore).
# After that, session_state is the source of truth  reading params on every
# rerun would override button-driven navigation and break the Next/Prev flow.
if not st.session_state.get("_url_params_applied", False):
    st.session_state["_url_params_applied"] = True
    _qp_init = st.query_params
    if "step" in _qp_init and st.session_state.get("dataset_loaded"):
        try:
            _url_step = int(_qp_init["step"])
            if 0 <= _url_step < len(STEPS):
                st.session_state["current_step"] = _url_step
        except (ValueError, TypeError):
            pass
    if _qp_init.get("page") == "regression":
        st.session_state["reg_mode"] = True
    elif _qp_init.get("page") == "classification":
        st.session_state["cls_mode"] = True
    elif _qp_init.get("page") == "unsupervised":
        st.session_state["unsup_mode"] = True
    elif _qp_init.get("page") == "ensemble":
        st.session_state["ens_mode"] = True

def _sync_url():
    """Write current page/step to the browser URL bar (write-only, never read after first load)."""
    try:
        if st.session_state.get("reg_mode"):
            st.query_params["page"] = "regression"
            if "step" in st.query_params:
                del st.query_params["step"]
        elif st.session_state.get("cls_mode"):
            st.query_params["page"] = "classification"
            if "step" in st.query_params:
                del st.query_params["step"]
        elif st.session_state.get("unsup_mode"):
            st.query_params["page"] = "unsupervised"
            if "step" in st.query_params:
                del st.query_params["step"]
        elif st.session_state.get("ens_mode"):
            st.query_params["page"] = "ensemble"
            if "step" in st.query_params:
                del st.query_params["step"]
        elif st.session_state.get("dataset_loaded"):
            st.query_params["page"] = "eda"
            st.query_params["step"] = str(st.session_state.get("current_step", 0))
        else:
            for _pk in ["page", "step"]:
                if _pk in st.query_params:
                    del st.query_params[_pk]
    except Exception:
        pass  # query_params unavailable in older Streamlit versions

_sync_url()

def simulate_animation(stage_name: str):
    """Simulates a premium ML analysis loading animation sequence."""
    status_placeholder = st.empty()
    messages = [
        f" Analyzing Dataset for {stage_name}...",
        f" Cleaning Data & Imputing...",
        f" Generating Plotly Visualizations...",
        f" Analysis Complete!"
    ]
    for msg in messages:
        status_placeholder.markdown(f"""
            <div style="background-color: #172033; border: 1px solid #263244; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem; text-align: center;">
                <div style="font-size: 1rem; color: #F97316; font-weight: 500; margin-bottom:0.5rem;">{msg}</div>
                <div class="progress-bar-container" style="background-color:#111827; height:4px; border-radius:2px; overflow:hidden;">
                    <div style="background-color:#F97316; height:100%; width: 50%; animation: shift 1.5s infinite ease-in-out;"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(0.3)
    status_placeholder.empty()

# ----------------- REGRESSION PIPELINE -----------------
if st.session_state.get("reg_mode", False):
    if _pipeline_start_clicked:
        st.session_state["reg_mode"] = False
        st.session_state["dataset_loaded"] = True
        st.rerun()
    if _classification_clicked:
        st.session_state["reg_mode"] = False
        st.session_state["cls_mode"] = True
        st.rerun()
    if _unsupervised_clicked:
        st.session_state["reg_mode"] = False
        st.session_state["unsup_mode"] = True
        st.rerun()
    if _ensemble_clicked:
        st.session_state["reg_mode"] = False
        st.session_state["ens_mode"] = True
        st.rerun()
    if st.button(" Back to EDA / Home", key="reg_back_btn"):
        st.session_state["reg_mode"] = False
        st.rerun()
    render_regression_page(api)
    st.stop()

# ----------------- CLASSIFICATION PIPELINE -----------------
elif st.session_state.get("cls_mode", False):
    if _pipeline_start_clicked:
        st.session_state["cls_mode"] = False
        st.session_state["dataset_loaded"] = True
        st.rerun()
    if _regression_clicked:
        st.session_state["cls_mode"] = False
        st.session_state["reg_mode"] = True
        st.rerun()
    if _unsupervised_clicked:
        st.session_state["cls_mode"] = False
        st.session_state["unsup_mode"] = True
        st.rerun()
    if _ensemble_clicked:
        st.session_state["cls_mode"] = False
        st.session_state["ens_mode"] = True
        st.rerun()
    if st.button(" Back to EDA / Home", key="cls_back_btn"):
        st.session_state["cls_mode"] = False
        st.rerun()
    render_classification_page(api)
    st.stop()

# ----------------- UNSUPERVISED PIPELINE -----------------
elif st.session_state.get("unsup_mode", False):
    if _pipeline_start_clicked:
        st.session_state["unsup_mode"] = False
        st.session_state["dataset_loaded"] = True
        st.rerun()
    if _regression_clicked:
        st.session_state["unsup_mode"] = False
        st.session_state["reg_mode"] = True
        st.rerun()
    if _classification_clicked:
        st.session_state["unsup_mode"] = False
        st.session_state["cls_mode"] = True
        st.rerun()
    if _ensemble_clicked:
        st.session_state["unsup_mode"] = False
        st.session_state["ens_mode"] = True
        st.rerun()
    if st.button(" Back to EDA / Home", key="unsup_back_btn"):
        st.session_state["unsup_mode"] = False
        st.rerun()
    render_unsupervised_page(api)
    st.stop()

# ----------------- ENSEMBLE PIPELINE -----------------
elif st.session_state.get("ens_mode", False):
    if _pipeline_start_clicked:
        st.session_state["ens_mode"] = False
        st.session_state["dataset_loaded"] = True
        st.rerun()
    if _regression_clicked:
        st.session_state["ens_mode"] = False
        st.session_state["reg_mode"] = True
        st.rerun()
    if _classification_clicked:
        st.session_state["ens_mode"] = False
        st.session_state["cls_mode"] = True
        st.rerun()
    if _unsupervised_clicked:
        st.session_state["ens_mode"] = False
        st.session_state["unsup_mode"] = True
        st.rerun()
    if st.button(" Back to EDA / Home", key="ens_back_btn"):
        st.session_state["ens_mode"] = False
        st.rerun()
    render_ensemble_page(api)
    st.stop()

# ----------------- LANDING PAGE -----------------
elif not st.session_state.get("dataset_loaded", False):
    # Main Header
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;">
            <h1 style="font-size: 3.5rem; background: linear-gradient(135deg, #FB923C 0%, #F97316 50%, #C2410C 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">IPL ML LAB</h1>
            <h3 style="font-size: 1.8rem; color: #E5E7EB; font-weight: 500; margin-bottom: 1rem;">Exploratory Data Analysis Visualizer</h3>
            <div style="font-size: 1.1rem; color: #9CA3AF; letter-spacing: 1px;">
                Understand  Clean  Analyze  Transform  Visualize  Prepare
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Columns for Local Datasets Selection
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader(f" Load Dataset: {st.session_state['active_dataset']}")
    st.markdown(f"""
        The active dataset is set to **{st.session_state['active_dataset']}**. 
        Click below to load this dataset and initialize the EDA pipeline. You can switch datasets at any time in the sidebar.
    """)
    
    load_btn = st.button(" Load Active Dataset", use_container_width=True, type="primary")
    if load_btn:
        with st.spinner(f"Loading {st.session_state['active_dataset']}..."):
            res = api.select_local_dataset(st.session_state['active_dataset'])
            if not res.get("error"):
                st.success(f"Dataset '{st.session_state['active_dataset']}' loaded successfully!")
                st.session_state["temp_metadata"] = res["metadata"]
                st.session_state["temp_loaded"] = True
    st.markdown("</div>", unsafe_allow_html=True)

    # Show Preview before starting EDA
    if st.session_state.get("temp_loaded"):
        if _pipeline_start_clicked:
            st.session_state["dataset_loaded"] = True
            st.session_state["current_step"] = 0
            st.session_state["max_completed_step"] = 0
            st.rerun()
            
        st.markdown("<hr style='border-color:#263244;' />", unsafe_allow_html=True)
        st.markdown("""
            <div style='margin-bottom:1rem;'>
                <span style='font-size:0.85rem; color:#F97316; font-weight:600; text-transform:uppercase; letter-spacing:1px;'>DATASET SNAPSHOT</span>
                <h3 style='margin:0.2rem 0 0;'> Quick Overview</h3>
            </div>""", unsafe_allow_html=True)

        meta = st.session_state["temp_metadata"]

        # Metrics row
        metrics = [
            {"label": "Total Rows",           "value": f"{meta['rows']:,}"},
            {"label": "Total Columns",         "value": f"{meta['columns']}"},
            {"label": "Memory Footprint",      "value": meta["memory_usage"]},
            {"label": "Numerical Features",    "value": f"{meta['numerical_cols_count']}"},
            {"label": "Categorical Features",  "value": f"{meta['categorical_cols_count']}"}
        ]
        render_metric_grid(metrics, cols_per_row=5)

        st.markdown("""
            <div style='margin-top:0.8rem; padding:0.7rem 1rem; background:rgba(249,115,22,0.08);
                        border:1px solid rgba(249,115,22,0.2); border-radius:8px; font-size:0.9rem; color:#9CA3AF;'>
                 Dataset loaded successfully. Click <b style='color:#F97316;'> START EDA PIPELINE</b>
                below to begin  the full data preview and column analysis await you in Step 1.
            </div>""", unsafe_allow_html=True)

        # START EDA PRIMARY BUTTON
        st.markdown("<div style='text-align: center; margin-top: 2rem; margin-bottom: 2rem;'>", unsafe_allow_html=True)
        start_eda_btn = st.button(" START EDA PIPELINE", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if start_eda_btn:
            st.session_state["dataset_loaded"] = True
            st.session_state["current_step"] = 0
            st.session_state["max_completed_step"] = 0
            st.rerun()

    # Handle Regression / Classification / Unsupervised clicks even from landing page
    if _regression_clicked:
        st.session_state["reg_mode"] = True
        st.session_state["cls_mode"] = False
        st.session_state["unsup_mode"] = False
        st.rerun()
    if _classification_clicked:
        st.session_state["cls_mode"] = True
        st.session_state["reg_mode"] = False
        st.session_state["unsup_mode"] = False
        st.rerun()
    if _unsupervised_clicked:
        st.session_state["unsup_mode"] = True
        st.session_state["reg_mode"] = False
        st.session_state["cls_mode"] = False
        st.rerun()

    # Always stop here  never fall through to EDA/Regression blocks
    st.stop()

# ----------------- MAIN EDA PIPELINE -----------------
else:
    # Use sidebar values captured once above
    clicked_step      = _clicked_step
    restart_clicked   = _restart_clicked
    pipeline_start_clicked = _pipeline_start_clicked
    regression_clicked = _regression_clicked

    if regression_clicked:
        st.session_state["reg_mode"] = True
        st.session_state["cls_mode"] = False
        st.session_state["unsup_mode"] = False
        st.rerun()
    if _classification_clicked:
        st.session_state["cls_mode"] = True
        st.session_state["reg_mode"] = False
        st.session_state["unsup_mode"] = False
        st.rerun()
    if _unsupervised_clicked:
        st.session_state["unsup_mode"] = True
        st.session_state["reg_mode"] = False
        st.session_state["cls_mode"] = False
        st.rerun()
    
    if restart_clicked:
        # Reset Session
        st.session_state["dataset_loaded"] = False
        st.session_state["current_step"] = 0
        st.session_state["max_completed_step"] = -1
        st.session_state["temp_loaded"] = False
        st.session_state["target_column"] = None
        st.rerun()
        
    if pipeline_start_clicked:
        st.session_state["dataset_loaded"] = True
        st.session_state["current_step"] = 0
        st.session_state["max_completed_step"] = max(0, st.session_state["max_completed_step"])
        st.rerun()
        
    if clicked_step is not None and clicked_step != st.session_state["current_step"]:
        st.session_state["current_step"] = clicked_step
        st.rerun()

    # --- Session validity check + auto-reload ---
    # Check without cache to properly detect if the backend wiped the session data on restart
    _health = api.get_preview(stage="original", limit=1, use_cache=False)
    if _health.get("error"):
        # Try to silently reload the original dataset from disk
        _reload = api.select_local_dataset(st.session_state["active_dataset"])
        if not _reload.get("error"):
            st.session_state["current_step"] = 0
            st.session_state["max_completed_step"] = 0
            _sync_url()
            st.toast(
                f" Backend restarted  **{st.session_state['active_dataset']}** reloaded from disk. "
                "Pipeline transformations were reset to Step 1.",
                icon="🔄"
            )
        else:
            # Backend unreachable entirely
            for _k in ["dataset_loaded", "temp_loaded", "current_step", "max_completed_step",
                       "target_column", "eda_started", "temp_metadata"]:
                st.session_state[_k] = {
                    "dataset_loaded": False, "temp_loaded": False, "current_step": 0,
                    "max_completed_step": -1, "target_column": None,
                    "eda_started": False, "temp_metadata": None
                }[_k]
            st.session_state["dataset_states"] = {}
            st.warning(" Could not reconnect to the backend. Please restart the backend server and refresh.")
            st.stop()

    current_step = st.session_state["current_step"]
    step_name = STEPS[current_step]
    
    # Horizontal Progress Timeline
    render_horizontal_timeline(current_step)
    
    # Page Header Info
    st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <span style="font-size:0.85rem; color:#F97316; font-weight:600; text-transform:uppercase; letter-spacing:1px;">EDA PIPELINE STAGE</span>
            <h2 style="margin:0; font-size:2.2rem; font-weight:700;">{step_name}</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # 1. Dataset Overview
    if current_step == 0:
        render_explanation(
            what="Dataset Overview gives a macroscopic summary of the shape, columns types, and stats of the raw dataset.",
            why="Before performing transformations, ML engineers inspect the data distribution, column formats, and sample rows to check variables consistency.",
            find="Original shape and data preview of the raw uploaded table."
        )
        
        overview_res = api.get_overview()
        if not overview_res.get("error"):
            cols = st.columns(4)
            with cols[0]:
                st.metric("Total Records", f"{overview_res['shape'][0]:,}")
            with cols[1]:
                st.metric("Total Features", f"{overview_res['shape'][1]}")
            with cols[2]:
                st.metric("Numerical Columns", f"{overview_res['classifications']['numerical']}")
            with cols[3]:
                st.metric("Categorical Columns", f"{overview_res['classifications']['categorical']}")
                
            st.markdown("###  Columns Statistical Attributes")
            # Build DataFrame summary table
            sum_df = pd.DataFrame(overview_res["summary"]).T
            st.dataframe(sum_df, use_container_width=True)
            
            st.markdown("###  Dataset Records")
            # Row-slice selector
            _slice_lp = st.radio(
                "Preview Rows:",
                ["First 10 rows", "First 25 rows", "First 50 rows", "Last 10 rows"],
                horizontal=True,
                key="overview_row_slice"
            )
            _lp_limit, _lp_type = 10, "first"
            if "25" in _slice_lp:  _lp_limit = 25
            elif "50" in _slice_lp: _lp_limit = 50
            elif "Last" in _slice_lp: _lp_type = "last"

            records_res = api.get_preview(stage="original", rows_type=_lp_type, limit=_lp_limit)
            if not records_res.get("error"):
                render_dataframe(records_res["records"], dtypes=records_res.get("dtypes"))
                
    # 2. Data Understanding
    elif current_step == 1:
        render_explanation(
            what="Data Understanding classifies variables into distinct physical and semantic concepts (Numerical, Categorical, Boolean, Date).",
            why="Categorical variables require encoding, numerical values require scaling, date fields need extraction, and boolean flags suggest binary markers. Treating columns correctly avoids pipeline bugs.",
            find="Classification summaries of features detected in the dataset."
        )
        
        types_res = api.get_dtypes()
        if not types_res.get("error"):
            # Metric blocks
            m_cols = st.columns(4)
            with m_cols[0]:
                st.metric(" Numerical Features", types_res["counts"]["numerical"])
            with m_cols[1]:
                st.metric(" Categorical Features", types_res["counts"]["categorical"])
            with m_cols[2]:
                st.metric(" Date Features", types_res["counts"]["date"])
            with m_cols[3]:
                st.metric(" Boolean Features", types_res["counts"]["boolean"])
                
            st.markdown("###  Variable Classification Audit")
            details_df = pd.DataFrame(types_res["details"])
            st.dataframe(details_df, use_container_width=True)
            
    # 3. Data Quality Check
    elif current_step == 2:
        render_explanation(
            what="Data Quality Check audits features for warnings or invalid entries like missing fields, constants, and high-cardinality keys.",
            why="Feeding constant values or unique IDs directly to model learning induces high overfitting, while extreme null ratios dilute signals.",
            find="A feature health summary, highlight issues as Warnings or Problems."
        )
        
        quality_res = api.get_quality_check()
        if not quality_res.get("error"):
            render_quality_card(quality_res["quality_score"])
            
            # Audit table
            st.markdown("###  Column-wise Quality Audit")
            audit_df = pd.DataFrame(quality_res["columns"])
            
            # Apply styling status mapping in tables
            # Streamlit dataframe supports HTML/Styler formatting
            def highlight_status(val):
                # [Function]: highlight_status
                # [Description]: Implements and executes the Highlight Status logic within this module pipeline.
                if val == "Problem":
                    return "color: #EF4444; font-weight: bold;"
                elif val == "Warning":
                    return "color: #F59E0B; font-weight: bold;"
                return "color: #10B981;"
            styler = audit_df.style
            if hasattr(styler, "map"):
                styled_df = styler.map(highlight_status, subset=["status"])
            else:
                styled_df = styler.applymap(highlight_status, subset=["status"])
            st.dataframe(styled_df, use_container_width=True)
            
    # 4. Data Cleaning
    elif current_step == 3:
        render_explanation(
            what="Data Cleaning consolidates handling duplicate records and filling missing values into clean datasets.",
            why="Incomplete rows trigger matrix errors in solvers, while duplicates bias the training distributions.",
            find="High-level cleaning summaries of missing elements and duplicates."
        )
        
        q_res = api.get_quality_check()
        if not q_res.get("error"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.subheader(" Missing Elements Status")
                null_cols = [c for c in q_res["columns"] if c["missing_count"] > 0]
                if null_cols:
                    st.warning(f"Found {len(null_cols)} columns containing missing values.")
                    for col_audit in null_cols:
                        st.markdown(f"- **`{col_audit['feature']}`**: {col_audit['missing_count']} missing ({col_audit['missing_percentage']:.1f}%)")
                else:
                    st.success("No missing values found in the dataset! ")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                st.subheader(" Duplicate Records Status")
                if q_res["duplicate_rows"] > 0:
                    st.warning(f"Found {q_res['duplicate_rows']} duplicate rows in the dataset.")
                    st.markdown("Duplicates will distort statistics if left uncleaned.")
                else:
                    st.success("No duplicate rows found! ")
                st.markdown("</div>", unsafe_allow_html=True)
                
    # 5. Missing Value Analysis
    elif current_step == 4:
        render_explanation(
            what="Missing Value Analysis isolates features with empty cells and imputes them using mathematical defaults.",
            why="Solvers like Linear Regression or SVM fail when matrices contain NaN values. Replacing NaNs with mean/median or mode restores data density.",
            find="Column-wise missing metrics and imputation controls."
        )
        
        miss_res = api.get_missing_values()
        if not miss_res.get("error"):
            # Plot
            render_plotly_chart(miss_res["chart"])
            
            # Imputation Form
            summary_cols = miss_res["summary"]["columns"]
            columns_with_nulls = [c for c in summary_cols if c["missing_count"] > 0]
            
            if columns_with_nulls:
                st.markdown("###  Interactive Imputation Panel")
                st.info("Impute values using recommended mathematical strategies or specify your own.")
                
                strategies = {}
                cols = st.columns(3)
                
                for idx, c in enumerate(columns_with_nulls):
                    col_name = c["column"]
                    rec = c["recommended_strategy"]
                    dtype = c["data_type"]
                    
                    with cols[idx % 3]:
                        # Select strategy
                        options = ["none", "mean", "median", "mode", "drop_rows", "drop_column"]
                        # Adjust options for types
                        if not any(t in dtype for t in ["int", "float"]):
                            # Remove mean/median for categories
                            options = ["none", "mode", "drop_rows", "drop_column"]
                            
                        # Recommend default
                        default_idx = options.index(rec) if rec in options else 0
                        
                        strategies[col_name] = st.selectbox(
                            f"Column: {col_name} ({dtype})",
                            options=options,
                            index=default_idx,
                            help=f"Missing: {c['missing_count']} rows ({c['missing_percentage']}%)"
                        )
                        
                impute_btn = st.button(" Apply Imputations", type="primary")
                if impute_btn:
                    with st.spinner("Imputing missing values..."):
                        clean_res = api.clean_missing_values(strategies)
                        if not clean_res.get("error"):
                            st.success("Imputation complete!")
                            st.rerun()
            else:
                st.success("Excellent! The dataset contains 0 missing values.")
                
    # 6. Duplicate Analysis
    elif current_step == 5:
        render_explanation(
            what="Duplicate Analysis detects duplicate records.",
            why="Duplicate records double-count identical outcomes, falsely inflating performance scores in validation.",
            find="Unique vs Duplicate comparisons."
        )
        
        dup_res = api.get_duplicates()
        if not dup_res.get("error"):
            render_plotly_chart(dup_res["chart"])
            
            dup_count = dup_res["summary"]["duplicate_rows"]
            if dup_count > 0:
                st.warning(f"The dataset contains {dup_count} duplicate rows.")
                remove_btn = st.button(" Remove All Duplicates", type="primary")
                if remove_btn:
                    with st.spinner("Deleting duplicate rows..."):
                        rem_res = api.remove_duplicates()
                        if not rem_res.get("error"):
                            st.success("All duplicates removed successfully!")
                            st.rerun()
            else:
                st.success("Perfect! 0 duplicate rows found in dataset.")
                
    # 7. Univariate Analysis
    elif current_step == 6:
        render_explanation(
            what="Univariate Analysis inspects features individually, plotting frequencies for categories or bounds for numerical variables.",
            why="Understanding individual variables shows range constraints, imbalance warnings, and potential coding errors.",
            find="Interactive univariate frequency or distribution plotter."
        )
        
        # Column selection list
        overview_res = api.get_overview()
        if not overview_res.get("error"):
            col_list = overview_res["columns"]
            
            selected_col = st.selectbox("Select Feature to Profile:", col_list)

            if selected_col:
                uni_res = api.get_univariate(selected_col)
                if not uni_res.get("error"):
                    stats_type = uni_res["stats"]["type"]

                    col1, col2 = st.columns([3, 2])
                    with col1:
                        render_plotly_chart(uni_res["chart"], key=f"uni_chart_{selected_col}")

                    with col2:
                        st.markdown(f"####  Statistics for `{selected_col}`")
                        if stats_type == "numeric":
                            s = uni_res["stats"]["stats"]
                            st.markdown(f"- **Mean**: {s['mean']:.4f}")
                            st.markdown(f"- **Median**: {s['median']:.4f}")
                            st.markdown(f"- **Std Deviation**: {s['std']:.4f}")
                            st.markdown(f"- **Min / Max**: {s['min']} / {s['max']}")
                            st.markdown(f"- **Q1 / Q3 (IQR)**: {s['q1']} / {s['q3']} ({s['q3']-s['q1']:.2f})")
                            st.markdown(f"- **Skewness**: {s['skewness']:.4f}")
                            st.markdown(f"- **Kurtosis**: {s['kurtosis']:.4f}")
                        else:
                            s = uni_res["stats"]["stats"]
                            st.markdown(f"- **Unique Values**: {s['unique_count']}")
                            st.markdown(f"- **Most Frequent (Mode)**: `{s['top_value']}` ({s['top_frequency']} occurrences)")

                            st.markdown("**Frequency Table (All)**:")
                            f_df = pd.DataFrame(s["freq_table"])
                            st.dataframe(f_df, use_container_width=True)
                                
    # 8. Distribution Analysis
    elif current_step == 7:
        render_explanation(
            what="Distribution Analysis checks how numerical values scale, measuring asymmetry via Skewness and peakedness via Kurtosis.",
            why="Algorithms like Linear Discriminant Analysis (LDA) and standard regression assume normally distributed residuals. Highly skewed inputs hinder stability.",
            find="Automatic shape descriptions based on actual statistical metrics."
        )
        
        dist_res = api.get_distribution()
        if not dist_res.get("error"):
            # Render a list of columns and their shape details
            for dist in dist_res["distributions"]:
                st.markdown(f"<div class='custom-card'>", unsafe_allow_html=True)
                st.subheader(f"Feature: {dist['column']}")
                
                col1, col2 = st.columns([3, 2])
                with col1:
                    render_plotly_chart(dist["chart"], key=f"dist_chart_{dist['column']}")
                    
                with col2:
                    st.markdown(f"**Shape Classification**: `{dist['shape']}`")
                    st.markdown(f"- **Mean**: {dist['mean']}")
                    st.markdown(f"- **Median**: {dist['median']}")
                    st.markdown(f"- **Skewness**: `{dist['skewness']}`")
                    st.markdown(f"- **Kurtosis**: `{dist['kurtosis']}`")
                    
                    st.markdown("#####  Interpretation:")
                    st.info(dist["interpretation"])
                st.markdown("</div>", unsafe_allow_html=True)
                
    # 9. Outlier Detection
    elif current_step == 8:
        render_explanation(
            what="Outlier Detection identifies extreme records, commonly using Interquartile Range (IQR) bounds or standard Z-Scores.",
            why="Outliers can distort mean-squared error loss values, pulling linear boundaries away from the general pattern.",
            find="Outliers boundary box plots and custom handling strategies."
        )
        
        types_res = api.get_dtypes()
        if not types_res.get("error"):
            num_cols = [c["column"] for c in types_res["details"] if c["type"] == "numerical"]
            
            col1, col2 = st.columns([1, 3])
            with col1:
                target_num = st.selectbox("Select Numeric Column:", num_cols)
                method = st.radio("Detection Method:", ["iqr", "zscore"])
                strategy = st.selectbox("Handling Action:", ["keep", "cap", "remove"])
                
                apply_btn = st.button("Apply Outlier Strategy", type="primary")
                
            with col2:
                if target_num:
                    out_res = api.get_outliers(target_num, method)
                    if not out_res.get("error"):
                        metrics = out_res["metrics"]
                        st.markdown(f"#### Outliers detected in `{target_num}`: **{metrics['outlier_count']}** ({metrics['outlier_percentage']}%)")
                        
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.markdown(f"- **Lower Boundary**: {metrics['lower_bound']:.2f}")
                        with col_m2:
                            st.markdown(f"- **Upper Boundary**: {metrics['upper_bound']:.2f}")
                            
                        render_plotly_chart(out_res["chart"], key=f"outlier_chart_{target_num}_{method}")
                        
            st.markdown("""
                > [!NOTE]
                > **Outliers detected  Outliers should always be removed.**
                > Sometimes outliers capture real physical variance (e.g. high-scoring T20 matches). Capping or keeping is often safer than flat removal.
            """, unsafe_allow_html=True)
            
            if apply_btn and target_num:
                with st.spinner("Applying outlier transformation..."):
                    handle_res = api.handle_outliers(target_num, strategy, method)
                    if not handle_res.get("error"):
                        st.success(f"Successfully processed outliers in '{target_num}' using {strategy}!")
                        st.rerun()
                        
    # 10. Bivariate Analysis
    elif current_step == 9:
        overview_res = api.get_overview()
        render_bivariate_page(api, overview_res)
                            
    # 11. Multivariate Analysis
    elif current_step == 10:
        render_explanation(
            what="Multivariate Analysis views multiple variables concurrently, analyzing systems instead of pairs.",
            why="Visualizing many dimensions helps find patterns, clusters, and dense relationships.",
            find="Multivariate Heatmap view."
        )
        
        corr_res = api.get_correlation()
        if not corr_res.get("error"):
            if "error" not in corr_res["correlation_data"]:
                render_plotly_chart(corr_res["chart"])
            else:
                st.info(corr_res["correlation_data"]["error"])
                
    # 12. Correlation Analysis
    elif current_step == 11:
        render_explanation(
            what="Correlation Analysis measures Pearson's coefficients from -1 (negative) to +1 (positive) to find linear relationships.",
            why="Strong correlations suggest predictive features, but highly collinear variables (r > 0.8) are redundant and confuse models.",
            find="Detailed Pearson correlations and multicollinearity reports."
        )
        
        corr_res = api.get_correlation()
        if not corr_res.get("error"):
            if "error" not in corr_res["correlation_data"]:
                cdata = corr_res["correlation_data"]
                
                col1, col2 = st.columns([3, 2])
                with col1:
                    render_plotly_chart(corr_res["chart"])
                    
                with col2:
                    st.markdown("####  Top Positive Correlations")
                    for p in cdata["top_positive"]:
                        st.markdown(f"- **`{p['feature_1']}`**  **`{p['feature_2']}`**: `+{p['coefficient']}`")
                        
                    st.markdown("####  Top Negative Correlations")
                    for p in cdata["top_negative"]:
                        st.markdown(f"- **`{p['feature_1']}`**  **`{p['feature_2']}`**: `{p['coefficient']}`")
                        
                    st.markdown("####  Multicollinearity Alerts (r > 0.8)")
                    if cdata["multicollinearity_warnings"]:
                        for w in cdata["multicollinearity_warnings"]:
                            st.warning(f"Redundant Pair: **`{w['feature_1']}`**  **`{w['feature_2']}`** (`{w['coefficient']}`)")
                    else:
                        st.success("No multicollinear feature pairs detected! ")
            else:
                st.info(corr_res["correlation_data"]["error"])
                
    # 13. Feature Selection
    elif current_step == 12:
        render_explanation(
            what="Feature Selection identifies useful features, removing constant columns or features uncorrelated with the target.",
            why="Reducing features speeds up model training and reduces overfitting.",
            find="Automated variable ratings based on Variance and Mutual Information (MI)."
        )
        
        overview_res = api.get_overview()
        if not overview_res.get("error"):
            col_list = overview_res["columns"]
            
            # Select Target
            target = st.selectbox(
                "Select Target Column (for MI calculation):", 
                options=[None] + col_list,
                index=col_list.index("winner") + 1 if "winner" in col_list else 0
            )
            
            if target:
                st.session_state["target_column"] = target
                
            # FORCE CLEAR FEATURE SELECTION CACHE TO FIX STUCK 0.0000 VALUES
            if "api_cache" in st.session_state:
                keys_to_del = [k for k in st.session_state["api_cache"].keys() if "/eda/feature-selection" in k]
                for k in keys_to_del:
                    del st.session_state["api_cache"][k]
                    
            fs_res = api.get_feature_selection(target)
            if not fs_res.get("error"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='custom-card' style='border-color:#10B981;'>", unsafe_allow_html=True)
                    st.markdown(" **Features Recommended to SELECT**")
                    for sf in fs_res["selected_features"]:
                        # Find recommendation info
                        rec_info = next((r for r in fs_res["recommendations"] if r["feature"] == sf), {})
                        mi_score = rec_info.get("mutual_info_score", 0.0)
                        st.markdown(f"- **`{sf}`** (MI: `{mi_score:.4f}`)")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with col2:
                    st.markdown("<div class='custom-card' style='border-color:#EF4444;'>", unsafe_allow_html=True)
                    st.markdown(" **Features Recommended to REMOVE**")
                    if fs_res["removed_features"]:
                        for rf in fs_res["removed_features"]:
                            st.markdown(f"- **`{rf['feature']}`**: *{rf['reason']}*")
                    else:
                        st.markdown("*No columns flagged for removal!*")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
    # 14. Categorical Encoding
    elif current_step == 13:
        render_explanation(
            what="Categorical Encoding maps text labels to numerical indices or boolean indicator matrices.",
            why="ML solvers read mathematical equations. Text categories like 'CSK' or 'RCB' must be mapped to numeric formats.",
            find="Interactive encoding options."
        )
        
        types_res = api.get_dtypes()
        if not types_res.get("error"):
            cat_cols = [c["column"] for c in types_res["details"] if c["type"] == "categorical"]
            
            # Filter target column and match ID from categorical inputs
            for filter_col in [st.session_state.get("target_column"), "match_id", "date"]:
                if filter_col in cat_cols:
                    cat_cols.remove(filter_col)
                    
            if cat_cols:
                st.info("Choose encoding strategies. We recommend: Binary -> Label, Nominal -> One-Hot, Ordered -> Ordinal.")
                encoding_map = {}
                
                cols = st.columns(3)
                for idx, col in enumerate(cat_cols):
                    # Guess default recommendation
                    preview_res = api.get_preview(stage="cleaned")
                    unique_count = 0
                    if not preview_res.get("error"):
                        df_preview = pd.DataFrame(preview_res["records"])
                        if col in df_preview.columns:
                            unique_count = df_preview[col].nunique()
                            
                    rec_method = "onehot" if unique_count > 2 else "label"
                    
                    with cols[idx % 3]:
                        encoding_map[col] = st.selectbox(
                            f"Column: {col} ({unique_count} uniques)",
                            options=["none", "label", "onehot", "ordinal"],
                            index=["none", "label", "onehot", "ordinal"].index(rec_method)
                        )
                        
                encode_btn = st.button("Apply Categorical Encoding", type="primary")
                if encode_btn:
                    with st.spinner("Encoding categorical columns..."):
                        enc_res = api.encode_categorical(encoding_map)
                        if not enc_res.get("error"):
                            st.success("Encoding completed successfully!")
                            st.rerun()
            else:
                st.success("0 categorical columns remain to encode.")
                
            # Preview Current Dataframe
            st.markdown("###  Active Preprocessing Data Preview")
            prev_res = api.get_preview()
            if not prev_res.get("error"):
                render_dataframe(prev_res["records"], dtypes=prev_res["dtypes"])
                
    # 15. Numerical Scaling
    elif current_step == 14:
        render_explanation(
            what="Numerical Scaling scales numerical columns into a shared numeric range.",
            why="Standardizing inputs ensures no single feature dominates gradient updates due to its magnitude.",
            find="Interactive scaler mappings."
        )
        
        types_res = api.get_dtypes()
        if not types_res.get("error"):
            num_cols = list(dict.fromkeys([c["column"] for c in types_res["details"] if c["type"] == "numerical"]))
            
            # Exclude match ID and target
            for filter_col in [st.session_state.get("target_column"), "match_id", "season"]:
                if filter_col in num_cols:
                    num_cols.remove(filter_col)
                    
            if num_cols:
                st.info("Map numerical variables to standard scalers: StandardScaler (z-score), MinMaxScaler (0-1), RobustScaler (outlier-friendly).")
                scaling_map = {}
                
                cols = st.columns(3)
                for idx, col in enumerate(num_cols):
                    # Check outlier count to recommend robust
                    out_check = api.get_outliers(col, method="iqr")
                    rec_scaler = "standard"
                    if not out_check.get("error"):
                        if out_check["metrics"]["outlier_count"] > 5:
                            rec_scaler = "robust"
                            
                    with cols[idx % 3]:
                        scaling_map[col] = st.selectbox(
                            f"Column: {col}",
                            options=["none", "standard", "minmax", "robust"],
                            index=["none", "standard", "minmax", "robust"].index(rec_scaler),
                            key=f"scale_selectbox_{col}_{idx}"
                        )
                        
                scale_btn = st.button("Apply Scaling Mappings", type="primary")
                if scale_btn:
                    with st.spinner("Scaling numerical values..."):
                        scale_res = api.scale_numerical(scaling_map)
                        if not scale_res.get("error"):
                            st.success("Scaling complete!")
                            st.rerun()
            else:
                st.success("0 numerical columns remain to scale.")
                
            # Preview Current Dataframe
            st.markdown("###  Active Preprocessing Data Preview")
            prev_res = api.get_preview()
            if not prev_res.get("error"):
                render_dataframe(prev_res["records"], dtypes=prev_res["dtypes"])
                
    # 16. Train/Test Preparation
    elif current_step == 15:
        render_explanation(
            what="Train/Test Split divides the dataset into training samples to train models and testing samples to evaluate them.",
            why="Evaluating models on the training data causes overfitting. Testing on unseen data provides an unbiased performance score.",
            find="Train-Test ratio simulator."
        )
        
        test_size = st.slider("Test Partition Ratio (Size %):", min_value=10, max_value=50, value=20, step=5)
        
        split_btn = st.button("Split Dataset", type="primary")
        
        if split_btn:
            with st.spinner("Splitting dataset..."):
                split_res = api.split_dataset(test_size / 100)
                if not split_res.get("error"):
                    col1, col2 = st.columns([2, 3])
                    with col1:
                        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
                        st.subheader(" Split Summary")
                        sinfo = split_res["split_info"]
                        st.metric("Training Rows (Learn)", f"{sinfo['train_count']:,} ({sinfo['train_percentage']}%)")
                        st.metric("Testing Rows (Evaluate)", f"{sinfo['test_count']:,} ({sinfo['test_percentage']}%)")
                        st.markdown("</div>", unsafe_allow_html=True)
                    with col2:
                        render_plotly_chart(split_res["chart"])
                        
    # 17. PCA
    elif current_step == 16:
        render_explanation(
            what="Principal Component Analysis (PCA) projects high-dimensional coordinates into orthogonal directions that maximize variance.",
            why="Applying PCA reduces features while retaining variance, improving computing speed and resolving multicollinearity.",
            find="Explained variance line charts."
        )
        
        types_res = api.get_dtypes()
        if not types_res.get("error"):
            num_cols = [c["column"] for c in types_res["details"] if c["type"] == "numerical"]
            # Filter non-PCA variables
            for filter_col in [st.session_state.get("target_column"), "match_id", "season"]:
                if filter_col in num_cols:
                    num_cols.remove(filter_col)
                    
            max_comp = len(num_cols)
            
            st.info(f"The dataset has {max_comp} features ready for PCA.")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                n_comp = st.slider("Number of Principal Components:", min_value=1, max_value=max(1, max_comp), value=min(3, max_comp))
                run_pca_btn = st.button("Run PCA Reduction", type="primary")
                
            with col2:
                if run_pca_btn:
                    with st.spinner("Executing Principal Component projection..."):
                        pca_res = api.run_pca(n_comp, st.session_state.get("target_column"))
                        if not pca_res.get("error"):
                            st.session_state["pca_completed_data"] = pca_res
                            st.success("PCA calculation completed!")
                            
            if "pca_completed_data" in st.session_state:
                pdata = st.session_state["pca_completed_data"]
                
                st.markdown("###  Variance Analysis")
                render_plotly_chart(pdata["charts"]["variance"])
                
                # Show individual weights table
                weights_data = []
                for idx, ev in enumerate(pdata["explained_variance_ratio"]):
                    weights_data.append({
                        "Component": f"PC{idx+1}",
                        "Explained Variance Ratio": f"{ev*100:.2f}%",
                        "Cumulative Variance": f"{pdata['cumulative_variance_ratio'][idx]*100:.2f}%"
                    })
                st.table(pd.DataFrame(weights_data))
                
    # 18. PCA Visualization
    elif current_step == 17:
        render_explanation(
            what="PCA Visualization projects high-dimensional coordinates into 2D and 3D scatter plots.",
            why="Humans cannot visualize more than 3 dimensions. PCA allows inspecting clusters, boundaries, and anomalies visually.",
            find="Interactive 2D and 3D PCA projection scatter plots."
        )
        
        if "pca_completed_data" in st.session_state:
            pdata = st.session_state["pca_completed_data"]
            
            dim = st.radio("Scatter Projection Dimensions:", ["2D PC1 vs PC2", "3D PC1 vs PC2 vs PC3"])
            
            if "2D" in dim:
                render_plotly_chart(pdata["charts"]["pca_2d"])
            else:
                if pdata["charts"].get("pca_3d") and pdata["charts"]["pca_3d"]:
                    render_plotly_chart(pdata["charts"]["pca_3d"])
                else:
                    st.warning("3D visualization is unavailable. Run PCA with at least 3 components first.")
        else:
            st.warning("PCA has not been executed yet. Return to the PCA step and run the calculation first.")
            
    # 19. Final Processed Dataset & Summary
    elif current_step == 18:
        render_explanation(
            what="The Final Processed Dataset & Summary displays the structured records ready for ML models and lists the consolidated report of findings.",
            why="After cleaning, encoding, scaling, and PCA projection, features are purely numerical, standardized, and clean.",
            find="A comparison dashboard showing before/after dimensions and a final summary report."
        )
        
        final_summary = api.get_summary(st.session_state.get("target_column"))
        if not final_summary.get("error"):
            # Metric blocks for Comparison


            cols = st.columns(4)
            with cols[0]:
                st.metric("Original Size", f"{final_summary['original_shape'][0]} × {final_summary['original_shape'][1]}")
            with cols[1]:
                st.metric("Processed Size", f"{final_summary['current_shape'][0]} × {final_summary['current_shape'][1]}")
            with cols[2]:
                st.metric("Cleaned Nulls", f"{final_summary['original_missing_values']}  {final_summary['current_missing_values']}")
            with cols[3]:
                st.metric("Cleaned Dups", f"{final_summary['original_duplicates']}  {final_summary['current_duplicates']}")
                
            st.markdown("###  Final Preprocessed Data Sample")
            preview_res = api.get_preview()
            if not preview_res.get("error"):
                render_dataframe(preview_res["records"], dtypes=preview_res["dtypes"])
                
            # Render Markdown Report (Without leading space indentation inside HTML)
            st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
            st.markdown(f"""<div class="custom-card">
<h3 style="margin-top:0; border-bottom:1px solid #263244; padding-bottom:0.5rem; color:#FB923C;"> EDA FINAL EXPERIMENT LOG</h3>

<h4>1. Dataset Attributes</h4>
<ul>
<li><b>Original Shape:</b> {final_summary['original_shape'][0]} rows × {final_summary['original_shape'][1]} features</li>
<li><b>Final Processed Shape:</b> {final_summary['current_shape'][0]} rows × {final_summary['current_shape'][1]} dimensions</li>
<li><b>Feature Classification:</b> {final_summary['numerical_features_count']} numerical features, {final_summary['categorical_features_count']} categorical features</li>
</ul>

<h4>2. Data Quality Checks</h4>
<ul>
<li><b>Initial Data Quality Score:</b> {final_summary['data_quality_score']}%</li>
<li><b>Missing values resolved:</b> {final_summary['original_missing_values']} NaNs  {final_summary['current_missing_values']} NaNs</li>
<li><b>Duplicate rows removed:</b> {final_summary['original_duplicates']}  {final_summary['current_duplicates']}</li>
<li><b>Multicollinear feature pairs detected:</b> {final_summary['multicollinear_pairs_count']}</li>
</ul>

<h4>3. Transformations Executed</h4>
<ol>
{"".join([f"<li>{t}</li>" for t in final_summary['transformations_applied']]) if final_summary['transformations_applied'] else "<li>No pipeline transformations were executed.</li>"}
</ol>

<div style="margin-top:1.5rem; background-color:#111827; border: 1px solid #263244; padding:1rem; border-radius:8px; border-left:4px solid #10B981; color:#D1D5DB; font-size:0.95rem;">
 <b>CONGRATULATIONS!</b> The dataset is preprocessed, scaled, and project-mapped into clean PCA principal components. It is ready for training classification or regression models!
</div>
</div>""", unsafe_allow_html=True)
            
            # Save processed dataset button
            save_name = f"processed_{st.session_state['active_dataset']}"
            if not save_name.endswith(".csv"):
                save_name += ".csv"
                
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f" Save Final Processed Dataset as '{save_name}'", type="primary", use_container_width=True):
                    with st.spinner(f"Saving to data/{save_name}..."):
                        save_res = api.save_processed_dataset(st.session_state["active_dataset"])
                        if not save_res.get("error"):
                            st.success(save_res.get("message", "Dataset saved successfully!"))
                            import time; time.sleep(1) # wait for user to see success msg
                            st.rerun()
            with col2:
                import os
                local_path = os.path.join("ipl_prediction_project", "data", save_name)
                if os.path.exists(local_path):
                    with open(local_path, "rb") as f:
                        csv_data = f.read()
                    st.download_button(
                        label=f"⬇️ Download '{save_name}' via Browser",
                        data=csv_data,
                        file_name=save_name,
                        mime="text/csv",
                        use_container_width=True
                    )

    # ----------------- NEXT / PREVIOUS NAVIGATION -----------------
    st.markdown("<hr style='border-color:#263244; margin-top: 3rem;' />", unsafe_allow_html=True)
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    
    with nav_col1:
        if current_step > 0:
            if st.button("⬅️ Previous Step", use_container_width=True):
                st.session_state["current_step"] = current_step - 1
                st.rerun()
                
    with nav_col3:
        if current_step < len(STEPS) - 1:
            if st.button("Next Step ➡️", type="primary", use_container_width=True):
                st.session_state["current_step"] = current_step + 1
                st.session_state["max_completed_step"] = max(st.session_state["max_completed_step"], current_step + 1)
                st.rerun()
