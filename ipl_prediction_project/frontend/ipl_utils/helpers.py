import streamlit as st
import time

def inject_custom_css():
    css = """
        <style>
    /* Import premium font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');

    /* ── App background ── */
    body, .stApp {
        background-color: #0F1117;
        color: #E8EDF5;
        font-family: 'Inter', sans-serif;
        font-size: 1.08rem !important;
    }

    /* ── Main Container Width ── */
    .block-container, 
    div[data-testid="stAppViewBlockContainer"], 
    div[data-testid="block-container"],
    .stMainBlockContainer {
        max-width: 1350px !important;
    }
    div[data-testid="stMainBlockContainer"] {
        max-width: 1350px !important;
    }
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #1A1D2E;
        border-right: 2px solid #1F3864;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label {
        color: #E8EDF5 !important;
    }

    /* ── Page headers ── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', 'Inter', sans-serif !important;
    }
    h1 {
        color: #FF6B00 !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px;
        border-bottom: 3px solid #1F3864;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    h2 {
        color: #FF6B00 !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
    }
    h3 {
        color: #C0D0E8 !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }
    h4 {
        color: #FFFFFF !important;
        font-size: 1.25rem !important;
    }

    /* ── Metric cards ── */
    div[data-testid="metric-container"] {
        background-color: #1F3864;
        border: 1px solid #FF6B00;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="metric-container"] label {
        color: #C0D0E8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #FF6B00 !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        color: #90EE90 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #FF6B00;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0.5rem 1.4rem;
        transition: background-color 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #e05a00;
        color: #FFFFFF;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1A1D2E;
        border-radius: 8px 8px 0 0;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #C0D0E8;
        background-color: #1A1D2E;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1F3864 !important;
        color: #FF6B00 !important;
        border-bottom: 3px solid #FF6B00;
    }

    /* ── DataFrames / Tables ── */
    .stDataFrame {
        border: 1px solid #1F3864;
        border-radius: 8px;
    }

    /* ── Select boxes & inputs ── */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background-color: #1A1D2E;
        border: 1px solid #1F3864;
        color: #E8EDF5;
    }
    
    /* Dropdown Menus Background Fix */
    ul[role="listbox"], div[role="listbox"], div[data-baseweb="menu"], div[data-baseweb="popover"] > div {
        background-color: #1A1D2E !important;
    }
    li[role="option"]:hover, div[role="option"]:hover, li[role="option"][aria-selected="true"], div[role="option"][aria-selected="true"] {
        background-color: #1F3864 !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background-color: #FF6B00;
    }

    /* ── Info / warning / success boxes ── */
    .stAlert {
        border-radius: 8px;
    }

    /* ── Divider ── */
    hr {
        border-color: #1F3864;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #FF6B00 !important;
    }

    /* ── Expander ── */
    details summary {
        color: #FF6B00 !important;
        font-weight: 600;
    }
    
    /* ── Widget Labels & Text ── */
    p, .stMarkdown p, .stMarkdown {
        font-size: 1.08rem !important;
        line-height: 1.6 !important;
        color: #E8EDF5 !important;
    }
    div[data-testid="stWidgetLabel"] p, 
    .stWidgetLabel p, 
    label[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] label p, 
    div[data-testid="stCheckbox"] label p,
    div[data-testid="stRadio"] label span,
    div[data-baseweb="select"] p,
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span,
    input {
        font-size: 1.1rem !important;
    }
    div[role="listbox"] div,
    div[role="option"] span,
    div[role="option"] div,
    ul[role="listbox"] li {
        font-size: 1.08rem !important;
    }
    form label, form p {
        font-size: 1.08rem !important;
    }
    div[data-testid="stCaptionContainer"], 
    .stCaption, 
    small,
    div[data-testid="stNotificationContent"] p {
        font-size: 0.95rem !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_explanation(what: str, why: str, find: str):
    """Render a standard explanation block."""
    with st.expander("📚 Concept Explanation & Pipeline Role"):
        st.markdown(f"**What it is:** {what}")
        st.markdown(f"**Why we do it:** {why}")
        st.markdown(f"**What to look for:** {find}")


def render_card(title: str, content: str, type: str = "info"):
    """Render a styled metric or info card."""
    st.markdown(f"**{title}**: {content}")


def save_current_dataset_state():
    """Save the active state in session state before dataset switch."""
    if "active_dataset" in st.session_state and st.session_state.get("dataset_loaded", False):
        ds = st.session_state["active_dataset"]
        state = {
            "current_step": st.session_state.get("current_step", 0),
            "max_completed_step": st.session_state.get("max_completed_step", -1),
            "dataset_loaded": True
        }
        if "dataset_states" not in st.session_state:
            st.session_state["dataset_states"] = {}
        st.session_state["dataset_states"][ds] = state


def load_dataset_state(dataset_name: str):
    """Load the state of a dataset when switching back."""
    if "dataset_states" in st.session_state and dataset_name in st.session_state["dataset_states"]:
        state = st.session_state["dataset_states"][dataset_name]
        st.session_state["current_step"] = state.get("current_step", 0)
        st.session_state["max_completed_step"] = state.get("max_completed_step", -1)
        st.session_state["dataset_loaded"] = state.get("dataset_loaded", True)
    else:
        st.session_state["current_step"] = 0
        st.session_state["max_completed_step"] = -1
        st.session_state["dataset_loaded"] = False
