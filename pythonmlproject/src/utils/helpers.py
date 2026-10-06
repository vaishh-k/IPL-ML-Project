from pathlib import Path
import joblib
import streamlit as st

try:
    from xgboost import XGBClassifier, XGBRegressor
    if hasattr(XGBClassifier, "__sklearn_tags__"):
        def _xgb_clf_tags(self):
            tags = super(XGBClassifier, self).__sklearn_tags__()
            tags.estimator_type = "classifier"
            return tags
        XGBClassifier.__sklearn_tags__ = _xgb_clf_tags
    if hasattr(XGBRegressor, "__sklearn_tags__"):
        def _xgb_reg_tags(self):
            tags = super(XGBRegressor, self).__sklearn_tags__()
            tags.estimator_type = "regressor"
            return tags
        XGBRegressor.__sklearn_tags__ = _xgb_reg_tags
except ImportError:
    pass


def get_project_root() -> Path:
    """Return the project root (3 levels up from src/utils/)."""
    return Path(__file__).parent.parent.parent


def apply_theme() -> str:
    """Return CSS string for the dark IPL theme."""
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
    return css


def load_model_safe(path):
    """Load a joblib model safely; return None if file not found."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


def model_missing_msg(name: str = "") -> bool:
    """
    Display a warning if model is missing.
    Returns True if missing (so caller can bail early), False otherwise.
    """
    label = f" ({name})" if name else ""
    st.warning(f"⚠️ Run train.py first to generate models{label}.")
    return True


def fmt_number(n: int) -> str:
    """Format an integer with comma separators."""
    return "{:,}".format(n)


IPL_TEAM_COLORS = {
    "Mumbai Indians": "#004BA0",
    "Chennai Super Kings": "#FDB913",
    "Royal Challengers Bengaluru": "#EC1C24",
    "Kolkata Knight Riders": "#3A225D",
    "Sunrisers Hyderabad": "#F7A721",
    "Delhi Capitals": "#0078BC",
    "Punjab Kings": "#DD1F2D",
    "Rajasthan Royals": "#254AA5",
    "Lucknow Super Giants": "#A72056",
    "Gujarat Titans": "#1C1C5F",
    "Rising Pune Supergiant": "#8B1A8B",
}


def get_team_color(team: str) -> str:
    """Return the hex brand color for a given IPL team."""
    return IPL_TEAM_COLORS.get(team, "#FF6B00")


@st.cache_data(show_spinner=False)
def load_dataset_stats() -> dict:
    """Load dataset stats (matches, seasons, venues, pom) from features.parquet safely."""
    feat_path = get_project_root() / "data" / "processed" / "features.parquet"
    stats = {"n_matches": 1169, "n_seasons": 18, "n_venues": 36, "n_pom": 275}
    if feat_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(feat_path)
            stats["n_matches"] = len(df)
            if "season" in df.columns:
                stats["n_seasons"] = int(df["season"].nunique())
            if "venue" in df.columns:
                stats["n_venues"] = int(df["venue"].nunique())
            if "player_of_match" in df.columns:
                stats["n_pom"] = int(df["player_of_match"].nunique())
        except Exception:
            pass
    return stats
