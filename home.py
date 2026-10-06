import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

# IMPLEMENTATION_ROOT = Path(__file__).resolve().parent / "pythonmlproject"
# if str(IMPLEMENTATION_ROOT) not in sys.path:
#     sys.path.insert(0, str(IMPLEMENTATION_ROOT))

# 1. Monkey-patch set_page_config FIRST (before anything else)
# This prevents Streamlit API Exceptions from the sub-pages when they try to call it again.
original_config = st.set_page_config
def noop_config(*args, **kwargs):
    pass
st.set_page_config = noop_config

# Call the real config once for the main app
original_config(page_title="Merged IPL Project", page_icon="🏏", layout="wide", initial_sidebar_state="expanded")

# 2. Start FastAPI Backend silently if not started
@st.cache_resource
def start_fastapi():
    cwd = Path(__file__).resolve().parent / "ipl_prediction_project" / "backend"
    api_port = os.environ.get("IPL_API_PORT", "8001")
    try:
        if os.name == 'nt':
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", api_port, "--reload"],
                cwd=str(cwd),
                creationflags=CREATE_NO_WINDOW
            )
        else:
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", api_port, "--reload"],
                cwd=str(cwd),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
    except Exception as e:
        print(f"Failed to start FastAPI backend: {e}")
    return True

start_fastapi()

# 3. Home Page Function
def home_page_view():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0B0F19;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(255, 107, 0, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 85% 30%, rgba(31, 56, 100, 0.15) 0%, transparent 50%);
        color: #E8EDF5;
    }
    
    /* Clean up default padding */
    .block-container {
        padding-top: 3rem !important;
        max-width: 1350px !important;
    }
        div[data-testid="stMainBlockContainer"] {
        max-width: 1350px !important;
    }
    /* Typography */
    .hero-title {
        text-align: center;
        font-family: 'Outfit', sans-serif;
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #A0B0C0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        text-align: center;
        font-size: 1.4rem;
        color: #8A9BAE;
        font-weight: 400;
        margin-bottom: 4rem;
        letter-spacing: 0.5px;
    }
    
    /* Module Cards (Columns) */
    div[data-testid="column"] {
        background: rgba(22, 27, 39, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 2.5rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 400px;
    }
    div[data-testid="column"]:hover {
        transform: translateY(-10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* Column 1 Specific Hover Glow (Learning) */
    div[data-testid="column"]:nth-child(1):hover {
        box-shadow: 0 20px 40px rgba(31, 56, 100, 0.5);
    }
    
    /* Column 2 Specific Hover Glow (Implementation) */
    div[data-testid="column"]:nth-child(2):hover {
        box-shadow: 0 20px 40px rgba(255, 107, 0, 0.25);
    }
    
    /* Buttons */
    div.stButton > button {
        width: 100%;
        height: 60px;
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease;
        margin-top: 1rem;
    }
    
    /* Button 1 (Learning) */
    div[data-testid="column"]:nth-child(1) div.stButton > button {
        background: linear-gradient(135deg, #1F3864 0%, #2A4C87 100%);
        color: white;
        box-shadow: 0 8px 20px rgba(31, 56, 100, 0.4);
    }
    div[data-testid="column"]:nth-child(1) div.stButton > button:hover {
        background: linear-gradient(135deg, #2A4C87 0%, #365D9E 100%);
        box-shadow: 0 12px 25px rgba(31, 56, 100, 0.6);
        transform: scale(1.02);
    }
    
    /* Button 2 (Implementation) */
    div[data-testid="column"]:nth-child(2) div.stButton > button {
        background: linear-gradient(135deg, #FF6B00 0%, #FF8E3C 100%);
        color: white;
        box-shadow: 0 8px 20px rgba(255, 107, 0, 0.3);
    }
    div[data-testid="column"]:nth-child(2) div.stButton > button:hover {
        background: linear-gradient(135deg, #FF8E3C 0%, #FFA25E 100%);
        box-shadow: 0 12px 25px rgba(255, 107, 0, 0.5);
        transform: scale(1.02);
    }
    
    /* Content styling inside cards */
    .card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    .card-desc {
        color: #9CA3AF;
        font-size: 1.1rem;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #6B7280;
        font-size: 0.95rem;
        margin-top: 5rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("<div class='hero-title'>IPL Analytics Platform</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>A unified ecosystem for Machine Learning & Predictive Analytics</div>", unsafe_allow_html=True)

    # Cards Section
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div>
            <div class='card-icon'>🔬</div>
            <div class='card-title'>Learning Lab</div>
            <div class='card-desc'>
                Explore datasets, perform Exploratory Data Analysis, handle missing values, 
                and run Dimensionality Reduction (PCA) in real-time.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Spacer to push button to bottom
        st.write("")
        st.write("")
        
        if st.button("Launch Learning Module", key="btn_learn"):
            st.switch_page(r"ipl_prediction_project\frontend\app.py")

    with col2:
        st.markdown("""
        <div>
            <div class='card-icon'>⚡</div>
            <div class='card-title'>Implementation</div>
            <div class='card-desc'>
                Deploy production-ready models on historical IPL data. 
                Predict match outcomes, simulate scores, analyze deep player profiles, 
                and optimize team compositions using advanced algorithms.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Spacer to push button to bottom
        st.write("")
        st.write("")
        
        if st.button("Launch Implementation", key="btn_impl"):
            st.switch_page(r"pythonmlproject\app.py")

    # Footer
   


# 4. Define Pages using Streamlit 1.36+ Navigation
home_page = st.Page(home_page_view, title="Home", icon="🏠", default=True)

learning_page = st.Page(r"ipl_prediction_project\frontend\app.py", title="Learning Module", icon="📚", url_path="learning")

impl_home = st.Page(r"pythonmlproject\app.py", title="Implementation Overview", icon="⚙️", url_path="implementation_home")
impl_history = st.Page(r"pythonmlproject\pages\02_history.py", title="History & EDA", icon="📜")
impl_match = st.Page(r"pythonmlproject\pages\03_match_prediction.py", title="Match Prediction", icon="🎯")
impl_score = st.Page(r"pythonmlproject\pages\04_score_prediction.py", title="Score Prediction", icon="📈")
impl_player = st.Page(r"pythonmlproject\pages\06_player_profiles.py", title="Player Analytics", icon="👤")
impl_team = st.Page(r"pythonmlproject\pages\08_team_optimizer.py", title="Team Optimizer", icon="🏆")
impl_algo = st.Page(r"pythonmlproject\pages\09_algo_lab.py", title="Algo Lab", icon="🔬")

# Group pages into sections in the sidebar
pg = st.navigation({
    "Main": [home_page],
    "Learning Section": [learning_page],
    "Implementation Section": [
        impl_home, impl_history, impl_match, impl_score, 
        impl_player, impl_team, impl_algo
    ]
})

pg.run()

