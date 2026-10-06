import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
os.chdir(APP_ROOT)

api_port = os.environ.setdefault("IPL_API_PORT", "8001")
os.environ.setdefault("IPL_API_URL", f"http://127.0.0.1:{api_port}")

original_set_page_config = st.set_page_config


def noop_set_page_config(*args, **kwargs):
    pass


st.set_page_config = noop_set_page_config
original_set_page_config(
    page_title="Merged IPL Project",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def start_fastapi():
    backend_dir = APP_ROOT / "ipl_prediction_project" / "backend"
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            api_port,
            "--reload",
        ],
        cwd=backend_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True


start_fastapi()


def home_page_view():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
        .stApp { background: #0B0F19; color: #E8EDF5; font-family: 'Inter', sans-serif; }
        .block-container { max-width: 1350px; padding-top: 3rem; }
        .hero-title {
            text-align: center; font-family: 'Outfit', sans-serif; font-size: 4rem;
            font-weight: 800; color: #F4F7FB; margin-bottom: .5rem;
        }
        .hero-subtitle { text-align: center; color: #AAB6C5; font-size: 1.2rem; margin-bottom: 3rem; }
        div[data-testid="column"] {
            background: #151B27; border: 1px solid #293345; border-radius: 16px;
            padding: 2rem; min-height: 350px;
        }
        .card-icon { font-size: 2.5rem; margin-bottom: .75rem; }
        .card-title { font: 700 1.8rem 'Outfit', sans-serif; color: #FFF; margin-bottom: .75rem; }
        .card-desc { color: #B5BFCC; font-size: 1rem; line-height: 1.6; }
        div.stButton > button { width: 100%; height: 3.5rem; border-radius: 8px; font-weight: 700; }
        </style>
        <div class="hero-title">IPL Analytics Platform</div>
        <div class="hero-subtitle">A unified ecosystem for machine learning and predictive analytics</div>
        """,
        unsafe_allow_html=True,
    )

    learning_column, implementation_column = st.columns(2, gap="large")

    with learning_column:
        st.markdown(
            """
            <div class="card-icon">🔬</div>
            <div class="card-title">Learning Lab</div>
            <div class="card-desc">Explore datasets, perform exploratory analysis, prepare data,
            and run dimensionality reduction and machine-learning workflows.</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Learning Module", key="btn_learn"):
            st.switch_page("ipl_prediction_project/frontend/app.py")

    with implementation_column:
        st.markdown(
            """
            <div class="card-icon">⚡</div>
            <div class="card-title">Implementation</div>
            <div class="card-desc">Predict match outcomes and scores, explore player profiles,
            and compare models trained on historical IPL data.</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Implementation", key="btn_impl"):
            st.switch_page("pythonmlproject/pythonmlproject/app.py")


home_page = st.Page(home_page_view, title="Home", icon="🏠", default=True)
learning_page = st.Page(
    "ipl_prediction_project/frontend/app.py",
    title="Learning Module",
    icon="📚",
    url_path="learning",
)
implementation_pages = [
    st.Page(
        "pythonmlproject/pythonmlproject/app.py",
        title="Implementation Overview",
        icon="⚙️",
        url_path="implementation_home",
    ),
    st.Page("pythonmlproject/pythonmlproject/pages/02_history.py", title="History & EDA", icon="📜"),
    st.Page("pythonmlproject/pythonmlproject/pages/03_match_prediction.py", title="Match Prediction", icon="🎯"),
    st.Page("pythonmlproject/pythonmlproject/pages/04_score_prediction.py", title="Score Prediction", icon="📈"),
    st.Page("pythonmlproject/pythonmlproject/pages/06_player_profiles.py", title="Player Analytics", icon="👤"),
    st.Page("pythonmlproject/pythonmlproject/pages/08_team_optimizer.py", title="Team Optimizer", icon="🏆"),
    st.Page("pythonmlproject/pythonmlproject/pages/09_algo_lab.py", title="Algo Lab", icon="🔬"),
]

navigation = st.navigation(
    {
        "Main": [home_page],
        "Learning Section": [learning_page],
        "Implementation Section": implementation_pages,
    }
)
navigation.run()