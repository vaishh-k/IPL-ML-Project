# IPL Machine Learning Exploratory Data Analysis (EDA) Visualizer

A premium, production-quality Exploratory Data Analysis (EDA) Interactive Laboratory designed for Machine Learning students. Built with **Streamlit** (Frontend) and **FastAPI** (Backend REST API), this application guides a user step-by-step through a complete data-science preprocessing, profiling, cleaning, transforming, and dimensionality reduction pipeline.

---

## 🚀 Project Overview

The application functions as an educational walkthrough. Users upload any arbitrary CSV/XLSX file or load the included mock **IPL matches dataset**, preview the original dataset, click **"Start EDA"**, and witness the step-by-step mathematical logic and interactive visualization of a 21-step data-science pipeline.

### Core Technology Stack
- **Frontend**: Streamlit, Custom HTML/CSS styling for a unified dark theme dashboard.
- **Backend**: FastAPI REST API, stateless per session using local state serialization.
- **Data & Computations**: Pandas, NumPy, Scikit-learn (imputations, scalers, encoders, PCA reduction).
- **Interactive Visualizations**: Plotly graphs styled with a dark theme color palette.

---

## 📁 Architecture Directory Structure

The project is structured modularly to separate frontend presentation from backend analytical services:

```text
d:/projects/ipl_prediction_project/
│
├── backend/
│   ├── main.py                # FastAPI entrypoint, middleware, and router mounts
│   ├── routes/                # REST API routers
│   │   ├── dataset.py         # Data uploading & slicing previews
│   │   ├── eda.py             # Feature analysis, cleaning, bivariate/multivariate charts
│   │   ├── preprocessing.py   # Label/One-hot encoding, scaling, train-test splits
│   │   └── pca.py             # Principal Component Analysis fitting
│   ├── services/              # Core business logic services
│   │   ├── data_loader.py     # Stream parsed files loaders
│   │   ├── cleaning.py        # Imputation, duplicates, IQR/Z-score outlier detection
│   │   ├── analysis.py        # Variable profiling, skewness shape analysis, MI metrics
│   │   ├── visualization.py   # Plotly figures generators (returns serializable dicts)
│   │   ├── preprocessing.py   # ML encoders, scalers, and data partitioners
│   │   └── pca_service.py     # PCA fitting, explained variance, coordinate projections
│   ├── utils/
│   │   └── helpers.py         # Session file management (to_pickle), nan sanitizers
│   └── temp_data/             # Auto-created temporary folder for session storage
│
├── frontend/
│   ├── app.py                 # Streamlit main orchestration file
│   ├── components/            # UI layout modular components
│   │   ├── sidebar.py         # Vertical pipeline checklists progress tracker
│   │   ├── metrics.py         # Responsive metric indicators grids
│   │   ├── charts.py          # Plotly figures deserializers
│   │   ├── dataframe.py       # Styled interactive tables & types viewers
│   │   ├── progress.py        # Previous/Next step navigation rows
│   │   └── cards.py           # Data quality gauges & custom info boxes
│   └── utils/
│       └── helpers.py         # CSS injector and educational explanations panels
│
├── data/
│   └── ipl_sample.csv         # Realistic sample IPL matches history dataset
│
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation (this file)
└── .gitignore                 # Untracked files configuration
```

---

## 📦 Installation & Setup

### Prerequisites
Make sure Python **3.12** and the Windows Python Launcher (`py`) are installed. Python 3.12 is required by the shared TensorFlow dependency set.

1. **Open the workspace root** (the parent directory containing `home.py` and both project folders):
   ```bash
   cd "d:\projects\iplproject data\merged ipl project"
   ```

2. **Create the shared virtual environment and install all project dependencies**:
   ```bash
   .\setup_venv.bat
   ```

---

## 🏃 Run Commands

### ⚡ Quick Start (Windows Batch Script)
We have provided a custom batch script `run.bat` at the project root to orchestrate services:
- **Interactive Menu**: Double-click `run.bat` or run `.\run.bat` in your terminal to select options.
- **Direct Commands**:
  ```bash
  .\run.bat both      # Runs both Backend (in a new window) and Frontend
  .\run.bat backend   # Starts the FastAPI Backend only
  .\run.bat frontend  # Starts the Streamlit Frontend only
  ```

---

The launcher uses the workspace-level `.venv` and shared `requirements.txt`; it does not use global Python or create per-project environments.
- The frontend dashboard will open in your default browser at: [http://localhost:8501](http://localhost:8501)

---

## 🏏 Sample IPL Dataset Details

The application comes with a mock IPL matches dataset (`data/ipl_sample.csv`) containing 508 matches (2019-2024 seasons). It contains:
- **Index/ID Columns**: `match_id` (numeric indices, ignored in scaling/PCA).
- **Date Column**: `date` (parsed for timelines).
- **Categorical Columns**: `venue`, `team1`, `team2`, `toss_winner`, `toss_decision`, `winner`, `player_of_match`.
- **Numerical Columns**: `season`, `runs_scored`, `wickets_lost`, `overs`, `strike_rate`, `margin`.
- **Boolean Column**: `is_playoff`.
- **Constant Column**: `league` (containing only `"IPL"` to trigger Data Quality problems).
- **Built-in Quality Issues**: Missing cells in `winner` (rainouts), `margin`, and `overs`; duplicate entries; and extreme run-score outliers (like `345` or `20`).

---

## ⚙️ The 21-Step EDA Workflow

The pipeline is laid out sequentially on the screen. Users can navigate using **← Previous** and **Next →** buttons, or jump back to unlocked steps using the sidebar selectbox.

1. **Dataset Overview**: Rows, columns, numerical/categorical summaries, and interactive tables.
2. **Data Understanding**: Semantic categories separation (Numbers, Categories, Dates, Booleans).
3. **Data Quality Check**: Calculates a global **Data Quality Score** and displays warning/problem indicators.
4. **Data Cleaning Overview**: Quick status check of nulls and duplicate cells.
5. **Missing Value Analysis**: Column-wise null bars and interactive mean/median/mode imputations.
6. **Duplicate Analysis**: Unique vs duplicate ratio charts and duplicate deletion controls.
7. **Data Type Analysis**: Schema review of datatypes before mathematical analyses.
8. **Univariate Analysis**: Custom histograms, box plots, and frequency tables for individual columns.
9. **Distribution Analysis**: Calculates Skewness and Kurtosis, displaying histogram KDE curves.
10. **Outlier Detection**: IQR & Z-score boundaries, box plots, and Keep/Cap/Remove strategies.
11. **Bivariate Analysis**: Scatter plots, box/violin plots, or grouped bars mapping two-variable interactions.
12. **Multivariate Analysis**: Correlation heatmap view of all dimensions.
13. **Correlation Analysis**: Pearson coefficients ranking top positive/negative pairs and collinear warnings.
14. **Feature Selection**: Selects a predictive target (e.g. `winner`) and scores columns using variance and Mutual Information.
15. **Categorical Encoding**: Recommendations for encoding categories (Label Encoding, One-Hot Encoding).
16. **Numerical Scaling**: Scale-adjustments comparison (StandardScaler, MinMaxScaler, RobustScaler).
17. **Train/Test Preparation**: Interactive training vs testing split divisions (e.g. 80/20 ratio).
18. **PCA**: Slider to fit a number of Principal Components and plot explained variance metrics.
19. **PCA 2D Visualization**: Projects observations onto a 2D scatter coordinates (PC1 vs PC2) colored by target.
20. **PCA 3D Visualization**: Projects observations onto an interactive 3D scatter coordinates (PC1, PC2, PC3).
21. **Final Processed Dataset & Summary**: Before-after dashboards showing shapes reduction and final summary logs.

---

## 💡 Educational Explanations

Each step in the EDA Visualizer incorporates three structured learning sections:
- **💡 WHAT?** - The conceptual definition of the active data science operation.
- **⚙️ WHY?** - The reason machine learning models require this processing step.
- **📊 WHAT DID WE FIND?** - Metrics dynamically calculated from the uploaded dataset, explaining the specific numbers.

---

## 🛠️ Troubleshooting

- **FastAPI connection fails**: Make sure the backend terminal is running and listening on `http://127.0.0.1:8000`. If you run on a different port, update the `BASE_URL` in `frontend/services/api_client.py`.
- **PCA crashes**: PCA only supports numerical columns. Make sure you apply **Categorical Encoding** on Step 15 before running PCA. Leaked missing values are automatically imputed with column medians by the service.
- **Data type warnings**: If numeric columns are mistakenly treated as objects, make sure there are no string characters in the cells (like `runs` having a text suffix) before uploading.
