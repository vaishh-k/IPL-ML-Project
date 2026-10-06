# 📊 IPL ML Lab — Lines of Code Report

> **Generated:** 2026-08-27 · Project: `ipl_prediction_project`

---

## 🔢 Grand Total

| Metric | Value |
|--------|-------|
| 🐍 Python Files | **28** |
| 📝 Python Lines of Code | **5,096** |
| 📄 Docs / Config Files | **2** (144 lines) |
| 🔢 **Grand Total (all files)** | **5,240 lines** |

---

## 🗂️ By Layer

| Layer | Files | Lines |
|-------|-------|-------|
| Backend (`backend/`) | 15 | ~2,545 |
| Frontend (`frontend/`) | 12 | ~2,418 |
| Root scripts | 1 | 133 |

---

## 📄 Per-File Breakdown

> Sorted by lines of code (descending)

| # | File Path | Lines |
|---|-----------|------:|
| 1 | `frontend/app.py` | **1,101** |
| 2 | `backend/routes/eda.py` | 411 |
| 3 | `backend/services/analysis.py` | 409 |
| 4 | `backend/services/visualization.py` | 357 |
| 5 | `frontend/components/regression_page.py` | 340 |
| 6 | `frontend/components/classification_page.py` | 317 |
| 7 | `backend/services/regression_service.py` | 227 |
| 8 | `backend/services/classification_service.py` | 209 |
| 9 | `frontend/utils/helpers.py` | 189 |
| 10 | `backend/services/cleaning.py` | 169 |
| 11 | `backend/routes/dataset.py` | 164 |
| 12 | `frontend/services/api_client.py` | 141 |
| 13 | `generate_sample_data.py` | 133 |
| 14 | `frontend/components/progress.py` | 123 |
| 15 | `backend/services/preprocessing.py` | 122 |
| 16 | `frontend/components/sidebar.py` | 108 |
| 17 | `backend/routes/preprocessing.py` | 107 |
| 18 | `backend/utils/helpers.py` | 80 |
| 19 | `backend/routes/pca.py` | 60 |
| 20 | `backend/services/pca_service.py` | 58 |
| 21 | `frontend/components/cards.py` | 49 |
| 22 | `backend/routes/regression.py` | 47 |
| 23 | `backend/routes/classification.py` | 47 |
| 24 | `backend/main.py` | 44 |
| 25 | `frontend/components/dataframe.py` | 28 |
| 26 | `frontend/components/metrics.py` | 22 |
| 27 | `frontend/components/charts.py` | 17 |
| 28 | `backend/services/data_loader.py` | 17 |

---

## 🏆 Top 5 Largest Files

```
1. frontend/app.py                          ████████████████████  1,101 lines
2. backend/routes/eda.py                    ████████               411 lines
3. backend/services/analysis.py             ████████               409 lines
4. backend/services/visualization.py        ███████                357 lines
5. frontend/components/regression_page.py   ██████                 340 lines
```

---

## 🗃️ Backend File Summary

| File | Role | Lines |
|------|------|------:|
| `backend/main.py` | FastAPI app entry point, router registration | 44 |
| `backend/routes/eda.py` | EDA analysis endpoints | 411 |
| `backend/routes/dataset.py` | Dataset load/select endpoints | 164 |
| `backend/routes/preprocessing.py` | Cleaning & transformation endpoints | 107 |
| `backend/routes/pca.py` | PCA reduction endpoints | 60 |
| `backend/routes/regression.py` | Regression train & predict endpoints | 47 |
| `backend/routes/classification.py` | Classification train & predict endpoints | 47 |
| `backend/services/analysis.py` | Statistical analysis logic | 409 |
| `backend/services/visualization.py` | Plotly chart generation | 357 |
| `backend/services/regression_service.py` | Regression ML pipeline & persistence | 227 |
| `backend/services/classification_service.py` | Classification ML pipeline & persistence | 209 |
| `backend/services/cleaning.py` | Data cleaning transformations | 169 |
| `backend/services/preprocessing.py` | Feature engineering logic | 122 |
| `backend/services/pca_service.py` | PCA dimensionality reduction | 58 |
| `backend/services/data_loader.py` | CSV loader utility | 17 |
| `backend/utils/helpers.py` | Backend utility functions | 80 |
| **Backend Total** | | **~2,545** |

---

## 🖥️ Frontend File Summary

| File | Role | Lines |
|------|------|------:|
| `frontend/app.py` | Main Streamlit app, all 20 EDA steps, routing | 1,101 |
| `frontend/components/regression_page.py` | Regression pipeline UI | 340 |
| `frontend/components/classification_page.py` | Classification pipeline UI | 317 |
| `frontend/components/progress.py` | Progress bar & timeline | 123 |
| `frontend/components/sidebar.py` | Sidebar navigation | 108 |
| `frontend/components/cards.py` | Info & quality card components | 49 |
| `frontend/components/dataframe.py` | DataFrame display component | 28 |
| `frontend/components/metrics.py` | Metric grid component | 22 |
| `frontend/components/charts.py` | Plotly chart wrapper | 17 |
| `frontend/services/api_client.py` | HTTP client for FastAPI backend | 141 |
| `frontend/utils/helpers.py` | CSS injection & UI utilities | 189 |
| **Frontend Total** | | **~2,418** |

---

## 📦 Project Architecture Overview

```
ipl_prediction_project/
├── backend/
│   ├── main.py                        (44)   FastAPI entry point
│   ├── routes/
│   │   ├── dataset.py                 (164)  Dataset endpoints
│   │   ├── eda.py                     (411)  EDA endpoints
│   │   ├── preprocessing.py           (107)  Preprocessing endpoints
│   │   ├── pca.py                     (60)   PCA endpoints
│   │   ├── regression.py              (47)   Regression endpoints
│   │   └── classification.py          (47)   Classification endpoints
│   ├── services/
│   │   ├── analysis.py                (409)  Statistical analysis
│   │   ├── visualization.py           (357)  Chart generation
│   │   ├── cleaning.py                (169)  Data cleaning
│   │   ├── preprocessing.py           (122)  Feature engineering
│   │   ├── regression_service.py      (227)  Regression ML
│   │   ├── classification_service.py  (209)  Classification ML
│   │   ├── pca_service.py             (58)   PCA service
│   │   └── data_loader.py             (17)   CSV loader
│   └── utils/
│       └── helpers.py                 (80)   Backend utilities
│
├── frontend/
│   ├── app.py                         (1101) Main Streamlit app
│   ├── components/
│   │   ├── regression_page.py         (340)  Regression UI
│   │   ├── classification_page.py     (317)  Classification UI
│   │   ├── progress.py                (123)  Progress timeline
│   │   ├── sidebar.py                 (108)  Sidebar nav
│   │   ├── cards.py                   (49)   Card components
│   │   ├── dataframe.py               (28)   DataFrame viewer
│   │   ├── metrics.py                 (22)   Metric grid
│   │   └── charts.py                  (17)   Chart wrapper
│   ├── services/
│   │   └── api_client.py              (141)  Backend HTTP client
│   └── utils/
│       └── helpers.py                 (189)  CSS & UI helpers
│
├── data/                              IPL CSV datasets
├── generate_sample_data.py            (133)  Sample data generator
└── LOC_REPORT.md                      This file
```

---

*Total: **28 Python files** · **5,096 Python LOC** · **5,240 lines across all files***
