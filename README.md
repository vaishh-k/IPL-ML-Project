# IPL Match & Performance Prediction System

A comprehensive **Machine Learning** framework built using **Python** to analyze historical Indian Premier League (IPL) data and forecast match outcomes and team performances. This project evaluates, implements, and benchmarks a wide array of machine learning algorithms to achieve maximum predictive accuracy.

## 🚀 Project Overview
Predicting outcomes in cricket is highly dynamic due to numerous variables like venue, toss, player form, and team composition. This system approaches the problem from multiple technical dimensions:
* **Regression Ensembles:** Utilized multiple regression algorithms to forecast continuous targets such as first-innings scores and precise run-chase targets.
* **Random Forest Models:** Deployed both Random Forest Regressors and Classifiers to cleanly navigate non-linear relationships within complex player and team statistics.
* **Algorithmic Benchmarking:** Evaluates and highlights the peak-performing model by directly comparing metrics like $R^2$ score and Mean Absolute Error (MAE).

## 📊 Key Features
* **Comprehensive Data Cleaning:** Preprocesses massive historical IPL datasets to handle missing values and encode categorical data.
* **Feature Engineering:** Extracts advanced match variables, including venue scoring trends, team head-to-head ratios, and live match situations.
* **Multi-Model Pipeline:** Trains, tests, and evaluates multiple regression and classification algorithms side-by-side.

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

## 📋 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd IPL-ML-Project
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

## 📈 Evaluation Results
The algorithms are rigorously evaluated and compared based on prediction accuracy, helping identify the most robust model for various match scenarios.
