import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from ipl_utils.helpers import save_current_dataset_state, load_dataset_state

# ==============================================================================
# IPL ML LAB - REGRESSION PIPELINE COMPONENT
# ==============================================================================
# This module implements the regression workflow including:
# 1. Selection of Regression Problem Statements (e.g. 1st Innings Score vs Batsman score).
# 2. Multi-algorithm configuration and selection (Linear, Ridge, Lasso, Decision Tree).
# 3. Model training, evaluation metric grids, and visual residual comparisons.
# 4. Interactive, real-time prediction screens for inference.
# ==============================================================================

# PROBLEM OPTIONS: Defines available target problems and their metadata
PROBLEM_OPTIONS = {
    "first_inning_score": {
        "label": "Predict IPL First Inning Score",
        "desc":  "Predict the total runs scored by a batting team in the first inning of a match using match-level features (venue, teams, toss).",
        "dataset": "ipl_matches.csv",
        "icon": "",
    },
    "batsman_score": {
        "label": "Predict IPL Score by Batsman",
        "desc":  "Predict the runs scored by a batsman on each delivery using ball-by-ball features (over, ball, batting team, bowler).",
        "dataset": "processed_ipl_ball_by_ball.csv",
        "icon": "",
    },
}

# ALGORITHM OPTIONS: Maps algorithm keys to user-friendly titles and baseline descriptions
ALGORITHM_OPTIONS = {
    "linear":        ("Linear Regression",  "Ordinary Least Squares. Fast, interpretable baseline."),
    "ridge":         ("Ridge Regression",    "L2 regularisation. Great when features are correlated."),
    "lasso":         ("Lasso Regression",    "L1 regularisation. Performs implicit feature selection."),
    "decision_tree": ("Decision Tree",       "Non-linear, handles mixed feature types well."),
}

METRIC_COLOR = "#F97316"

def _metric_card(label, value, delta_label=""):
    # [Function]: _metric_card
    # [Description]: Implements and executes the  Metric Card logic within this module pipeline.
    return f"""
    <div style="background:rgba(249,115,22,0.08); border:1px solid rgba(249,115,22,0.25);
                border-radius:10px; padding:1rem 1.2rem; text-align:center; height:100%;">
      <div style="font-size:0.75rem; color:#F97316; font-weight:700; letter-spacing:1px; text-transform:uppercase;">{label}</div>
      <div style="font-size:1.8rem; font-weight:800; color:#F3F4F6; margin:0.3rem 0;">{value}</div>
      <div style="font-size:0.75rem; color:#9CA3AF;">{delta_label}</div>
    </div>"""

def render_regression_page(api):
    """Main regression pipeline page rendered inside app.py."""

    if "reg_subpage" not in st.session_state:
        st.session_state["reg_subpage"] = "evaluation"

    if st.session_state["reg_subpage"] == "prediction":
        render_prediction_screen(api)
        return

    #  Page header 
    st.markdown("""
        <div style="margin-bottom:1.5rem;">
            <span style="font-size:0.85rem; color:#F97316; font-weight:600; text-transform:uppercase; letter-spacing:1px;">ALGORITHMS</span>
            <h2 style="margin:0; font-size:2.2rem; font-weight:700;">Regression Pipeline</h2>
            <p style="color:#9CA3AF; margin:0.5rem 0 0;">Train, evaluate and compare regression models on IPL data end-to-end.</p>
        </div>
    """, unsafe_allow_html=True)

    #  Flow diagram 
    with st.expander("Pipeline Flow", expanded=False):
        st.markdown("""
        ```
        REGRESSION
           
        Select Problem Statement
        (First Inning Score  |  IPL Score by Batsman)
           
        Select Algorithms
        (Linear · Ridge · Lasso · Decision Tree)
           
        TRAIN    PREDICT    EVALUATE
           
        COMPARE MODELS    SHOW BEST MODEL    REALTIME PREDICTIONS
        ```
        """)

    st.markdown("---")

    #  STEP 1  Problem Statement 
    st.markdown("### Step 1  Select Problem Statement")
    prob_col1, prob_col2 = st.columns(2)
    selected_problem = st.session_state.get("reg_problem", None)

    for idx, (key, info) in enumerate(PROBLEM_OPTIONS.items()):
        col = prob_col1 if idx == 0 else prob_col2
        with col:
            active = selected_problem == key
            border_color = "#F97316" if active else "rgba(255,255,255,0.1)"
            bg_color = "rgba(249,115,22,0.12)" if active else "rgba(255,255,255,0.03)"
            
            st.markdown(f"""
            <div style="border:2px solid {border_color}; background:{bg_color}; border-radius:12px;
                        padding:1.2rem; margin-bottom:0.5rem; cursor:pointer;">
                <div style="font-weight:700; font-size:1.05rem; color:#F3F4F6;">{info['label']}</div>
                <div style="font-size:1rem; color:#9CA3AF; margin-top:0.3rem;">{info['desc']}</div>
            </div>""", unsafe_allow_html=True)
            
            # Choose dataset inside the problem statement column
            files = st.session_state.get("available_datasets", [])
            tables = st.session_state.get("available_db_tables", [])
            source_mode = st.session_state.get("data_source_mode", "CSV Files")
            options_list = files if source_mode == "CSV Files" else tables
            
            active_ds_key = f"active_dataset_{key}"
            if active_ds_key not in st.session_state:
                if key == "first_inning_score":
                    st.session_state[active_ds_key] = "ipl_matches.csv" if source_mode == "CSV Files" else "ipl_matches"
                else:
                    st.session_state[active_ds_key] = "processed_ipl_ball_by_ball.csv" if source_mode == "CSV Files" else "processed_ipl_ball_by_ball"
            
            try:
                ds_idx = options_list.index(st.session_state[active_ds_key])
            except ValueError:
                ds_idx = 0
                st.session_state[active_ds_key] = options_list[0] if options_list else ""
                
            selected_ds = st.selectbox(
                f"Choose dataset:",
                options=options_list,
                index=ds_idx,
                key=f"ds_select_{key}"
            )
            
            if selected_ds != st.session_state[active_ds_key]:
                st.session_state[active_ds_key] = selected_ds
                save_current_dataset_state()
                st.session_state["active_dataset"] = selected_ds
                load_dataset_state(selected_ds)
                if source_mode == "CSV Files":
                    api.select_local_dataset(selected_ds)
                else:
                    api.select_db_table(selected_ds)
                st.rerun()

            btn_label = "Selected" if active else "Select & Load Dataset"
            if st.button(btn_label, key=f"prob_btn_{key}", use_container_width=True):
                st.session_state["reg_problem"] = key
                st.session_state["reg_results"] = None
                st.session_state["reg_subpage"] = "evaluation"
                
                save_current_dataset_state()
                st.session_state["active_dataset"] = selected_ds
                load_dataset_state(selected_ds)
                if source_mode == "CSV Files":
                    api.select_local_dataset(selected_ds)
                else:
                    api.select_db_table(selected_ds)
                st.rerun()

    if not selected_problem:
        st.info("Select a problem statement to continue.")
        return

    st.markdown("---")

    #  STEP 2  Algorithm Selection 
    st.markdown("### Step 2  Select Algorithms")
    st.caption("Select one or more algorithms to train and compare.")

    if "reg_selected_algos" not in st.session_state:
        st.session_state["reg_selected_algos"] = ["linear"]

    algo_cols = st.columns(4)
    selected_algos = []
    for idx, (algo_key, (algo_label, algo_desc)) in enumerate(ALGORITHM_OPTIONS.items()):
        with algo_cols[idx]:
            checked = st.checkbox(
                algo_label,
                value=(algo_key in st.session_state["reg_selected_algos"]),
                key=f"algo_cb_{algo_key}",
                help=algo_desc
            )
            if checked:
                selected_algos.append(algo_key)

    st.session_state["reg_selected_algos"] = selected_algos

    if not selected_algos:
        st.warning(" Select at least one algorithm.")
        return

    st.markdown("---")

    #  STEP 3  Train Button 
    st.markdown("###  Step 3  Train & Predict")
    prob_info = PROBLEM_OPTIONS[selected_problem]
    algo_labels = [ALGORITHM_OPTIONS[a][0] for a in selected_algos]

    st.markdown(f"""
    <div style="background:rgba(249,115,22,0.07); border:1px solid rgba(249,115,22,0.2);
                border-radius:10px; padding:1rem 1.5rem; margin-bottom:1rem;">
        <b>Problem:</b> {prob_info['label']}<br>
        <b>Algorithms:</b> {' · '.join(algo_labels)}<br>
        <b>Dataset:</b> {prob_info['dataset']}
    </div>""", unsafe_allow_html=True)

    train_btn = st.button(" Train All Selected Models", type="primary", use_container_width=True, key="train_models_btn")

    if train_btn:
        with st.spinner(" Loading data, training models & evaluating (this may take 2060 seconds for large datasets)"):
            result = api.train_regression(selected_problem, selected_algos)
            if result.get("error"):
                st.error(f"Training failed: {result.get('detail', 'Unknown error')}")
                return
            st.session_state["reg_results"] = result
            st.session_state["reg_subpage"] = "evaluation"
            st.success(" All models trained successfully!")

    results_data = st.session_state.get("reg_results")
    if not results_data:
        return

    st.markdown("---")

    #  STEP 4  Evaluation Metrics 
    st.markdown("###  Step 4  Evaluation Technique")
    st.caption(f"Test set size: **{results_data['test_size']:,}** samples · Target: {results_data['target_column']}")

    model_results = results_data.get("results", {})
    best_key = results_data.get("best_model")

    for algo_key, metrics in model_results.items():
        is_best = (algo_key == best_key)
        badge = "  **BEST**" if is_best else ""
        border = "2px solid #F97316" if is_best else "1px solid rgba(255,255,255,0.1)"
        with st.container():
            st.markdown(f"#### {metrics['label']}{badge}")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.markdown(_metric_card("MAE",  f"{metrics['mae']:.4f}",  "Mean Abs Error"), unsafe_allow_html=True)
            c2.markdown(_metric_card("MSE",  f"{metrics['mse']:.4f}",  "Mean Sq Error"),  unsafe_allow_html=True)
            c3.markdown(_metric_card("RMSE", f"{metrics['rmse']:.4f}", "Root MSE"),        unsafe_allow_html=True)
            c4.markdown(_metric_card("R²",   f"{metrics['r2']:.4f}",   "R-Squared"),       unsafe_allow_html=True)
            c5.markdown(_metric_card("CV R²",f"{metrics['cv_r2']:.4f}","3-Fold CV R²"),    unsafe_allow_html=True)

            # Scatter: Actual vs Predicted
            preds = results_data.get("predictions", {}).get(algo_key, {})
            if preds:
                actual    = preds["actual"]
                predicted = preds["predicted"]
                fig = go.Figure()
                min_v, max_v = min(actual), max(actual)
                fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v],
                                         mode="lines", name="Perfect Fit",
                                         line=dict(color="rgba(249,115,22,0.5)", dash="dash")))
                fig.add_trace(go.Scatter(x=actual, y=predicted, mode="markers",
                                         name="Predictions",
                                         marker=dict(color="#F97316", size=5, opacity=0.6)))
                fig.update_layout(
                    title=f"{metrics['label']}  Actual vs Predicted",
                    xaxis_title="Actual", yaxis_title="Predicted",
                    template="plotly_dark", height=350,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F3F4F6"),
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    st.markdown("---")

    #  STEP 5  Compare Models 
    st.markdown("###  Step 5  Compare Models")

    compare_data = {
        "Model": [m["label"] for m in model_results.values()],
        "MAE":   [m["mae"]   for m in model_results.values()],
        "RMSE":  [m["rmse"]  for m in model_results.values()],
        "R²":    [m["r2"]    for m in model_results.values()],
        "CV R²": [m["cv_r2"] for m in model_results.values()],
    }
    compare_df = pd.DataFrame(compare_data)

    def highlight_best(row):
        # [Function]: highlight_best
        # [Description]: Implements and executes the Highlight Best logic within this module pipeline.
        style = []
        for col in row.index:
            if col in ("R²", "CV R²"):
                val = row[col]
                best_val = compare_df[col].max()
                style.append("background-color: rgba(249,115,22,0.25); font-weight:bold;" if val == best_val else "")
            elif col in ("MAE", "RMSE"):
                val = row[col]
                best_val = compare_df[col].min()
                style.append("background-color: rgba(249,115,22,0.25); font-weight:bold;" if val == best_val else "")
            else:
                style.append("")
        return style

    styled_df = compare_df.style.apply(highlight_best, axis=1).format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}", "CV R²": "{:.4f}"})
    st.dataframe(styled_df, use_container_width=True)

    fig_bar = go.Figure()
    colors = ["#F97316" if k == best_key else "#374151" for k in model_results.keys()]
    fig_bar.add_trace(go.Bar(
        x=[m["label"] for m in model_results.values()],
        y=[m["r2"]    for m in model_results.values()],
        marker_color=colors,
        text=[f"{m['r2']:.4f}" for m in model_results.values()],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Model Comparison  R² Score (Higher is Better)",
        xaxis_title="Model", yaxis_title="R² Score",
        template="plotly_dark", height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F3F4F6"),
        yaxis=dict(range=[min(0, min([m["r2"] for m in model_results.values()])-0.05), 1.05]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    #  STEP 6  Best Model 
    st.markdown("### Step 6  Best Model")

    if best_key and best_key in model_results:
        bm = model_results[best_key]
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, rgba(249,115,22,0.18) 0%, rgba(251,146,60,0.08) 100%);
                    border:2px solid #F97316; border-radius:16px; padding:2rem; text-align:center;">
            <h2 style="color:#F97316; margin:0.5rem 0;">{bm['label']}</h2>
            <p style="color:#9CA3AF; margin-bottom:1.2rem;">Selected as the best-performing model based on R² score.</p>
            <div style="display:flex; justify-content:center; gap:2rem; flex-wrap:wrap;">
                <div><div style="font-size:1rem; color:#F97316; font-weight:700;">R²</div><div style="font-size:2rem; font-weight:800;">{bm['r2']:.4f}</div></div>
                <div><div style="font-size:1rem; color:#F97316; font-weight:700;">RMSE</div><div style="font-size:2rem; font-weight:800;">{bm['rmse']:.4f}</div></div>
                <div><div style="font-size:1rem; color:#F97316; font-weight:700;">MAE</div><div style="font-size:2rem; font-weight:800;">{bm['mae']:.4f}</div></div>
                <div><div style="font-size:1rem; color:#F97316; font-weight:700;">CV R²</div><div style="font-size:2rem; font-weight:800;">{bm['cv_r2']:.4f}</div></div>
            </div>
            <div style="margin-top:1rem; padding:1rem; background:rgba(249,115,22,0.1); border-radius:8px; font-size:0.85rem; color:#D1D5DB;">
                Features used: {', '.join(f'<code>{f}</code>' for f in results_data['feature_columns'])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        #  NAVIGATION TO REAL-TIME PREDICTIONS 
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        if st.button("Proceed to Real-Time Predictions ", type="primary", use_container_width=True, key="go_to_predict_btn"):
            st.session_state["reg_subpage"] = "prediction"
            st.rerun()
    else:
        st.warning("No best model identified.")


def render_prediction_screen(api):
    """Prediction view to calculate score inferences on user input."""
    results_data = st.session_state.get("reg_results")
    selected_problem = st.session_state.get("reg_problem")
    
    if not results_data or not selected_problem:
        st.warning("No trained models found. Please train models first.")
        st.session_state["reg_subpage"] = "evaluation"
        st.rerun()

    st.markdown("""
        <div style="margin-bottom:1.5rem;">
            <span style="font-size:0.85rem; color:#F97316; font-weight:600; text-transform:uppercase; letter-spacing:1px;">REAL-TIME INFERENCE</span>
            <h2 style="margin:0; font-size:2.2rem; font-weight:700;">Real-Time Predictor</h2>
            <p style="color:#9CA3AF; margin:0.5rem 0 0;">Enter match attributes below to get real-time predictions from your trained models.</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button(" Back to Evaluation & Comparison", key="back_to_eval_btn", use_container_width=True):
        st.session_state["reg_subpage"] = "evaluation"
        st.rerun()

    st.markdown("---")

    model_results = results_data.get("results", {})
    best_key = results_data.get("best_model")

    #  Model selection 
    st.markdown("### Model Configuration")
    trained_algos = list(model_results.keys())
    
    default_idx = trained_algos.index(best_key) if best_key in trained_algos else 0
    selected_algo = st.selectbox(
        "Choose Algorithm for Prediction:",
        options=trained_algos,
        index=default_idx,
        format_func=lambda k: f"{model_results[k]['label']} (R²: {model_results[k]['r2']:.4f}) {'(Best)' if k == best_key else ''}"
    )

    st.markdown("---")

    #  Input Form 
    st.markdown("### Enter Features")
    st.caption("Provide the inputs for prediction. Categorical lists and ranges are extracted directly from the dataset.")

    inputs = {}
    features_meta = results_data.get("features_metadata", [])

    col1, col2 = st.columns(2)
    for idx, f in enumerate(features_meta):
        col = col1 if idx % 2 == 0 else col2
        with col:
            name_label = f['name'].replace('_', ' ').title()
            if f['type'] == 'categorical':
                inputs[f['name']] = st.selectbox(
                    f"Select {name_label}:",
                    options=f['options'],
                    key=f"input_{f['name']}"
                )
            else:
                min_val = f['min']
                max_val = f['max']
                default_val = f['default']
                step = 1 if int(min_val) == min_val and int(max_val) == max_val else 0.1
                inputs[f['name']] = st.number_input(
                    f"Enter {name_label} (Range: {min_val} to {max_val}):",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=float(step),
                    key=f"input_{f['name']}"
                )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("Calculate Real-Time Prediction", type="primary", use_container_width=True, key="calc_pred_btn")

    if predict_clicked:
        with st.spinner("Calculating prediction..."):
            pred_res = api.predict_regression(selected_problem, selected_algo, inputs)
            if pred_res.get("error"):
                st.error(f"Prediction failed: {pred_res.get('detail', 'Unknown error')}")
            else:
                prediction_val = pred_res["prediction"]
                
                target_label = results_data['target_column'].replace('_', ' ').title()
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, rgba(249,115,22,0.18) 0%, rgba(251,146,60,0.08) 100%);
                            border:2px solid #F97316; border-radius:16px; padding:2rem; text-align:center; margin-top:2rem;">
                    <span style="font-size:1rem; color:#F97316; font-weight:700; letter-spacing:1px; text-transform:uppercase;">PREDICTED VALUE</span>
                    <h1 style="color:#FFFFFF; margin:0.5rem 0; font-size:3.5rem; font-weight:800;">{prediction_val:.2f}</h1>
                    <p style="color:#9CA3AF; margin-bottom:0.5rem;">Target: <b>{target_label}</b></p>
                    <div style="font-size:0.85rem; color:#F97316; font-weight:600;">Model: {model_results[selected_algo]['label']}</div>
                </div>
                """, unsafe_allow_html=True)
