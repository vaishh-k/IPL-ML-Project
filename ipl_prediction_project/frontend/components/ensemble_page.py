import streamlit as st
import pandas as pd
from services.api_client import ApiClient
from ipl_utils.helpers import save_current_dataset_state, load_dataset_state

# ==============================================================================
# IPL ML LAB - ENSEMBLE LEARNING LAB COMPONENT
# ==============================================================================
# This module implements the ensemble learning workflow including:
# 1. Ensemble goal selection (1st Innings Score Regression vs Match Winner Classification).
# 2. Multi-model ensemble training (Random Forest, Gradient Boosting, XGBoost, AdaBoost).
# 3. Model comparison table highlighting the best-performing model dynamically.
# 4. Interactive, real-time prediction forms for regression and multiclass classification.
# ==============================================================================

# ENSEMBLE OPTIONS: Configures task goals, descriptions, and dataset contexts
ENSEMBLE_OPTIONS = {
    "regression": {
        "label": "Predict 1st Innings Score (Regression)",
        "desc": "Ensemble regression models: Predict 1st innings score based on venue, teams, toss decisions, etc.",
        "dataset": "ipl_matches.csv",
    },
    "classification": {
        "label": "Predict Match Winner (Classification)",
        "desc": "Ensemble classification models: Predict winning team name based on venue, teams, toss decisions, etc.",
        "dataset": "ipl_matches.csv",
    },
}

def render_ensemble_page(api: ApiClient):
    # Header Section
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); padding: 2rem; border-radius: 12px; border: 1px solid #334155; margin-bottom: 2rem;'>
            <h2 style='margin:0; font-size:2.2rem; font-weight:700; color: #F97316;'>Ensemble Learning Lab</h2>
            <p style='margin:0.5rem 0 0 0; color:#94A3B8; font-size:1.1rem;'>
                Compare and predict using Random Forest, Gradient Boosting, XGBoost, and AdaBoost models.
            </p>
        </div>
    """, unsafe_allow_html=True)

    #  1. SELECT MODE (CARDS) 
    st.markdown("### Step 1 - Select Problem Statement")
    
    if "ens_mode_selected" not in st.session_state:
        st.session_state["ens_mode_selected"] = "regression"
        
    mode = st.session_state["ens_mode_selected"]
    
    prob_col1, prob_col2 = st.columns(2)
    for idx, (key, info) in enumerate(ENSEMBLE_OPTIONS.items()):
        col = prob_col1 if idx == 0 else prob_col2
        with col:
            active = (mode == key)
            bc = "#F97316" if active else "rgba(255,255,255,0.1)"
            bg = "rgba(249,115,22,0.12)" if active else "rgba(255,255,255,0.03)"
            st.markdown(
                f"<div style='border:2px solid {bc}; background:{bg};"
                " border-radius:12px; padding:1.2rem; margin-bottom:0.5rem; min-height:120px;'>"
                f"<div style='font-weight:700; font-size:1rem; color:#F3F4F6;'>{info['label']}</div>"
                f"<div style='font-size:1rem; color:#9CA3AF; margin-top:0.3rem;'>{info['desc']}</div>"
                f"<div style='font-size:1rem; color:#F97316; margin-top:0.5rem;'>Dataset: {info['dataset']}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            btn_label = "Selected" if active else "Select"
            if st.button(btn_label, key=f"ens_mode_btn_{key}", use_container_width=True):
                st.session_state["ens_mode_selected"] = key
                st.rerun()

    st.markdown("---")

    col_config, col_run = st.columns([2, 1])

    with col_config:
        st.markdown("### Ensemble Algorithms")
        st.caption("Select the ensemble models to train and compare:")
        
        selected_algos = []
        if st.checkbox("Random Forest", value=True, key="ens_cb_rf"):
            selected_algos.append("rf")
        if st.checkbox("Gradient Boosting", value=True, key="ens_cb_gb"):
            selected_algos.append("gb")
        if st.checkbox("XGBoost", value=True, key="ens_cb_xgb"):
            selected_algos.append("xgb")
        if st.checkbox("AdaBoost", value=True, key="ens_cb_ada"):
            selected_algos.append("ada")

    with col_run:
        st.markdown("### Action")
        st.write("Train and evaluate the selected advanced ensemble models on the matches dataset.")
        train_clicked = st.button("Train Ensemble Models", type="primary", use_container_width=True, key="train_ens_btn")

    res_key = f"ensemble_results_{mode}"

    if train_clicked:
        if not selected_algos:
            st.error("Please select at least one algorithm to train.")
        else:
            with st.spinner("Training ensemble models..."):
                if mode == "regression":
                    res = api.train_ensemble_reg(selected_algos)
                else:
                    res = api.train_ensemble_cls(selected_algos)
                
                if "error" in res:
                    st.error(f"Error training: {res.get('detail', 'Unknown error')}")
                    st.session_state[res_key] = None
                else:
                    st.session_state[res_key] = res
                    st.toast("Training Complete!")

    results = st.session_state.get(res_key, None)

    if results:
        #  2. COMPARE RESULTS 
        st.markdown("### Model Performance Comparison")
        metrics_dict = results["results"]
        
        if not metrics_dict:
            st.warning("No models were successfully trained.")
            return
 
        best_key = results["best_model"]
        
        # Build comparative dataframe
        rows = []
        for key, metrics in metrics_dict.items():
            row = {"Algorithm": metrics["label"]}
            for k, v in metrics.items():
                if k != "label":
                    metric_name = k.replace("_", " ").upper()
                    row[metric_name] = v
            rows.append(row)
        
        compare_df = pd.DataFrame(rows)
        
        # Highlight best model row helper
        def highlight_best(val):
            # If accuracy is target for classification, or R2 for regression
            best_label = metrics_dict[best_key]["label"]
            style = []
            for col in compare_df.columns:
                if col == "Algorithm":
                    style.append("background-color: rgba(249,115,22,0.2); font-weight:bold;" if val["Algorithm"] == best_label else "")
                else:
                    style.append("background-color: rgba(249,115,22,0.2); font-weight:bold;" if val["Algorithm"] == best_label else "")
            return style

        styled_df = compare_df.style.apply(highlight_best, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Best Model Callout Card
        bm = metrics_dict[best_key]
        st.markdown(f"""
            <div style="background:linear-gradient(135deg, rgba(249,115,22,0.15) 0%, rgba(251,146,60,0.05) 100%);
                        border: 1px solid rgba(249,115,22,0.3); padding: 1.5rem; border-radius: 12px; text-align: center; margin: 1.5rem 0;">
                <h3 style="color:#F97316; margin:0.25rem 0;">{bm['label']}</h3>
                <p style="color:#94A3B8; font-size: 0.95rem; margin-bottom: 1rem;">Selected as the best-performing model based on { 'Accuracy' if mode == 'classification' else 'R² Score' }.</p>
            </div>
        """, unsafe_allow_html=True)

        #  3. CHOOSE ALGORITHM & PREDICT 
        st.markdown("---")
        st.markdown("### Real-Time Ensemble Prediction")
        
        trained_keys = list(metrics_dict.keys())
        default_idx = trained_keys.index(best_key) if best_key in trained_keys else 0
        
        selected_predict_algo = st.selectbox(
            "Choose Algorithm for Prediction:",
            options=trained_keys,
            index=default_idx,
            format_func=lambda k: f"{metrics_dict[k]['label']} {'(Best)' if k == best_key else ''}",
            key="ens_pred_algo_select"
        )

        st.markdown("#### Enter Input Features")
        st.caption("Provide match conditions to calculate predictions in real-time.")

        inputs = {}
        features_meta = results.get("features_metadata", [])
        
        col1, col2 = st.columns(2)
        for idx, f in enumerate(features_meta):
            col = col1 if idx % 2 == 0 else col2
            name_label = f["name"].replace("_", " ").title()
            with col:
                if f["type"] == "categorical":
                    inputs[f["name"]] = st.selectbox(
                        f"Select {name_label}:", options=f["options"],
                        key=f"ens_input_{f['name']}",
                    )
                else:
                    step = 1 if int(f["min"]) == f["min"] and int(f["max"]) == f["max"] else 0.1
                    inputs[f["name"]] = st.number_input(
                        f"{name_label} ({f['min']:.0f} to {f['max']:.0f}):",
                        min_value=float(f["min"]), max_value=float(f["max"]),
                        value=float(f["default"]), step=float(step),
                        key=f"ens_input_{f['name']}",
                    )

        st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("Calculate Ensemble Prediction", type="primary", use_container_width=True, key="ens_calc_btn")

        if predict_btn:
            with st.spinner("Calculating prediction..."):
                if mode == "regression":
                    pred_res = api.predict_ensemble_reg(selected_predict_algo, inputs)
                    if "error" in pred_res:
                        st.error(f"Prediction failed: {pred_res.get('detail', 'Unknown error')}")
                    else:
                        prediction = pred_res["prediction"]
                        st.markdown(f"""
                            <div style="background: rgba(24, 24, 37, 0.9); border: 2px solid #F97316; border-radius: 12px; padding: 2rem; text-align: center; margin-top: 1.5rem;">
                                <div style="font-size: 0.85rem; color: #F97316; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.5rem;">Predicted First Innings Score</div>
                                <div style="font-size: 4rem; font-weight: 900; color: #F3F4F6;">{prediction:.0f}</div>
                                <div style="font-size: 0.9rem; color: #9CA3AF; margin-top: 0.5rem;">Runs</div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    pred_res = api.predict_ensemble_cls(selected_predict_algo, inputs)
                    if "error" in pred_res:
                        st.error(f"Prediction failed: {pred_res.get('detail', 'Unknown error')}")
                    else:
                        pred_label = pred_res.get("prediction_label", "Unknown")
                        probas = pred_res.get("probabilities")
                        label_names = pred_res.get("label_names")
                        
                        st.markdown(f"""
                            <div style="background: rgba(24, 24, 37, 0.9); border: 2px solid #F97316; border-radius: 12px; padding: 2rem; text-align: center; margin-top: 1.5rem;">
                                <div style="font-size: 0.85rem; color: #F97316; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.5rem;">Predicted Winning Team</div>
                                <div style="font-size: 2.8rem; font-weight: 900; color: #F3F4F6; line-height:1.2;">{pred_label}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if probas and label_names:
                            st.markdown("#### Probability Breakdown")
                            prob_data = [{"Team": name, "Probability": f"{p*100:.2f}%"} for name, p in zip(label_names, probas) if p > 0.01]
                            df_probs = pd.DataFrame(prob_data).sort_values("Probability", ascending=False)
                            st.dataframe(df_probs, use_container_width=True, hide_index=True)
    else:
        st.info("Click **'Train Ensemble Models'** above to run the machine learning ensemble models and compare stats.")
        st.info("Click **'Train Ensemble Models'** above to run the machine learning ensemble models and compare stats.")
