import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from ipl_utils.helpers import save_current_dataset_state, load_dataset_state

# ==============================================================================
# IPL ML LAB - CLASSIFICATION PIPELINE COMPONENT
# ==============================================================================
# This module implements the classification workflow including:
# 1. Selection of Classification Problem Statements (Target Defend vs Batsman Fifty).
# 2. Algorithm selection (Logistic Regression, Decision Tree, SVM, Naive Bayes).
# 3. Model training, evaluation metric grids, confusion matrices, and ROC-AUC scores.
# 4. Interactive inference widget for real-time match predictions.
# ==============================================================================

# PROBLEM OPTIONS: Defines available classification targets and datasets
PROBLEM_OPTIONS = {
    "defend_target": {
        "label": "Predict Whether a Team Will Defend Its Target",
        "desc":  "Binary classification: Given match context (teams, venue, toss), predict if the batting-first team defends its total.",
        "dataset": "ipl_matches.csv",
    },
    "batsman_fifty": {
        "label": "Predict Whether a Batsman Will Score 50+",
        "desc":  "Binary classification: Given ball-by-ball features (batter, teams, venue), predict if a batsman will reach a fifty.",
        "dataset": "processed_ipl_ball_by_ball.csv",
    },
}

# ALGORITHM OPTIONS: Configures classifier descriptions and algorithms
ALGORITHM_OPTIONS = {
    "logistic":      ("Logistic Regression", "Linear decision boundary. Fast, interpretable baseline."),
    "decision_tree": ("Decision Tree",        "Non-linear splits. Handles mixed features well."),
    "svm":           ("SVM (RBF Kernel)",     "Powerful non-linear classifier. Best for complex boundaries."),
    "naive_bayes":   ("Naive Bayes",          "Probabilistic model. Fast training, great for small data."),
}


def _metric_card(label, value, sub=""):
    # [Function]: _metric_card
    # [Description]: Implements and executes the  Metric Card logic within this module pipeline.
    return (
        "<div style='background:rgba(249,115,22,0.08); border:1px solid rgba(249,115,22,0.25);"
        " border-radius:10px; padding:1rem 1.2rem; text-align:center; height:100%;'>"
        f"<div style='font-size:0.72rem; color:#F97316; font-weight:700;"
        f" letter-spacing:1px; text-transform:uppercase;'>{label}</div>"
        f"<div style='font-size:1.8rem; font-weight:800; color:#F3F4F6; margin:0.3rem 0;'>{value}</div>"
        f"<div style='font-size:0.72rem; color:#9CA3AF;'>{sub}</div>"
        "</div>"
    )


def _confusion_matrix_chart(cm, label_names, title):
    # [Function]: _confusion_matrix_chart
    # [Description]: Implements and executes the  Confusion Matrix Chart logic within this module pipeline.
    z   = [[cm[i][j] for j in range(len(cm[i]))] for i in range(len(cm))]
    txt = [[str(cm[i][j]) for j in range(len(cm[i]))] for i in range(len(cm))]
    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Pred: " + l for l in label_names],
        y=["Act: " + l  for l in label_names],
        text=txt, texttemplate="%{text}", colorscale="Oranges", showscale=False,
    ))
    fig.update_layout(
        title=title, template="plotly_dark", height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F3F4F6"), margin=dict(t=40, b=20, l=20, r=20),
    )
    return fig


def render_classification_page(api):
    # [Function]: render_classification_page
    # [Description]: Implements and executes the Render Classification Page logic within this module pipeline.
    if "cls_subpage" not in st.session_state:
        st.session_state["cls_subpage"] = "evaluation"

    if st.session_state["cls_subpage"] == "prediction":
        _render_prediction_screen(api)
        return

    #  Header 
    st.markdown(
        "<div style='margin-bottom:1.5rem;'>"
        "<span style='font-size:0.85rem; color:#F97316; font-weight:600;"
        " text-transform:uppercase; letter-spacing:1px;'>ALGORITHMS</span>"
        "<h2 style='margin:0; font-size:2.2rem; font-weight:700;'>Classification Pipeline</h2>"
        "<p style='color:#9CA3AF; margin:0.5rem 0 0;'>Train, evaluate and compare"
        " classification models on IPL data end-to-end.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Pipeline Flow", expanded=False):
        flow_text = (
            "Select Problem Statement (Defend Target | Batsman 50+) -> "
            "Select Algorithms (Logistic Regression, Decision Tree, SVM, Naive Bayes) -> "
            "TRAIN -> EVALUATE (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix) -> "
            "COMPARE MODELS -> BEST MODEL -> REAL-TIME PREDICTION"
        )
        st.code(flow_text, language="text")


    st.markdown("---")

    #  STEP 1: Problem Statement 
    st.markdown("### Step 1 - Select Problem Statement")
    selected_problem = st.session_state.get("cls_problem", None)
    prob_col1, prob_col2 = st.columns(2)
    for idx, (key, info) in enumerate(PROBLEM_OPTIONS.items()):
        col = prob_col1 if idx == 0 else prob_col2
        with col:
            active = selected_problem == key
            bc = "#F97316" if active else "rgba(255,255,255,0.1)"
            bg = "rgba(249,115,22,0.12)" if active else "rgba(255,255,255,0.03)"
            st.markdown(
                f"<div style='border:2px solid {bc}; background:{bg};"
                " border-radius:12px; padding:1.2rem; margin-bottom:0.5rem;'>"
                f"<div style='font-weight:700; font-size:1.05rem; color:#F3F4F6;'>{info['label']}</div>"
                f"<div style='font-size:1rem; color:#9CA3AF; margin-top:0.3rem;'>{info['desc']}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            
            # Choose dataset inside the problem statement column
            files = st.session_state.get("available_datasets", [])
            tables = st.session_state.get("available_db_tables", [])
            source_mode = st.session_state.get("data_source_mode", "CSV Files")
            options_list = files if source_mode == "CSV Files" else tables
            
            active_ds_key = f"active_dataset_cls_{key}"
            if active_ds_key not in st.session_state:
                if key == "defend_target":
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
                key=f"cls_ds_select_{key}"
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
            if st.button(btn_label, key=f"cls_prob_btn_{key}", use_container_width=True):
                st.session_state["cls_problem"] = key
                st.session_state["cls_results"] = None
                st.session_state["cls_subpage"] = "evaluation"
                
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

    #  STEP 2: Algorithm Selection 
    st.markdown("### Step 2 - Select Algorithms")
    st.caption("Select one or more classification algorithms to train and compare.")
    if "cls_selected_algos" not in st.session_state:
        st.session_state["cls_selected_algos"] = ["logistic"]
    algo_cols = st.columns(4)
    selected_algos = []
    for idx, (key, (lbl, desc)) in enumerate(ALGORITHM_OPTIONS.items()):
        with algo_cols[idx]:
            if st.checkbox(lbl, value=(key in st.session_state["cls_selected_algos"]),
                           key=f"cls_algo_{key}", help=desc):
                selected_algos.append(key)
    st.session_state["cls_selected_algos"] = selected_algos
    if not selected_algos:
        st.warning("Select at least one algorithm.")
        return

    st.markdown("---")

    #  STEP 3: Train 
    st.markdown("### Step 3 - Train and Evaluate")
    pinfo = PROBLEM_OPTIONS[selected_problem]
    algo_labels = ", ".join(ALGORITHM_OPTIONS[a][0] for a in selected_algos)
    st.markdown(
        "<div style='background:rgba(249,115,22,0.07); border:1px solid rgba(249,115,22,0.2);"
        " border-radius:10px; padding:1rem 1.5rem; margin-bottom:1rem;'>"
        f"<b>Problem:</b> {pinfo['label']}<br>"
        f"<b>Algorithms:</b> {algo_labels}<br>"
        f"<b>Dataset:</b> {pinfo['dataset']}"
        "</div>",
        unsafe_allow_html=True,
    )
    if st.button("Train All Selected Classifiers", type="primary",
                 use_container_width=True, key="cls_train_btn"):
        with st.spinner("Loading data, training classifiers and evaluating..."):
            result = api.train_classification(selected_problem, selected_algos)
            if result.get("error"):
                st.error(f"Training failed: {result.get('detail', 'Unknown error')}")
                return
            st.session_state["cls_results"] = result
            st.success("All classifiers trained successfully!")

    results_data = st.session_state.get("cls_results")
    if not results_data:
        return

    st.markdown("---")

    #  STEP 4: Evaluation 
    label_names   = results_data.get("label_names", ["No", "Yes"])
    model_results = results_data.get("results", {})
    best_key      = results_data.get("best_model")
    conf_matrices = results_data.get("confusion_matrices", {})

    st.markdown("### Step 4 - Evaluation")
    st.caption(
        f"Test set size: **{results_data['test_size']:,}** samples | "
        f"Target: **{results_data['target_column']}** | Classes: {label_names}"
    )

    for algo_key, metrics in model_results.items():
        is_best = (algo_key == best_key)
        badge = "  **[BEST MODEL]**" if is_best else ""
        st.markdown(f"#### {metrics['label']}{badge}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.markdown(_metric_card("Accuracy",  f"{metrics['accuracy']:.3f}",  "Overall correct"), unsafe_allow_html=True)
        c2.markdown(_metric_card("Precision", f"{metrics['precision']:.3f}", "TP / (TP+FP)"),    unsafe_allow_html=True)
        c3.markdown(_metric_card("Recall",    f"{metrics['recall']:.3f}",    "TP / (TP+FN)"),    unsafe_allow_html=True)
        c4.markdown(_metric_card("F1 Score",  f"{metrics['f1']:.3f}",        "Harmonic mean"),   unsafe_allow_html=True)
        roc_val = f"{metrics['roc_auc']:.3f}" if metrics.get("roc_auc") is not None else "N/A"
        c5.markdown(_metric_card("ROC-AUC",   roc_val,                        "Area under ROC"), unsafe_allow_html=True)
        c6.markdown(_metric_card("CV Acc",    f"{metrics['cv_accuracy']:.3f}", "3-Fold CV"),     unsafe_allow_html=True)

        if algo_key in conf_matrices:
            st.plotly_chart(
                _confusion_matrix_chart(conf_matrices[algo_key], label_names,
                                        f"Confusion Matrix - {metrics['label']}"),
                use_container_width=True,
            )
        st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("---")

    #  STEP 5: Compare Models 
    st.markdown("### Step 5 - Compare Models")
    compare_df = pd.DataFrame([
        {
            "Model":     m["label"],
            "Accuracy":  m["accuracy"],
            "Precision": m["precision"],
            "Recall":    m["recall"],
            "F1":        m["f1"],
            "ROC-AUC":   m["roc_auc"] if m.get("roc_auc") is not None else 0,
            "CV Acc":    m["cv_accuracy"],
        }
        for m in model_results.values()
    ])

    def highlight_best_cls(row):
        # [Function]: highlight_best_cls
        # [Description]: Implements and executes the Highlight Best Cls logic within this module pipeline.
        styles = []
        for col in row.index:
            if col in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV Acc"):
                styles.append(
                    "background-color: rgba(249,115,22,0.25); font-weight:bold;"
                    if row[col] == compare_df[col].max() else ""
                )
            else:
                styles.append("")
        return styles

    fmt = {"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
           "F1": "{:.4f}", "ROC-AUC": "{:.4f}", "CV Acc": "{:.4f}"}
    st.dataframe(compare_df.style.apply(highlight_best_cls, axis=1).format(fmt),
                 use_container_width=True)

    colors  = ["#F97316" if k == best_key else "#374151" for k in model_results]
    min_acc = min(m["accuracy"] for m in model_results.values())
    fig_bar = go.Figure(go.Bar(
        x=[m["label"] for m in model_results.values()],
        y=[m["accuracy"] for m in model_results.values()],
        marker_color=colors,
        text=[f"{m['accuracy']:.4f}" for m in model_results.values()],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Model Comparison - Accuracy (Higher is Better)",
        xaxis_title="Model", yaxis_title="Accuracy",
        template="plotly_dark", height=350,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F3F4F6"),
        yaxis=dict(range=[max(0, min_acc - 0.05), 1.05]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    #  STEP 6: Best Model 
    st.markdown("### Step 6 - Best Model")
    if best_key and best_key in model_results:
        bm = model_results[best_key]
        roc_display = f"{bm['roc_auc']:.4f}" if bm.get("roc_auc") is not None else "N/A"
        st.markdown(
            "<div style='background:linear-gradient(135deg, rgba(249,115,22,0.18) 0%,"
            " rgba(251,146,60,0.08) 100%); border:2px solid #F97316;"
            " border-radius:16px; padding:2rem; text-align:center;'>"
            f"<h2 style='color:#F97316; margin:0.5rem 0;'>{bm['label']}</h2>"
            "<p style='color:#9CA3AF; margin-bottom:1.2rem;'>Selected as the best-performing model based on Accuracy.</p>"
            "<div style='display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap;'>"
            f"<div><div style='font-size:0.75rem; color:#F97316; font-weight:700;'>ACCURACY</div>"
            f"<div style='font-size:2rem; font-weight:800;'>{bm['accuracy']:.4f}</div></div>"
            f"<div><div style='font-size:0.75rem; color:#F97316; font-weight:700;'>F1</div>"
            f"<div style='font-size:2rem; font-weight:800;'>{bm['f1']:.4f}</div></div>"
            f"<div><div style='font-size:0.75rem; color:#F97316; font-weight:700;'>PRECISION</div>"
            f"<div style='font-size:2rem; font-weight:800;'>{bm['precision']:.4f}</div></div>"
            f"<div><div style='font-size:0.75rem; color:#F97316; font-weight:700;'>RECALL</div>"
            f"<div style='font-size:2rem; font-weight:800;'>{bm['recall']:.4f}</div></div>"
            f"<div><div style='font-size:0.75rem; color:#F97316; font-weight:700;'>ROC-AUC</div>"
            f"<div style='font-size:2rem; font-weight:800;'>{roc_display}</div></div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
        if st.button("Proceed to Real-Time Prediction Screen", type="primary",
                     use_container_width=True, key="cls_go_predict_btn"):
            st.session_state["cls_subpage"] = "prediction"
            st.rerun()
    else:
        st.warning("No best model identified.")


def _render_prediction_screen(api):
    # [Function]: _render_prediction_screen
    # [Description]: Implements and executes the  Render Prediction Screen logic within this module pipeline.
    results_data     = st.session_state.get("cls_results")
    selected_problem = st.session_state.get("cls_problem")
    if not results_data or not selected_problem:
        st.warning("No trained classifiers found. Please train first.")
        st.session_state["cls_subpage"] = "evaluation"
        st.rerun()

    label_names   = results_data.get("label_names", ["No", "Yes"])
    model_results = results_data.get("results", {})
    best_key      = results_data.get("best_model")

    st.markdown(
        "<div style='margin-bottom:1.5rem;'>"
        "<span style='font-size:0.85rem; color:#F97316; font-weight:600;"
        " text-transform:uppercase; letter-spacing:1px;'>REAL-TIME INFERENCE</span>"
        "<h2 style='margin:0; font-size:2.2rem; font-weight:700;'>Real-Time Classifier</h2>"
        "<p style='color:#9CA3AF; margin:0.5rem 0 0;'>Enter match/player attributes"
        " to instantly classify using your trained models.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("Back to Evaluation and Comparison", key="cls_back_eval_btn", use_container_width=True):
        st.session_state["cls_subpage"] = "evaluation"
        st.rerun()

    st.markdown("---")
    st.markdown("### Model Configuration")

    trained_algos = list(model_results.keys())
    default_idx   = trained_algos.index(best_key) if best_key in trained_algos else 0

    def algo_label_fmt(k):
        # [Function]: algo_label_fmt
        # [Description]: Implements and executes the Algo Label Fmt logic within this module pipeline.
        acc      = model_results[k]["accuracy"]
        best_tag = " (Best)" if k == best_key else ""
        return f"{model_results[k]['label']} (Acc: {acc:.4f}){best_tag}"

    selected_algo = st.selectbox(
        "Choose Algorithm for Classification:",
        options=trained_algos,
        index=default_idx,
        format_func=algo_label_fmt,
    )

    st.markdown("---")
    st.markdown("### Enter Features")
    st.caption("Fill in the attributes below. Dropdowns are populated directly from dataset values.")

    inputs        = {}
    features_meta = results_data.get("features_metadata", [])
    col1, col2    = st.columns(2)
    for idx, f in enumerate(features_meta):
        col = col1 if idx % 2 == 0 else col2
        name_label = f["name"].replace("_", " ").title()
        with col:
            if f["type"] == "categorical":
                inputs[f["name"]] = st.selectbox(
                    f"Select {name_label}:", options=f["options"],
                    key=f"cls_input_{f['name']}",
                )
            else:
                step = 1 if int(f["min"]) == f["min"] and int(f["max"]) == f["max"] else 0.1
                inputs[f["name"]] = st.number_input(
                    f"{name_label} ({f['min']:.0f} to {f['max']:.0f}):",
                    min_value=float(f["min"]), max_value=float(f["max"]),
                    value=float(f["default"]), step=float(step),
                    key=f"cls_input_{f['name']}",
                )

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("Run Classification", type="primary", use_container_width=True, key="cls_calc_btn"):
        with st.spinner("Running classifier..."):
            pred_res = api.predict_classification(selected_problem, selected_algo, inputs)
            if pred_res.get("error"):
                st.error(f"Prediction failed: {pred_res.get('detail', 'Unknown error')}")
            else:
                pred_label  = pred_res.get("prediction_label", "Unknown")
                pred_idx    = pred_res.get("prediction", 0)
                probas      = pred_res.get("probabilities")
                lns         = pred_res.get("label_names", label_names)
                is_positive = pred_idx == 1
                color       = "#10B981" if is_positive else "#EF4444"

                st.markdown(
                    "<div style='background:rgba(249,115,22,0.08);"
                    f" border:2px solid {color}; border-radius:16px;"
                    " padding:2rem; text-align:center; margin-top:1.5rem;'>"
                    f"<span style='font-size:1rem; color:{color}; font-weight:700;"
                    " letter-spacing:1px; text-transform:uppercase;'>PREDICTION</span>"
                    f"<h1 style='color:#FFFFFF; margin:0.5rem 0; font-size:2.8rem;"
                    f" font-weight:800;'>{pred_label}</h1>"
                    f"<p style='color:#9CA3AF;'>Model: <b>{model_results[selected_algo]['label']}</b>"
                    f" | Target: <b>{results_data['target_column']}</b></p>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                if probas and len(probas) == len(lns):
                    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
                    st.markdown("#### Class Probabilities")
                    bar_colors = ["#EF4444", "#10B981"] if len(lns) == 2 else ["#F97316"] * len(lns)
                    prob_fig = go.Figure(go.Bar(
                        x=lns,
                        y=[p * 100 for p in probas],
                        marker_color=bar_colors,
                        text=[f"{p*100:.1f}%" for p in probas],
                        textposition="outside",
                    ))
                    prob_fig.update_layout(
                        yaxis_title="Probability (%)", template="plotly_dark", height=280,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#F3F4F6"), yaxis=dict(range=[0, 110]),
                    )
                    st.plotly_chart(prob_fig, use_container_width=True, key=f"cls_prob_chart_{selected_algo}")
