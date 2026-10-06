# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: bivariate_page.py
# Path: frontend/components/bivariate_page.py
# Description: Modular bivariate analysis dashboard for Numerical vs Numerical,
#              Categorical vs Numerical, and Categorical vs Categorical feature pairs.
# ==============================================================================

import streamlit as st
import pandas as pd
from ipl_utils.helpers import render_explanation
from components.charts import render_plotly_chart
from components.dataframe import render_dataframe

def render_bivariate_page(api, overview_res: dict):
    """
    Renders the complete Bivariate Analysis module in the EDA pipeline.
    """
    render_explanation(
        what="Bivariate Analysis explores pairwise relationships, dependency patterns, and distributions between two variables across Numerical vs. Numerical, Categorical vs. Numerical, and Categorical vs. Categorical types.",
        why="Exploring dependencies shows interactions with variables, indicating linear correlation or mean variance.",
        find="Interactive bivariate plotter."
    )
    
    if not overview_res or overview_res.get("error"):
        st.error("Failed to load dataset overview for Bivariate Analysis.")
        return
        
    col_list = overview_res.get("columns", [])
    classified = overview_res.get("classified_columns", {})
    
    num_cols = classified.get("numerical", [])
    cat_cols = classified.get("categorical", []) + classified.get("boolean", [])
    
    # Fallbacks if classified is empty
    if not num_cols and not cat_cols:
        num_cols = col_list
        cat_cols = col_list
    elif not num_cols:
        num_cols = col_list
    elif not cat_cols:
        cat_cols = col_list
        
    st.markdown("""
    <style>
    .biv-mode-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .badge-num { background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid #3B82F6; }
    .badge-cat-num { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
    .badge-cat-cat { background: rgba(168, 85, 247, 0.2); color: #C084FC; border: 1px solid #A855F7; }
    
    .stat-metric-card {
        background: #172033;
        border: 1px solid #263244;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.75rem;
    }
    .stat-metric-title {
        font-size: 0.8rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .stat-metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0.2rem 0;
    }
    .stat-metric-sub {
        font-size: 0.8rem;
        color: #F97316;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Select Analysis Mode & Variables")
    
    mode_options = [
        "🔵 Numerical vs. Numerical",
        "🟢 Categorical vs. Numerical",
        "🟣 Categorical vs. Categorical",
        "🔍 Free Auto-Detect"
    ]
    
    selected_mode = st.radio(
        "Relationship Mode:",
        options=mode_options,
        index=0,
        horizontal=True,
        key="biv_mode_radio"
    )
    
    st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)
    
    # Render column pickers based on mode
    col_x = None
    col_y = None
    
    if selected_mode == "🔵 Numerical vs. Numerical":
        st.caption("📈 **Numerical vs. Numerical**: Visualizes continuous variable relationships using Scatter Plots with OLS regression lines and Pearson's / Spearman's correlation coefficients.")
        c1, c2 = st.columns(2)
        with c1:
            col_x = st.selectbox("Select Numerical Feature X:", num_cols, index=0, key="biv_nx")
        with c2:
            default_y_idx = 1 if len(num_cols) > 1 else 0
            col_y = st.selectbox("Select Numerical Feature Y:", num_cols, index=default_y_idx, key="biv_ny")
            
    elif selected_mode == "🟢 Categorical vs. Numerical":
        st.caption("📦 **Categorical vs. Numerical**: Compares continuous variable distributions across categories using Box / Violin Plots, Mean / Median Bar Charts, and ANOVA tests.")
        c1, c2 = st.columns(2)
        with c1:
            col_x = st.selectbox("Select Categorical Feature (Group):", cat_cols, index=0, key="biv_cn_cat")
        with c2:
            col_y = st.selectbox("Select Numerical Feature (Measurement):", num_cols, index=0, key="biv_cn_num")
            
    elif selected_mode == "🟣 Categorical vs. Categorical":
        st.caption("📊 **Categorical vs. Categorical**: Analyzes joint frequency distributions using Contingency Tables (Cross-Tabulations), Heatmaps, Stacked Bar Charts, and Chi-Square tests.")
        c1, c2 = st.columns(2)
        with c1:
            col_x = st.selectbox("Select Categorical Feature X:", cat_cols, index=0, key="biv_cc_x")
        with c2:
            default_cc_y = 1 if len(cat_cols) > 1 else 0
            col_y = st.selectbox("Select Categorical Feature Y:", cat_cols, index=default_cc_y, key="biv_cc_y")
            
    else:  # Free Auto-Detect
        st.caption("🔍 **Free Auto-Detect**: Select any two features from the dataset and the engine will automatically determine the appropriate bivariate analysis method.")
        c1, c2 = st.columns(2)
        with c1:
            col_x = st.selectbox("Select Feature X:", col_list, index=0, key="biv_free_x")
        with c2:
            default_free_y = 1 if len(col_list) > 1 else 0
            col_y = st.selectbox("Select Feature Y:", col_list, index=default_free_y, key="biv_free_y")
            
    if not col_x or not col_y:
        st.warning("Please select two features to analyze.")
        return
        
    st.markdown("---")
    
    # Fetch Bivariate API Analysis
    with st.spinner(f"Analyzing bivariate relationship between '{col_x}' and '{col_y}'..."):
        biv_res = api.get_bivariate(col_x, col_y)
        
    if biv_res.get("error"):
        error_msg = biv_res.get("detail", str(biv_res.get("error")))
        st.error(f"Error performing bivariate analysis: {error_msg}")
        return
        
    if biv_res.get("too_many"):
        st.markdown(f"""
        <div style="background:rgba(245,158,11,0.1); border:1px solid #F59E0B; border-radius:10px;
                    padding:1.2rem 1.5rem; margin-top:1rem; margin-bottom:1.5rem;">
            <div style="font-size:1.1rem; font-weight:700; color:#F59E0B; margin-bottom:0.4rem;">
                ⚠️ Cardinality Notice
            </div>
            <div style="color:#D1D5DB; font-size:0.95rem; line-height:1.5;">{biv_res.get('message', 'Too many unique categories to visualize clearly.')}</div>
        </div>""", unsafe_allow_html=True)
        return
        
    rel_type = biv_res.get("relationship_type", "")
    stats_data = biv_res.get("stats", {})
    charts = biv_res.get("charts", {})
    
    # -------------------------------------------------------------------------
    # 1. NUMERICAL VS NUMERICAL
    # -------------------------------------------------------------------------
    if rel_type == "num_vs_num":
        st.markdown(f"#### 🔵 Numerical vs. Numerical: `{col_x}` vs `{col_y}`")
        
        # Interactive Controls
        ctrl_col1, ctrl_col2 = st.columns([1, 2])
        with ctrl_col1:
            show_trendline = st.checkbox("📈 Show OLS Trendline", value=True, key=f"trendline_{col_x}_{col_y}")
            
        if not show_trendline and "scatter" in charts:
            scatter_plot = api.get_bivariate(col_x, col_y, show_trendline=False).get("charts", {}).get("scatter", biv_res.get("chart"))
        else:
            scatter_plot = charts.get("scatter", biv_res.get("chart"))
            
        # Top KPI Metric Cards
        pearson_r = stats_data.get("pearson_r", 0.0)
        r_sq_pct = stats_data.get("r_squared_pct", 0.0)
        spearman_rho = stats_data.get("spearman_rho", 0.0)
        p_val = stats_data.get("pearson_p_value", 1.0)
        strength = stats_data.get("strength", "None")
        direction = stats_data.get("direction", "None")
        equation = stats_data.get("equation", "N/A")
        
        p_val_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.4f}"
        sig_color = "#10B981" if p_val < 0.05 else "#EF4444"
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="stat-metric-card">
                <div class="stat-metric-title">Pearson's r</div>
                <div class="stat-metric-val">{pearson_r:+.3f}</div>
                <div class="stat-metric-sub">{strength} {direction}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="stat-metric-card">
                <div class="stat-metric-title">Variance Explained (R²)</div>
                <div class="stat-metric-val">{r_sq_pct:.1f}%</div>
                <div class="stat-metric-sub">Linear Goodness of Fit</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="stat-metric-card">
                <div class="stat-metric-title">Spearman's Rank (ρ)</div>
                <div class="stat-metric-val">{spearman_rho:+.3f}</div>
                <div class="stat-metric-sub">Monotonic Relationship</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="stat-metric-card">
                <div class="stat-metric-title">Statistical Significance</div>
                <div class="stat-metric-val" style="color:{sig_color};">p {p_val_str}</div>
                <div class="stat-metric-sub">{'Significant (p < 0.05)' if p_val < 0.05 else 'Not Significant'}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Plot + Details Layout
        p_col, d_col = st.columns([3, 2], gap="medium")
        with p_col:
            st.markdown("##### 📌 Scatter Plot")
            render_plotly_chart(scatter_plot, key=f"scatter_{col_x}_{col_y}")
            
        with d_col:
            st.markdown("##### 💡 Relationship Interpretation")
            st.info(stats_data.get("interpretation", biv_res.get("interpretation", "")))
            
            st.markdown("##### 📐 Linear Fit Details")
            reg_table = pd.DataFrame([
                {"Parameter": "Linear Equation", "Value": equation},
                {"Parameter": "Slope (m)", "Value": f"{stats_data.get('slope', 0.0):.4f}"},
                {"Parameter": "Intercept (c)", "Value": f"{stats_data.get('intercept', 0.0):.4f}"},
                {"Parameter": "Covariance", "Value": f"{stats_data.get('covariance', 0.0):.4f}"},
                {"Parameter": "Sample Size (N)", "Value": f"{stats_data.get('sample_size', 0):,}"},
            ])
            render_dataframe(reg_table, hide_index=True)
            
            st.markdown("""
            <div style="background:rgba(255,255,255,0.03); border:1px solid #263244; border-radius:8px; padding:0.9rem; margin-top:0.8rem; font-size:0.85rem; color:#9CA3AF;">
                <b style="color:#F97316;">ML Takeaway:</b> Highly correlated numerical pairs (|r| > 0.80) introduce multicollinearity in Linear Regression. Tree models (XGBoost, Random Forest) handle collinearity better, but feature selection can improve model interpretability.
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. CATEGORICAL VS NUMERICAL
    # -------------------------------------------------------------------------
    elif rel_type == "cat_vs_num":
        cat_c = biv_res.get("cat_col", col_x)
        num_c = biv_res.get("num_col", col_y)
        st.markdown(f"#### 🟢 Categorical vs. Numerical: `{num_c}` across `{cat_c}`")
        
        # Sub-tabs for Box/Violin, Bar Charts, and Group Statistics
        tab_dist, tab_bar, tab_stats = st.tabs([
            "📦 Box & Violin Plots (Distribution)",
            "📊 Bar Charts (Mean / Median)",
            "📋 Group Statistics & Hypothesis Test"
        ])
        
        with tab_dist:
            dist_c1, dist_c2 = st.columns([1, 1])
            with dist_c1:
                plot_type = st.radio("Distribution Plot Type:", ["Box Plot", "Violin Plot"], horizontal=True, key=f"dist_type_{cat_c}_{num_c}")
            with dist_c2:
                show_pts = st.checkbox("Show Individual Data Points (Jitter)", value=True, key=f"pts_{cat_c}_{num_c}")
                
            col_chart, col_desc = st.columns([3, 2], gap="medium")
            with col_chart:
                if plot_type == "Box Plot":
                    box_fig = charts.get("box", biv_res.get("chart"))
                    render_plotly_chart(box_fig, key=f"box_{cat_c}_{num_c}")
                else:
                    violin_fig = charts.get("violin", biv_res.get("chart"))
                    render_plotly_chart(violin_fig, key=f"violin_{cat_c}_{num_c}")
                    
            with col_desc:
                st.markdown("##### 💡 Distribution Insights")
                st.info(stats_data.get("interpretation", biv_res.get("interpretation", "")))
                
                st.markdown(f"""
                <div style="background:#172033; border:1px solid #263244; border-radius:10px; padding:1.2rem; margin-top:0.5rem;">
                    <div style="color:#10B981; font-weight:700; font-size:0.95rem; margin-bottom:0.4rem;">📦 Why Box & Violin Plots?</div>
                    <div style="color:#D1D5DB; font-size:0.85rem; line-height:1.5;">
                        • <b>Box Plots</b> reveal the Median, Interquartile Range (IQR), and extreme Outliers for each category.<br>
                        • <b>Violin Plots</b> combine box plot quartiles with Kernel Density Estimation (KDE) to show multimodal distributions and cluster densities.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with tab_bar:
            bar_c1, bar_c2 = st.columns([1, 2])
            with bar_c1:
                agg_choice = st.radio("Aggregation Metric:", ["Mean (Average)", "Median"], horizontal=True, key=f"bar_agg_{cat_c}_{num_c}")
                
            b_chart_col, b_desc_col = st.columns([3, 2], gap="medium")
            with b_chart_col:
                if agg_choice == "Median":
                    bar_fig = charts.get("bar_median", biv_res.get("chart"))
                else:
                    bar_fig = charts.get("bar_mean", biv_res.get("chart"))
                render_plotly_chart(bar_fig, key=f"bar_{cat_c}_{num_c}_{agg_choice}")
                
            with b_desc_col:
                st.markdown("##### 💡 Aggregation Comparison")
                st.write(f"This bar chart compares the **{agg_choice.lower()}** value of `{num_c}` grouped by `{cat_c}`.")
                st.info(stats_data.get("interpretation", ""))
                
        with tab_stats:
            st.markdown(f"##### 📊 Summary Statistics of `{num_c}` by `{cat_c}`")
            g_stats = stats_data.get("group_stats", [])
            if g_stats:
                stat_df = pd.DataFrame(g_stats)
                stat_df = stat_df.rename(columns={
                    "category": f"Category ({cat_c})",
                    "count": "Sample Count (N)",
                    "mean": "Mean",
                    "median": "Median",
                    "std": "Std Dev",
                    "min": "Min",
                    "max": "Max",
                    "q1": "Q1 (25%)",
                    "q3": "Q3 (75%)"
                })
                render_dataframe(stat_df, hide_index=True)
                
            # ANOVA & Kruskal test cards
            anova_f = stats_data.get("anova_f", 0.0)
            anova_p = stats_data.get("anova_p_value", 1.0)
            kruskal_h = stats_data.get("kruskal_h", 0.0)
            kruskal_p = stats_data.get("kruskal_p_value", 1.0)
            
            anova_p_str = "< 0.001" if anova_p < 0.001 else f"= {anova_p:.4f}"
            kruskal_p_str = "< 0.001" if kruskal_p < 0.001 else f"= {kruskal_p:.4f}"
            
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(f"""
                <div class="stat-metric-card" style="text-align:left;">
                    <div style="font-weight:700; color:#10B981; font-size:1rem; margin-bottom:0.25rem;">🔬 One-Way ANOVA Test</div>
                    <div style="font-size:0.9rem; color:#D1D5DB;"><b>F-Statistic:</b> <code>{anova_f:.3f}</code> | <b>p-value:</b> <code>{anova_p_str}</code></div>
                    <div style="font-size:0.8rem; color:#9CA3AF; margin-top:0.4rem;">
                        {' Significant difference in group means across categories.' if anova_p < 0.05 else '❌ No significant difference in group means across categories.'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with tc2:
                st.markdown(f"""
                <div class="stat-metric-card" style="text-align:left;">
                    <div style="font-weight:700; color:#3B82F6; font-size:1rem; margin-bottom:0.25rem;">🔬 Kruskal-Wallis H-Test (Non-Parametric)</div>
                    <div style="font-size:0.9rem; color:#D1D5DB;"><b>H-Statistic:</b> <code>{kruskal_h:.3f}</code> | <b>p-value:</b> <code>{kruskal_p_str}</code></div>
                    <div style="font-size:0.8rem; color:#9CA3AF; margin-top:0.4rem;">
                        {' Significant difference in median ranks across categories.' if kruskal_p < 0.05 else '❌ No significant difference in median ranks.'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. CATEGORICAL VS CATEGORICAL
    # -------------------------------------------------------------------------
    elif rel_type == "cat_vs_cat":
        st.markdown(f"#### 🟣 Categorical vs. Categorical: `{col_x}` vs `{col_y}`")
        
        tab_ct, tab_stack = st.tabs([
            "📋 Contingency Table (Cross-Tabulation)",
            "📊 Stacked Bar Charts"
        ])
        
        with tab_ct:
            ct_c1, ct_c2 = st.columns([2, 1])
            with ct_c1:
                ct_mode = st.radio(
                    "Display Format:",
                    ["Raw Frequency Counts", "Row Percentages (Row %)", "Column Percentages (Col %)", "Total Percentages (Total %)"],
                    horizontal=True,
                    key=f"ct_mode_{col_x}_{col_y}"
                )
            with ct_c2:
                show_heatmap = st.checkbox("🔥 Display as Interactive Heatmap", value=False, key=f"hm_{col_x}_{col_y}")
                
            # Pick matrix dict
            if "Row" in ct_mode:
                matrix_dict = stats_data.get("contingency_row_pct", {})
                hm_fig = charts.get("heatmap_row_pct", biv_res.get("chart"))
                unit_label = " (%)"
            elif "Col" in ct_mode:
                matrix_dict = stats_data.get("contingency_col_pct", {})
                hm_fig = charts.get("heatmap_col_pct", biv_res.get("chart"))
                unit_label = " (%)"
            elif "Total" in ct_mode:
                matrix_dict = stats_data.get("contingency_tot_pct", {})
                hm_fig = charts.get("heatmap_tot_pct", biv_res.get("chart"))
                unit_label = " (%)"
            else:
                matrix_dict = stats_data.get("contingency_counts", {})
                hm_fig = charts.get("heatmap_counts", biv_res.get("chart"))
                unit_label = " (Count)"
                
            if show_heatmap:
                render_plotly_chart(hm_fig, key=f"heatmap_{col_x}_{col_y}_{ct_mode}")
            else:
                if matrix_dict and "index" in matrix_dict and "columns" in matrix_dict and "data" in matrix_dict:
                    ct_df = pd.DataFrame(
                        matrix_dict["data"],
                        index=matrix_dict["index"],
                        columns=[f"{c}{unit_label}" for c in matrix_dict["columns"]]
                    )
                    st.markdown(f"**Cross-Tabulation Matrix: `{col_x}` (Rows) vs `{col_y}` (Columns)**")
                    st.dataframe(ct_df, use_container_width=True)
                else:
                    render_plotly_chart(hm_fig, key=f"heatmap_fallback_{col_x}_{col_y}")
                    
            # Chi-Square Test & Association KPI
            chi2_stat = stats_data.get("chi2_stat", 0.0)
            chi2_p = stats_data.get("chi2_p_value", 1.0)
            dof = stats_data.get("dof", 0)
            cramers_v = stats_data.get("cramers_v", 0.0)
            v_strength = stats_data.get("cramers_v_strength", "None")
            top_combo = stats_data.get("top_combination", {})
            
            chi2_p_str = "< 0.001" if chi2_p < 0.001 else f"= {chi2_p:.4f}"
            
            st.markdown("##### 🔬 Statistical Independence & Association")
            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(f"""
                <div class="stat-metric-card">
                    <div class="stat-metric-title">Chi-Square Test (χ²)</div>
                    <div class="stat-metric-val">{chi2_stat:.2f}</div>
                    <div class="stat-metric-sub">dof = {dof}, p {chi2_p_str}</div>
                </div>
                """, unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div class="stat-metric-card">
                    <div class="stat-metric-title">Cramer's V Association</div>
                    <div class="stat-metric-val">{cramers_v:.3f}</div>
                    <div class="stat-metric-sub">{v_strength}</div>
                </div>
                """, unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                <div class="stat-metric-card">
                    <div class="stat-metric-title">Most Frequent Co-occurrence</div>
                    <div class="stat-metric-val" style="font-size:1.15rem;">{top_combo.get('x', 'N/A')} & {top_combo.get('y', 'N/A')}</div>
                    <div class="stat-metric-sub">{top_combo.get('count', 0):,} matches</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.info(stats_data.get("interpretation", biv_res.get("interpretation", "")))
            
        with tab_stack:
            st.markdown(f"##### 📊 Stacked Category Visualizations: `{col_y}` within `{col_x}`")
            stack_style = st.radio(
                "Bar Chart Style:",
                ["100% Normalized Proportions (Percentage)", "Stacked Absolute Counts", "Grouped (Side-by-Side)"],
                horizontal=True,
                key=f"stack_style_{col_x}_{col_y}"
            )
            
            if stack_style == "100% Normalized Proportions (Percentage)":
                stack_fig = charts.get("stacked_pct", biv_res.get("chart"))
            elif stack_style == "Stacked Absolute Counts":
                stack_fig = charts.get("stacked_counts", biv_res.get("chart"))
            else:
                stack_fig = charts.get("grouped_bar", biv_res.get("chart"))
                
            render_plotly_chart(stack_fig, key=f"stack_chart_{col_x}_{col_y}_{stack_style}")
            
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid #263244; border-radius:8px; padding:0.9rem; margin-top:0.8rem; font-size:0.85rem; color:#9CA3AF;">
                <b style="color:#C084FC;">Interpretation Tip:</b> <b>100% Normalized Stacked Bar Charts</b> allow comparing the internal composition and relative proportions of <code>{col_y}</code> across categories of <code>{col_x}</code>, eliminating sample-size distortion.
            </div>
            """, unsafe_allow_html=True)
