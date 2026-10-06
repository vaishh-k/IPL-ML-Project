# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: visualization.py
# Path: backend/services/visualization.py
# Description: Service layer providing plotting configurations and dataset overview metrics.
# ==============================================================================

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Professional Dark Theme Constants
BG_COLOR = "#111827"
PANEL_COLOR = "#172033"
BORDER_COLOR = "#263244"
TEXT_COLOR = "#E5E7EB"
ACCENT_BLUE = "#F97316"
ACCENT_GREEN = "#10B981"
ACCENT_YELLOW = "#F59E0B"
ACCENT_RED = "#EF4444"
ACCENT_PURPLE = "#8B5CF6"

def apply_dark_theme(fig):
    """Applies a professional dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", # Transparent to match page container
        plot_bgcolor="#111827",
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        xaxis=dict(
            gridcolor=BORDER_COLOR,
            zerolinecolor=BORDER_COLOR,
            title_font=dict(color=TEXT_COLOR),
            tickfont=dict(color=TEXT_COLOR)
        ),
        yaxis=dict(
            gridcolor=BORDER_COLOR,
            zerolinecolor=BORDER_COLOR,
            title_font=dict(color=TEXT_COLOR),
            tickfont=dict(color=TEXT_COLOR)
        ),
        legend=dict(
            bgcolor="#172033",
            bordercolor=BORDER_COLOR,
            font=dict(color=TEXT_COLOR)
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        title_font=dict(color=TEXT_COLOR, size=16)
    )
    return fig

def get_missing_values_chart(df: pd.DataFrame) -> dict:
    """Returns a bar chart showing missing value counts by column."""
    missing = df.isna().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing = missing[missing["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False)
    
    if missing.empty:
        # Create a dummy success chart
        fig = go.Figure()
        fig.add_annotation(text="No missing values found! ", showarrow=False, font=dict(size=18, color=ACCENT_GREEN))
        apply_dark_theme(fig)
        return fig.to_dict()
        
    fig = px.bar(
        missing, 
        x="Column", 
        y="Missing Count", 
        title="Missing Values Count by Feature",
        color_discrete_sequence=[ACCENT_RED]
    )
    apply_dark_theme(fig)
    return fig.to_dict()

def get_duplicates_chart(df: pd.DataFrame) -> dict:
    """Returns a pie/donut chart showing unique vs duplicate rows."""
    total = len(df)
    duplicates = int(df.duplicated().sum())
    unique = total - duplicates
    
    fig = go.Figure(data=[go.Pie(
        labels=["Unique Rows", "Duplicate Rows"],
        values=[unique, duplicates],
        hole=.4,
        marker=dict(colors=[ACCENT_GREEN, ACCENT_RED])
    )])
    fig.update_layout(title_text="Dataset Rows: Unique vs Duplicates")
    apply_dark_theme(fig)
    return fig.to_dict()

def get_univariate_chart(df: pd.DataFrame, col: str) -> dict:
    """
    Returns an appropriate univariate visualization:
    - Numerical: Histogram with KDE curve or overlayed box plot
    - Categorical: Horizontal or vertical count plot
    """
    if col not in df.columns:
        return {}
        
    if pd.api.types.is_numeric_dtype(df[col]):
        # Numerical - Plot histogram
        series = df[col].dropna()
        fig = px.histogram(
            df, 
            x=col, 
            marginal="box", 
            title=f"Univariate Distribution of {col}",
            color_discrete_sequence=[ACCENT_BLUE],
            opacity=0.8
        )
        apply_dark_theme(fig)
        return fig.to_dict()
    else:
        # Categorical - Plot counts
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "Count"]
        # Cap cardinality for plotting
        if len(counts) > 15:
            other_count = counts.iloc[15:]["Count"].sum()
            counts = counts.head(15)
            # Add an 'Other' row using concat
            other_df = pd.DataFrame([{col: "Other (high cardinality)", "Count": other_count}])
            counts = pd.concat([counts, other_df], ignore_index=True)
            
        fig = px.bar(
            counts,
            y=col,
            x="Count",
            orientation="h",
            title=f"Univariate Frequency of {col} (Top 15)",
            color_discrete_sequence=[ACCENT_PURPLE]
        )
        # Reverse yaxis to show highest on top
        fig.update_layout(yaxis=dict(autorange="reversed"))
        apply_dark_theme(fig)
        return fig.to_dict()

def get_distribution_chart(df: pd.DataFrame, col: str) -> dict:
    """Returns a distribution histogram + KDE approximation for skewness analysis."""
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return {}
    
    series = df[col].dropna()
    fig = px.histogram(
        df, 
        x=col, 
        title=f"Detailed Distribution & Density for {col}",
        color_discrete_sequence=[ACCENT_BLUE],
        histnorm="probability density",
        opacity=0.7
    )
    apply_dark_theme(fig)
    return fig.to_dict()

def get_outliers_chart(df: pd.DataFrame, col: str, lower_bound: float, upper_bound: float) -> dict:
    """Returns a Box Plot with custom threshold indicators for outlier visualization."""
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return {}
        
    fig = px.box(
        df, 
        y=col, 
        title=f"Outlier Boundary Boxplot: {col}",
        color_discrete_sequence=[ACCENT_YELLOW]
    )
    
    # Add horizontal lines for upper and lower bounds
    fig.add_hline(y=upper_bound, line_dash="dash", line_color=ACCENT_RED, annotation_text="Upper Limit")
    fig.add_hline(y=lower_bound, line_dash="dash", line_color=ACCENT_RED, annotation_text="Lower Limit")
    
    apply_dark_theme(fig)
    return fig.to_dict()

def get_bivariate_scatter_chart(df: pd.DataFrame, col_x: str, col_y: str, show_trendline: bool = True) -> dict:
    """
    Returns an interactive scatter plot for two numerical columns.
    Adds an OLS linear regression fit line when show_trendline is True.
    """
    if col_x not in df.columns or col_y not in df.columns:
        return {}
        
    valid_df = df[[col_x, col_y]].dropna()
    if valid_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No valid data points for scatter plot", showarrow=False)
        apply_dark_theme(fig)
        return fig.to_dict()
        
    fig = px.scatter(
        valid_df,
        x=col_x,
        y=col_y,
        title=f"Scatter Plot: {col_x} vs {col_y}",
        color_discrete_sequence=[ACCENT_BLUE],
        opacity=0.75
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="rgba(255,255,255,0.4)")))
    
    if show_trendline and len(valid_df) >= 2:
        x_vals = valid_df[col_x].values
        y_vals = valid_df[col_y].values
        if np.var(x_vals) > 0:
            poly = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(np.min(x_vals), np.max(x_vals), 100)
            y_line = np.polyval(poly, x_line)
            
            eq_label = f"Fit: y = {poly[0]:.2f}x + {poly[1]:.2f}" if poly[1] >= 0 else f"Fit: y = {poly[0]:.2f}x - {abs(poly[1]):.2f}"
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=eq_label,
                line=dict(color=ACCENT_YELLOW, width=3, dash="dash")
            ))
            
    apply_dark_theme(fig)
    return fig.to_dict()

def get_bivariate_box_chart(df: pd.DataFrame, cat_col: str, num_col: str, show_points: bool = True) -> dict:
    """Returns a Box Plot comparing distributions of a continuous variable across categories."""
    if cat_col not in df.columns or num_col not in df.columns:
        return {}
    valid_df = df[[cat_col, num_col]].dropna()
    if valid_df.empty:
        return {}
        
    unique_cats = valid_df[cat_col].nunique()
    if unique_cats > 50:
        return {
            "too_many": True,
            "message": f"Column **`{cat_col}`** has **{unique_cats} unique categories** — too many for a box plot. Select a categorical column with ≤ 50 unique categories."
        }
        
    if unique_cats > 20:
        top_cats = valid_df[cat_col].value_counts().head(20).index
        valid_df = valid_df[valid_df[cat_col].isin(top_cats)]
        
    fig = px.box(
        valid_df,
        x=cat_col,
        y=num_col,
        color=cat_col,
        title=f"Box Plot: Distribution of {num_col} across {cat_col}",
        points="all" if show_points else "outliers"
    )
    apply_dark_theme(fig)
    return fig.to_dict()

def get_bivariate_violin_chart(df: pd.DataFrame, cat_col: str, num_col: str, show_points: bool = True) -> dict:
    """Returns a Violin Plot comparing probability densities across categories."""
    if cat_col not in df.columns or num_col not in df.columns:
        return {}
    valid_df = df[[cat_col, num_col]].dropna()
    if valid_df.empty:
        return {}
        
    unique_cats = valid_df[cat_col].nunique()
    if unique_cats > 50:
        return {
            "too_many": True,
            "message": f"Column **`{cat_col}`** has **{unique_cats} unique categories** — too many for a violin plot. Select a categorical column with ≤ 50 unique categories."
        }
        
    if unique_cats > 20:
        top_cats = valid_df[cat_col].value_counts().head(20).index
        valid_df = valid_df[valid_df[cat_col].isin(top_cats)]
        
    fig = px.violin(
        valid_df,
        x=cat_col,
        y=num_col,
        color=cat_col,
        title=f"Violin Plot: Density & Quartiles of {num_col} across {cat_col}",
        box=True,
        points="all" if show_points else "outliers"
    )
    apply_dark_theme(fig)
    return fig.to_dict()

def get_bivariate_bar_chart(df: pd.DataFrame, cat_col: str, num_col: str, agg_type: str = "mean") -> dict:
    """
    Returns an aggregated bar chart comparing the mean or median of a numerical variable for each category.
    """
    if cat_col not in df.columns or num_col not in df.columns:
        return {}
    valid_df = df[[cat_col, num_col]].dropna()
    if valid_df.empty:
        return {}
        
    unique_cats = valid_df[cat_col].nunique()
    if unique_cats > 50:
        return {
            "too_many": True,
            "message": f"Column **`{cat_col}`** has **{unique_cats} unique categories**."
        }
        
    if unique_cats > 20:
        top_cats = valid_df[cat_col].value_counts().head(20).index
        valid_df = valid_df[valid_df[cat_col].isin(top_cats)]
        
    if agg_type.lower() == "median":
        agg_df = valid_df.groupby(cat_col)[num_col].median().reset_index(name="AggValue")
        title_text = f"Category Comparison: Median {num_col} by {cat_col}"
        y_label = f"Median of {num_col}"
    else:
        agg_df = valid_df.groupby(cat_col)[num_col].mean().reset_index(name="AggValue")
        title_text = f"Category Comparison: Mean (Average) {num_col} by {cat_col}"
        y_label = f"Mean of {num_col}"
        
    agg_df = agg_df.sort_values(by="AggValue", ascending=False)
    
    fig = px.bar(
        agg_df,
        x=cat_col,
        y="AggValue",
        color=cat_col,
        text_auto=".2f",
        title=title_text,
        labels={"AggValue": y_label, cat_col: cat_col}
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    apply_dark_theme(fig)
    return fig.to_dict()

def get_bivariate_contingency_heatmap(df: pd.DataFrame, col_x: str, col_y: str, normalize: str = "none") -> dict:
    """
    Returns a Contingency Table Heatmap displaying frequency or normalized percentage distribution.
    """
    if col_x not in df.columns or col_y not in df.columns:
        return {}
    valid_df = df[[col_x, col_y]].dropna().astype(str)
    if valid_df.empty:
        return {}
        
    if valid_df[col_x].nunique() > 40 or valid_df[col_y].nunique() > 40:
        return {
            "too_many": True,
            "message": "Too many unique categories for contingency heatmap."
        }
        
    if valid_df[col_x].nunique() > 20:
        top_x = valid_df[col_x].value_counts().head(20).index
        valid_df = valid_df[valid_df[col_x].isin(top_x)]
    if valid_df[col_y].nunique() > 20:
        top_y = valid_df[col_y].value_counts().head(20).index
        valid_df = valid_df[valid_df[col_y].isin(top_y)]
        
    if normalize == "row":
        ct = pd.crosstab(valid_df[col_x], valid_df[col_y], normalize="index") * 100
        title = f"Contingency Heatmap (Row %): {col_x} vs {col_y}"
        color_scale = "Viridis"
    elif normalize == "column":
        ct = pd.crosstab(valid_df[col_x], valid_df[col_y], normalize="columns") * 100
        title = f"Contingency Heatmap (Column %): {col_x} vs {col_y}"
        color_scale = "Viridis"
    elif normalize == "all":
        ct = pd.crosstab(valid_df[col_x], valid_df[col_y], normalize="all") * 100
        title = f"Contingency Heatmap (Total %): {col_x} vs {col_y}"
        color_scale = "Viridis"
    else:
        ct = pd.crosstab(valid_df[col_x], valid_df[col_y])
        title = f"Contingency Heatmap (Frequency Counts): {col_x} vs {col_y}"
        color_scale = "Blues"
        
    fig = px.imshow(
        ct,
        text_auto=True,
        color_continuous_scale=color_scale,
        title=title,
        aspect="auto",
        labels=dict(x=col_y, y=col_x, color="Value")
    )
    apply_dark_theme(fig)
    return fig.to_dict()

def get_bivariate_stacked_bar_chart(df: pd.DataFrame, col_x: str, col_y: str, normalize: bool = False) -> dict:
    """
    Returns a Stacked Bar Chart visually comparing proportions or counts of categories.
    When normalize is True, creates a 100% normalized proportion stacked bar chart.
    """
    if col_x not in df.columns or col_y not in df.columns:
        return {}
    valid_df = df[[col_x, col_y]].dropna().astype(str)
    if valid_df.empty:
        return {}
        
    if valid_df[col_x].nunique() > 40 or valid_df[col_y].nunique() > 40:
        return {
            "too_many": True,
            "message": "Too many unique categories for stacked bar chart."
        }
        
    if valid_df[col_x].nunique() > 20:
        top_x = valid_df[col_x].value_counts().head(20).index
        valid_df = valid_df[valid_df[col_x].isin(top_x)]
    if valid_df[col_y].nunique() > 20:
        top_y = valid_df[col_y].value_counts().head(20).index
        valid_df = valid_df[valid_df[col_y].isin(top_y)]
        
    if normalize:
        ct = pd.crosstab(valid_df[col_x], valid_df[col_y], normalize="index") * 100
        melted = ct.reset_index().melt(id_vars=col_x, value_name="Percentage", var_name=col_y)
        fig = px.bar(
            melted,
            x=col_x,
            y="Percentage",
            color=col_y,
            barmode="relative",
            title=f"100% Normalized Stacked Bar Chart: {col_y} Proportions within {col_x}",
            text_auto=".1f",
            labels={"Percentage": "Proportion (%)"}
        )
        fig.update_layout(yaxis=dict(range=[0, 100]))
    else:
        grouped = valid_df.groupby([col_x, col_y]).size().reset_index(name="Count")
        fig = px.bar(
            grouped,
            x=col_x,
            y="Count",
            color=col_y,
            barmode="stack",
            title=f"Stacked Bar Chart (Frequency Counts): {col_y} within {col_x}",
            text_auto=True
        )
        
    apply_dark_theme(fig)
    return fig.to_dict()

def get_bivariate_grouped_bar_chart(df: pd.DataFrame, col_x: str, col_y: str) -> dict:
    """Returns a side-by-side Grouped Bar Chart comparing category frequencies."""
    if col_x not in df.columns or col_y not in df.columns:
        return {}
    valid_df = df[[col_x, col_y]].dropna().astype(str)
    if valid_df.empty:
        return {}
        
    if valid_df[col_x].nunique() > 40 or valid_df[col_y].nunique() > 40:
        return {
            "too_many": True,
            "message": "Too many unique categories for grouped bar chart."
        }
        
    if valid_df[col_x].nunique() > 20:
        top_x = valid_df[col_x].value_counts().head(20).index
        valid_df = valid_df[valid_df[col_x].isin(top_x)]
    if valid_df[col_y].nunique() > 20:
        top_y = valid_df[col_y].value_counts().head(20).index
        valid_df = valid_df[valid_df[col_y].isin(top_y)]
        
    grouped = valid_df.groupby([col_x, col_y]).size().reset_index(name="Count")
    fig = px.bar(
        grouped,
        x=col_x,
        y="Count",
        color=col_y,
        barmode="group",
        title=f"Grouped Bar Chart: {col_x} vs {col_y}",
        text_auto=True
    )
    apply_dark_theme(fig)
    return fig.to_dict()

def get_bivariate_chart(df: pd.DataFrame, col_x: str, col_y: str) -> dict:
    """
    Legacy helper: Returns bivariate visualization depending on column types:
    - Num vs Num: Scatter Plot
    - Cat vs Num: Box / Violin Plot
    - Cat vs Cat: Grouped Bar Chart
    """
    if col_x not in df.columns or col_y not in df.columns:
        return {}
        
    is_x_num = pd.api.types.is_numeric_dtype(df[col_x])
    is_y_num = pd.api.types.is_numeric_dtype(df[col_y])
    
    if is_x_num and is_y_num:
        return get_bivariate_scatter_chart(df, col_x, col_y, show_trendline=True)
    elif (is_x_num and not is_y_num) or (not is_x_num and is_y_num):
        num_col = col_x if is_x_num else col_y
        cat_col = col_y if is_x_num else col_x
        return get_bivariate_violin_chart(df, cat_col, num_col)
    else:
        return get_bivariate_stacked_bar_chart(df, col_x, col_y, normalize=False)

def get_correlation_heatmap(df: pd.DataFrame, num_cols: list[str]) -> dict:
    """Returns a correlation heatmap plot for numerical columns."""
    if len(num_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="At least 2 numerical columns are required for a heatmap.", showarrow=False)
        apply_dark_theme(fig)
        return fig.to_dict()
        
    corr = df[num_cols].corr()
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Heatmap"
    )
    apply_dark_theme(fig)
    return fig.to_dict()

def get_pca_variance_chart(explained_variance: list[float]) -> dict:
    """Returns a line chart showing explained variance ratio per component."""
    components = [f"PC{i+1}" for i in range(len(explained_variance))]
    cumulative = np.cumsum(explained_variance).tolist()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=components, 
        y=explained_variance, 
        name="Individual Explained Variance",
        marker_color=ACCENT_BLUE
    ))
    fig.add_trace(go.Scatter(
        x=components, 
        y=cumulative, 
        name="Cumulative Explained Variance",
        mode="lines+markers",
        line=dict(color=ACCENT_GREEN, width=3),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title="PCA Variance Explained Ratio",
        xaxis_title="Principal Components",
        yaxis_title="Variance Ratio",
        legend=dict(x=0.02, y=0.98)
    )
    apply_dark_theme(fig)
    return fig.to_dict()

def get_pca_2d_chart(projection_df: pd.DataFrame, target_col: str = None) -> dict:
    """Returns a 2D PCA scatter plot projection (PC1 vs PC2)."""
    if "PC1" not in projection_df.columns or "PC2" not in projection_df.columns:
        return {}
        
    if target_col and target_col in projection_df.columns:
        # Check target type
        is_num = pd.api.types.is_numeric_dtype(projection_df[target_col])
        color_seq = None
        if not is_num:
            # Limit colors for categories
            projection_df[target_col] = projection_df[target_col].astype(str)
            
        fig = px.scatter(
            projection_df, 
            x="PC1", 
            y="PC2", 
            color=target_col,
            title="PCA 2D Projection (PC1 vs PC2)",
            opacity=0.8,
            color_continuous_scale="Viridis" if is_num else None
        )
    else:
        fig = px.scatter(
            projection_df, 
            x="PC1", 
            y="PC2", 
            title="PCA 2D Projection (PC1 vs PC2)",
            color_discrete_sequence=[ACCENT_BLUE],
            opacity=0.8
        )
        
    apply_dark_theme(fig)
    return fig.to_dict()

def get_pca_3d_chart(projection_df: pd.DataFrame, target_col: str = None) -> dict:
    """Returns a 3D PCA scatter plot projection (PC1 vs PC2 vs PC3)."""
    if "PC1" not in projection_df.columns or "PC2" not in projection_df.columns or "PC3" not in projection_df.columns:
        return {}
        
    if target_col and target_col in projection_df.columns:
        is_num = pd.api.types.is_numeric_dtype(projection_df[target_col])
        if not is_num:
            projection_df[target_col] = projection_df[target_col].astype(str)
            
        fig = px.scatter_3d(
            projection_df, 
            x="PC1", 
            y="PC2", 
            z="PC3", 
            color=target_col,
            title="PCA 3D Projection",
            opacity=0.8,
            color_continuous_scale="Viridis" if is_num else None
        )
    else:
        fig = px.scatter_3d(
            projection_df, 
            x="PC1", 
            y="PC2", 
            z="PC3", 
            title="PCA 3D Projection",
            color_discrete_sequence=[ACCENT_BLUE],
            opacity=0.8
        )
        
    # Apply dark theme properties for 3D graphs
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        scene=dict(
            xaxis=dict(backgroundcolor="#111827", gridcolor=BORDER_COLOR, title="PC1", color=TEXT_COLOR),
            yaxis=dict(backgroundcolor="#111827", gridcolor=BORDER_COLOR, title="PC2", color=TEXT_COLOR),
            zaxis=dict(backgroundcolor="#111827", gridcolor=BORDER_COLOR, title="PC3", color=TEXT_COLOR),
        ),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig.to_dict()
