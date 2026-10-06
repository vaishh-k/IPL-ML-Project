# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: analysis.py
# Path: backend/services/analysis.py
# Description: Data analysis service containing statistical calculation functions, correlation matrices, and distribution metrics.
# ==============================================================================

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Classifies dataframe columns into four main types:
    numerical, categorical, date, and boolean.
    """
    classifications = {
        "numerical": [],
        "categorical": [],
        "date": [],
        "boolean": []
    }
    
    for col in df.columns:
        # Check date/datetime first
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            classifications["date"].append(col)
            continue
            
        col_lower = col.lower()
        if "date" in col_lower or "time" in col_lower or "year" in col_lower:
            # Check if it can be parsed as datetime easily
            try:
                # If it's numeric, check range. Years like 2019 are numeric, let's treat them as numeric
                if pd.api.types.is_numeric_dtype(df[col]) and not df[col].astype(str).str.contains('-|/').any():
                    # If it's just years like 2008-2024, it's numerical
                    classifications["numerical"].append(col)
                else:
                    pd.to_datetime(df[col].dropna().head(10))
                    classifications["date"].append(col)
                    continue
            except Exception:
                pass
                
        # Check boolean
        if pd.api.types.is_bool_dtype(df[col]):
            classifications["boolean"].append(col)
            continue
            
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False, 'True', 'False', 'yes', 'no', 'Y', 'N'}):
            classifications["boolean"].append(col)
            continue
            
        # Numerical
        if pd.api.types.is_numeric_dtype(df[col]):
            classifications["numerical"].append(col)
        else:
            # Categorical
            classifications["categorical"].append(col)
            
    return classifications

def get_data_quality(df: pd.DataFrame) -> dict:
    """
    Generates a data quality dashboard summary with a row-wise feature audit
    and a global data quality score.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Calculate duplicates
    dup_rows = int(df.duplicated().sum())
    
    # Classify cols
    types = classify_columns(df)
    
    # Analyze columns
    column_audit = []
    total_problems = 0
    total_warnings = 0
    constant_columns_count = 0
    
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        missing_count = int(df[col].isna().sum())
        missing_pct = float(missing_count / total_rows * 100) if total_rows > 0 else 0.0
        unique_count = int(df[col].nunique())
        
        # Determine status
        status = "Good"
        reason = ""
        
        # Check constant column
        if unique_count <= 1:
            status = "Problem"
            reason = "Constant column (no variance)"
            total_problems += 1
            constant_columns_count += 1
        # Check missing values
        elif missing_pct > 20.0:
            status = "Problem"
            reason = f"High missing values ({missing_pct:.1f}%)"
            total_problems += 1
        elif missing_pct > 0.0:
            status = "Warning"
            reason = f"Contains missing values ({missing_pct:.1f}%)"
            total_warnings += 1
        # Check cardinality
        elif col in types["categorical"] and unique_count > 50:
            # Exclude potential IDs
            if "id" in col.lower() or "name" in col.lower():
                status = "Warning"
                reason = "High cardinality ID/Name column"
                total_warnings += 1
            else:
                status = "Warning"
                reason = f"High cardinality categorical ({unique_count} uniques)"
                total_warnings += 1
                
        column_audit.append({
            "feature": col,
            "data_type": dtype_str,
            "missing_count": missing_count,
            "missing_percentage": round(missing_pct, 2),
            "unique_values": unique_count,
            "status": status,
            "reason": reason
        })
        
    # Calculate overall data quality score
    # Score starts at 100
    # Subtract points for duplicates, missingness, constant columns, etc.
    score_deduction = 0
    if total_rows > 0:
        score_deduction += (dup_rows / total_rows) * 15 # Deduct up to 15% for duplicates
        
    # Deduct for column problems
    for audit in column_audit:
        if audit["status"] == "Problem":
            score_deduction += 5
        elif audit["status"] == "Warning":
            score_deduction += 2.5
            
    quality_score = max(5, int(100 - score_deduction))
    
    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "duplicate_rows": dup_rows,
        "constant_columns": constant_columns_count,
        "numerical_features_count": len(types["numerical"]),
        "categorical_features_count": len(types["categorical"]),
        "date_features_count": len(types["date"]),
        "boolean_features_count": len(types["boolean"]),
        "quality_score": quality_score,
        "columns": column_audit
    }

def get_univariate_stats(df: pd.DataFrame, col: str) -> dict:
    """Calculates comprehensive statistics for a column based on its type."""
    if col not in df.columns:
        return {"error": f"Column {col} does not exist."}
        
    total_count = len(df)
    missing_count = int(df[col].isna().sum())
    non_null_count = total_count - missing_count
    
    if pd.api.types.is_numeric_dtype(df[col]):
        series = df[col].dropna()
        if series.empty:
            return {"type": "numeric", "stats": {}, "message": "Column has no data."}
            
        stats = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
            "q1": float(series.quantile(0.25)),
            "q3": float(series.quantile(0.75)),
            "skewness": float(series.skew()) if len(series) > 2 else 0.0,
            "kurtosis": float(series.kurt()) if len(series) > 3 else 0.0,
            "non_null_count": non_null_count,
            "missing_count": missing_count
        }
        return {"type": "numeric", "stats": stats}
    else:
        # Categorical/String/Date/Boolean
        series = df[col].dropna().astype(str)
        if series.empty:
            return {"type": "categorical", "stats": {}, "message": "Column has no data."}
            
        unique_count = int(df[col].nunique())
        value_counts = df[col].value_counts()
        freq_table = []
        
        # Take all values for detail
        for val, count in value_counts.items():
            freq_table.append({
                "value": str(val),
                "count": int(count),
                "percentage": round(float(count / non_null_count * 100), 2)
            })
            
        stats = {
            "unique_count": unique_count,
            "top_value": str(value_counts.index[0]) if not value_counts.empty else None,
            "top_frequency": int(value_counts.iloc[0]) if not value_counts.empty else 0,
            "non_null_count": non_null_count,
            "missing_count": missing_count,
            "freq_table": freq_table
        }
        return {"type": "categorical", "stats": stats}

def get_distribution_analysis(df: pd.DataFrame) -> dict:
    """
    Calculates distribution metrics (skewness, kurtosis) for all numerical columns
    and dynamically generates educational explanations.
    """
    types = classify_columns(df)
    num_cols = types["numerical"]
    
    results = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 3:
            continue
            
        skew = float(series.skew())
        kurt = float(series.kurt())
        
        # Classify shape
        if abs(skew) < 0.5:
            shape = "Approximately Symmetric (Normal)"
            desc = (f"The distribution of '{col}' is symmetric. The mean ({series.mean():.1f}) "
                    f"and median ({series.median():.1f}) are close, suggesting a bell-shaped normal curve. "
                    f"Machine learning algorithms generally perform optimally with symmetric data.")
        elif skew >= 0.5:
            shape = "Right Skewed (Positive)"
            desc = (f"The feature '{col}' is right-skewed (skewness = {skew:.2f}). Most observations "
                    f"are concentrated on the lower side, while a few high values pull the mean ({series.mean():.1f}) "
                    f"higher than the median ({series.median():.1f}). Capping outliers or applying log transforms "
                    f"may help algorithms learn better.")
        else:
            shape = "Left Skewed (Negative)"
            desc = (f"The feature '{col}' is left-skewed (skewness = {skew:.2f}). Observations "
                    f"are concentrated on the higher side, with tail values dragging the mean ({series.mean():.1f}) "
                    f"below the median ({series.median():.1f}). Special transformations can help normalize this feature.")
            
        results.append({
            "column": col,
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "skewness": round(skew, 2),
            "kurtosis": round(kurt, 2),
            "shape": shape,
            "interpretation": desc
        })
        
    return {"distributions": results}

def get_correlation_analysis(df: pd.DataFrame) -> dict:
    """
    Performs Pearson correlation analysis, extracts top pairs,
    and flags potential multicollinearity.
    """
    types = classify_columns(df)
    num_cols = types["numerical"]
    
    if len(num_cols) < 2:
        return {"error": "Not enough numerical features to calculate correlation.", "correlations": [], "multicollinearity_warnings": []}
        
    corr_matrix = df[num_cols].corr()
    
    # Extract top positive and negative correlations (excluding self-correlation)
    pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            col1 = num_cols[i]
            col2 = num_cols[j]
            coeff = float(corr_matrix.loc[col1, col2])
            if not np.isnan(coeff):
                pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "coefficient": round(coeff, 3),
                    "abs_coeff": abs(coeff)
                })
                
    # Sort pairs
    pairs_sorted = sorted(pairs, key=lambda x: x["coefficient"], reverse=True)
    
    top_positive = [p for p in pairs_sorted if p["coefficient"] > 0.1][:5]
    top_negative = sorted([p for p in pairs_sorted if p["coefficient"] < -0.1], key=lambda x: x["coefficient"])[:5]
    
    # Multicollinearity check (coeff > 0.8)
    multicollinearity = [p for p in pairs if p["abs_coeff"] > 0.8]
    
    # Format correlation matrix for JSON representation
    matrix_data = {}
    for col in num_cols:
        matrix_data[col] = {other: round(float(corr_matrix.loc[col, other]), 3) if not np.isnan(corr_matrix.loc[col, other]) else None for other in num_cols}
        
    return {
        "columns": num_cols,
        "matrix": matrix_data,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "multicollinearity_warnings": multicollinearity
    }

def get_feature_selection_recommendations(df: pd.DataFrame, target_col: str = None) -> dict:
    """
    Recommends feature inclusion/exclusion based on:
    - Zero/Low variance (where standard deviation < 0.01)
    - High missing values (> 40%)
    - Correlation / Mutual Information score with target (if target exists)
    """
    total_rows = len(df)
    recommendations = []
    
    # 1. Check variance and missingness
    for col in df.columns:
        if col == target_col:
            continue
            
        missing_count = df[col].isna().sum()
        missing_pct = float(missing_count / total_rows * 100)
        
        status = "Select"
        reason = "Good feature candidate."
        var_val = 1.0
        
        # Missing value check
        if missing_pct > 40.0:
            status = "Remove"
            reason = f"High missing values ({missing_pct:.1f}%). Imputation might bias model."
        
        # Variance check
        elif pd.api.types.is_numeric_dtype(df[col]):
            var_val = float(df[col].var())
            if var_val < 0.001 or np.isnan(var_val):
                status = "Remove"
                reason = f"Extremely low variance ({var_val:.5f}). Feature provides no predictive signal."
                
        # Unique/Constant check for categories
        elif df[col].nunique() <= 1:
            status = "Remove"
            reason = "Constant column. Values do not change."
            
        # ID check
        elif "id" in col.lower() and df[col].nunique() == total_rows:
            status = "Remove"
            reason = "Unique Identifier (High-cardinality index with no physical signal)."
            
        recommendations.append({
            "feature": col,
            "status": status,
            "reason": reason,
            "variance": round(var_val, 4) if pd.api.types.is_numeric_dtype(df[col]) else None,
            "missing_percentage": round(missing_pct, 2),
            "mutual_info_score": 0.0,
            "correlation_with_target": 0.0
        })
        
    # 2. Calculate Mutual Information & target correlation if target selected
    if target_col and target_col in df.columns:
        temp_df = df.copy().dropna(subset=[target_col])
        is_target_numeric = pd.api.types.is_numeric_dtype(temp_df[target_col])
        
        # Impute or encode features to calculate MI
        # We only calculate MI for columns that aren't already flagged for removal
        mi_features = [r["feature"] for r in recommendations if r["status"] == "Select"]
        
        if mi_features:
            encoded_df = pd.DataFrame()
            for col in mi_features:
                if pd.api.types.is_numeric_dtype(temp_df[col]):
                    encoded_df[col] = temp_df[col].fillna(temp_df[col].median())
                else:
                    # Fill na and Label Encode
                    filled = temp_df[col].fillna("Missing_Val").astype(str)
                    encoded_df[col] = LabelEncoder().fit_transform(filled)
                    
            y = temp_df[target_col]
            if not is_target_numeric:
                y = LabelEncoder().fit_transform(y.astype(str))
            else:
                y = y.fillna(y.median())
                
            try:
                # Calculate MI
                if is_target_numeric:
                    mi_scores = mutual_info_regression(encoded_df, y)
                else:
                    mi_scores = mutual_info_classif(encoded_df, y)
                    
                mi_map = dict(zip(mi_features, mi_scores))
                
                # Update recommendations
                for r in recommendations:
                    feat = r["feature"]
                    if feat in mi_map:
                        score = float(mi_map[feat])
                        r["mutual_info_score"] = round(score, 4)
                        
                        # Add correlation if both target and feature are numeric
                        if is_target_numeric and pd.api.types.is_numeric_dtype(df[feat]):
                            corr_coeff = df[feat].corr(df[target_col])
                            r["correlation_with_target"] = round(float(corr_coeff), 3) if not np.isnan(corr_coeff) else 0.0
                            
                        # If MI score is zero, warn user
                        if score < 0.005:
                            r["status"] = "Remove"
                            r["reason"] = f"No mutual information with target ({score:.4f}). Very weak dependency."
            except Exception as e:
                import traceback
                with open("d:/projects/merged ipl project/mi_error.log", "w") as f:
                    f.write(traceback.format_exc())
                # Fallback silently on errors in calculation
                pass
                
    return recommendations

def get_num_vs_num_bivariate(df: pd.DataFrame, col_x: str, col_y: str) -> dict:
    """
    Computes statistical relationships between two continuous/numerical variables:
    - Pearson correlation (r), p-value, strength & direction
    - Spearman rank correlation (rho)
    - Coefficient of determination (R^2)
    - Covariance
    - OLS Linear regression (slope, intercept, equation)
    """
    if col_x not in df.columns or col_y not in df.columns:
        return {"error": f"Columns {col_x} or {col_y} not found in dataframe."}
        
    valid_df = df[[col_x, col_y]].dropna()
    if len(valid_df) < 3:
        return {"error": "Not enough valid numeric data points for bivariate analysis."}
        
    x = valid_df[col_x].values
    y = valid_df[col_y].values
    
    # Check variance
    if np.var(x) == 0 or np.var(y) == 0:
        return {
            "error": "One of the selected columns has zero variance (all values are constant).",
            "pearson_r": 0.0,
            "spearman_rho": 0.0,
            "r_squared": 0.0
        }
        
    # Pearson
    pearson_r, pearson_p = stats.pearsonr(x, y)
    pearson_r = 0.0 if np.isnan(pearson_r) else float(pearson_r)
    pearson_p = 1.0 if np.isnan(pearson_p) else float(pearson_p)
    
    # Spearman
    spearman_rho, spearman_p = stats.spearmanr(x, y)
    spearman_rho = 0.0 if np.isnan(spearman_rho) else float(spearman_rho)
    spearman_p = 1.0 if np.isnan(spearman_p) else float(spearman_p)
    
    # R-squared
    r_squared = float(pearson_r ** 2)
    
    # Covariance
    covariance = float(valid_df[col_x].cov(valid_df[col_y]))
    
    # OLS Regression line
    lin_reg = stats.linregress(x, y)
    slope = float(lin_reg.slope) if not np.isnan(lin_reg.slope) else 0.0
    intercept = float(lin_reg.intercept) if not np.isnan(lin_reg.intercept) else 0.0
    
    # Qualitative interpretation
    abs_r = abs(pearson_r)
    if abs_r >= 0.7:
        strength = "Strong"
    elif abs_r >= 0.4:
        strength = "Moderate"
    elif abs_r >= 0.2:
        strength = "Weak"
    else:
        strength = "Very Weak / Negligible"
        
    if pearson_r > 0.05:
        direction = "Positive"
    elif pearson_r < -0.05:
        direction = "Negative"
    else:
        direction = "Neutral / No Linear Direction"
        
    sig_text = "statistically significant (p < 0.05)" if pearson_p < 0.05 else "not statistically significant (p >= 0.05)"
    
    desc = (
        f"**Pearson's r = {pearson_r:+.3f}** indicates a **{strength} {direction}** linear relationship "
        f"between `{col_x}` and `{col_y}` which is {sig_text}. "
        f"The coefficient of determination (**R² = {r_squared*100:.1f}%**) indicates that {r_squared*100:.1f}% "
        f"of the variance in `{col_y}` can be explained by linear variation in `{col_x}`. "
    )
    if abs(spearman_rho) > abs_r + 0.15:
        desc += f"Spearman's rank correlation (rho = {spearman_rho:+.3f}) is noticeably higher, suggesting a non-linear monotonic pattern."
        
    equation = f"{col_y} = {slope:.3f} * {col_x} + {intercept:.3f}" if intercept >= 0 else f"{col_y} = {slope:.3f} * {col_x} - {abs(intercept):.3f}"
    
    return {
        "sample_size": int(len(valid_df)),
        "pearson_r": round(pearson_r, 4),
        "pearson_p_value": float(pearson_p),
        "spearman_rho": round(spearman_rho, 4),
        "spearman_p_value": float(spearman_p),
        "r_squared": round(r_squared, 4),
        "r_squared_pct": round(r_squared * 100, 2),
        "covariance": round(covariance, 4),
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "equation": equation,
        "strength": strength,
        "direction": direction,
        "interpretation": desc
    }

def get_cat_vs_num_bivariate(df: pd.DataFrame, cat_col: str, num_col: str) -> dict:
    """
    Computes statistical relationships between a categorical and a numerical variable:
    - Group-wise statistics (Mean, Median, Std, Min, Max, Q1, Q3, Count)
    - ANOVA F-statistic & p-value (Hypothesis test for difference in group means)
    - Kruskal-Wallis H-test (Non-parametric test for group medians)
    """
    if cat_col not in df.columns or num_col not in df.columns:
        return {"error": f"Columns {cat_col} or {num_col} not found in dataframe."}
        
    valid_df = df[[cat_col, num_col]].dropna()
    if valid_df.empty:
        return {"error": "No valid data points found."}
        
    unique_cats = valid_df[cat_col].nunique()
    cardinality_note = ""
    if unique_cats > 25:
        cardinality_note = f"Displaying top 25 most frequent categories out of {unique_cats} unique values for clear visualization."
        top_cats = valid_df[cat_col].value_counts().head(25).index
        valid_df = valid_df[valid_df[cat_col].isin(top_cats)]
        
    # Group by category and compute stats
    grouped = valid_df.groupby(cat_col)[num_col]
    group_stats = []
    arrays_for_anova = []
    
    for cat_name, group_data in grouped:
        clean_vals = group_data.dropna().values
        if len(clean_vals) > 0:
            arrays_for_anova.append(clean_vals)
            group_stats.append({
                "category": str(cat_name),
                "count": int(len(clean_vals)),
                "mean": round(float(np.mean(clean_vals)), 2),
                "median": round(float(np.median(clean_vals)), 2),
                "std": round(float(np.std(clean_vals, ddof=1)), 2) if len(clean_vals) > 1 else 0.0,
                "min": round(float(np.min(clean_vals)), 2),
                "max": round(float(np.max(clean_vals)), 2),
                "q1": round(float(np.percentile(clean_vals, 25)), 2),
                "q3": round(float(np.percentile(clean_vals, 75)), 2)
            })
            
    # Sort group stats by count descending
    group_stats = sorted(group_stats, key=lambda x: x["count"], reverse=True)
    
    # Statistical Tests
    anova_f = 0.0
    anova_p = 1.0
    kruskal_h = 0.0
    kruskal_p = 1.0
    
    if len(arrays_for_anova) >= 2 and all(len(a) > 0 for a in arrays_for_anova):
        try:
            f_res = stats.f_oneway(*arrays_for_anova)
            anova_f = float(f_res.statistic) if not np.isnan(f_res.statistic) else 0.0
            anova_p = float(f_res.pvalue) if not np.isnan(f_res.pvalue) else 1.0
        except Exception:
            pass
            
        try:
            k_res = stats.kruskal(*arrays_for_anova)
            kruskal_h = float(k_res.statistic) if not np.isnan(k_res.statistic) else 0.0
            kruskal_p = float(k_res.pvalue) if not np.isnan(k_res.pvalue) else 1.0
        except Exception:
            pass
            
    # Key insights
    if group_stats:
        max_mean_group = max(group_stats, key=lambda x: x["mean"])
        min_mean_group = min(group_stats, key=lambda x: x["mean"])
        max_med_group = max(group_stats, key=lambda x: x["median"])
        min_med_group = min(group_stats, key=lambda x: x["median"])
        
        diff_sig = "significant difference across group means (p < 0.05)" if anova_p < 0.05 else "no statistically significant difference across group means (p >= 0.05)"
        
        interpretation = (
            f"Comparing `{num_col}` across categories of `{cat_col}`: One-Way ANOVA test (F = {anova_f:.2f}, p = {anova_p:.3e}) "
            f"indicates **{diff_sig}**. "
            f"The category **'{max_mean_group['category']}'** achieves the highest average `{num_col}` ({max_mean_group['mean']:.2f}, median: {max_mean_group['median']:.2f}), "
            f"whereas **'{min_mean_group['category']}'** has the lowest average ({min_mean_group['mean']:.2f}, median: {min_mean_group['median']:.2f})."
        )
        if cardinality_note:
            interpretation += f" ({cardinality_note})"
    else:
        interpretation = "Insufficient data to compute group statistics."
        
    return {
        "cat_col": cat_col,
        "num_col": num_col,
        "categories_count": len(group_stats),
        "total_unique_categories": unique_cats,
        "cardinality_note": cardinality_note,
        "group_stats": group_stats,
        "anova_f": round(anova_f, 3),
        "anova_p_value": float(anova_p),
        "kruskal_h": round(kruskal_h, 3),
        "kruskal_p_value": float(kruskal_p),
        "interpretation": interpretation
    }

def get_cat_vs_cat_bivariate(df: pd.DataFrame, col_x: str, col_y: str) -> dict:
    """
    Computes statistical relationships between two categorical variables:
    - Contingency table / Cross-tabulation (Counts, Row %, Col %, Total %)
    - Chi-Square test of independence (chi2, p-value, degrees of freedom)
    - Cramer's V association metric
    - Top co-occurrences
    """
    if col_x not in df.columns or col_y not in df.columns:
        return {"error": f"Columns {col_x} or {col_y} not found in dataframe."}
        
    valid_df = df[[col_x, col_y]].dropna().astype(str)
    if valid_df.empty:
        return {"error": "No valid data points found."}
        
    x_uniques = valid_df[col_x].nunique()
    y_uniques = valid_df[col_y].nunique()
    
    cardinality_note = ""
    if x_uniques > 20 or y_uniques > 20:
        cardinality_note = f"Showing top 20 categories per axis (out of {x_uniques} and {y_uniques} unique values) for readable cross-tabulation."
        if x_uniques > 20:
            top_x = valid_df[col_x].value_counts().head(20).index
            valid_df = valid_df[valid_df[col_x].isin(top_x)]
        if y_uniques > 20:
            top_y = valid_df[col_y].value_counts().head(20).index
            valid_df = valid_df[valid_df[col_y].isin(top_y)]
        
    # Cross-tabulation (raw count)
    ct_counts = pd.crosstab(valid_df[col_x], valid_df[col_y], margins=True, margins_name="Total")
    ct_row_pct = pd.crosstab(valid_df[col_x], valid_df[col_y], normalize="index") * 100
    ct_col_pct = pd.crosstab(valid_df[col_x], valid_df[col_y], normalize="columns") * 100
    ct_tot_pct = pd.crosstab(valid_df[col_x], valid_df[col_y], normalize="all") * 100
    
    # Chi-Square test on raw counts (excluding totals margin)
    ct_raw = pd.crosstab(valid_df[col_x], valid_df[col_y])
    
    chi2_stat = 0.0
    chi2_p = 1.0
    dof = 0
    cramers_v = 0.0
    
    if ct_raw.shape[0] >= 2 and ct_raw.shape[1] >= 2:
        try:
            chi2_res = stats.chi2_contingency(ct_raw)
            chi2_stat = float(chi2_res[0])
            chi2_p = float(chi2_res[1])
            dof = int(chi2_res[2])
            
            n = len(valid_df)
            min_dim = min(ct_raw.shape[0] - 1, ct_raw.shape[1] - 1)
            if min_dim > 0 and n > 0:
                cramers_v = float(np.sqrt(chi2_stat / (n * min_dim)))
                cramers_v = min(1.0, max(0.0, cramers_v))
        except Exception:
            pass
            
    # Cramer's V qualitative strength
    if cramers_v >= 0.5:
        v_strength = "Very Strong Association"
    elif cramers_v >= 0.3:
        v_strength = "Strong Association"
    elif cramers_v >= 0.15:
        v_strength = "Moderate Association"
    elif cramers_v >= 0.05:
        v_strength = "Weak Association"
    else:
        v_strength = "Little to No Association"
        
    # Find most frequent pair
    stacked = ct_raw.stack()
    if not stacked.empty:
        top_pair_idx = stacked.idxmax()
        top_pair_count = int(stacked.loc[top_pair_idx])
        top_pair_x = str(top_pair_idx[0])
        top_pair_y = str(top_pair_idx[1])
    else:
        top_pair_x, top_pair_y, top_pair_count = "N/A", "N/A", 0
        
    dep_text = "statistically dependent / associated (p < 0.05)" if chi2_p < 0.05 else "independent with no significant association (p >= 0.05)"
    
    interpretation = (
        f"Contingency Cross-Tabulation between `{col_x}` and `{col_y}`: "
        f"Chi-Square test (Chi2 = {chi2_stat:.2f}, dof = {dof}, p = {chi2_p:.3e}) indicates the two categorical features are **{dep_text}**. "
        f"Cramer's V association metric is **{cramers_v:.3f}** ({v_strength}). "
        f"The most frequent combination is **'{col_x}' = '{top_pair_x}'** with **'{col_y}' = '{top_pair_y}'** ({top_pair_count:,} occurrences)."
    )
    
    # Format tables for JSON serialization
    def format_crosstab_dict(ct_df):
        return {
            "index": [str(i) for i in ct_df.index],
            "columns": [str(c) for c in ct_df.columns],
            "data": [[round(float(val), 2) for val in row] for row in ct_df.values]
        }
        
    return {
        "col_x": col_x,
        "col_y": col_y,
        "total_samples": int(len(valid_df)),
        "chi2_stat": round(chi2_stat, 3),
        "chi2_p_value": float(chi2_p),
        "dof": int(dof),
        "cramers_v": round(cramers_v, 4),
        "cramers_v_strength": v_strength,
        "top_combination": {
            "x": top_pair_x,
            "y": top_pair_y,
            "count": top_pair_count
        },
        "contingency_counts": format_crosstab_dict(ct_counts),
        "contingency_row_pct": format_crosstab_dict(ct_row_pct),
        "contingency_col_pct": format_crosstab_dict(ct_col_pct),
        "contingency_tot_pct": format_crosstab_dict(ct_tot_pct),
        "interpretation": interpretation
    }

