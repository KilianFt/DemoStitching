#!/usr/bin/env python3
"""
Improved LaTeX Summary Generator for DS Method Evaluation Results

This script provides functions to generate publication-ready LaTeX tables
with the following improvements:
1. Remove underscores from method names and metric labels for readability
2. Bold the best method's metric values automatically
3. Combine mean and standard deviation into a single column formatted as "mean ± std"
"""

import pandas as pd
import numpy as np


def clean_method_names(df):
    """
    Clean DS method names by removing underscores and making them more readable.
    
    Args:
        df: DataFrame with 'ds_method' column
        
    Returns:
        DataFrame with cleaned method names
    """
    df = df.copy()
    method_name_mapping = {
        'reuse': 'Reuse',
        'recompute_all': 'Recompute All', 
        'recompute_ds': 'Recompute DS'
    }
    
    # Apply mapping if the original names exist, otherwise keep current names
    if 'ds_method' in df.columns:
        df['ds_method'] = df['ds_method'].replace(method_name_mapping)
    
    return df


def clean_metric_names(metric_names):
    """
    Clean metric names by removing underscores and making them more readable.
    
    Args:
        metric_names: List of metric column names
        
    Returns:
        Dictionary mapping original names to cleaned names
    """
    metric_name_mapping = {
        'cosine_dissimilarity': 'Cosine Dissimilarity',
        'prediction_rmse': 'Prediction RMSE',
        'dtw_distance_mean': 'DTW Distance',
        'distance_to_attractor_mean': 'Distance to Attractor',
        'trajectory_length_mean': 'Trajectory Length',
        'compute_time': 'Compute Time',
        'ds_compute_time': 'ds_compute_time'
    }
    
    return {name: metric_name_mapping.get(name, name.replace('_', ' ').title()) 
            for name in metric_names}


def identify_best_methods(summary_stats, metrics_cols):
    """
    Identify the best performing method for each metric.
    For all metrics, lower values are better (indicated by ↓ in plots).
    
    Args:
        summary_stats: DataFrame with multi-level columns (metric, statistic)
        metrics_cols: List of metric column names
        
    Returns:
        Dictionary mapping (metric, method) tuples to True if best
    """
    best_methods = {}
    
    for metric in metrics_cols:
        if (metric, 'mean') in summary_stats.columns:
            # Find method with minimum mean value (lower is better for all metrics)
            best_method_idx = summary_stats[(metric, 'mean')].idxmin()
            best_methods[(metric, best_method_idx)] = True
    
    return best_methods


def format_mean_std_with_bold(mean_val, std_val, is_best=False):
    """
    Format mean ± std with optional bold formatting for best values.
    
    Args:
        mean_val: Mean value
        std_val: Standard deviation value
        is_best: Whether this is the best value (should be bolded)
        
    Returns:
        Formatted string for LaTeX
    """
    # Format the mean ± std string with 2 decimal places
    formatted = f"{mean_val:.2f} ± {std_val:.2f}"
    
    # Bold if it's the best value
    if is_best:
        formatted = f"\\textbf{{{formatted}}}"
    
    return formatted


def generate_latex_summary(df, metrics_cols=None):
    """
    Generate an LaTeX summary table with the requested enhancements.
    
    Args:
        df: DataFrame with experimental results
        metrics_cols: List of metric columns to include (optional)
        
    Returns:
        String containing LaTeX table code
    """
    if metrics_cols is None:
        metrics_cols = ['cosine_dissimilarity', 'prediction_rmse', 'dtw_distance_mean', 
                       'distance_to_attractor_mean', 'trajectory_length_mean']
        if 'compute_time' in df.columns:
            metrics_cols.append('compute_time')
        if 'ds_compute_time' in df.columns:
            metrics_cols.append('ds_compute_time')
    
    # Clean method names
    df_clean = clean_method_names(df)
    
    # Generate summary statistics with 2 decimal places
    summary_stats = df_clean.groupby('ds_method')[metrics_cols].agg(['mean', 'std']).round(2)
    
    # Identify best methods for each metric
    best_methods = identify_best_methods(summary_stats, metrics_cols)
    
    # Clean metric names for display
    metric_name_mapping = clean_metric_names(metrics_cols)
    
    # Create the improved summary table
    latex_rows = []
    
    # Header
    header_cols = ['Method'] + [metric_name_mapping[col] for col in metrics_cols]
    latex_rows.append(' & '.join(header_cols) + ' \\\\')
    latex_rows.append('\\hline')
    
    # Data rows
    for method in summary_stats.index:
        row_data = [method.replace('_', ' ')]  # Clean method name
        
        for metric in metrics_cols:
            mean_val = summary_stats.loc[method, (metric, 'mean')]
            std_val = summary_stats.loc[method, (metric, 'std')]
            is_best = best_methods.get((metric, method), False)
            
            formatted_cell = format_mean_std_with_bold(mean_val, std_val, is_best)
            row_data.append(formatted_cell)
        
        latex_rows.append(' & '.join(row_data) + ' \\\\')
    
    # Combine into full LaTeX table
    latex_table = """\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison of DS Methods (Mean ± Standard Deviation)}
\\label{tab:ds_method_comparison}
\\begin{tabular}{""" + 'l' + 'c' * len(metrics_cols) + """}
\\hline
""" + '\n'.join(latex_rows) + """
\\hline
\\end{tabular}
\\note{Lower values indicate better performance for all metrics. Best values are shown in bold.}
\\end{table}"""
    
    return latex_table


def print_summary(df, metrics_cols=None):
    """
    Print the LaTeX summary to console.
    
    Args:
        df: DataFrame with experimental results
        metrics_cols: List of metric columns to include (optional)
    """
    print("=== Statistical Summary by DS Method ===")
    print()
    latex_output = generate_latex_summary(df, metrics_cols)
    print(latex_output)
    print()
    print("=== End of LaTeX Summary ===")


if __name__ == "__main__":
    # Example usage - this would be called from the notebook
    print("Improved LaTeX Summary Generator")
    print("Import this module in your notebook and call:")
    print("from improved_latex_summary import print_improved_summary")
    print("print_improved_summary(df)")
