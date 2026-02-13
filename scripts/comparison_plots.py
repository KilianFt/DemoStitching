import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up the plotting style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# Load the data (assuming the data is already loaded as 'df' in your notebook)
# df1 = pd.read_csv("./dataset/stitching/presentation/figures/reuse/results.csv")
# df2 = pd.read_csv("./dataset/stitching/presentation/figures/recompute_all/results.csv")
# df3 = pd.read_csv("./dataset/stitching/presentation/figures/recompute_ds/results.csv")
# df = pd.concat([df1, df2, df3])

def create_comparison_plots(df):
    """
    Create comprehensive comparison plots for DS methods across three key metrics
    """
    
    # Define the metrics to compare
    metrics = ['cosine_dissimilarity', 'prediction_rmse', 'dtw_distance_mean']
    metric_labels = ['Cosine Dissimilarity', 'Prediction RMSE', 'DTW Distance Mean']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DS Method Comparison Across Key Metrics', fontsize=18, fontweight='bold')
    
    # Plot 1: Bar plots for each metric
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Create bar plot with error bars
        sns.barplot(data=df, x="ds_method", y=metric, errorbar="sd", ax=ax, 
                   palette=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        ax.set_title(f'{label} by DS Method', fontweight='bold')
        ax.set_xlabel('DS Method')
        ax.set_ylabel(label)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', rotation=0, fontsize=10)
    
    # Plot 4: Combined comparison (normalized values)
    ax = axes[1, 1]
    
    # Normalize metrics to 0-1 scale for comparison
    df_normalized = df.copy()
    for metric in metrics:
        df_normalized[f'{metric}_norm'] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
    
    # Calculate mean normalized values per method
    method_means = df_normalized.groupby('ds_method')[[f'{m}_norm' for m in metrics]].mean()
    
    # Create grouped bar plot
    x = np.arange(len(method_means.index))
    width = 0.25
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax.bar(x + i*width, method_means[f'{metric}_norm'], width, 
               label=label, alpha=0.8)
    
    ax.set_title('Normalized Metric Comparison\n(Lower is Better)', fontweight='bold')
    ax.set_xlabel('DS Method')
    ax.set_ylabel('Normalized Score (0-1)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(method_means.index, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_detailed_comparison_table(df):
    """
    Create a detailed comparison table with statistics
    """
    metrics = ['cosine_dissimilarity', 'prediction_rmse', 'dtw_distance_mean', 'compute_time']
    
    # Calculate statistics for each method
    stats_table = df.groupby('ds_method')[metrics].agg(['mean', 'std', 'min', 'max']).round(4)
    
    print("=== DS Method Comparison Statistics ===")
    print(stats_table)
    
    return stats_table

def create_pairwise_comparison_plot(df):
    """
    Create scatter plots to show relationships between metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Pairwise Metric Relationships by DS Method', fontsize=16, fontweight='bold')
    
    # Define metric pairs
    pairs = [
        ('cosine_dissimilarity', 'prediction_rmse'),
        ('cosine_dissimilarity', 'dtw_distance_mean'),
        ('prediction_rmse', 'dtw_distance_mean')
    ]
    
    for i, (x_metric, y_metric) in enumerate(pairs):
        ax = axes[i]
        
        # Create scatter plot with different colors for each method
        for method in df['ds_method'].unique():
            method_data = df[df['ds_method'] == method]
            ax.scatter(method_data[x_metric], method_data[y_metric], 
                      label=method, alpha=0.7, s=60)
        
        ax.set_xlabel(x_metric.replace('_', ' ').title())
        ax.set_ylabel(y_metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_performance_radar_chart(df):
    """
    Create a radar chart comparing DS methods across all metrics
    """
    from math import pi
    
    # Calculate mean values for each method
    metrics = ['cosine_dissimilarity', 'prediction_rmse', 'dtw_distance_mean']
    method_means = df.groupby('ds_method')[metrics].mean()
    
    # Normalize values (invert so higher is better for visualization)
    for metric in metrics:
        method_means[metric] = 1 - (method_means[metric] - method_means[metric].min()) / (method_means[metric].max() - method_means[metric].min())
    
    # Set up radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Number of metrics
    N = len(metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Plot each method
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (method, color) in enumerate(zip(method_means.index, colors)):
        values = method_means.loc[method].tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title('DS Method Performance Radar Chart\n(Higher is Better)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)
    
    return fig

# Example usage (copy this into your notebook):
"""
# In your Jupyter notebook, after loading the data:

# Load data
df1 = pd.read_csv("./dataset/stitching/presentation/figures/reuse/results.csv")
df2 = pd.read_csv("./dataset/stitching/presentation/figures/recompute_all/results.csv")
df3 = pd.read_csv("./dataset/stitching/presentation/figures/recompute_ds/results.csv")
df = pd.concat([df1, df2, df3])

# Create comparison plots
fig1 = create_comparison_plots(df)
plt.show()

# Create detailed statistics table
stats = create_detailed_comparison_table(df)

# Create pairwise comparison plots
fig2 = create_pairwise_comparison_plot(df)
plt.show()

# Create radar chart (if you want a more advanced visualization)
fig3 = create_performance_radar_chart(df)
plt.show()
"""
