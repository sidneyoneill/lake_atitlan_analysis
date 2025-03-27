import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def plot_feature_heatmap(df_imputed, feature_name, output_path=None):
    """Create a heatmap of feature concentrations over depth and time."""
    if output_path is None:
        output_path = f'output/figures/{feature_name}_heatmap.png'
    
    # Convert index to numeric depths
    df = df_imputed.copy()
    df.index = df.index.astype(float)
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        df,
        cmap='viridis',
        yticklabels=20,
        cbar_kws={'label': f'{feature_name.capitalize()}'}
    )
    
    plt.title(f'{feature_name.capitalize()} Over Depth and Time')
    plt.xlabel('Date')
    plt.ylabel('Depth (m)')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_depth_group_timeseries(df_grouped, feature_name, output_path=None):
    """Create separate time series plots for each depth group."""
    if output_path is None:
        output_path = f'output/figures/{feature_name}_depth_group_timeseries.png'
    
    # Convert date to datetime
    df_grouped['date'] = pd.to_datetime(df_grouped['date'])
    
    # Get unique depth groups
    depth_groups = sorted(df_grouped['depth_group'].unique())
    n_groups = len(depth_groups)
    
    # Create subplots
    fig, axes = plt.subplots(n_groups, 1, figsize=(15, 4*n_groups))
    fig.suptitle(f'{feature_name.capitalize()} by Depth Group Over Time', y=1.02, fontsize=14)
    
    # Plot each depth group in its own subplot
    for ax, group in zip(axes, depth_groups):
        group_data = df_grouped[df_grouped['depth_group'] == group]
        
        # Plot mean line with markers
        ax.plot(
            group_data['date'],
            group_data[f'mean_{feature_name}'],
            label='Mean',
            color='blue',
            marker='o'
        )
        
        # Add error bands
        ax.fill_between(
            group_data['date'],
            group_data[f'mean_{feature_name}'] - group_data[f'std_{feature_name}'],
            group_data[f'mean_{feature_name}'] + group_data[f'std_{feature_name}'],
            alpha=0.2,
            color='blue',
            label='±1 SD'
        )
        
        # Customize subplot
        ax.set_title(f'{group} Layer')
        ax.set_xlabel('Date')
        ax.set_ylabel(feature_name.capitalize())
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_surface_middepth_comparison(df_grouped, feature_name, output_path=None):
    """Create a single plot comparing Surface and Mid-Depth feature values."""
    if output_path is None:
        output_path = f'output/figures/{feature_name}_surface_middepth_comparison.png'
    
    # Convert date to datetime
    df_grouped['date'] = pd.to_datetime(df_grouped['date'])
    
    # Filter for Surface and Mid-Depth groups only
    comparison_groups = ['Surface', 'Mid-Depth']
    df_comparison = df_grouped[df_grouped['depth_group'].isin(comparison_groups)]
    
    plt.figure(figsize=(15, 8))
    colors = {'Surface': 'blue', 'Mid-Depth': 'red'}
    
    for group in comparison_groups:
        group_data = df_comparison[df_comparison['depth_group'] == group]
        
        plt.plot(
            group_data['date'],
            group_data[f'mean_{feature_name}'],
            label=f'{group} Mean',
            color=colors[group],
            marker='o'
        )
        
        plt.fill_between(
            group_data['date'],
            group_data[f'mean_{feature_name}'] - group_data[f'std_{feature_name}'],
            group_data[f'mean_{feature_name}'] + group_data[f'std_{feature_name}'],
            alpha=0.2,
            color=colors[group],
            label=f'{group} ±1 SD'
        )
    
    plt.title(f'Comparison of Surface and Mid-Depth {feature_name.capitalize()} Values', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel(feature_name.capitalize())
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_data_quality_report(df_assigned, df_interpolated, df_imputed, df_grouped, feature_name):
    """Generate a data quality report."""
    report = []
    
    report.append(f"=== {feature_name.capitalize()} Data Quality Report ===")
    report.append(f"Original missing values: {df_assigned.isna().sum().sum()}")
    report.append(f"After vertical interpolation: {df_interpolated.isna().sum().sum()}")
    report.append(f"After horizontal imputation: {df_imputed.isna().sum().sum()}")
    
    report.append("\n=== Value Range Checks ===")
    report.append(f"Min {feature_name}: {df_imputed.min().min():.3f}")
    report.append(f"Max {feature_name}: {df_imputed.max().max():.3f}")
    
    # Depth group statistics
    report.append("\n=== Depth Group Statistics ===")
    group_stats = df_grouped.groupby('depth_group').agg({
        f'mean_{feature_name}': ['mean', 'std', 'min', 'max'],
        'measurement_count': 'mean'
    }).round(3)
    report.append(group_stats.to_string())
    
    # Save report with feature name
    report_text = '\n'.join(report)
    with open(f'output/{feature_name}_data_quality_report.txt', 'w') as f:
        f.write(report_text)
    
    return report_text