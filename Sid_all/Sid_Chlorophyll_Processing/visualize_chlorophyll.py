import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def plot_chlorophyll_heatmap(df_imputed, output_path='output/figures/chlorophyll_heatmap.png'):
    """Create a heatmap of chlorophyll concentrations over depth and time."""
    # Convert index to numeric depths
    df = df_imputed.copy()
    df.index = df.index.astype(float)
    
    # Create the heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        df,
        cmap='viridis',
        yticklabels=20,  # Show fewer depth labels for clarity
        cbar_kws={'label': 'Chlorophyll (μg/L)'}
    )
    
    plt.title('Chlorophyll Concentration Over Depth and Time')
    plt.xlabel('Date')
    plt.ylabel('Depth (m)')
    plt.gca().invert_yaxis()  # Invert y-axis to show depth increasing downward
    
    # Save and close
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_depth_group_timeseries(df_grouped, output_path='output/figures/depth_group_timeseries.png'):
    """Create separate time series plots for each depth group."""
    # Convert date to datetime
    df_grouped['date'] = pd.to_datetime(df_grouped['date'])
    
    # Get unique depth groups
    depth_groups = sorted(df_grouped['depth_group'].unique())
    n_groups = len(depth_groups)
    
    # Create subplots
    fig, axes = plt.subplots(n_groups, 1, figsize=(15, 4*n_groups))
    fig.suptitle('Chlorophyll Concentration by Depth Group Over Time', y=1.02, fontsize=14)
    
    # Plot each depth group in its own subplot
    for ax, group in zip(axes, depth_groups):
        group_data = df_grouped[df_grouped['depth_group'] == group]
        
        # Plot mean line with markers
        ax.plot(
            group_data['date'],
            group_data['mean_chlorophyll'],
            label='Mean',
            color='blue',
            marker='o'
        )
        
        # Add error bands
        ax.fill_between(
            group_data['date'],
            group_data['mean_chlorophyll'] - group_data['std_chlorophyll'],
            group_data['mean_chlorophyll'] + group_data['std_chlorophyll'],
            alpha=0.2,
            color='blue',
            label='±1 SD'
        )
        
        # Customize subplot
        ax.set_title(f'{group} Layer')
        ax.set_xlabel('Date')
        ax.set_ylabel('Chlorophyll (μg/L)')
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

def plot_surface_middepth_comparison(df_grouped, output_path='output/figures/surface_middepth_comparison.png'):
    """Create a single plot comparing Surface and Mid-Depth chlorophyll concentrations."""
    # Convert date to datetime
    df_grouped['date'] = pd.to_datetime(df_grouped['date'])
    
    # Filter for Surface and Mid-Depth groups only
    comparison_groups = ['Surface', 'Mid-Depth']
    df_comparison = df_grouped[df_grouped['depth_group'].isin(comparison_groups)]
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Plot each depth group with different colors
    colors = {'Surface': 'blue', 'Mid-Depth': 'red'}
    
    for group in comparison_groups:
        group_data = df_comparison[df_comparison['depth_group'] == group]
        
        # Plot mean line with markers
        plt.plot(
            group_data['date'],
            group_data['mean_chlorophyll'],
            label=f'{group} Mean',
            color=colors[group],
            marker='o'
        )
        
        # Add error bands
        plt.fill_between(
            group_data['date'],
            group_data['mean_chlorophyll'] - group_data['std_chlorophyll'],
            group_data['mean_chlorophyll'] + group_data['std_chlorophyll'],
            alpha=0.2,
            color=colors[group],
            label=f'{group} ±1 SD'
        )
    
    # Customize plot
    plt.title('Comparison of Surface and Mid-Depth Chlorophyll Concentrations', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Chlorophyll (μg/L)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis limits
    plt.ylim(0, 10)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_data_quality_report(df_assigned, df_interpolated, df_imputed, df_grouped):
    """Generate a data quality report."""
    report = []
    
    # Missing values summary
    report.append("=== Missing Values Summary ===")
    report.append(f"Original missing values: {df_assigned.isna().sum().sum()}")
    report.append(f"After vertical interpolation: {df_interpolated.isna().sum().sum()}")
    report.append(f"After horizontal imputation: {df_imputed.isna().sum().sum()}")
    
    # Value range checks
    report.append("\n=== Value Range Checks ===")
    report.append(f"Min chlorophyll: {df_imputed.min().min():.3f} μg/L")
    report.append(f"Max chlorophyll: {df_imputed.max().max():.3f} μg/L")
    
    # Depth group statistics
    report.append("\n=== Depth Group Statistics ===")
    group_stats = df_grouped.groupby('depth_group').agg({
        'mean_chlorophyll': ['mean', 'std', 'min', 'max'],
        'measurement_count': 'mean'
    }).round(3)
    report.append(group_stats.to_string())
    
    # Save report
    report_text = '\n'.join(report)
    with open('output/data_quality_report.txt', 'w') as f:
        f.write(report_text)
    
    return report_text