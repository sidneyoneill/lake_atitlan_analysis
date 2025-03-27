import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def plot_paired_comparisons(feature, depth_group, location):
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Read the datasets
    old_data = pd.read_csv('output/grouped_depths.csv')
    sid_data = pd.read_csv('data/SID_LIMNO_no_outliers_v2.csv')
    evan_data = pd.read_csv('data/Lake_data_clean_final_v2.csv')
    
    # Convert date formats
    old_data['date'] = pd.to_datetime(old_data['date'])
    sid_data['date'] = pd.to_datetime(sid_data['date'])
    evan_data['date'] = pd.to_datetime(evan_data['date'], format='%d/%m/%Y')
    
    # Convert EVAN depth_group format to match others
    evan_data['depth_group'] = evan_data['depth_group'].str.replace(' ', '')
    
    # Filter data for specified conditions
    old_filtered = old_data[
        (old_data['location'] == location) & 
        (old_data['depth_group'] == depth_group)
    ]
    
    sid_filtered = sid_data[
        (sid_data['location'] == location) & 
        (sid_data['depth_group'] == depth_group)
    ]
    
    evan_filtered = evan_data[
        (evan_data['location'] == location) & 
        (evan_data['depth_group'] == depth_group)
    ]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Add a main title for both subplots
    fig.suptitle('Comparing raw and processed pre-processed Chl-a time-series', fontsize=16, y=1.02)
    
    # Plot SID comparison on top subplot
    ax1.plot(old_filtered['date'], old_filtered[feature], 
             label='Original', linestyle='-', alpha=0.7)
    ax1.plot(sid_filtered['date'], sid_filtered[feature], 
             label='Feature-Specific Processed', linestyle='-', alpha=0.7)
    ax1.set_ylabel('Chl-a (μg/L)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot EVAN comparison on bottom subplot
    print("\nDebugging EVAN data:")
    print("EVAN filtered columns:", evan_filtered.columns.tolist())
    print("Feature requested:", feature)
    print("EVAN filtered head:", evan_filtered.head())
    
    ax2.plot(old_filtered['date'], old_filtered[feature], 
             label='Original', linestyle='-', alpha=0.7, color='blue')
    try:
        ax2.plot(evan_filtered['date'], evan_filtered[feature], 
                 label='SVD', linestyle='-', alpha=0.7, color='red')
        print("Successfully plotted EVAN data")
    except Exception as e:
        print(f"Error plotting EVAN data: {e}")
    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('Chl-a (μg/L)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'plots/paired_{feature}_{location}_{depth_group.replace("-", "to")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# Example usage:
plot_paired_comparisons('chlorophyll_a', '0-10m', 'WG')
