import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def plot_data_comparison(feature, depth_group, location):
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Read the datasets
    old_data = pd.read_csv('output/grouped_depths.csv')
    sid_data = pd.read_csv('data/SID_LIMNO_no_outliers_v2.csv')
    evan_data = pd.read_csv('data/Lake_data_clean_final_v3.csv')
    
    # Convert date formats and sort
    old_data['date'] = pd.to_datetime(old_data['date'])
    sid_data['date'] = pd.to_datetime(sid_data['date'])
    evan_data['date'] = pd.to_datetime(evan_data['date'])
    
    # Print date ranges for debugging
    print("\nDate ranges in each dataset:")
    print(f"Old data: {old_data['date'].min()} to {old_data['date'].max()}")
    print(f"SID data: {sid_data['date'].min()} to {sid_data['date'].max()}")
    print(f"EVAN data: {evan_data['date'].min()} to {evan_data['date'].max()}")
    
    # Sort all datasets by date
    old_data = old_data.sort_values('date')
    sid_data = sid_data.sort_values('date')
    evan_data = evan_data.sort_values('date')
    
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original data on top subplot
    ax1.plot(old_filtered['date'], old_filtered[feature], 
             label='Original', linestyle='-', alpha=0.7)
    ax1.set_title(f'Original {feature} at {location} ({depth_group})')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(feature)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot imputed data on bottom subplot
    ax2.plot(sid_filtered['date'], sid_filtered[feature], 
             label='SID Imputed', linestyle='-', alpha=0.7)
    ax2.plot(evan_filtered['date'], evan_filtered[feature], 
             label='EVAN Imputed', linestyle='-', alpha=0.7)
    ax2.set_title(f'Imputed {feature} at {location} ({depth_group})')
    ax2.set_xlabel('Date')
    ax2.set_ylabel(feature)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'plots/{feature}_{location}_{depth_group.replace("-", "to")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# Example usage:
plot_data_comparison('temp', '0-10m', 'WG')
