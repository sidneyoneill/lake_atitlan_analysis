#!/usr/bin/env python
"""
remove_outliers_v2.py - Modified to better detect significant deviations from local trends

This script automatically detects suspect outliers based on manual visual inspection.
For each feature (e.g. chlorophyll_a, dissolved_oxygen) at a given (location, depth_group)
and a specified suspect date/value, a rolling time window (± months) is used to search for records that deviate more than a
user-defined percentage tolerance from the suspect value.

The anomalies (flagged outliers) are saved to a CSV file and time series plots for each grouping are generated.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def detect_suspect_outliers(data_path, suspect_definitions, output_path="output/suspect_anomalies_list.csv"):
    """
    Detect suspect outliers based on user-provided suspect definitions.
    Modified to compare values against local means rather than a fixed value.
    """
    # Read CSV file and convert dates
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    
    suspect_outliers = []
    
    # Process each suspect definition
    for spec in suspect_definitions:
        location = spec['location']
        depth_group = spec['depth_group']
        feature = spec['feature']
        suspect_date = pd.to_datetime(spec['suspect_date'], format='%d/%m/%Y')
        window_months = spec.get('window_months', 6)
        zscore_threshold = spec.get('zscore_threshold', 3.0)

        print(f"\nAnalyzing: {location}, {depth_group}, {feature}, date: {suspect_date}")
        
        # Create a time window (± window_months)
        dt_lower = suspect_date - pd.DateOffset(months=window_months)
        dt_upper = suspect_date + pd.DateOffset(months=window_months)
        
        # Filter data matching the group and within the time window
        group_df = df[
            (df['location'] == location) & 
            (df['depth_group'] == depth_group) &
            (df['date'] >= dt_lower) & 
            (df['date'] <= dt_upper)
        ]
        
        if group_df.empty:
            print(f"WARNING: No data found for this group within time window {dt_lower} to {dt_upper}")
            # Print counts for each filter to help identify the issue
            print(f"Records matching location {location}: {len(df[df['location'] == location])}")
            print(f"Records matching depth_group {depth_group}: {len(df[df['depth_group'] == depth_group])}")
            print(f"Records in time window: {len(df[(df['date'] >= dt_lower) & (df['date'] <= dt_upper)])}")
            continue

        print(f"Found {len(group_df)} records in the time window")

        # Calculate local statistics
        local_mean = group_df[feature].mean()
        local_std = group_df[feature].std()
        
        print(f"Local statistics for {feature}:")
        print(f"  Mean: {local_mean:.2f}")
        print(f"  Std:  {local_std:.2f}")

        # Calculate z-scores
        z_scores = (group_df[feature] - local_mean) / local_std

        # Find outliers based on z-score
        outlier_mask = abs(z_scores) > zscore_threshold
        outlier_count = outlier_mask.sum()
        
        print(f"Found {outlier_count} outliers exceeding z-score threshold of {zscore_threshold}")
        if outlier_count > 0:
            print("Outlier details:")
            outliers = group_df[outlier_mask]
            for _, row in outliers.iterrows():
                print(f"  Date: {row['date']}, Value: {row[feature]:.2f}, Z-score: {z_scores[_]:.2f}")

        # Add outliers to the list
        for idx, row in group_df[outlier_mask].iterrows():
            suspect_outliers.append({
                'location': location,
                'depth_group': depth_group,
                'feature': feature,
                'date': row['date'].strftime('%d/%m/%Y'),
                'observed_value': row[feature],
                'local_mean': local_mean,
                'z_score': z_scores[idx],
                'time_window_start': dt_lower.strftime('%d/%m/%Y'),
                'time_window_end': dt_upper.strftime('%d/%m/%Y')
            })

    # Create a DataFrame for anomalies and write to CSV
    anomalies_df = pd.DataFrame(suspect_outliers)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anomalies_df.to_csv(output_path, index=False)
    print(f"Suspect anomalies saved to {output_path}")
    return anomalies_df

def plot_suspect_outliers(data_path, anomalies_df, output_dir="output/plots_suspect"):
    """
    Plot time series for each feature and group (location, depth_group) where suspect anomalies were flagged.

    Args:
        data_path (str): Path to the original CSV data.
        anomalies_df (pd.DataFrame): DataFrame containing flagged anomalies.
        output_dir (str): Directory to save the plots.
    """
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    
    # Group the anomalies list by (location, depth_group, feature)
    grouped = anomalies_df.groupby(['location', 'depth_group', 'feature'])
    
    for (location, depth_group, feature), group in grouped:
        # Filter the original data for this specific grouping.
        subset = df[(df['location'] == location) & (df['depth_group'] == depth_group)]
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the readings over time.
        ax.plot(subset['date'], subset[feature], 'o-', label='Regular readings', alpha=0.6)
        
        # Mark the suspect anomalies.
        anomaly_dates = pd.to_datetime(group['date'], format='%d/%m/%Y')
        anomaly_values = group['observed_value']
        ax.plot(anomaly_dates, anomaly_values, 'ro', label='Suspect Anomalies', markersize=10)
        
        ax.set_title(f"{feature} at {location}, Depth Group: {depth_group}")
        ax.set_xlabel("Date")
        ax.set_ylabel(feature)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format the x-axis dates.
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        
        # Create directories for the output plot.
        group_dir = os.path.join(output_dir, feature, depth_group, location)
        os.makedirs(group_dir, exist_ok=True)
        plot_file = os.path.join(group_dir, "time_series.png")
        plt.savefig(plot_file, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {plot_file}")

if __name__ == "__main__":
    suspect_definitions = [
        {
            "location": "WP",
            "depth_group": "0-10m",
            "feature": "ammonium",
            "suspect_date": "01/03/2021",  # Approximate date of the suspect outlier
            "window_months": 12,
            "zscore_threshold": 3.0
        },
# SA - 0-10m
        {
            "location": "SA",
            "depth_group": "0-10m",
            "feature": "chlorophyll_a",
            "suspect_date": "01/09/2023",
            "window_months": 12,
            "zscore_threshold": 3.0
        },
        
        # SA - 10-30m
        {
            "location": "SA",
            "depth_group": "10-30m",
            "feature": "chlorophyll_a",
            "suspect_date": "01/09/2023",
            "window_months": 12,
            "zscore_threshold": 3.0
        },

        # WP - 0-10m
        {
            "location": "WP",
            "depth_group": "0-10m",
            "feature": "chlorophyll_a",
            "suspect_date": "01/09/2023",
            "window_months": 12,
            "zscore_threshold": 3.0
        },

        # WP - 10-30m
        {
            "location": "WP",
            "depth_group": "10-30m",
            "feature": "chlorophyll_a",
            "suspect_date": "01/09/2023",
            "window_months": 12,
            "zscore_threshold": 3.0
        },

        # WP - 30m+
        {
            "location": "WP",
            "depth_group": "30m+",
            "feature": "chlorophyll_a",
            "suspect_date": "01/09/2023",
            "window_months": 12,
            "zscore_threshold": 3.0
        },

        # WG - 0-10m
        {
            "location": "WG",
            "depth_group": "0-10m",
            "feature": "chlorophyll_a",
            "suspect_date": "01/09/2023",
            "window_months": 12,
            "zscore_threshold": 3.0
        },

        # WG - 10-30m
        {
            "location": "WG",
            "depth_group": "10-30m",
            "feature": "chlorophyll_a",
            "suspect_date": "01/09/2023",
            "window_months": 12,
            "zscore_threshold": 3.0
        },

        # WG - 30m+
        {
            "location": "WG",
            "depth_group": "30m+",
            "feature": "chlorophyll_a",
            "suspect_date": "01/09/2023",
            "window_months": 12,
            "zscore_threshold": 3.0
        },
        {
            "location": "SA",
            "depth_group": "30m+",
            "feature": "dissolved_oxygen",
            "suspect_date": "01/11/2019",
            "window_months": 12,
            "zscore_threshold": 3.0
        },
        {
            "location": "WG",
            "depth_group": "30m+",
            "feature": "dissolved_oxygen",
            "suspect_date": "01/11/2019",
            "window_months": 12,
            "zscore_threshold": 3.0
        },
        {
            "location": "SA",
            "depth_group": "0-10m",
            "feature": "phosphate",
            "suspect_date": "01/05/2018",
            "window_months": 12,
            "zscore_threshold": 3.0
        },
        {
            "location": "WG",
            "depth_group": "0-10m",
            "feature": "temp",
            "suspect_date": "01/11/2017",
            "window_months": 12,
            "zscore_threshold": 1.5
        },
        {
            "location": "WG",
            "depth_group": "10-30m",
            "feature": "temp",
            "suspect_date": "01/11/2017",  # Approximate based on unknown date
            "window_months": 12,
            "zscore_threshold": 2.0
        },
        {
            "location": "WG",
            "depth_group": "30m+",
            "feature": "temp",
            "suspect_date": "01/11/2017",  # Approximate based on unknown date
            "window_months": 12,
            "zscore_threshold": 3.0
        },
        {
            "location": "SA",
            "depth_group": "0-10m",
            "feature": "turbidity",
            "suspect_date": "01/08/2018",
            "window_months": 12,
            "zscore_threshold": 3.0
        },
        {
            "location": "WG",
            "depth_group": "0-10m",
            "feature": "turbidity",
            "suspect_date": "01/08/2018",  # Approximate based on unknown date
            "window_months": 12,
            "zscore_threshold": 3.0
        }
    ]

    data_file = "data/SID_LIMNO_processed_v5.csv"
    output_anomalies_file = "output/suspect_anomalies_list.csv"

    # Run the suspect outlier detection
    anomalies_df = detect_suspect_outliers(data_file, suspect_definitions, output_anomalies_file)

    # Generate plots that highlight the flagged anomalies
    plot_suspect_outliers(data_file, anomalies_df, output_dir="output/plots_suspect")

    print("Suspect outlier detection completed.")

