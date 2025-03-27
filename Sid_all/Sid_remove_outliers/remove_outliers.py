import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os

def replace_outliers_with_adjacent_means(data_path, output_path, zscore_threshold=2):
    """
    Replace outliers with mean of adjacent time values for same location and depth.
    
    Args:
        data_path: Path to input CSV file
        output_path: Path to save processed CSV file
        zscore_threshold: Z-score threshold for outlier detection (default=2)
    
    Returns:
        DataFrame with replaced outliers and list of outliers found
    """
    # Read the CSV file
    df = pd.read_csv(data_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    
    # Sort by location, depth_group, and date
    df = df.sort_values(['location', 'depth_group', 'date'])
    
    # Select only numerical columns (excluding date and location columns)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Store outlier information
    outliers_info = []
    
    # For each numerical column
    for col in numerical_cols:
        # Calculate z-scores within each location-depth group
        for (loc, depth), group in df.groupby(['location', 'depth_group']):
            # Calculate z-scores for this specific location and depth group
            z_scores = (group[col] - group[col].mean()) / group[col].std()
            
            # Identify outliers using z-scores
            outlier_mask = abs(z_scores) > zscore_threshold
            
            # For each outlier
            for idx in group[outlier_mask].index:
                # Get original value and date
                date = df.loc[idx, 'date']
                orig_value = df.loc[idx, col]
                orig_zscore = z_scores[idx]
                
                # Get previous and next values for same location and depth
                same_group = group  # We already have the filtered group for this location and depth
                prev_value = same_group[same_group['date'] < date][col].iloc[-1] if len(same_group[same_group['date'] < date]) > 0 else None
                next_value = same_group[same_group['date'] > date][col].iloc[0] if len(same_group[same_group['date'] > date]) > 0 else None
                
                # Calculate replacement value
                if prev_value is not None and next_value is not None:
                    replacement = (prev_value + next_value) / 2
                elif prev_value is not None:
                    replacement = prev_value
                elif next_value is not None:
                    replacement = next_value
                else:
                    replacement = group[col].mean()  # Use mean of this location-depth group
                
                # Store outlier information
                outliers_info.append({
                    'location': loc,
                    'depth_group': depth,
                    'date': date,
                    'feature': col,
                    'original_value': orig_value,
                    'z_score': orig_zscore,
                    'replacement_value': replacement
                })
                
                # Replace the outlier
                df.loc[idx, col] = replacement
    
    # Convert date back to original format before saving
    df['date'] = df['date'].dt.strftime('%d/%m/%Y')
    
    # Create outliers DataFrame
    outliers_df = pd.DataFrame(outliers_info)
    
    # Save processed data and outliers list
    df.to_csv(output_path, index=False)
    outliers_df.to_csv('output/outliers_list.csv', index=False)
    
    # Count outliers by feature
    outlier_counts = outliers_df['feature'].value_counts()
    
    # Save summary to file
    with open('output/outliers_summary.txt', 'w') as f:
        f.write("Number of outliers found in each feature:\n")
        for feature, count in outlier_counts.items():
            f.write(f"{feature}: {count} outliers\n")
    
    return df, outliers_df

def plot_time_series_with_outliers(original_data, outliers_df, output_dir='output/plots'):
    """
    Create time series plots for each feature and depth group, with separate subplots.
    
    Args:
        original_data: Path to original CSV file
        outliers_df: DataFrame containing outlier information
        output_dir: Directory to save plots
    """
    # Read original data
    df = pd.read_csv(original_data)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create plots for each feature
    for feature in numerical_cols:
        # Create feature directory
        feature_dir = os.path.join(output_dir, feature)
        
        # Get unique depth groups and locations
        depth_groups = sorted(df['depth_group'].unique())
        locations = sorted(df['location'].unique())
        
        # For each depth group
        for depth in depth_groups:
            # Create depth group directory
            depth_dir = os.path.join(feature_dir, depth)
            
            # For each location
            for loc in locations:
                # Create location directory
                loc_dir = os.path.join(depth_dir, loc)
                os.makedirs(loc_dir, exist_ok=True)
                
                # Create figure for this specific combination
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Get data for this location and depth
                mask = (df['location'] == loc) & (df['depth_group'] == depth)
                location_depth_data = df[mask]
                
                # Plot regular points
                ax.plot(location_depth_data['date'], location_depth_data[feature], 'o-', 
                       label='Regular readings', alpha=0.6)
                
                # Get outliers for this feature, depth and location
                feature_outliers = outliers_df[
                    (outliers_df['feature'] == feature) & 
                    (outliers_df['depth_group'] == depth) &
                    (outliers_df['location'] == loc)
                ]
                
                # Plot outliers in red
                if not feature_outliers.empty:
                    ax.plot(
                        pd.to_datetime(feature_outliers['date']),
                        feature_outliers['original_value'],
                        'ro',
                        label='Outliers',
                        markersize=10
                    )
                
                ax.set_title(f'{feature} at {loc}, Depth Group: {depth}')
                ax.set_xlabel('Date')
                ax.set_ylabel(feature)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Format x-axis dates
                ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
                # Adjust layout and save
                plt.tight_layout()
                plt.savefig(os.path.join(loc_dir, 'time_series.png'), bbox_inches='tight')
                plt.close()

# Run the outlier detection first
processed_df, outliers_list = replace_outliers_with_adjacent_means(
    'data/SID_LIMNO_processed_v5.csv',
    'output/SID_LIMNO_processed_v6.csv'
)

# Create the plots
plot_time_series_with_outliers(
    'data/SID_LIMNO_processed_v5.csv',
    outliers_list,
    'output/plots'
)
