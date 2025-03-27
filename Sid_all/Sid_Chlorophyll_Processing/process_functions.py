import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

def create_depth_grid(photic_zone_depth=30, max_depth=250, 
                     photic_interval=1, deep_interval=10):
    """
    Create a depth grid with fine intervals in the photic zone and coarser intervals below.
    
    Parameters:
    -----------
    photic_zone_depth : int
        Maximum depth of the photic zone in meters (default: 30)
    max_depth : int
        Maximum depth of the grid in meters (default: 250)
    photic_interval : int
        Interval between depth points in photic zone in meters (default: 1)
    deep_interval : int
        Interval between depth points below photic zone in meters (default: 10)
    
    Returns:
    --------
    numpy.ndarray
        Array of depth values combining photic and deep zones
    """
    # Create fine-resolution grid for photic zone (0 to photic_zone_depth)
    photic_grid = np.arange(0, photic_zone_depth + photic_interval, photic_interval)
    
    # Create coarser grid for deeper waters (photic_zone_depth to max_depth)
    # Start from the next interval after photic_zone_depth to avoid duplicates
    deep_start = photic_zone_depth + deep_interval
    deep_grid = np.arange(deep_start, max_depth + deep_interval, deep_interval)
    
    # Combine the two grids
    depth_grid = np.concatenate([photic_grid, deep_grid])
    
    return depth_grid

def assign_to_grid_depth(measurement_matrix_path, depth_grid, column_name):
    """
    Assign measurements from the measurement matrix to the closest depth in the grid.
    
    Parameters:
    -----------
    measurement_matrix_path : str
        Path to the measurement matrix CSV file
    depth_grid : numpy.ndarray
        Array of regularly spaced depth values
    column_name : str
        Name of the feature column to process
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with measurements mapped to closest grid depths
    """
    # Read the measurement matrix
    df = pd.read_csv(measurement_matrix_path)
    
    # Debug: Print column names and first few rows
    print("Available columns:", df.columns.tolist())
    print("\nFirst few rows of data:")
    print(df.head())
    
    # Pivot the data to get feature values by depth and date
    df_pivot = df.pivot(index='depth', columns='date', values=column_name)
    
    # Get original depths (already numeric)
    original_depths = df_pivot.index.values
    
    # Find closest grid depth for each original depth
    assigned_depths = np.full_like(original_depths, np.nan, dtype=float)
    mask = (original_depths >= np.min(depth_grid)) & (original_depths <= np.max(depth_grid))
    valid_depths = original_depths[mask]
    
    if len(valid_depths) > 0:
        indices = np.abs(valid_depths[:, np.newaxis] - depth_grid).argmin(axis=1)
        assigned_depths[mask] = depth_grid[indices]
    
    # Create new DataFrame with assigned depths
    new_index = [f"{d:.1f}" for d in assigned_depths]
    df_assigned = df_pivot.copy()
    df_assigned.index = new_index
    
    # Group by assigned depths (in case of duplicates)
    df_assigned = df_assigned.groupby(df_assigned.index).mean()
    
    # Sort by depth (converting index to float for proper numeric sorting)
    df_assigned = df_assigned.reindex(index=[f"{d:.1f}" for d in sorted([float(d) for d in df_assigned.index])])
    
    return df_assigned

def interpolate_vertical_gaps(df, column_name):
    """
    Vertically interpolate missing values within each date,
    only interpolating between existing measurements.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with Assigned_Depth as index and dates as columns
    column_name : str
        Name of the feature being processed (for logging/validation)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with interpolated values, no negative values allowed
    """
    # Create a copy to avoid modifying the original
    df_interp = df.copy()
    
    # Convert string NaN and zeros to actual NaN for interpolation
    df_interp = df_interp.replace(['N/A', 0], np.nan)
    
    # Convert to numeric, coercing any remaining strings to NaN
    df_interp = df_interp.apply(pd.to_numeric, errors='coerce')
    
    # Interpolate vertically (along depths) for each date
    # limit_area='inside' ensures we only interpolate between valid values
    df_interp = df_interp.interpolate(
        method='linear',
        axis=0,
        limit_direction='both',
        limit_area='inside'  # Only interpolate between valid values
    )
    
    # Clip negative values to 0
    df_interp = df_interp.clip(lower=0)
    
    return df_interp

def impute_horizontal_gaps(df, n_neighbors=5, weights='distance'):
    """
    Impute missing values horizontally (across dates) using KNN imputation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with depths as index and dates as columns
    n_neighbors : int
        Number of neighbors to use for KNN imputation (default: 5)
    weights : str
        Weight function used in prediction: 'uniform' or 'distance' (default: 'distance')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with imputed values across dates
    """
    # Create a copy to avoid modifying the original
    df_imputed = df.copy()
    
    # Initialize the KNN imputer
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    
    # Perform imputation
    imputed_data = imputer.fit_transform(df_imputed.T)  # Transpose so dates are rows
    
    # Convert back to DataFrame with original structure
    df_imputed = pd.DataFrame(
        imputed_data.T,  # Transpose back to original orientation
        index=df_imputed.index,
        columns=df_imputed.columns
    )
    
    # Clip negative values to 0
    df_imputed = df_imputed.clip(lower=0)
    
    return df_imputed

def aggregate_depth_groups(df, column_name):
    """
    Aggregate measurements into ecologically relevant depth groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with depths as index and dates as columns
    column_name : str
        Name of the feature being processed
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with aggregated statistics by depth group and date
    """
    # Convert index to float for depth comparisons
    df = df.copy()
    df.index = df.index.astype(float)
    
    # Define depth group ranges
    depth_groups = {
        'Surface': (0, 10),
        'Mid-Depth': (10, 20),
        'Lower Photic': (20, 30),
        'Deep': (30, float('inf'))
    }
    
    # Create a dictionary to store results
    results = []
    
    # Process each date column
    for column in df.columns:
        for group_name, (min_depth, max_depth) in depth_groups.items():
            mask = (df.index >= min_depth) & (df.index < max_depth)
            group_data = df.loc[mask, column]
            
            if not group_data.empty:
                results.append({
                    'date': column,
                    'depth_group': group_name,
                    f'mean_{column_name}': group_data.mean(),
                    f'std_{column_name}': group_data.std(),
                    'measurement_count': group_data.count()
                })
    
    # Convert results to DataFrame
    df_grouped = pd.DataFrame(results)
    
    # Pivot the results for easier analysis
    df_pivot = df_grouped.pivot(
        index='depth_group',
        columns='date',
        values=[f'mean_{column_name}', f'std_{column_name}', 'measurement_count']
    )
    
    return df_grouped, df_pivot

def analyze_original_data(measurement_matrix_path, depth_grid, column_name):
    """
    Analyze original data before interpolation/imputation and generate a report.
    
    Parameters:
    -----------
    measurement_matrix_path : str
        Path to the original measurement matrix CSV file
    depth_grid : numpy.ndarray
        Array of depth values used for grouping
    column_name : str
        Name of the feature being analyzed
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing value analysis by depth group and date
    """
    # Read original data
    df = pd.read_csv(measurement_matrix_path)
    
    # Pivot the data
    df_pivot = df.pivot(index='depth', columns='date', values=column_name)
    
    # Define depth groups
    depth_groups = {
        'Surface': (0, 10),
        'Mid-Depth': (10, 20),
        'Lower Photic': (20, 30),
        'Deep': (30, float('inf'))
    }
    
    # Create results list for analysis
    results = []
    
    # Convert index to numeric
    df_pivot.index = pd.to_numeric(df_pivot.index, errors='coerce')
    
    # Process each date column
    for column in df_pivot.columns:
        for group_name, (min_depth, max_depth) in depth_groups.items():
            # Select depths within the group range
            mask = (df_pivot.index >= min_depth) & (df_pivot.index < max_depth)
            group_data = df_pivot.loc[mask, column]
            
            if not group_data.empty:
                results.append({
                    'date': column,
                    'depth_group': group_name,
                    'total_measurements': len(group_data),
                    'missing_values': group_data.isna().sum(),
                    'missing_proportion': group_data.isna().mean(),
                    f'mean_{column_name}': group_data.mean(),
                    f'std_{column_name}': group_data.std()
                })
    
    # Convert to DataFrame
    df_analysis = pd.DataFrame(results)
    df_analysis['date'] = pd.to_datetime(df_analysis['date'])
    
    # Generate report text
    report = []
    report.append(f"=== Original {column_name} Data Quality Report ===\n")
    
    # Overall statistics
    total_measurements = df.size
    total_missing = df.isna().sum().sum()
    report.append("Overall Statistics:")
    report.append(f"Total measurements possible: {total_measurements}")
    report.append(f"Total missing values: {total_missing}")
    report.append(f"Overall missing proportion: {(total_missing/total_measurements):.3f}")
    
    # Depth group statistics
    report.append("\nMissing Values by Depth Group:")
    group_stats = df_analysis.groupby('depth_group').agg({
        'missing_proportion': ['mean', 'min', 'max'],
        'total_measurements': 'mean',
        f'mean_{column_name}': ['mean', 'std', 'min', 'max'],
        f'std_{column_name}': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    for group in depth_groups.keys():
        stats = group_stats.loc[group]
        report.append(f"\n{group} Layer:")
        report.append(f"  Missing values (mean): {stats['missing_proportion']['mean']:.3f}")
        report.append(f"  Missing values (range): {stats['missing_proportion']['min']:.3f} - {stats['missing_proportion']['max']:.3f}")
        report.append(f"  Average measurements per date: {stats['total_measurements']['mean']:.1f}")
        report.append(f"  Mean {column_name}: {stats[f'mean_{column_name}']['mean']:.3f} ± {stats[f'mean_{column_name}']['std']:.3f}")
        report.append(f"  {column_name} range: {stats[f'mean_{column_name}']['min']:.3f} - {stats[f'mean_{column_name}']['max']:.3f}")
    
    # Save report with feature name
    report_text = '\n'.join(report)
    with open(f'output/original_data_quality_report_{column_name}.txt', 'w') as f:
        f.write(report_text)
    
    return df_analysis