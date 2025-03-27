from visualize_functions import (plot_depth_group_timeseries,
                                 plot_surface_middepth_comparison,
                                 create_data_quality_report,
                                 plot_feature_heatmap)
import numpy as np
import pandas as pd
import os
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

def assign_to_grid_depth(measurement_matrix_path, depth_grid, feature_name):
    """
    Assign measurements from the measurement matrix to the closest depth in the grid.
    
    Parameters:
    -----------
    measurement_matrix_path : str
        Path to the measurement matrix CSV file
    depth_grid : numpy.ndarray
        Array of regularly spaced depth values
    feature_name : str
        Name of the feature column to process (e.g., 'chlorophyll', 'temp')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with measurements mapped to closest grid depths
    """
    # Read the measurement matrix
    df = pd.read_csv(measurement_matrix_path, index_col='depth')
    
    # Verify feature exists in the dataframe
    if feature_name not in measurement_matrix_path:
        raise ValueError(f"Feature '{feature_name}' not found in measurement matrix path")
    
    # Get original depths (already numeric)
    original_depths = df.index.values
    
    # Find closest grid depth for each original depth
    assigned_depths = np.full_like(original_depths, np.nan, dtype=float)
    mask = (original_depths >= np.min(depth_grid)) & (original_depths <= np.max(depth_grid))
    valid_depths = original_depths[mask]
    
    if len(valid_depths) > 0:
        indices = np.abs(valid_depths[:, np.newaxis] - depth_grid).argmin(axis=1)
        assigned_depths[mask] = depth_grid[indices]
    
    # Create new DataFrame with assigned depths
    new_index = [f"{d:.1f}" for d in assigned_depths]
    df_assigned = df.copy()
    df_assigned.index = new_index
    
    # Group by assigned depths (in case of duplicates)
    df_assigned = df_assigned.groupby(df_assigned.index).mean()
    
    # Sort by depth (converting index to float for proper numeric sorting)
    df_assigned = df_assigned.reindex(index=[f"{d:.1f}" for d in sorted([float(d) for d in df_assigned.index])])
    
    return df_assigned

def interpolate_vertical_gaps(df):
    """
    Vertically interpolate missing chlorophyll values within each date,
    only interpolating between existing measurements.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with Assigned_Depth as index and dates as columns
    
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

def aggregate_depth_groups(df, feature_name):
    """
    Aggregate measurements into ecologically relevant depth groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with depths as index and dates as columns
    feature_name : str
        Name of the feature being analyzed
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
            # Select depths within the group range
            mask = (df.index >= min_depth) & (df.index < max_depth)
            group_data = df.loc[mask, column]
            
            if not group_data.empty:
                results.append({
                    'date': column,
                    'depth_group': group_name,
                    f'mean_{feature_name}': group_data.mean(),
                    f'std_{feature_name}': group_data.std(),
                    'measurement_count': group_data.count()
                })
    
    # Convert results to DataFrame
    df_grouped = pd.DataFrame(results)
    
    # Pivot the results for easier analysis
    df_pivot = df_grouped.pivot(
        index='depth_group',
        columns='date',
        values=[f'mean_{feature_name}', f'std_{feature_name}', 'measurement_count']
    )
    
    return df_grouped, df_pivot

def analyze_original_data(measurement_matrix_path, depth_grid, feature_name):
    """
    Analyze original data before interpolation/imputation and generate a report.
    
    Parameters:
    -----------
    measurement_matrix_path : str
        Path to the original measurement matrix CSV file
    depth_grid : numpy.ndarray
        Array of depth values used for grouping
    feature_name : str
        Name of the feature being analyzed (e.g., 'chlorophyll', 'temp')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing value analysis by depth group and date
    """
    # Read original data
    df = pd.read_csv(measurement_matrix_path, index_col='depth')
    
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
    df.index = pd.to_numeric(df.index, errors='coerce')
    
    # Process each date column
    for column in df.columns:
        for group_name, (min_depth, max_depth) in depth_groups.items():
            # Select depths within the group range
            mask = (df.index >= min_depth) & (df.index < max_depth)
            group_data = df.loc[mask, column]
            
            if not group_data.empty:
                results.append({
                    'date': column,
                    'depth_group': group_name,
                    'total_measurements': len(group_data),
                    'missing_values': group_data.isna().sum(),
                    'missing_proportion': group_data.isna().mean(),
                    f'mean_{feature_name}': group_data.mean(),
                    f'std_{feature_name}': group_data.std()
                })
    
    # Convert to DataFrame
    df_analysis = pd.DataFrame(results)
    df_analysis['date'] = pd.to_datetime(df_analysis['date'])
    
    # Generate report text
    report = []
    report.append(f"=== Original {feature_name.capitalize()} Data Quality Report ===\n")
    
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
        f'mean_{feature_name}': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    for group in depth_groups.keys():
        stats = group_stats.loc[group]
        report.append(f"\n{group} Layer:")
        report.append(f"  Missing values (mean): {stats['missing_proportion']['mean']:.3f}")
        report.append(f"  Missing values (range): {stats['missing_proportion']['min']:.3f} - {stats['missing_proportion']['max']:.3f}")
        report.append(f"  Average measurements per date: {stats['total_measurements']['mean']:.1f}")
        report.append(f"  Mean {feature_name}: {stats[f'mean_{feature_name}']['mean']:.3f} ± {stats[f'mean_{feature_name}']['std']:.3f}")
        report.append(f"  {feature_name.capitalize()} range: {stats[f'mean_{feature_name}']['min']:.3f} - {stats[f'mean_{feature_name}']['max']:.3f}")
    
    # Save report
    report_text = '\n'.join(report)
    with open(f'output/{feature_name}_original_data_quality_report.txt', 'w') as f:
        f.write(report_text)
    
    return df_analysis

def add_missing_elements(df1, df2, feature1_name, feature2_name):
    """
    Add missing rows and columns to each dataframe with NaN values.
    
    Parameters:
    -----------
    df1, df2 : pandas.DataFrame
        The dataframes to align
    feature1_name, feature2_name : str
        Names of the features for logging purposes
    
    Returns:
    --------
    tuple
        (df1_updated, df2_updated) with aligned shapes and indices
    """
    # Create copies to avoid modifying original dataframes
    df1_updated = df1.copy()
    df2_updated = df2.copy()
    
    # Find missing elements
    rows_in_1_not_2 = set(df1.index) - set(df2.index)
    rows_in_2_not_1 = set(df2.index) - set(df1.index)
    cols_in_1_not_2 = set(df1.columns) - set(df2.columns)
    cols_in_2_not_1 = set(df2.columns) - set(df1.columns)
    
    # Add missing rows
    for rows, source_df, target_df, source_name, target_name in [
        (rows_in_1_not_2, df1, df2_updated, feature1_name, feature2_name),
        (rows_in_2_not_1, df2, df1_updated, feature2_name, feature1_name)
    ]:
        if rows:
            print(f"\nRows in {source_name} but not in {target_name}:")
            print(sorted(rows))
            print(f"Adding missing rows to {target_name} from {source_name}:")
            for row in sorted(rows):
                print(f"  - Adding row '{row}' with NaN values")
                target_df.loc[row] = pd.Series(np.nan, index=target_df.columns)
    
    # Add missing columns
    for cols, source_df, target_df, source_name, target_name in [
        (cols_in_1_not_2, df1, df2_updated, feature1_name, feature2_name),
        (cols_in_2_not_1, df2, df1_updated, feature2_name, feature1_name)
    ]:
        if cols:
            print(f"\nColumns in {source_name} but not in {target_name}:")
            print(sorted(cols))
            print(f"Adding missing columns to {target_name} from {source_name}:")
            for col in sorted(cols):
                print(f"  - Adding column '{col}' with NaN values")
                target_df[col] = pd.Series(np.nan, index=target_df.index)
    
    # Ensure alignment
    all_columns = sorted(set(df1_updated.columns) | set(df2_updated.columns))
    all_indices = sorted(set(df1_updated.index) | set(df2_updated.index))
    
    df1_updated = df1_updated.reindex(index=all_indices, columns=all_columns)
    df2_updated = df2_updated.reindex(index=all_indices, columns=all_columns)
    
    if any([rows_in_1_not_2, rows_in_2_not_1, cols_in_1_not_2, cols_in_2_not_1]):
        print("\nMatrices have been aligned successfully.")
    
    return df1_updated, df2_updated

def compare_shapes(df1, df2, location, feature1_name, feature2_name):
    """
    Compare shapes of two feature matrices for a given location.
    
    Parameters:
    -----------
    df1, df2 : pandas.DataFrame
        The dataframes to compare
    location : str
        Location identifier
    feature1_name, feature2_name : str
        Names of the features being compared
    
    Returns:
    --------
    dict
        Comparison results including shapes and missing elements
    """
    shape1 = df1.shape
    shape2 = df2.shape
    
    print(f"\n{location} Matrix Shapes:")
    print(f"{feature1_name}: {shape1} (rows × columns)")
    print(f"{feature2_name}: {shape2} (rows × columns)")
    
    if shape1 == shape2:
        print("✓ Matrices have matching shapes")
        print("No updates needed")
        return {
            feature1_name: {'shape': shape1},
            feature2_name: {'shape': shape2},
            'matched': True,
            f'missing_rows_{feature1_name.lower()}': [],
            f'missing_rows_{feature2_name.lower()}': [],
            f'missing_cols_{feature1_name.lower()}': [],
            f'missing_cols_{feature2_name.lower()}': []
        }
    
    print("✗ Matrices have different shapes")
    
    # Identify missing elements and align matrices
    df1_updated, df2_updated = add_missing_elements(df1, df2, feature1_name, feature2_name)
    
    return {
        feature1_name: {'shape': df1_updated.shape},
        feature2_name: {'shape': df2_updated.shape},
        'matched': df1_updated.shape == df2_updated.shape,
        f'missing_rows_{feature1_name.lower()}': list(set(df1.index) - set(df2.index)),
        f'missing_rows_{feature2_name.lower()}': list(set(df2.index) - set(df1.index)),
        f'missing_cols_{feature1_name.lower()}': list(set(df1.columns) - set(df2.columns)),
        f'missing_cols_{feature2_name.lower()}': list(set(df2.columns) - set(df1.columns))
    }

def compare_feature_assigned_matrices(location, features):
    """
    Compare and align assigned_matrix.csv files of different features for a given location.
    
    Parameters:
    -----------
    location : str
        Location identifier
    features : list
        List of feature names to compare (e.g., ['temp', 'chlorophyll_a'])
    
    Returns:
    --------
    dict
        Dictionary containing comparison results and alignment actions
    """
    if len(features) != 2:
        raise ValueError("Exactly two features must be provided for comparison")
    
    # Initialize matrices and results
    matrices = {}
    feature_display_names = {}
    
    # Create mapping for display names (can be extended for more features)
    display_name_map = {
        'temp': 'Temperature',
        'chlorophyll_a': 'Chlorophyll-a',
        # Add more mappings as needed
    }
    
    # Initialize comparison result with dynamic feature names
    comparison_result = {
        'matched': True
    }
    
    # Load matrices
    for feature in features:
        display_name = display_name_map.get(feature, feature.capitalize())
        feature_display_names[feature] = display_name
        filepath = f'output/{location}/{feature}/assigned_matrix.csv'
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0)
            matrices[display_name] = df
            comparison_result[display_name] = {'shape': df.shape}
        else:
            print(f"Assigned matrix for feature '{feature}' at location '{location}' does not exist.")
            return comparison_result
    
    # Get display names for the two features
    feature1_name = feature_display_names[features[0]]
    feature2_name = feature_display_names[features[1]]
    
    # Compare shapes and get updated matrices
    df1 = matrices[feature1_name]
    df2 = matrices[feature2_name]
    
    comparison = compare_shapes(df1, df2, location, feature1_name, feature2_name)
    comparison_result.update(comparison)
    
    # Always perform the alignment, regardless of 'matched' status
    df1_updated, df2_updated = add_missing_elements(df1, df2, feature1_name, feature2_name)
    
    # Save updated matrices
    for feature, df_updated in zip(features, [df1_updated, df2_updated]):
        output_path = f'output/{location}/{feature}/assigned_matrix.csv'
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save with index
        df_updated.to_csv(output_path)
        print(f"Saved {feature_display_names[feature]} matrix to: {output_path}")
        
        # Verify the save by reading it back
        loaded_df = pd.read_csv(output_path, index_col=0)
        print(f"Verified {feature_display_names[feature]} shape after save: {loaded_df.shape}")
        
        # Update the comparison result with final shapes
        comparison_result[feature_display_names[feature]]['shape'] = loaded_df.shape
    
    # Verify final alignment
    comparison_result['matched'] = (df1_updated.shape == df2_updated.shape)
    
    if not comparison_result['matched']:
        print("WARNING: Matrices still don't match after alignment!")
        print(f"Shape 1: {df1_updated.shape}, Shape 2: {df2_updated.shape}")
    
    return comparison_result

def add_missing_elements_multi(df1, df2, feature1_name, feature2_name):
    """
    Add missing rows and columns to each dataframe with NaN values.
    Similar to original function but returns whether updates were made.
    
    Parameters:
    -----------
    df1, df2 : pandas.DataFrame
        The dataframes to align
    feature1_name, feature2_name : str
        Names of the features for logging purposes
    
    Returns:
    --------
    tuple
        (df1_updated, df2_updated, updates_made)
    """
    # Create copies to avoid modifying original dataframes
    df1_updated = df1.copy()
    df2_updated = df2.copy()
    
    # Track if any updates were made
    updates_made = False
    
    # Find missing elements
    rows_in_1_not_2 = set(df1.index) - set(df2.index)
    rows_in_2_not_1 = set(df2.index) - set(df1.index)
    cols_in_1_not_2 = set(df1.columns) - set(df2.columns)
    cols_in_2_not_1 = set(df2.columns) - set(df1.columns)
    
    # Add missing rows and columns
    for rows, cols, source_df, target_df, source_name, target_name in [
        (rows_in_1_not_2, cols_in_1_not_2, df1, df2_updated, feature1_name, feature2_name),
        (rows_in_2_not_1, cols_in_2_not_1, df2, df1_updated, feature2_name, feature1_name)
    ]:
        if rows or cols:
            updates_made = True
            if rows:
                print(f"\nRows in {source_name} but not in {target_name}:")
                print(sorted(rows))
                for row in sorted(rows):
                    target_df.loc[row] = pd.Series(np.nan, index=target_df.columns)
            
            if cols:
                print(f"\nColumns in {source_name} but not in {target_name}:")
                print(sorted(cols))
                for col in sorted(cols):
                    target_df[col] = pd.Series(np.nan, index=target_df.index)
    
    # Ensure alignment
    all_columns = sorted(set(df1_updated.columns) | set(df2_updated.columns))
    all_indices = sorted(set(df1_updated.index) | set(df2_updated.index))
    
    df1_updated = df1_updated.reindex(index=all_indices, columns=all_columns)
    df2_updated = df2_updated.reindex(index=all_indices, columns=all_columns)
    
    return df1_updated, df2_updated, updates_made

def compare_feature_matrices_multi(location, features):
    """
    Compare and align multiple feature matrices for a given location.
    
    Parameters:
    -----------
    location : str
        Location identifier
    features : list
        List of feature names to compare
    
    Returns:
    --------
    dict
        Dictionary containing the final aligned matrices and comparison results
    """
    if len(features) < 2:
        raise ValueError("At least two features must be provided for comparison")
    
    # Initialize matrices and display names
    matrices = {}
    display_name_map = {
        'temp': 'Temperature',
        'chlorophyll_a': 'Chlorophyll-a',
        # Add more mappings as needed
    }
    
    # Load initial matrices
    for feature in features:
        display_name = display_name_map.get(feature, feature.capitalize())
        filepath = f'output/{location}/{feature}/assigned_matrix.csv'
        
        if os.path.exists(filepath):
            matrices[feature] = {
                'data': pd.read_csv(filepath, index_col=0),
                'display_name': display_name
            }
        else:
            print(f"Assigned matrix for feature '{feature}' at location '{location}' does not exist.")
            return None
    
    def compare_pair(feature1, feature2, matrices_dict):
        """Helper function to compare and align a pair of matrices"""
        print(f"\nComparing {matrices_dict[feature1]['display_name']} with {matrices_dict[feature2]['display_name']}")
        
        df1, df2, updates_made = add_missing_elements_multi(
            matrices_dict[feature1]['data'],
            matrices_dict[feature2]['data'],
            matrices_dict[feature1]['display_name'],
            matrices_dict[feature2]['display_name']
        )
        
        if updates_made:
            matrices_dict[feature1]['data'] = df1
            matrices_dict[feature2]['data'] = df2
            return True
        return False
    
    # Recursive comparison function
    def recursive_compare(features_list, matrices_dict, iteration=1):
        if iteration > 10:  # Prevent infinite loops
            print("Warning: Maximum iterations reached. Some matrices might not be fully aligned.")
            return matrices_dict
        
        print(f"\nIteration {iteration}")
        updates_made = False
        
        # Compare all possible pairs
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                feature1, feature2 = features_list[i], features_list[j]
                if compare_pair(feature1, feature2, matrices_dict):
                    updates_made = True
        
        # If any updates were made, recurse to ensure all matrices are aligned
        if updates_made:
            return recursive_compare(features_list, matrices_dict, iteration + 1)
        return matrices_dict
    
    # Start recursive comparison
    final_matrices = recursive_compare(features, matrices)
    
    # Save final aligned matrices
    for feature in features:
        output_path = f'output/{location}/{feature}/assigned_matrix.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_matrices[feature]['data'].to_csv(output_path)
        print(f"\nSaved aligned {final_matrices[feature]['display_name']} matrix to: {output_path}")
        
        # Verify the save
        loaded_df = pd.read_csv(output_path, index_col=0)
        print(f"Verified {final_matrices[feature]['display_name']} shape after save: {loaded_df.shape}")
    
    # Verify final alignment
    shapes = [final_matrices[feature]['data'].shape for feature in features]
    all_aligned = all(shape == shapes[0] for shape in shapes)
    
    if not all_aligned:
        print("\nWARNING: Not all matrices are aligned after processing!")
        for feature in features:
            print(f"{final_matrices[feature]['display_name']}: {final_matrices[feature]['data'].shape}")
    else:
        print("\nAll matrices successfully aligned!")
    
    return final_matrices

def create_removed_values_matrix(matrix, removal_percentage=0.05):
    """
    Create a matrix where random values are removed from the input matrix
    and stored in a separate matrix with the same shape.
    
    Parameters:
    -----------
    matrix : pandas.DataFrame
        Input matrix with values
    removal_percentage : float
        Percentage of values to remove (default: 0.05 for 5%)
    
    Returns:
    --------
    tuple
        (modified_matrix, removed_values_matrix) where removed values are stored
        in removed_values_matrix and replaced with NaN in modified_matrix
    """
    # Create a copy of the input matrix
    matrix_copy = matrix.copy()
    removed_matrix = pd.DataFrame(np.nan, index=matrix.index, columns=matrix.columns)
    
    # Get non-null values
    non_null_mask = matrix_copy.notna()
    non_null_coords = list(zip(*non_null_mask.values.nonzero()))
    
    # Calculate number of values to remove
    n_values = len(non_null_coords)
    n_remove = int(n_values * removal_percentage)
    
    # Randomly select indices to remove
    remove_indices = np.random.choice(len(non_null_coords), n_remove, replace=False)
    
    # Remove values and store them in removed_matrix
    for idx in remove_indices:
        i, j = non_null_coords[idx]
        removed_value = matrix_copy.iloc[i, j]
        removed_matrix.iloc[i, j] = removed_value
        matrix_copy.iloc[i, j] = np.nan
    
    return matrix_copy, removed_matrix

def calculate_validation_metrics(imputed_matrix, removed_matrix):
    """
    Calculate RMSE between imputed and actual values.
    
    Parameters:
    -----------
    imputed_matrix : pandas.DataFrame
        Matrix with imputed values
    removed_matrix : pandas.DataFrame
        Matrix containing the actual values that were removed
    
    Returns:
    --------
    dict
        Dictionary containing RMSE
    """
    from sklearn.metrics import mean_squared_error
    
    # Get actual and predicted values where values were removed
    mask = ~removed_matrix.isna()
    actual = removed_matrix[mask].values.flatten()
    predicted = imputed_matrix[mask].values.flatten()
    
    # Remove any remaining NaN values
    valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]
    
    # Check if we have any valid pairs for comparison
    if len(actual) == 0 or len(predicted) == 0:
        return {'rmse': np.nan}
    
    # Calculate RMSE
    try:
        rmse = np.sqrt(mean_squared_error(actual, predicted))
    except Exception as e:
        print(f"Error calculating RMSE: {str(e)}")
        return {'rmse': np.nan}
    
    return {'rmse': rmse}

def process_location(location, depth_grid, feature_name, removal_percentage=0.05, n_validation_runs=10):
    """
    Process feature data for a specific location with multiple validation runs.
    
    Parameters:
    -----------
    location : str
        Location identifier
    depth_grid : numpy.ndarray
        Array of depth values
    feature_name : str
        Name of the feature being processed
    removal_percentage : float
        Percentage of values to remove for validation (default: 0.05 for 5%)
    n_validation_runs : int
        Number of validation runs to perform (default: 10)
    """
    # Create feature-specific output directory with figures subdirectory
    os.makedirs(f'output/{location}/{feature_name}/figures', exist_ok=True)
    
    # Measurement matrix path
    measurement_matrix_path = f'output/{location}/{feature_name}/measurement_matrix.csv'

    # Analyze original data
    analyze_original_data(measurement_matrix_path, depth_grid, feature_name)

    # First create the assigned matrix
    df_assigned = assign_to_grid_depth(measurement_matrix_path, depth_grid, feature_name)
    
    # Save the assigned matrix
    assigned_matrix_path = f'output/{location}/{feature_name}/assigned_matrix.csv'
    df_assigned.to_csv(assigned_matrix_path)
    
    # Perform multiple validation runs
    validation_results = []
    final_imputed = None
    
    print(f"\nPerforming {n_validation_runs} validation runs for {location}...")
    
    for run in range(n_validation_runs):
        print(f"Run {run + 1}/{n_validation_runs}")
        
        # Create removed values matrix
        modified_matrix, removed_matrix = create_removed_values_matrix(df_assigned, removal_percentage)
        
        # Process the modified matrix
        df_interpolated = interpolate_vertical_gaps(modified_matrix)
        df_imputed = impute_horizontal_gaps(df_interpolated, n_neighbors=5, weights='distance')
        
        # Calculate validation metrics for this run
        metrics = calculate_validation_metrics(df_imputed, removed_matrix)
        metrics['run'] = run + 1
        validation_results.append(metrics)
        
        # Save the final imputed matrix from the last run
        if run == n_validation_runs - 1:
            final_imputed = df_imputed
            removed_matrix.to_csv(f'output/{location}/{feature_name}/removed_matrix.csv')
            df_imputed.to_csv(f'output/{location}/{feature_name}/imputed_matrix.csv')
    
    # Save validation results
    df_validation = pd.DataFrame(validation_results)
    df_validation.to_csv(f'output/{location}/{feature_name}/validation_results.csv', index=False)
    
    # Calculate and print average metrics
    print("\nValidation Results (averaged over all runs):")
    mean_metrics = df_validation.mean()
    std_metrics = df_validation.std()
    print(f"RMSE: {mean_metrics['rmse']:.4f} ± {std_metrics['rmse']:.4f}")
    
    # Process final results
    df_grouped, df_pivot = aggregate_depth_groups(final_imputed, feature_name)
    df_grouped.to_csv(f'output/{location}/{feature_name}/depth_groups_long.csv', index=False)
    df_pivot.to_csv(f'output/{location}/{feature_name}/depth_groups_pivot.csv')
    
    # Create visualizations and quality report
    plot_feature_heatmap(final_imputed, feature_name, 
                        output_path=f'output/{location}/{feature_name}/figures/heatmap.png')
    plot_depth_group_timeseries(df_grouped, feature_name, 
                               output_path=f'output/{location}/{feature_name}/figures/depth_group_timeseries.png')
    plot_surface_middepth_comparison(df_grouped, feature_name, 
                                   output_path=f'output/{location}/{feature_name}/figures/surface_middepth_comparison.png')
    report = create_data_quality_report(df_assigned, df_interpolated, final_imputed, df_grouped, feature_name)
    
    # Save report
    with open(f'output/{location}/{feature_name}/data_quality_report.txt', 'w') as f:
        f.write(report)
    
    return final_imputed, df_grouped, df_pivot, df_validation

def combine_depth_groups(locations, feature_name):
    """
    Combine depth group data from all locations into a single DataFrame for a specific feature.

    Parameters:
    -----------
    locations : list
        List of location identifiers
    feature_name : str
        Name of the feature being processed

    Returns:
    --------
    pandas.DataFrame
        Combined depth groups data from all locations
    """
    # List to store DataFrames from each location
    dfs = []
    
    # Read and combine data from each location
    for location in locations:
        try:
            # Read the depth groups long format file
            df = pd.read_csv(f'output/{location}/{feature_name}/depth_groups_long.csv')
            
            # Add location column
            df['location'] = location
            
            # Append to list
            dfs.append(df)
        except FileNotFoundError:
            print(f"Depth groups file not found for feature '{feature_name}' at location '{location}'. Skipping.")
            continue
    
    if not dfs:
        print(f"No depth group data available to combine for feature '{feature_name}'.")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Combine all DataFrames
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Reorder columns to put location first
    cols = ['location'] + [col for col in df_combined.columns if col != 'location']
    df_combined = df_combined[cols]
    
    return df_combined

