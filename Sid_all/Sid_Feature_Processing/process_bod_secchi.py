import pandas as pd
import os
from pathlib import Path
import numpy as np
from sklearn.impute import KNNImputer

def load_measurement_matrix(feature, location):
    """Load measurement matrix for a given feature and location."""
    filepath = Path(f"output/{location}/{feature}/measurement_matrix.csv")
    if not filepath.exists():
        raise FileNotFoundError(f"No measurement matrix found for {feature} at {filepath}")
    return pd.read_csv(filepath)

def compare_dates_with_master(feature_dates, master_dates):
    """Compare dates from a feature against master dates list."""
    feature_dates = set(pd.to_datetime(feature_dates))
    master_dates = set(pd.to_datetime(master_dates))
    
    missing_from_feature = master_dates - feature_dates
    extra_in_feature = feature_dates - master_dates
    
    return {
        'missing_dates_count': len(missing_from_feature),
        'extra_dates_count': len(extra_in_feature),
        'missing_dates': sorted(list(missing_from_feature)),
        'extra_dates': sorted(list(extra_in_feature))
    }

def align_matrix_with_master_dates(matrix, master_dates):
    """
    Align measurement matrix with master dates by adding missing dates as empty columns.
    
    Args:
        matrix (pd.DataFrame): Original measurement matrix
        master_dates (pd.Series): Master list of dates
    
    Returns:
        pd.DataFrame: Aligned matrix with all master dates
    """
    # Convert master dates to datetime
    master_dates = pd.to_datetime(master_dates)
    
    # Get current dates (excluding first column which typically contains station names)
    current_dates = pd.to_datetime(matrix.columns[1:])
    
    # Create new column list with station names column and all master dates
    new_columns = [matrix.columns[0]] + [d.strftime('%Y-%m-%d') for d in sorted(master_dates)]
    
    # Create new empty DataFrame with desired columns
    aligned_matrix = pd.DataFrame(columns=new_columns)
    
    # Copy station names from original matrix
    aligned_matrix[matrix.columns[0]] = matrix[matrix.columns[0]]
    
    # Fill in existing data
    for date in current_dates:
        date_str = date.strftime('%Y-%m-%d')
        if date_str in aligned_matrix.columns:
            aligned_matrix[date_str] = matrix[date_str]
    
    return aligned_matrix

def merge_shallow_depths(matrix):
    """
    Merge values from 0.1m depth into 0.0m depth.
    If a value exists at both depths, keep the 0.0m value.
    
    Args:
        matrix (pd.DataFrame): Original measurement matrix
    
    Returns:
        pd.DataFrame: Matrix with 0.1m values merged into 0.0m
    """
    # Create a copy to avoid modifying the original
    merged_matrix = matrix.copy()
    
    # Check if both depths exist
    if 0.0 not in merged_matrix['depth'].values or 0.1 not in merged_matrix['depth'].values:
        return merged_matrix
    
    # Get rows for both depths
    row_0 = merged_matrix[merged_matrix['depth'] == 0.0].iloc[0]
    row_01 = merged_matrix[merged_matrix['depth'] == 0.1].iloc[0]
    
    # For each date column (skipping 'depth' column)
    for col in merged_matrix.columns[1:]:
        # If 0.0m is NaN and 0.1m has a value, use the 0.1m value
        if pd.isna(row_0[col]) and not pd.isna(row_01[col]):
            merged_matrix.loc[merged_matrix['depth'] == 0.0, col] = row_01[col]
    
    # Remove the 0.1m depth row
    merged_matrix = merged_matrix[merged_matrix['depth'] != 0.1]
    
    return merged_matrix

def knn_impute_timeseries(matrix, k=3):
    """
    Custom KNN imputation for time series data where temporal proximity matters.
    
    Args:
        matrix (pd.DataFrame): DataFrame where:
            - First column is 'depth'
            - Remaining columns are dates with measurements
            - Only one row of data (index 0)
        k (int): Number of nearest neighbors to consider
    
    Returns:
        pd.DataFrame: Matrix with imputed values
    """
    result_matrix = matrix.copy()
    
    # Convert date strings to datetime objects for temporal distance calculation
    dates = pd.to_datetime(matrix.columns[1:])
    values = matrix.iloc[0, 1:].values
    
    # Find indices of missing and non-missing values
    missing_indices = np.where(pd.isna(values))[0]
    valid_indices = np.where(~pd.isna(values))[0]
    
    # For each missing value
    for missing_idx in missing_indices:
        missing_date = dates[missing_idx]
        
        # Calculate temporal distance (in days) to all non-missing values
        time_distances = np.abs([(missing_date - dates[idx]).days for idx in valid_indices])
        
        # Get k nearest neighbors
        nearest_k_indices = valid_indices[np.argsort(time_distances)[:k]]
        nearest_k_distances = time_distances[np.argsort(time_distances)[:k]]
        
        # Calculate weights based on temporal distance (inverse distance weighting)
        weights = 1 / (nearest_k_distances + 1)  # Add 1 to avoid division by zero
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Calculate weighted average of k nearest neighbors
        imputed_value = np.sum(values[nearest_k_indices] * weights)
        
        # Update the result matrix with the imputed value
        result_matrix.iloc[0, missing_idx + 1] = imputed_value
    
    return result_matrix

def create_depth_groups(imputed_matrix, location, feature):
    """
    Transform imputed matrix into depth groups and save to a new file.
    
    Args:
        imputed_matrix (pd.DataFrame): The imputed matrix with depth and measurements
        location (str): Location code (e.g., 'SA', 'WG', 'WP')
        feature (str): Feature name (e.g., 'biochemical_oxygen_demand', 'secchi')
    """
    # Create new dataframe with depth groups
    depth_groups = pd.DataFrame(columns=imputed_matrix.columns)
    
    # Define depth group names
    group_names = ['Surface', 'Mid-Depth', 'Lower Photic', 'Deep']
    
    # Process each depth group
    for group in group_names:
        if group == 'Surface':
            # Copy the original measurements for Surface (previously depth 0)
            surface_row = imputed_matrix.iloc[0].copy()
            surface_row['depth'] = group
            depth_groups = pd.concat([depth_groups, surface_row.to_frame().T])
        else:
            # Create row with 'only measured at surface' for other depth groups
            new_row = pd.Series('only measured at surface', index=imputed_matrix.columns[1:])
            new_row['depth'] = group
            depth_groups = pd.concat([depth_groups, new_row.to_frame().T])
    
    # Save to new file
    output_path = Path(f"output/{location}/{feature}/depth_groups.csv")
    depth_groups.to_csv(output_path, index=False)
    print(f"Saved depth groups matrix to: {output_path}")

def combine_feature_data(feature, locations):
    """
    Combine processed depth groups data for a given feature across all locations.
    
    Args:
        feature (str): Feature name (e.g., 'biochemical_oxygen_demand', 'secchi')
        locations (list): List of location codes (e.g., ['SA', 'WG', 'WP'])
    """
    combined_data = pd.DataFrame()
    
    for location in locations:
        try:
            # Load depth groups file for this location
            filepath = Path(f"output/{location}/{feature}/depth_groups.csv")
            location_data = pd.read_csv(filepath)
            
            # Add location column
            location_data.insert(0, 'location', location)
            
            # Append to combined data
            combined_data = pd.concat([combined_data, location_data], ignore_index=True)
            
        except Exception as e:
            print(f"Error loading data for {feature} at {location}: {str(e)}")
    
    # Save combined data
    output_path = Path(f"output/{feature}_processed.csv")
    combined_data.to_csv(output_path, index=False)
    print(f"Saved combined {feature} data to: {output_path}")

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

def calculate_validation_metrics(original_values, predicted_values):
    """
    Calculate validation metrics between original and predicted values.
    
    Parameters:
    -----------
    original_values : pd.Series
        Original values that were removed
    predicted_values : pd.Series
        Predicted values from the imputation process
    
    Returns:
    --------
    dict
        Dictionary containing validation metrics
    """
    # Remove any paired values where either is NaN
    mask = ~(original_values.isna() | predicted_values.isna())
    original = original_values[mask]
    predicted = predicted_values[mask]
    
    if len(original) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'n_points': 0
        }
    
    rmse = np.sqrt(((original - predicted) ** 2).mean())
    mae = np.abs(original - predicted).mean()
    r2 = 1 - (((original - predicted) ** 2).sum() / 
              ((original - original.mean()) ** 2).sum())
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_points': len(original)
    }

def create_validation_report(metrics, feature, location):
    """Create a validation report from the metrics."""
    report = [
        f"=== Validation Report for {feature} at {location} ===\n",
        f"Number of validation points: {metrics['n_points']}",
        f"Root Mean Square Error (RMSE): {metrics['rmse']:.3f}",
        f"Mean Absolute Error (MAE): {metrics['mae']:.3f}",
        f"R-squared (R2): {metrics['r2']:.3f}\n"
    ]
    return '\n'.join(report)

def main():
    # Load master dates list
    master_dates_path = Path("output/comparison/unique_dates.csv")
    master_dates = pd.read_csv(master_dates_path)['date']
    print(f"Total dates in master list: {len(master_dates)}")
    
    features = ['biochemical_oxygen_demand', 'secchi']
    locations = ['SA', 'WG', 'WP']
    removal_percentage = 0.05  # 5% of values removed for validation
    n_validation_runs = 10  # Number of validation runs
    
    validation_results = {}
    
    for location in locations:
        print(f"\n=== Processing location: {location} ===")
        validation_results[location] = {}
        
        for feature in features:
            try:
                print(f"\nProcessing {feature}")
                print(f"Performing {n_validation_runs} validation runs, removing {removal_percentage*100:.1f}% of values each time")
                
                # Load measurement matrix
                matrix = load_measurement_matrix(feature, location)
                
                # Track validation results for this feature/location
                feature_validation_results = []
                final_imputed = None
                final_removed = None
                
                # Perform multiple validation runs
                for run in range(n_validation_runs):
                    print(f"Run {run + 1}/{n_validation_runs}")
                    
                    # Create validation set
                    modified_matrix, removed_matrix = create_removed_values_matrix(
                        matrix, removal_percentage=removal_percentage
                    )
                    
                    # Save removed values from final run
                    if run == n_validation_runs - 1:
                        final_removed = removed_matrix
                    
                    # Merge shallow depths and continue processing with modified matrix
                    modified_matrix = merge_shallow_depths(modified_matrix)
                    
                    feature_dates = modified_matrix.columns[1:]
                    comparison_results = compare_dates_with_master(feature_dates, master_dates)
                    
                    # Align matrix with master dates
                    aligned_matrix = align_matrix_with_master_dates(modified_matrix, master_dates)
                    
                    # Perform KNN imputation
                    imputed_matrix = knn_impute_timeseries(aligned_matrix, k=3)
                    
                    # Save final imputed matrix from last run
                    if run == n_validation_runs - 1:
                        final_imputed = imputed_matrix
                        removed_path = Path(f"output/{location}/{feature}/removed_matrix.csv")
                        imputed_path = Path(f"output/{location}/{feature}/imputed_matrix.csv")
                        removed_matrix.to_csv(removed_path, index=True)
                        imputed_matrix.to_csv(imputed_path, index=False)
                    
                    # Calculate validation metrics for this run
                    metrics = calculate_validation_metrics(
                        pd.Series(imputed_matrix.values.flatten()),
                        pd.Series(removed_matrix.values.flatten())
                    )
                    metrics['run'] = run + 1
                    feature_validation_results.append(metrics)
                
                # Convert validation results to DataFrame
                df_validation = pd.DataFrame(feature_validation_results)
                validation_results[location][feature] = df_validation
                
                # Save validation results
                validation_path = Path(f"output/{location}/{feature}/validation_results.csv")
                df_validation.to_csv(validation_path, index=False)
                
                # Calculate and print average metrics for this location/feature
                mean_rmse = df_validation['rmse'].mean()
                std_rmse = df_validation['rmse'].std()
                print(f"\nValidation results for {feature} at {location}:")
                print(f"RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
                
                # Create and save depth groups with final imputed matrix
                create_depth_groups(final_imputed, location, feature)
                
                print(f"\nOutputs saved to:")
                print(f"- output/{location}/{feature}/removed_matrix.csv")
                print(f"- output/{location}/{feature}/imputed_matrix.csv")
                print(f"- output/{location}/{feature}/validation_results.csv")
                print(f"- output/{location}/{feature}/depth_groups.csv")
                
            except Exception as e:
                print(f"Error processing {feature} at {location}: {str(e)}")
                validation_results[location][feature] = None
    
    # Calculate and display overall RMSE for each feature
    print("\n=== Overall Results ===")
    for feature in features:
        feature_rmse_values = []
        for location in locations:
            if (validation_results.get(location) and 
                validation_results[location].get(feature) is not None):
                mean_rmse = validation_results[location][feature]['rmse'].mean()
                feature_rmse_values.append(mean_rmse)
        
        if feature_rmse_values:
            overall_mean = np.mean(feature_rmse_values)
            overall_std = np.std(feature_rmse_values)
            print(f"\nOverall RMSE for {feature} across all locations:")
            print(f"Mean RMSE: {overall_mean:.4f} ± {overall_std:.4f}")
    
    # Combine feature data across locations
    for feature in features:
        try:
            combine_feature_data(feature, locations)
        except Exception as e:
            print(f"\nError combining data for {feature}: {str(e)}")
    
    return validation_results

if __name__ == "__main__":
    main()
