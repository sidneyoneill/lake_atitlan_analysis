import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

def load_measurement_matrix(location, feature):
    """
    Load the measurement matrix for a given location and feature.
    
    Args:
        location (str): Location code (e.g., 'SA', 'WG', 'WP')
        feature (str): Feature name (e.g., 'total_dissolved_solids')
    
    Returns:
        pd.DataFrame: Loaded measurement matrix
    """
    filepath = Path(f"output/{location}/{feature}/measurement_matrix.csv")
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded measurement matrix for {feature} at {location}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def clip_matrix_by_date(df, cutoff_date='12/03/2019'):
    """
    Remove all data before the cutoff date.
    
    Args:
        df (pd.DataFrame): Input measurement matrix
        cutoff_date (str): Date to clip before (DD/MM/YYYY format)
    
    Returns:
        pd.DataFrame: Clipped measurement matrix
    """
    # Convert date column (column names) to datetime, assuming DD/MM/YYYY format
    date_columns = [col for col in df.columns if col != 'depth']
    
    # First convert cutoff date to datetime
    cutoff = pd.to_datetime(cutoff_date, format='%d/%m/%Y')
    
    try:
        # Try parsing dates as they appear in the file (DD/MM/YYYY)
        dates = pd.to_datetime(date_columns, format='%d/%m/%Y')
    except:
        # If that fails, try parsing as YYYY-MM-DD
        dates = pd.to_datetime(date_columns)
    
    # Get columns for dates >= cutoff_date
    keep_columns = ['depth'] + [col for col, date in zip(date_columns, dates) 
                               if date >= cutoff]
    
    # Select only the columns we want to keep
    clipped_df = df[keep_columns].copy()
    
    return clipped_df

def save_clipped_matrix(df, location, feature):
    """
    Save the clipped matrix to CSV.
    
    Args:
        df (pd.DataFrame): Clipped measurement matrix to save
        location (str): Location code
        feature (str): Feature name
    """
    output_path = Path(f"output/{location}/{feature}/clipped_matrix.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved clipped matrix to: {output_path}")

def group_depths(df):
    """
    Group depth measurements into four zones and calculate average values.
    
    Args:
        df (pd.DataFrame): Input measurement matrix with 'depth' column
    
    Returns:
        pd.DataFrame: Matrix with 4 rows for depth groups and averaged values
    """
    # Create depth bins and labels
    bins = [-float('inf'), 10, 20, 30, float('inf')]
    labels = ['Surface', 'Mid-Depth', 'Lower Photic', 'Deep']
    
    # Create depth group column
    df['depth_group'] = pd.cut(df['depth'], bins=bins, labels=labels)
    
    # Calculate mean for each group and date
    # First melt the dataframe to make it easier to group
    melted = df.melt(id_vars=['depth', 'depth_group'], 
                    var_name='date', 
                    value_name='value')
    
    # Convert N/A to NaN so they don't affect averages
    melted['value'] = pd.to_numeric(melted['value'].replace('N/A', np.nan))
    
    # Group and calculate means
    grouped = melted.groupby(['depth_group', 'date'])['value'].mean().unstack()
    
    # Ensure groups are in the correct order
    grouped = grouped.reindex(labels)
    
    return grouped

def load_unique_dates(filepath):
    """
    Load the unique dates from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file containing unique dates
    
    Returns:
        set: Set of unique dates as datetime objects
    """
    dates_df = pd.read_csv(filepath)
    return set(pd.to_datetime(dates_df['date']))

def compare_dates_with_reference(feature_dates, reference_dates):
    """
    Compare dates between a feature and reference dates.
    
    Args:
        feature_dates (set): Set of dates from the feature
        reference_dates (set): Set of reference dates
    
    Returns:
        dict: Dictionary containing comparison metrics
    """
    missing_from_feature = reference_dates - feature_dates
    extra_in_feature = feature_dates - reference_dates
    
    return {
        'missing_dates_count': len(missing_from_feature),
        'extra_dates_count': len(extra_in_feature),
        'total_feature_dates': len(feature_dates),
        'total_reference_dates': len(reference_dates),
        'overlap_percentage': len(feature_dates.intersection(reference_dates)) / len(reference_dates) * 100
    }

def get_feature_dates(df):
    """
    Extract unique dates from a measurement matrix.
    
    Args:
        df (pd.DataFrame): Measurement matrix
    
    Returns:
        set: Set of unique dates as datetime objects
    """
    date_columns = [col for col in df.columns if col != 'depth']
    try:
        dates = pd.to_datetime(date_columns, format='%d/%m/%Y')
    except:
        dates = pd.to_datetime(date_columns)
    return set(dates)

def align_matrix_with_reference_dates(grouped_df, reference_dates):
    """
    Add missing dates from reference dates to the grouped measurement matrix.
    
    Args:
        grouped_df (pd.DataFrame): Grouped measurement matrix with depth zones as index
        reference_dates (set): Set of reference dates to align with
    
    Returns:
        pd.DataFrame: Matrix with all reference dates included (missing values as NaN)
    """
    # Create new dataframe with same index (depth zones)
    aligned_df = pd.DataFrame(index=grouped_df.index)
    
    # Convert current date columns to datetime
    try:
        current_dates = pd.to_datetime(grouped_df.columns, format='%d/%m/%Y')
    except:
        current_dates = pd.to_datetime(grouped_df.columns)
    
    # Create mapping of old column names to datetime objects
    date_mapping = dict(zip(grouped_df.columns, current_dates))
    
    # Add data for each reference date
    for ref_date in sorted(reference_dates):
        # Find if this date exists in original data
        matching_cols = [col for col, date in date_mapping.items() 
                        if date == ref_date]
        
        if matching_cols:
            # If date exists, copy the data
            aligned_df[ref_date.strftime('%d/%m/%Y')] = grouped_df[matching_cols[0]]
        else:
            # If date doesn't exist, add empty column
            aligned_df[ref_date.strftime('%d/%m/%Y')] = np.nan
    
    return aligned_df

def mark_early_data_as_not_measured(df, cutoff_date='12/03/2019'):
    """
    Set any data point before the cutoff date to 'not measured'.
    
    Args:
        df (pd.DataFrame): Input measurement matrix
        cutoff_date (str): Date to mark before (DD/MM/YYYY format)
    
    Returns:
        pd.DataFrame: Modified measurement matrix
    """
    # Create a copy to avoid modifying the original
    modified_df = df.copy()
    
    # Convert cutoff date to datetime
    cutoff = pd.to_datetime(cutoff_date, format='%d/%m/%Y')
    
    # Get date columns
    date_columns = [col for col in df.columns if col != 'depth']
    
    try:
        # Try parsing dates as DD/MM/YYYY
        dates = pd.to_datetime(date_columns, format='%d/%m/%Y')
    except:
        # If that fails, try parsing as YYYY-MM-DD
        dates = pd.to_datetime(date_columns)
    
    # Find columns before cutoff date
    early_columns = [col for col, date in zip(date_columns, dates) 
                    if date < cutoff]
    
    # Set early data to 'not measured'
    modified_df[early_columns] = 'not measured'
    
    return modified_df

def impute_with_knn(df, k=3):
    """
    Impute missing values using k-nearest neighbors through time.
    Ignores 'not measured' values in the computation.
    
    Args:
        df (pd.DataFrame): Input matrix with missing values
        k (int): Number of nearest neighbors to use
    
    Returns:
        pd.DataFrame: Matrix with imputed values
    """
    # Create a copy to avoid modifying the original
    imputed_df = df.copy()
    
    # Convert 'not measured' to np.nan
    imputed_df = imputed_df.replace('not measured', np.nan)
    
    # Convert all values to float (will convert any remaining strings to nan)
    imputed_df = imputed_df.astype(float)
    
    # Process each depth zone separately
    for idx in imputed_df.index:
        row = imputed_df.loc[idx]
        
        # Find indices of missing values (nan) in this row
        missing_indices = row.index[row.isna()]
        
        # Find indices of valid values in this row
        valid_indices = row.index[~row.isna()]
        
        # Skip if no valid values or no missing values
        if len(valid_indices) == 0 or len(missing_indices) == 0:
            continue
        
        # Convert dates to numerical values for distance calculation
        valid_dates = pd.to_datetime(valid_indices)
        
        # Process each missing value
        for missing_idx in missing_indices:
            missing_date = pd.to_datetime(missing_idx)
            
            # Calculate time differences in days
            time_diffs = abs((valid_dates - missing_date).days)
            
            # Get k nearest dates
            nearest_k_indices = valid_indices[np.argsort(time_diffs)[:k]]
            
            # Calculate weighted average based on time difference
            weights = 1 / (time_diffs[np.argsort(time_diffs)[:k]] + 1)  # Add 1 to avoid division by zero
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Calculate weighted average
            imputed_value = np.average(row[nearest_k_indices], weights=weights)
            imputed_df.loc[idx, missing_idx] = imputed_value
    
    return imputed_df

def combine_feature_data(feature, locations):
    """
    Combine processed marked matrix data for a given feature across all locations.
    
    Args:
        feature (str): Feature name (e.g., 'total_dissolved_solids')
        locations (list): List of location codes (e.g., ['SA', 'WG', 'WP'])
    """
    combined_data = pd.DataFrame()
    
    for location in locations:
        try:
            # Load marked matrix file for this location
            filepath = Path(f"output/{location}/{feature}/marked_matrix.csv")
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

def calculate_validation_metrics(original_values, predicted_values):
    """
    Calculate validation metrics between original and predicted values.
    
    Parameters:
    -----------
    original_values : pd.Series
        Original values that were removed
    predicted_values : pd.Series
        Predicted values from imputation
    
    Returns:
    --------
    dict
        Dictionary containing RMSE, MAE, and R² metrics
    """
    # Remove any remaining NaN values
    mask = ~(np.isnan(original_values) | np.isnan(predicted_values))
    original_values = original_values[mask]
    predicted_values = predicted_values[mask]
    
    if len(original_values) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'n_points': 0
        }
    
    rmse = np.sqrt(mean_squared_error(original_values, predicted_values))
    mae = mean_absolute_error(original_values, predicted_values)
    r2 = r2_score(original_values, predicted_values)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_points': len(original_values)
    }

def create_validation_report(metrics, feature, location):
    """
    Create a detailed validation report.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing validation metrics
    feature : str
        Name of the feature being validated
    location : str
        Location identifier
    
    Returns:
    --------
    str
        Formatted validation report
    """
    report = [
        f"=== Validation Report for {feature} at {location} ===\n",
        f"Number of validation points: {metrics['n_points']}",
        f"Root Mean Square Error (RMSE): {metrics['rmse']:.3f}",
        f"Mean Absolute Error (MAE): {metrics['mae']:.3f}",
        f"R² Score: {metrics['r2']:.3f}\n",
        "Interpretation:",
        "- RMSE: Lower values indicate better prediction accuracy",
        "- MAE: Lower values indicate better prediction accuracy",
        "- R²: Values closer to 1 indicate better fit (range: -∞ to 1)",
        "\nNote: These metrics compare the imputed values with the original",
        "values that were randomly removed for validation."
    ]
    
    return "\n".join(report)

def save_validation_results(validation_results, output_path="output/tds_validation_results.csv"):
    """
    Save validation results to a CSV file.
    
    Parameters:
    -----------
    validation_results : dict
        Nested dictionary containing validation results for each location and feature
    output_path : str
        Path to save the CSV file
    """
    # Convert nested dictionary to DataFrame
    rows = []
    for location in validation_results:
        for feature in validation_results[location]:
            metrics = validation_results[location][feature]
            if metrics is not None:
                row = {
                    'location': location,
                    'feature': feature,
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2'],
                    'n_points': metrics['n_points']
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nValidation results saved to: {output_path}")

def plot_validation_results(original_values, predicted_values, feature, location, 
                          output_dir="output"):
    """
    Create scatter plot of original vs predicted values.
    
    Parameters:
    -----------
    original_values : array-like
        Original values that were removed
    predicted_values : array-like
        Predicted values from imputation
    feature : str
        Name of the feature being validated
    location : str
        Location identifier
    output_dir : str
        Directory to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(original_values, predicted_values, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(original_values), min(predicted_values))
    max_val = max(max(original_values), max(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    plt.xlabel('Original Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{feature} Validation: Original vs Predicted Values\nLocation: {location}')
    plt.legend()
    
    # Add R² value to plot
    r2 = r2_score(original_values, predicted_values)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Save plot
    output_path = os.path.join(output_dir, location, feature, 'validation_plot.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Validation plot saved to: {output_path}")

def compare_matrices(imputed_df, removed_df):
    """
    Compare imputed values with the removed values and calculate RMSE.
    
    Args:
        imputed_df (pd.DataFrame): Matrix with imputed values
        removed_df (pd.DataFrame): Matrix with removed values
    
    Returns:
        float: RMSE between imputed and removed values
    """
    # Get mask where removed_df has values
    mask = ~removed_df.isna()
    
    # Extract actual and predicted values using the mask
    actual_values = removed_df[mask].values.flatten()
    predicted_values = imputed_df[mask].values.flatten()
    
    # Remove any remaining NaN values
    valid_mask = ~(np.isnan(actual_values) | np.isnan(predicted_values))
    actual_values = actual_values[valid_mask]
    predicted_values = predicted_values[valid_mask]
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
    
    return rmse

def create_removed_values_matrix(df):
    """
    Create a matrix with randomly removed values for validation.
    
    Args:
        df (pd.DataFrame): Input matrix to remove values from
    
    Returns:
        tuple: (matrix_with_removed_values, removed_values_matrix)
    """
    # Create a copy of the input matrix
    df_copy = df.copy()
    removed_df = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    
    # Get non-null values
    non_null_mask = df_copy.notna()
    non_null_coords = list(zip(*non_null_mask.values.nonzero()))
    
    # Calculate number of values to remove (5% of non-null values)
    n_values = len(non_null_coords)
    n_remove = int(n_values * 0.05)  # Remove 5% of values
    
    # Randomly select indices to remove
    remove_indices = np.random.choice(len(non_null_coords), n_remove, replace=False)
    
    # Remove values and store them in removed_df
    for idx in remove_indices:
        i, j = non_null_coords[idx]
        value = df_copy.iloc[i, j]
        removed_df.iloc[i, j] = value
        df_copy.iloc[i, j] = np.nan
    
    return df_copy, removed_df

def main():
    # Set parameters
    features = ['total_dissolved_solids']
    locations = ['SA', 'WG', 'WP']
    cutoff_date = '12/03/2019'
    n_runs = 10  # Number of validation runs
    
    # Load reference dates
    reference_dates = load_unique_dates('output/comparison/unique_dates.csv')
    
    # Dictionary to store RMSE values for each location and run
    all_rmse_values = {loc: [] for loc in locations}
    
    # Perform multiple validation runs
    for run in range(n_runs):
        print(f"\nValidation Run {run + 1}/{n_runs}")
        
        # Process each location and feature
        for location in locations:
            for feature in features:
                try:
                    # Load data
                    df = load_measurement_matrix(location, feature)
                    if df is None:
                        continue
                    
                    # Process the data
                    clipped_df = clip_matrix_by_date(df, cutoff_date)
                    grouped_df = group_depths(clipped_df)
                    
                    # Create aligned matrix from grouped matrix
                    aligned_df = align_matrix_with_reference_dates(grouped_df, reference_dates)

                    # Create removed values matrix
                    matrix_with_removed, removed_df = create_removed_values_matrix(aligned_df)

                    # Perform KNN imputation
                    imputed_df = impute_with_knn(matrix_with_removed)

                    # Compare imputed and removed matrices
                    rmse = compare_matrices(imputed_df, removed_df)
                    all_rmse_values[location].append(rmse)
                    
                    # Only save matrices for the last run
                    if run == n_runs - 1:
                        removed_path = Path(f"output/{location}/{feature}/removed_matrix.csv")
                        removed_df.to_csv(removed_path)
                        
                        imputed_path = Path(f"output/{location}/{feature}/imputed_matrix.csv")
                        imputed_df.to_csv(imputed_path)
                        
                        marked_df = mark_early_data_as_not_measured(imputed_df, cutoff_date)
                        marked_path = Path(f"output/{location}/{feature}/marked_matrix.csv")
                        marked_df.to_csv(marked_path)

                except Exception as e:
                    print(f"Error processing {location} - {feature}: {str(e)}")
    
    # Calculate and display statistics for all runs
    print("\nOverall Validation Results:")
    
    # Calculate statistics for each location
    location_stats = {}
    for location in locations:
        if all_rmse_values[location]:
            mean_rmse = np.mean(all_rmse_values[location])
            std_rmse = np.std(all_rmse_values[location])
            location_stats[location] = {'mean': mean_rmse, 'std': std_rmse}
            print(f"\n{location}:")
            print(f"Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
            print(f"Individual runs: {', '.join([f'{x:.4f}' for x in all_rmse_values[location]])}")
    
    # Calculate overall statistics across all locations and runs
    all_values = [rmse for loc_values in all_rmse_values.values() for rmse in loc_values]
    if all_values:
        overall_mean = np.mean(all_values)
        overall_std = np.std(all_values)
        print(f"\nOverall RMSE across all locations and runs: {overall_mean:.4f} ± {overall_std:.4f}")
    
    # Save validation results to CSV
    results_df = pd.DataFrame(all_rmse_values)
    results_df.to_csv('output/tds_validation_results.csv', index=False)
    print("\nValidation results saved to: output/tds_validation_results.csv")

    # Continue with existing feature combination code
    for feature in features:
        combine_feature_data(feature, locations)
    
    return all_rmse_values

if __name__ == "__main__":
    main()

