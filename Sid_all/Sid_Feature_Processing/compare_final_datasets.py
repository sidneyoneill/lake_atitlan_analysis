import pandas as pd
import os

def load_and_prepare_data(feature_name):
    """
    Load and prepare feature data from CSV file.
    
    Parameters:
    -----------
    feature_name : str
        Name of the feature (e.g., 'temp', 'chlorophyll')
    
    Returns:
    --------
    pandas.DataFrame
        Prepared dataset with standardized columns
    """
    file_path = f'output/{feature_name}_processed.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data file found for feature: {feature_name}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Standardize column names for the feature
    feature_cols = [col for col in df.columns if col.startswith(f'mean_{feature_name}')]
    if not feature_cols:
        raise ValueError(f"No mean value column found for feature: {feature_name}")
    
    feature_col = feature_cols[0]
    df = df.rename(columns={feature_col: 'value'})
    
    return df

def compare_multiple_datasets(feature_names):
    """
    Compare multiple feature datasets and identify differences between all pairs.
    
    Parameters:
    -----------
    feature_names : list
        List of feature names to compare
    
    Returns:
    --------
    dict
        Dictionary containing comparison results for all feature pairs
    """
    if len(feature_names) < 2:
        raise ValueError("At least two features are required for comparison")
    
    print(f"\nComparing features: {', '.join(feature_names)}...")
    
    # Load and prepare all datasets
    datasets = {}
    for feature in feature_names:
        try:
            datasets[feature] = load_and_prepare_data(feature)
        except Exception as e:
            print(f"Error loading {feature}: {str(e)}")
            return None
    
    # After loading all datasets, get unique dates across all features
    all_dates = set()
    for feature, df in datasets.items():
        all_dates.update(df['date'].dt.date.unique())
    
    # Convert to sorted list and save to CSV
    unique_dates = pd.DataFrame({'date': sorted(list(all_dates))})
    unique_dates.to_csv('output/comparison/unique_dates.csv', index=False)
    print(f"\nUnique dates saved to: output/comparison/unique_dates.csv")
    
    # Create output directory
    os.makedirs('output/comparison', exist_ok=True)
    
    # Initialize results dictionary with unique dates
    results = {
        'merged_dfs': {},
        'summaries': {},
        'differences': {},
        'non_matching': {},
        'unique_dates': unique_dates
    }
    
    # Compare all possible pairs
    for i, feature1 in enumerate(feature_names):
        for feature2 in feature_names[i+1:]:
            print(f"\nComparing {feature1} with {feature2}...")
            
            # Perform full outer join
            merged = pd.merge(
                datasets[feature1][['location', 'date', 'depth_group', 'value', 'measurement_count']],
                datasets[feature2][['location', 'date', 'depth_group', 'value', 'measurement_count']],
                on=['location', 'date', 'depth_group'],
                how='outer',
                suffixes=(f'_{feature1}', f'_{feature2}')
            )
            
            # Calculate summary statistics
            total_rows = len(merged)
            matching_rows = merged[f'value_{feature1}'].notna() & merged[f'value_{feature2}'].notna()
            only_feature1 = merged[f'value_{feature1}'].notna() & merged[f'value_{feature2}'].isna()
            only_feature2 = merged[f'value_{feature1}'].isna() & merged[f'value_{feature2}'].notna()
            
            summary = {
                'total_rows': total_rows,
                'matching_rows': matching_rows.sum(),
                'only_feature1': only_feature1.sum(),
                'only_feature2': only_feature2.sum()
            }
            
            # Print summary
            print("\nComparison Summary:")
            print(f"Total unique location-date-depth combinations: {summary['total_rows']}")
            print(f"Matching rows: {summary['matching_rows']}")
            print(f"Only in {feature1}: {summary['only_feature1']}")
            print(f"Only in {feature2}: {summary['only_feature2']}")
            
            # Analyze differences in matching rows
            matching_mask = matching_rows
            differences = merged[matching_mask].copy()
            differences['value_difference'] = differences[f'value_{feature1}'] - differences[f'value_{feature2}']
            differences['measurement_count_difference'] = (
                differences[f'measurement_count_{feature1}'] - 
                differences[f'measurement_count_{feature2}']
            )
            
            # Identify non-matching rows
            non_matching = pd.concat([
                merged[only_feature1].assign(missing_in=feature2),
                merged[only_feature2].assign(missing_in=feature1)
            ]).sort_values(['date', 'location', 'depth_group'])
            
            # Save results
            pair_key = f"{feature1}_vs_{feature2}"
            results['merged_dfs'][pair_key] = merged
            results['summaries'][pair_key] = summary
            results['differences'][pair_key] = differences
            results['non_matching'][pair_key] = non_matching
            
            # Save to files
            for name, df, suffix in [
                ('merged', merged, 'merged'),
                ('differences', differences, 'differences'),
                ('non_matching', non_matching, 'non_matching')
            ]:
                output_path = f'output/comparison/{feature1}_vs_{feature2}_{suffix}.csv'
                df.to_csv(output_path, index=False)
                print(f"{name.capitalize()} analysis saved to: {output_path}")
    
    return results

def main(feature_names):
    """
    Main function to compare multiple feature datasets.
    
    Parameters:
    -----------
    feature_names : list
        List of feature names to compare
    
    Returns:
    --------
    dict
        Results dictionary containing all comparisons
    """
    try:
        results = compare_multiple_datasets(feature_names)
        return results
    except Exception as e:
        print(f"\nError comparing datasets: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage with multiple features
    features_to_compare = ["temp", "chlorophyll_a", "ph", "dissolved_oxygen"]
    results = main(features_to_compare)
