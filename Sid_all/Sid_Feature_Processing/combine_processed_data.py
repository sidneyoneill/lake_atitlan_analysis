import pandas as pd

def load_feature_data(feature):
    """Load processed data for a specific feature."""
    df = pd.read_csv(f'output/{feature}_processed.csv')
    # Ensure date is in consistent format
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

def combine_processed_data():
    # List of features to combine
    features = ['temp', 'chlorophyll_a', 'ph', 'dissolved_oxygen']
    
    # Load first feature to get the basic structure
    combined_df = load_feature_data(features[0])
    
    # Initialize the final dataframe with base columns
    result_df = combined_df[['location', 'depth_group', 'date']].copy()
    
    # Add all features in a single loop
    for feature in features:
        feature_df = load_feature_data(feature)
        result_df[feature] = feature_df[f'mean_{feature}']
    
    # Save the combined dataset
    result_df.to_csv('output/combined_1.csv', index=False)

    print(f"Combined data saved to output/combined_1.csv")
    return result_df

def reshape_measurement_matrix(df):
    """
    Reshape a matrix where dates are columns into a long format dataframe.
    
    Input format:
    location | depth_group | date_1 | date_2 | ...
    
    Output format:
    location | depth_group | date | value
    """
    # Melt the dataframe to convert date columns to rows
    date_columns = [col for col in df.columns if col not in ['location', 'depth_group']]
    melted_df = pd.melt(
        df,
        id_vars=['location', 'depth_group'],
        value_vars=date_columns,
        var_name='date',
        value_name='measurement'
    )
    
    # Sort the dataframe to match the structure of other processed files
    return melted_df.sort_values(['location', 'depth_group', 'date']).reset_index(drop=True)

def load_and_process_matrix_data(feature, filepath):
    """Load and process data that's in the matrix format."""
    # Read the matrix-format data
    matrix_df = pd.read_csv(filepath)
    
    # Reshape to match the structure of other processed files
    processed_df = reshape_measurement_matrix(matrix_df)
    
    # Ensure date is in consistent format
    processed_df['date'] = pd.to_datetime(processed_df['date']).dt.date
    
    # Rename the measurement column to match the naming convention
    processed_df = processed_df.rename(columns={'measurement': f'mean_{feature}'})
    
    return processed_df

def process_matrix_features(filepaths):
    """
    Process multiple features from matrix format, handling 'only measure' values.
    
    Args:
        filepaths: dict of feature_name: filepath pairs
    """
    combined_df = None
    
    for feature, filepath in filepaths.items():
        # Read the matrix-format data
        matrix_df = pd.read_csv(filepath)
        
        # Get date columns (all columns except location and depth_group)
        date_cols = [col for col in matrix_df.columns if col not in ['location', 'depth_group']]
        
        # Melt the dataframe to long format
        melted_df = pd.melt(
            matrix_df,
            id_vars=['location', 'depth_group'],
            value_vars=date_cols,
            var_name='date',
            value_name=feature
        )
        
        # Convert date to datetime
        melted_df['date'] = pd.to_datetime(melted_df['date'], format='%d/%m/%Y').dt.date
        
        # Replace 'only measure' with NaN
        melted_df[feature] = melted_df[feature].replace('only measure', pd.NA)
        
        # Convert feature to float
        melted_df[feature] = pd.to_numeric(melted_df[feature], errors='coerce')
        
        # If this is the first feature, use it as the base
        if combined_df is None:
            combined_df = melted_df
        else:
            # Merge with existing data
            combined_df = pd.merge(
                combined_df,
                melted_df,
                on=['location', 'depth_group', 'date'],
                how='outer'
            )
    
    return combined_df

def combine_all_data():
    """Combine both regular processed data and matrix-format data."""
    # First get the combined data from the original process
    combined_df = combine_processed_data()
    
    # Define the matrix-format features and their file paths
    matrix_features = {
        'secchi': 'output/secchi_processed.csv',
        'biochemical_oxygen_demand': 'output/biochemical_oxygen_demand_processed.csv',
        'total_dissolved_solids': 'output/total_dissolved_solids_processed.csv',
        'turbidity': 'output/turbidity_processed.csv',
        'nitrate': 'output/nitrate_processed.csv',
        'phosphate': 'output/phosphate_processed.csv',
        'ammonium': 'output/ammonium_processed.csv',
        'phosphorus': 'output/phosphorus_processed.csv',
        # Add more features here as needed, for example:
        # 'feature2': 'output/feature2_processed.csv',
        # 'feature3': 'output/feature3_processed.csv',
    }
    
    # Load and process the matrix format data
    matrix_df = process_matrix_features(matrix_features)
    
    # Merge with the combined dataset
    final_df = pd.merge(
        combined_df,
        matrix_df,
        on=['location', 'depth_group', 'date'],
        how='left'
    )
    
    # Save the final combined dataset
    final_df.to_csv('output/combined_2.csv', index=False)
    print(f"Final combined data saved to output/combined_2.csv")
    
    return final_df

if __name__ == "__main__":
    combine_all_data()

