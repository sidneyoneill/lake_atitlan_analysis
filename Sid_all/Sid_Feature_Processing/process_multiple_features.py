from process_functions import (
    process_location,
    combine_depth_groups,
    create_depth_grid,
    compare_feature_assigned_matrices,
    compare_feature_matrices_multi,
    assign_to_grid_depth
)
import os
import pandas as pd

def process_multiple_features(feature_list):
    """
    Process and analyze data for multiple features in a structured sequence:
    1. Create assigned_matrix.csv for all features per location.
    2. Compare and align assigned matrices across features.
    3. Perform interpolation and imputation on aligned matrices.
    4. Combine depth group data across all locations for each feature.

    Parameters:
    -----------
    feature_list : list
        List of feature names to process

    Returns:
    --------
    dict
        Dictionary containing results for each feature
    """
    # Create the depth grid (same for all features and locations)
    depth_grid = create_depth_grid(
        photic_zone_depth=50,  # 50m photic zone
        max_depth=300,          # Maximum depth of 250m
        photic_interval=1,      # 1m intervals in photic zone
        deep_interval=10        # 10m intervals in deep zone
    )
    
    all_results = {}
    locations = ['SA', 'WG', 'WP']
    
    for location in locations:
        print("\n" + "-"*50)
        print(f"Processing location {location}")
        print("-"*50)
        
        # Step 1: Create assigned_matrix.csv for all features
        print("\nStep 1: Creating assigned matrices for all features at", location)
        for feature_name in feature_list:
            try:
                # Assign and save the assigned matrix
                assign_and_save_assigned_matrix(location, depth_grid, feature_name)
                print(f"Assigned matrix created for feature '{feature_name}' at location '{location}'.")
            except Exception as e:
                print(f"Error creating assigned_matrix for feature '{feature_name}' at location '{location}': {e}")
        
        # Step 2: Compare and align assigned matrices across features
        print("\nStep 2: Comparing and aligning assigned matrices for each feature at", location)
        try:
            comparison_result = compare_feature_matrices_multi(location, feature_list)
        except Exception as e:
            print(f"Error comparing assigned matrices for location '{location}': {e}")
            continue  # Skip further processing for this location if comparison fails
        
        # Step 3: Perform interpolation and imputation on aligned matrices
        print("\nStep 3: Interpolation and imputation of features at", location)
        feature_results = {}
        
        for feature_name in feature_list:
            try:
                # Proceed with the full processing now that matrices are aligned
                df_imputed, df_grouped, df_pivot = process_location(location, depth_grid, feature_name)
                feature_results[feature_name] = {
                    'imputed_matrix': df_imputed,
                    'depth_groups_long': df_grouped,
                    'depth_groups_pivot': df_pivot
                }
                
                print(f"\nOutputs for feature '{feature_name}' at location '{location}' saved successfully.")
            except Exception as e:
                print(f"Error processing feature '{feature_name}' at location '{location}': {e}")
        
        if feature_results:
            all_results[location] = {
                'feature_results': feature_results
            }
            print(f"\nProcessing complete for location '{location}'.")
        else:
            print(f"\nNo features were successfully processed for location '{location}'.")
            all_results[location] = None
    
    # Step 4: Combine depth groups across all locations for each feature
    print("\n" + "-"*50)
    print("Combining Depth Groups Across All Locations for Each Feature")
    print("-"*50)
    
    for feature_name in feature_list:
        try:
            df_combined = combine_depth_groups(locations, feature_name)
            output_path = f'output/{feature_name}_processed.csv'
            df_combined.to_csv(output_path, index=False)
            print(f"Combined depth groups for feature '{feature_name}' saved to '{output_path}'.")
        except Exception as e:
            print(f"Error combining depth groups for feature '{feature_name}': {e}")
    
    print("\nEnd of processing")
    return all_results

def assign_and_save_assigned_matrix(location, depth_grid, feature_name):
    """
    Assign measurements to grid depths and save the assigned_matrix.csv.

    Parameters:
    -----------
    location : str
        Location identifier
    depth_grid : numpy.ndarray
        Array of regularly spaced depth values
    feature_name : str
        Name of the feature being processed

    Returns:
    --------
    pandas.DataFrame
        DataFrame with measurements mapped to closest grid depths
    """
    # Define measurement matrix path
    measurement_matrix_path = f'output/{location}/{feature_name}/measurement_matrix.csv'
    
    # Assign measurements to closest grid depths
    df_assigned = assign_to_grid_depth(measurement_matrix_path, depth_grid, feature_name)
    
    # Save the assigned_matrix.csv
    assigned_matrix_path = f'output/{location}/{feature_name}/assigned_matrix.csv'
    df_assigned.to_csv(assigned_matrix_path)
    
    return df_assigned

def main():
    """
    Main function to process multiple features.
    """
    # Define the features to analyze
    features = ['temp', 'chlorophyll_a', 'total_dissolved_solids', 'ph', 'dissolved_oxygen', 'nitrate', 'phosphate', 'ammonium', 'phosphorus', 'nitrogen', 'biochemical_oxygen_demand', 'secchi']
    
    try:
        results = process_multiple_features(features)
        
        # Print final summary
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        for location, result in results.items():
            if result:
                feature_names = list(result['feature_results'].keys())
                print(f"\nLocation '{location}':")
                print(f"- Processed features: {', '.join(feature_names)}")
            else:
                print(f"\nLocation '{location}': Processing failed or no features processed.")
                    
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")

if __name__ == "__main__":
    main()
