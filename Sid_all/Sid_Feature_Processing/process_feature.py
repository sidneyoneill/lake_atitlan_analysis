from process_functions import (create_depth_grid, process_location, 
                                combine_depth_groups)
import pandas as pd
import os
import numpy as np

def main(feature_name, removal_percentage=0.05, n_validation_runs=10):
    """
    Process and analyze feature data for all locations.
    
    Parameters:
    -----------
    feature_name : str
        Name of the feature to process
    removal_percentage : float
        Percentage of values to remove for validation (default: 0.05 for 5%)
    n_validation_runs : int
        Number of validation runs to perform (default: 10)
    """
    # Create the depth grid (same for all locations)
    depth_grid = create_depth_grid(
        photic_zone_depth=30,  # 30m photic zone
        max_depth=250,         # Maximum depth of 250m
        photic_interval=1,     # 1m intervals in photic zone
        deep_interval=10       # 10m intervals in deep zone
    )
    
    # Get list of locations from measurement matrices
    locations = ['SA', 'WG', 'WP']  
    
    if not locations:
        print(f"\nNo locations found with {feature_name} measurement matrices!")
        return None, None
    
    print(f"\nProcessing {feature_name} data for locations: {', '.join(locations)}")
    print(f"Performing {n_validation_runs} validation runs, removing {removal_percentage*100:.1f}% of values each time")
    
    # Process each location
    results = {}
    processed_locations = []
    all_rmse_values = []
    
    for location in locations:
        try:
            final_imputed, df_grouped, df_pivot, df_validation = process_location(
                location, depth_grid, feature_name, removal_percentage, n_validation_runs)
            
            results[location] = {
                'imputed': final_imputed,
                'grouped': df_grouped,
                'pivot': df_pivot,
                'validation': df_validation
            }
            processed_locations.append(location)
            
            # Store mean RMSE for this location
            mean_rmse = df_validation['rmse'].mean()
            std_rmse = df_validation['rmse'].std()
            all_rmse_values.append(mean_rmse)
            
            print(f"\nOutputs for {location} saved to:")
            print(f"- output/{location}/{feature_name}/imputed_matrix.csv")
            print(f"- output/{location}/{feature_name}/validation_results.csv")
            print(f"- output/{location}/{feature_name}/depth_groups_long.csv")
            print(f"- output/{location}/{feature_name}/depth_groups_pivot.csv")
            print(f"- output/{location}/{feature_name}/data_quality_report.txt")
            print(f"- output/{location}/figures/{feature_name}_heatmap.png")
            print(f"- output/{location}/figures/{feature_name}_depth_group_timeseries.png")
            print(f"- output/{location}/figures/{feature_name}_surface_middepth_comparison.png")
        except Exception as e:
            print(f"\nError processing location {location}")
            print(f"Error message: {str(e)}")
    
    if not processed_locations:
        print("\nNo locations were successfully processed.")
        return None, None
    
    try:
        # Calculate and display average RMSE across all locations
        overall_mean_rmse = np.mean(all_rmse_values)
        overall_std_rmse = np.std(all_rmse_values)
        print(f"\nOverall RMSE across all locations for {feature_name}:")
        print(f"Mean RMSE: {overall_mean_rmse:.4f} ± {overall_std_rmse:.4f}")
        
        # Only attempt to combine if we have processed locations
        df_combined = combine_depth_groups(processed_locations, feature_name)
        print(f"\nProcessing complete for {feature_name} data across all locations.")
        return results, df_combined
    except Exception as e:
        print(f"\nError combining depth groups: {str(e)}")
        return results, None

if __name__ == "__main__":
    main("ph", removal_percentage=0.05)  # or any other percentage