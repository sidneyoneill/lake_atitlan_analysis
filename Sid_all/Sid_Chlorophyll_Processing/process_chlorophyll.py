from process_functions import (create_depth_grid, assign_to_grid_depth, 
                             interpolate_vertical_gaps, impute_horizontal_gaps,
                             aggregate_depth_groups, analyze_original_data)
from visualize_chlorophyll import (plot_chlorophyll_heatmap, 
                                 plot_depth_group_timeseries,
                                 plot_surface_middepth_comparison,
                                 create_data_quality_report)
import pandas as pd
import os

def process_location(location, depth_grid, column='chlorophyll'):
    """
    Process chlorophyll data for a specific location.
    
    Parameters:
    -----------
    location : str
        Location identifier
    depth_grid : numpy.ndarray
        Array of regularly spaced depth values
    column : str
        Name of the column to process (default: 'chlorophyll')
        
    Returns:
    --------
    tuple
        (df_imputed, df_grouped, df_pivot) for the location
    """
    print(f"\nProcessing location: {location} for {column}")
    
    # Create location-specific output directory
    os.makedirs(f'output/{location}/figures', exist_ok=True)
    
    # Analyze original data
    measurement_matrix_path = f'output/{location}/measurement_matrix.csv'
    # df_original_analysis = analyze_original_data(measurement_matrix_path, depth_grid, column)
    
    # Assign measurements to closest grid depths
    df_assigned = assign_to_grid_depth(measurement_matrix_path, depth_grid, column)
    
    # Interpolate vertical gaps
    df_interpolated = interpolate_vertical_gaps(df_assigned)
    
    # Impute horizontal gaps using KNN
    df_imputed = impute_horizontal_gaps(df_interpolated, n_neighbors=5, weights='distance')
    
    # Aggregate into depth groups
    df_grouped, df_pivot = aggregate_depth_groups(df_imputed)
    
    # Save the results with column name in file paths
    df_imputed.to_csv(f'output/{location}/{column}_imputed_matrix.csv')
    df_grouped.to_csv(f'output/{location}/{column}_depth_groups_long.csv', index=False)
    df_pivot.to_csv(f'output/{location}/{column}_depth_groups_pivot.csv')
    
    # Create visualizations and quality report with column name in file paths
    plot_chlorophyll_heatmap(df_imputed, output_path=f'output/{location}/figures/{column}_heatmap.png')
    plot_depth_group_timeseries(df_grouped, output_path=f'output/{location}/figures/{column}_depth_group_timeseries.png')
    plot_surface_middepth_comparison(df_grouped, output_path=f'output/{location}/figures/{column}_surface_middepth_comparison.png')
    report = create_data_quality_report(df_assigned, df_interpolated, df_imputed, df_grouped)
    
    # Save report with column name
    with open(f'output/{location}/{column}_data_quality_report.txt', 'w') as f:
        f.write(report)
    
    # Print location-specific summary
    print(f"\nProcessing complete for {location} {column}:")
    print(f"- Original missing values: {df_assigned.isna().sum().sum()}")
    print(f"- After vertical interpolation: {df_interpolated.isna().sum().sum()}")
    print(f"- After horizontal imputation: {df_imputed.isna().sum().sum()}")
    
    return df_imputed, df_grouped, df_pivot

def combine_depth_groups(locations, column='chlorophyll'):
    """
    Combine depth group data from all locations into a single DataFrame.
    
    Parameters:
    -----------
    locations : list
        List of location identifiers
    column : str
        Name of the column being processed (default: 'chlorophyll')
        
    Returns:
    --------
    pandas.DataFrame
        Combined depth groups data from all locations
    """
    dfs = []
    
    for location in locations:
        df = pd.read_csv(f'output/{location}/{column}_depth_groups_long.csv')
        df['location'] = location
        dfs.append(df)
    
    # Combine all DataFrames
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Reorder columns to put location first
    cols = ['location'] + [col for col in df_combined.columns if col != 'location']
    df_combined = df_combined[cols]
    
    # Save combined file with column name
    df_combined.to_csv(f'output/combined_{column}_depth_groups.csv', index=False)
    
    # Print summary
    print(f"\nCombined {column} Depth Groups Summary:")
    print(f"Total rows: {len(df_combined)}")
    print("\nMeasurements per location:")
    print(df_combined.groupby('location').size())
    print(f"\nCombined file saved to: output/combined_{column}_depth_groups.csv")
    
    return df_combined

def main():
    # Create the depth grid (same for all locations)
    depth_grid = create_depth_grid(
        photic_zone_depth=30,  # 30m photic zone
        max_depth=250,         # Maximum depth of 250m
        photic_interval=1,     # 1m intervals in photic zone
        deep_interval=10       # 10m intervals in deep zone
    )
    
    # Get list of locations from measurement matrices
    locations = [d for d in os.listdir('output') 
                if os.path.isdir(os.path.join('output', d)) and 
                os.path.exists(f'output/{d}/measurement_matrix.csv')]
    
    # Process each location
    results = {}
    column = 'temp'  # Default column, could be made configurable
    for location in locations:
        results[location] = process_location(location, depth_grid, column)
        
        print(f"\nOutputs for {location} saved to:")
        print(f"- output/{location}/{column}_imputed_matrix.csv")
        print(f"- output/{location}/{column}_depth_groups_long.csv")
        print(f"- output/{location}/{column}_depth_groups_pivot.csv")
        print(f"- output/{location}/{column}_data_quality_report.txt")
        print(f"- output/{location}/figures/{column}_heatmap.png")
        print(f"- output/{location}/figures/{column}_depth_group_timeseries.png")
        print(f"- output/{location}/figures/{column}_surface_middepth_comparison.png")
    
    # Combine depth groups from all locations
    df_combined = combine_depth_groups(locations, column)
    
    return results, df_combined

if __name__ == "__main__":
    main()