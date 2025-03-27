import pandas as pd
import os
from visualize_functions import (
    plot_feature_heatmap,
    plot_depth_group_timeseries,
    plot_surface_middepth_comparison,
    create_data_quality_report
)

def plot_processed_feature(feature_name, location):
    """
    Create visualizations for a processed feature using its depth_groups_long data.
    
    Parameters:
    -----------
    feature_name : str
        Name of the feature to plot (e.g., 'temp', 'chlorophyll_a')
    location : str
        Location identifier (e.g., 'SA', 'WG', 'WP')
    """
    # Create output directory for figures if it doesn't exist
    os.makedirs(f'output/{location}/figures', exist_ok=True)
    
    try:
        # Read the processed data from the correct path
        df = pd.read_csv(f'output/{location}/{feature_name}/depth_groups_long.csv')
        
        # Convert the long format data to wide format for heatmap
        df_pivot = df.pivot(
            index='depth',
            columns='date',
            values=f'mean_{feature_name}'
        )
        
        # Generate all plots with location-specific output paths
        plot_feature_heatmap(df_pivot, feature_name, 
                           f'output/{location}/figures/{feature_name}_heatmap.png')
        plot_depth_group_timeseries(df, feature_name,
                                  f'output/{location}/figures/{feature_name}_depth_group_timeseries.png')
        plot_surface_middepth_comparison(df, feature_name,
                                       f'output/{location}/figures/{feature_name}_surface_middepth_comparison.png')
        
        # Create data quality report
        report = create_data_quality_report(
            df_pivot,  # Using pivot as assigned
            df_pivot,  # Using pivot as interpolated
            df_pivot,  # Using pivot as imputed
            df,        # Using long format as grouped
            feature_name
        )
        
        print(f"\nPlots and report generated successfully for {feature_name} at {location}!")
        print(f"Check 'output/{location}/figures' directory for the following files:")
        print(f"- {feature_name}_heatmap.png")
        print(f"- {feature_name}_depth_group_timeseries.png")
        print(f"- {feature_name}_surface_middepth_comparison.png")
        print(f"Data quality report saved as: output/{location}/{feature_name}_data_quality_report.txt")
        
    except Exception as e:
        print(f"Error processing {feature_name} at {location}: {str(e)}")

def main():
    """
    Main function to plot processed features for each location.
    """
    # List of features and locations to plot
 # Define the features to analyze
    features = ['temp', 'chlorophyll_a', 'total_dissolved_solids', 'ph', 'dissolved_oxygen', 'nitrate', 'phosphate', 'ammonium', 'phosphorus', 'nitrogen', 'biochemical_oxygen_demand', 'secchi']
    locations = ['SA', 'WG', 'WP']
    
    print("Starting to generate plots for processed features...")
    
    for location in locations:
        print(f"\nProcessing location: {location}")
        for feature in features:
            print(f"Processing {feature}...")
            plot_processed_feature(feature, location)
    
    print("\nPlotting complete!")

if __name__ == "__main__":
    main()
