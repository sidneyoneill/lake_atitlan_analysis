import pandas as pd
import matplotlib.pyplot as plt
import os
from analysis_functions import analyze_correlations, analyze_feature_depths, create_measurement_matrix, analyze_feature_missingness
from visualize_functions import plot_depth_group_timeseries

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists('output'):
        os.makedirs('output')
        os.makedirs('output/figures')

def prepare_grouped_data(df, feature_name):
    """Prepare data for depth group time series plotting"""
    # Create depth groups
    df['depth_group'] = pd.cut(df['depth'],
                              bins=[-float('inf'), 2, 15, float('inf')],
                              labels=['Surface', 'Mid-Depth', 'Bottom'])
    
    # Group by date and depth_group
    grouped = df.groupby(['date', 'depth_group']).agg({
        feature_name: ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['date', 'depth_group', 
                      f'mean_{feature_name}', 
                      f'std_{feature_name}', 
                      'measurement_count']
    
    return grouped

def main():
    # Create output directory
    ensure_output_dir()
    
    # Define the features to analyze
    features = ['turbidity']
    
    # Load the dataset
    df = pd.read_excel('data/LIMNO.xlsx')
    
    # Get unique locations
    locations = df['location'].unique()
    
    # Open summary file
    with open('output/measurement_matrices_summary.txt', 'w') as summary_file:
        # Process each feature
        for feature_name in features:
            print(f"\nProcessing feature: {feature_name}")
            summary_file.write(f"\nProcessing feature: {feature_name}\n")
            
            # Create measurement matrices for each location
            matrices, matrix_figs = create_measurement_matrix(df, feature_name=feature_name)
            
            # Create depth group time series for each location
            for location in locations:
                # Create directories
                os.makedirs(f'output/{location}/{feature_name}', exist_ok=True)
                
                # Filter data for this location
                location_data = df[df['location'] == location].copy()
                
                # Prepare grouped data
                grouped_data = prepare_grouped_data(location_data, feature_name)
                
                # Create and save time series plot
                plot_depth_group_timeseries(
                    grouped_data, 
                    feature_name,
                    output_path=f'output/{location}/{feature_name}/depth_group_timeseries.png'
                )
                
                # Save matrix
                matrices[location].to_csv(f'output/{location}/{feature_name}/measurement_matrix.csv')
                
                # Save matrix visualization
                matrix_figs[location].savefig(f'output/{location}/{feature_name}/presence_matrix.png')
                plt.close(matrix_figs[location])
                
                # Print and write matrix summary
                summary = f"\nMeasurement Matrix Summary for {feature_name} - {location}:\n"
                summary += f"Total dates: {matrices[location].shape[1]}\n"
                summary += f"Total depths: {matrices[location].shape[0]}\n"
                summary += f"Total possible measurements: {matrices[location].shape[0] * matrices[location].shape[1]}\n"
                summary += f"Matrix saved to: output/{location}/{feature_name}/measurement_matrix.csv\n"
                summary += f"Time series plot saved to: output/{location}/{feature_name}/depth_group_timeseries.png\n"
                
                print(summary)
                summary_file.write(summary)
            
            # Analyze missingness patterns
            missingness_figs = analyze_feature_missingness(df, feature_name=feature_name)
            for location in locations:
                missingness_figs[location].savefig(f'output/{location}/{feature_name}/missingness.png')
                plt.close(missingness_figs[location])
            
            print(f"\nProcessing complete for {feature_name}.")
            summary_file.write(f"\nProcessing complete for {feature_name}.\n")
        
        print("\nAll features have been processed. Check the output directory for results.")
        summary_file.write("\nAll features have been processed. Check the output directory for results.\n")

if __name__ == "__main__":
    main()
