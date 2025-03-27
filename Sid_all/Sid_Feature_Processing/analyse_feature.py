import pandas as pd
import matplotlib.pyplot as plt
import os
from analysis_functions import  analyze_correlations, analyze_feature_depths, create_measurement_matrix, analyze_feature_missingness

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    if not os.path.exists('output'):
        os.makedirs('output')

def main():
    # Create output directory
    ensure_output_dir()
    
    # Define the feature to analyze
    feature_name = 'turbidity'  # This can be changed to any column name
    
    # Load the Excel dataset
    df = pd.read_excel('data/LIMNO.xlsx')
    
    # Get unique locations
    locations = df['location'].unique()
    
    # Create measurement matrices for each location
    matrices, matrix_figs = create_measurement_matrix(df, feature_name=feature_name)
    
    # Save matrices and figures for each location
    for location in locations:
        # Create location and feature-specific directory
        os.makedirs(f'output/{location}/{feature_name}', exist_ok=True)
        
        # Save matrix
        matrices[location].to_csv(f'output/{location}/{feature_name}/measurement_matrix.csv')
        
        # Save matrix visualization
        matrix_figs[location].savefig(f'output/{location}/{feature_name}/presence_matrix.png')
        plt.close(matrix_figs[location])
        
        # Print matrix summary for this location
        print(f"\nMeasurement Matrix Summary for {feature_name} - {location}:")
        print(f"Total dates: {matrices[location].shape[1]}")
        print(f"Total depths: {matrices[location].shape[0]}")
        print(f"Total possible measurements: {matrices[location].shape[0] * matrices[location].shape[1]}")
        print(f"Matrix saved to: output/{location}/{feature_name}/measurement_matrix.csv")
    
    # Analyze missingness patterns for each location
    missingness_figs = analyze_feature_missingness(df, feature_name=feature_name)
    for location in locations:
        missingness_figs[location].savefig(f'output/{location}/{feature_name}/missingness.png')
        plt.close(missingness_figs[location])
    
    print(f"\nProcessing complete. Check the output directory for location-specific {feature_name} results.")

if __name__ == "__main__":
    main()
