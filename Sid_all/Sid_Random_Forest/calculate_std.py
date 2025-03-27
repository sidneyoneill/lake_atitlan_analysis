import pandas as pd
import numpy as np

def calculate_feature_std():
    # Read the CSV file
    df = pd.read_csv('data/SID_LIMNO_processed_v5.csv')
    
    # List of features to calculate std for (excluding grouping columns and date)
    features = ['temp', 'chlorophyll_a', 'ph', 'dissolved_oxygen', 'secchi', 
                'biochemical_oxygen_demand', 'total_dissolved_solids', 'turbidity', 
                'nitrate', 'phosphate', 'ammonium', 'phosphorus']
    
    # Group by location and depth_group and calculate std for each feature
    std_by_group = df.groupby(['location', 'depth_group'])[features].std()
    
    # Reset index to make location and depth_group regular columns
    std_by_group = std_by_group.reset_index()
    
    # Save results to CSV
    std_by_group.to_csv('feature_std_by_location_depth.csv', index=False)
    
    return std_by_group

if __name__ == "__main__":
    std_results = calculate_feature_std()
    print("Standard deviations calculated and saved to 'feature_std_by_location_depth.csv'")
    print("\nFirst few rows of results:")
    print(std_results.head())
