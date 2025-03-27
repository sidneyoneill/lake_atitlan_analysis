import pandas as pd
import os
from pathlib import Path

def count_nonzero_entries(matrix_path):
    """
    Count number of non-zero entries in a measurement matrix
    """
    try:
        df = pd.read_csv(matrix_path)
        return (df != 0).sum().sum()
    except Exception as e:
        print(f"Error reading {matrix_path}: {e}")
        return 0

def get_total_nonzero_entries(feature, output_dir="output"):
    """
    Calculate total non-zero entries for a feature across all locations
    """
    total = 0
    locations = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    
    for location in locations:
        matrix_path = Path(output_dir) / location / feature / "measurement_matrix.csv"
        if matrix_path.exists():
            count = count_nonzero_entries(matrix_path)
            print(f"Location {location}, Feature {feature}: {count} non-zero entries")
            total += count
        else:
            print(f"No measurement matrix found for {feature} in {location}")
    
    return total

def main():
    # List of main features to analyze
    features = [
        "temp",
        "chlorophyll_a",
        "ph",
        "dissolved_oxygen",
        "total_dissolved_solids",
        "turbidity",
        "biochemical_oxygen_demand",
        "secchi",
        "nitrate",
        "phosphate",
        "ammonium",
        "phosphorus",
    ]
    
    print("Calculating total non-zero entries for each feature...")
    
    for feature in features:
        total = get_total_nonzero_entries(feature)
        print(f"\nTotal non-zero entries for {feature}: {total}")
        print("-" * 50)

if __name__ == "__main__":
    main()
