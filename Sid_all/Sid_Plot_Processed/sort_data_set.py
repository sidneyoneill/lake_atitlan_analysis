import pandas as pd

def sort_dataset(df):
    """
    Sort the dataset by date primarily, then by location.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        
    Returns:
        pandas.DataFrame: Sorted DataFrame
    """
    # Create a copy of the dataframe
    df = df.copy()
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date first, then location
    df_sorted = df.sort_values(['date', 'location'], ascending=[True, True])
    
    return df_sorted

# Load the csv file
file_path = "output/grouped_depths.csv"
df = pd.read_csv(file_path)

# Sort the dataset
df_sorted = sort_dataset(df)

# Save the sorted dataset to a new CSV file in the output directory
output_path = "output/sorted_depths.csv"
df_sorted.to_csv(output_path, index=False)

