import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path):
    """
    Load and clean water quality data from Excel file.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with temporal and static features
    """
    try:
        # Load the Excel file
        excel_file = pd.ExcelFile(file_path)
        
        # Verify that Sheet1 exists
        if 'Sheet1' not in excel_file.sheet_names:
            raise ValueError("Sheet1 not found in Excel file")
            
        # Load the data
        df = excel_file.parse('Sheet1')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Handle missing values in time
        if 'time' in df.columns:
            df['time'] = df['time'].fillna('00:00:00')
        
        # Step 2: Create numeric time index
        if 'date' in df.columns:
            df['time_idx'] = (df['date'] - df['date'].min()).dt.days
        else:
            raise ValueError("Date column not found in dataset")
            
        # Step 3: Create group identifier
        df['group_id'] = df['location'] + "_" + df['depth (m)'].astype(str)
        
        # Step 4: Handle missing values for depth-dependent variables
        depth_dependent_vars = [
            "temp. (c)", "ph (units)", "dissolved oxygen (mg l)", "turbidity (ntu)"
        ]
        existing_depth_dependent_vars = [
            var for var in depth_dependent_vars if var in df.columns
        ]
        df[existing_depth_dependent_vars] = df[existing_depth_dependent_vars].interpolate(method='linear', axis=0)
        
        # Step 5: Handle depth-independent temporal variables
        depth_independent_vars = ["biochemical oxygen demand (mg l)", "secchi depth (m)"]
        existing_depth_independent_vars = [
            var for var in depth_independent_vars if var in df.columns
        ]
        for var in existing_depth_independent_vars:
            # Propagate depth-independent variables across depths
            df[var] = df.groupby(['location', 'time_idx'])[var].transform('first')
        
        # Step 6: Define feature groups
        static_features = ["location", "depth (m)"]
        temporal_features = ["time_idx"] + existing_depth_dependent_vars + existing_depth_independent_vars
        
        # Step 7: Sort by group_id and time index
        df = df.sort_values(by=["group_id", "time_idx"])
            
        # Print basic data quality information
        print(f"Loaded {len(df)} rows of data")
        print("\nMissing values summary:")
        print(df.isnull().sum())
        
        print("\nFeature groups:")
        print("Temporal features:", temporal_features)
        print("Static features:", static_features)
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None



def save_cleaned_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame to save
        output_path (str): Path where the CSV file should be saved
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the DataFrame to CSV
        df.to_csv(output_path, index=False)
        print(f"Successfully saved cleaned data to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        return False

# Load and clean the dataset
data_file_path = 'data/water_quality.xlsx'
output_file_path = 'output/cleaned_water_quality_v4.csv'
df_uploaded = load_and_clean_data(data_file_path)

if df_uploaded is not None:
    # Display the first few rows of cleaned data
    print("\nFirst few rows of cleaned data:")
    print(df_uploaded.head())
    
    # Save the cleaned data
    save_cleaned_data(df_uploaded, output_file_path)