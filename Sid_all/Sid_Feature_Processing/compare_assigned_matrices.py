import pandas as pd
import os

def load_matrix(location, feature):
    """
    Load a matrix from the specified location and feature.
    
    Args:
        location (str): Location code (SA, WG, or WP)
        feature (str): Feature name (temp or chlorophyll_a)
        
    Returns:
        pandas.DataFrame: Loaded matrix
    """
    filepath = f"output/{location}/{feature}/assigned_matrix.csv"
    return pd.read_csv(filepath, index_col=0)

def find_missing_elements(df1, df2, df1_name, df2_name):
    """
    Find missing rows and columns between two dataframes.
    
    Args:
        df1 (pandas.DataFrame): First dataframe
        df2 (pandas.DataFrame): Second dataframe
        df1_name (str): Name of first dataframe
        df2_name (str): Name of second dataframe
    """
    # Compare row indices
    rows_in_1_not_2 = set(df1.index) - set(df2.index)
    rows_in_2_not_1 = set(df2.index) - set(df1.index)
    
    # Compare column names
    cols_in_1_not_2 = set(df1.columns) - set(df2.columns)
    cols_in_2_not_1 = set(df2.columns) - set(df1.columns)
    
    # Print results if there are differences
    if rows_in_1_not_2:
        print(f"\nRows in {df1_name} but not in {df2_name}:")
        print(sorted(rows_in_1_not_2))
    
    if rows_in_2_not_1:
        print(f"\nRows in {df2_name} but not in {df1_name}:")
        print(sorted(rows_in_2_not_1))
    
    if cols_in_1_not_2:
        print(f"\nColumns in {df1_name} but not in {df2_name}:")
        print(sorted(cols_in_1_not_2))
    
    if cols_in_2_not_1:
        print(f"\nColumns in {df2_name} but not in {df1_name}:")
        print(sorted(cols_in_2_not_1))

def check_duplicates(df, matrix_name, location):
    """
    Check for duplicate rows and columns in a matrix.
    
    Args:
        df (pandas.DataFrame): Matrix to check
        matrix_name (str): Name of the matrix (Temperature or Chlorophyll-a)
        location (str): Location code
        
    Returns:
        bool: True if no duplicates found, False otherwise
    """
    has_duplicates = False
    
    # Check for duplicate row indices
    duplicate_rows = df.index[df.index.duplicated()].unique()
    if len(duplicate_rows) > 0:
        print(f"\nWarning: Found duplicate rows in {location} {matrix_name} matrix:")
        print(f"Duplicate rows: {sorted(duplicate_rows)}")
        has_duplicates = True
    
    # Check for duplicate column names
    duplicate_cols = df.columns[df.columns.duplicated()].unique()
    if len(duplicate_cols) > 0:
        print(f"\nWarning: Found duplicate columns in {location} {matrix_name} matrix:")
        print(f"Duplicate columns: {sorted(duplicate_cols)}")
        has_duplicates = True
    
    return not has_duplicates

def add_missing_elements(df1, df2, df1_name, df2_name):
    """
    Add missing rows and columns to each dataframe with NaN values.
    
    Args:
        df1 (pandas.DataFrame): First dataframe
        df2 (pandas.DataFrame): Second dataframe
        df1_name (str): Name of first dataframe
        df2_name (str): Name of second dataframe
        
    Returns:
        tuple: (updated_df1, updated_df2) with aligned rows and columns
    """
    # Create copies to avoid modifying original dataframes
    df1_updated = df1.copy()
    df2_updated = df2.copy()
    
    # Find missing rows in each dataframe
    rows_in_1_not_2 = set(df1.index) - set(df2.index)
    rows_in_2_not_1 = set(df2.index) - set(df1.index)
    
    # Find missing columns in each dataframe
    cols_in_1_not_2 = set(df1.columns) - set(df2.columns)
    cols_in_2_not_1 = set(df2.columns) - set(df1.columns)
    
    # Add missing rows to df2
    if rows_in_1_not_2:
        print(f"\nAdding missing rows to {df2_name} from {df1_name}:")
        print(sorted(rows_in_1_not_2))
        for row in rows_in_1_not_2:
            df2_updated.loc[row] = float('nan')
    
    # Add missing rows to df1
    if rows_in_2_not_1:
        print(f"\nAdding missing rows to {df1_name} from {df2_name}:")
        print(sorted(rows_in_2_not_1))
        for row in rows_in_2_not_1:
            df1_updated.loc[row] = float('nan')
    
    # Add missing columns to df2
    if cols_in_1_not_2:
        print(f"\nAdding missing columns to {df2_name} from {df1_name}:")
        print(sorted(cols_in_1_not_2))
        for col in cols_in_1_not_2:
            df2_updated[col] = float('nan')
    
    # Add missing columns to df1
    if cols_in_2_not_1:
        print(f"\nAdding missing columns to {df1_name} from {df2_name}:")
        print(sorted(cols_in_2_not_1))
        for col in cols_in_2_not_1:
            df1_updated[col] = float('nan')
    
    # Sort indices and columns to ensure alignment
    df1_updated.sort_index(inplace=True)
    df2_updated.sort_index(inplace=True)
    df1_updated = df1_updated.reindex(sorted(df1_updated.columns), axis=1)
    df2_updated = df2_updated.reindex(sorted(df2_updated.columns), axis=1)
    
    return df1_updated, df2_updated

def compare_shapes(temp_df, chl_df, location):
    """
    Compare shapes of temperature and chlorophyll-a matrices for a given location.
    
    Args:
        temp_df (pandas.DataFrame): Temperature matrix
        chl_df (pandas.DataFrame): Chlorophyll-a matrix
        location (str): Location code
        
    Returns:
        tuple: (updated_temp_df, updated_chl_df) with aligned dimensions
    """
    # First check for duplicates
    temp_ok = check_duplicates(temp_df, "Temperature", location)
    chl_ok = check_duplicates(chl_df, "Chlorophyll-a", location)
    
    if not (temp_ok and chl_ok):
        print(f"\n✗ Found duplicate entries in {location} matrices")
        return temp_df, chl_df
    
    temp_shape = temp_df.shape
    chl_shape = chl_df.shape
    
    print(f"\n{location} Matrix Shapes:")
    print(f"Temperature: {temp_shape} (rows × columns)")
    print(f"Chlorophyll-a: {chl_shape} (rows × columns)")
    
    if temp_shape == chl_shape:
        print("✓ Matrices have matching shapes")
        return temp_df, chl_df
    else:
        print("✗ Matrices have different shapes")
        find_missing_elements(temp_df, chl_df, "Temperature", "Chlorophyll-a")
        # Add missing rows and columns to both matrices
        updated_temp_df, updated_chl_df = add_missing_elements(temp_df, chl_df, "Temperature", "Chlorophyll-a")
        return updated_temp_df, updated_chl_df

def save_matrix(df, location, feature):
    """
    Save a matrix to its corresponding CSV file in the output folder.
    
    Args:
        df (pandas.DataFrame): Matrix to save
        location (str): Location code (SA, WG, or WP)
        feature (str): Feature name (temp or chlorophyll_a)
    """
    filepath = f"output/{location}/{feature}/assigned_matrix.csv"
    df.to_csv(filepath)
    print(f"Updated matrix saved to: {filepath}")

def main():
    # Load all matrices
    try:
        # South Atlantic (SA)
        sa_temp = load_matrix('SA', 'temp')
        sa_chl = load_matrix('SA', 'chlorophyll_a')
        
        # Weddell Gyre (WG)
        wg_temp = load_matrix('WG', 'temp')
        wg_chl = load_matrix('WG', 'chlorophyll_a')
        
        # Western Pacific (WP)
        wp_temp = load_matrix('WP', 'temp')
        wp_chl = load_matrix('WP', 'chlorophyll_a')
        
        print("All matrices loaded successfully")
        
        # Compare shapes and get updated matrices for each location
        sa_temp_updated, sa_chl_updated = compare_shapes(sa_temp, sa_chl, 'SA')
        wg_temp_updated, wg_chl_updated = compare_shapes(wg_temp, wg_chl, 'WG')
        wp_temp_updated, wp_chl_updated = compare_shapes(wp_temp, wp_chl, 'WP')
        
        # Save updated matrices if they were modified
        if not sa_temp.equals(sa_temp_updated) or not sa_chl.equals(sa_chl_updated):
            save_matrix(sa_temp_updated, 'SA', 'temp')
            save_matrix(sa_chl_updated, 'SA', 'chlorophyll_a')
            
        if not wg_temp.equals(wg_temp_updated) or not wg_chl.equals(wg_chl_updated):
            save_matrix(wg_temp_updated, 'WG', 'temp')
            save_matrix(wg_chl_updated, 'WG', 'chlorophyll_a')
            
        if not wp_temp.equals(wp_temp_updated) or not wp_chl.equals(wp_chl_updated):
            save_matrix(wp_temp_updated, 'WP', 'temp')
            save_matrix(wp_chl_updated, 'WP', 'chlorophyll_a')
        
    except FileNotFoundError as e:
        print(f"Error loading matrices: {e}")

if __name__ == "__main__":
    main()
