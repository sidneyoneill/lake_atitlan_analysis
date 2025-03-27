import pandas as pd

def fill_empty_values(df):
    """
    Fill empty values in secchi and biochemical_demand columns by copying surface values
    to all deeper measurements within the same group.
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Get surface values for each group
    surface_values = df[df['depth_group'] == 'Surface'].copy()
    
    # Columns to fill
    columns_to_fill = ['secchi', 'biochemical_oxygen_demand']
    
    # For each unique date-station combination
    for _, surface_row in surface_values.iterrows():
        # Find all rows with same date and station
        mask = ((df['date'] == surface_row['date']) & 
                (df['location'] == surface_row['location']))
        
        # Copy surface values to all depths for these columns
        for col in columns_to_fill:
            df.loc[mask, col] = surface_row[col]
    
    return df

def main():
    # Read the input file
    input_path = 'data/SID_LIMNO_processed.csv'
    output_path = 'output/SID_LIMNO_processed_V2.csv'
    
    # Read the CSV file
    df = pd.read_csv(input_path)
    
    # Apply the filling function
    df_filled = fill_empty_values(df)
    
    # Save the processed dataframe
    df_filled.to_csv(output_path, index=False)
    print(f"Processed file saved to: {output_path}")

if __name__ == "__main__":
    main()