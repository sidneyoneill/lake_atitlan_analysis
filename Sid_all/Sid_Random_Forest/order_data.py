import pandas as pd

def order_data(input_file, output_file='data_ordered.xlsx', sort_columns=None, ascending=True):
    """
    Read data, sort it, and save to a new Excel file.
    
    Args:
        input_file (str): Path to input file (csv or xlsx)
        output_file (str): Path to save ordered data (default: 'data_ordered.xlsx')
        sort_columns (str or list): Column(s) to sort by (default: None)
        ascending (bool or list): Sort ascending or descending (default: True)
    """
    # Read the data
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file, parse_dates=True)
    else:
        df = pd.read_excel(input_file, parse_dates=True)
    
    # Convert date columns to datetime
    date_columns = df.select_dtypes(include=['object']).columns
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            continue
    
    # Sort the data if columns specified
    if sort_columns:
        df = df.sort_values(by=sort_columns, ascending=ascending)
    
    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Ordered data saved to {output_file}")

def main():
    # Example usage
    input_file = "data/EVAN_LIMNO_processed_v4.csv"
    sort_columns = ["date"]  # Adjust this to your actual date column name
    
    order_data(
        input_file=input_file,
        output_file='data_ordered.xlsx',
        sort_columns=sort_columns,
        ascending=True
    )

if __name__ == "__main__":
    main()
