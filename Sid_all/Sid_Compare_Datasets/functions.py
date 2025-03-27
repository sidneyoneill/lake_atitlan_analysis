import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os

def read_csv_file(file_path, **kwargs):
    """
    Read a CSV file and return a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv()
        
    Returns:
        pandas.DataFrame: DataFrame containing the CSV data
    """
    try:
        df = pd.read_csv(file_path, **kwargs)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None

def read_excel_file(file_path, sheet_name=0, **kwargs):
    """
    Read an Excel (XLSX) file and return a pandas DataFrame.
    
    Args:
        file_path (str): Path to the Excel file
        sheet_name (str or int): Name or index of the sheet to read (default: 0)
        **kwargs: Additional arguments to pass to pd.read_excel()
        
    Returns:
        pandas.DataFrame: DataFrame containing the Excel data
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return None

def combine_depth_rows(df):
    """
    Combine consecutive Mid-Depth and Lower Photic rows by averaging their values.
    Maintains original data order and only combines pairs of rows.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        
    Returns:
        pandas.DataFrame: DataFrame with combined depth rows
    """
    # Create a copy of the dataframe
    df = df.copy()
    
    # Convert date column to datetime if not already
    df.iloc[:, 2] = pd.to_datetime(df.iloc[:, 2])
    
    # Get indices where Mid-Depth and Lower Photic appear consecutively
    depth_col = df.iloc[:, 1]
    rows_to_combine = []
    
    for i in range(len(df)-1):
        if (depth_col.iloc[i] in ['Mid-Depth', 'Lower Photic'] and 
            depth_col.iloc[i+1] in ['Mid-Depth', 'Lower Photic']):
            rows_to_combine.append(i)
    
    # Create new dataframe with combined rows
    result = []
    i = 0
    while i < len(df):
        if i in rows_to_combine:
            # Average the two rows
            row1 = df.iloc[i]
            row2 = df.iloc[i+1]
            combined = row1.copy()
            # Average numerical values (columns from index 3 onwards)
            combined.iloc[3:] = (row1.iloc[3:] + row2.iloc[3:]) / 2
            combined.iloc[1] = 'Mid-Depth Combined'
            result.append(combined)
            i += 2
        else:
            result.append(df.iloc[i])
            i += 1
    
    return pd.DataFrame(result)

def rename_depth_groups(df):
    """
    Rename depth groups to their corresponding depth ranges.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        
    Returns:
        pandas.DataFrame: DataFrame with renamed depth groups
    """
    # Create a copy of the dataframe
    df = df.copy()
    
    # Define depth mapping
    depth_mapping = {
        'Surface': '0-10m',
        'Mid-Depth Combined': '10-30m',
        'Deep': '30m+'
    }
    
    # Apply mapping to the depth column (column index 1)
    df.iloc[:, 1] = df.iloc[:, 1].replace(depth_mapping)
    
    return df

def rename_depth_groups_evan(df):
    """
    Rename depth groups to standardized format without spaces.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        
    Returns:
        pandas.DataFrame: DataFrame with standardized depth group names
    """
    # Create a copy of the dataframe
    df = df.copy()
    
    # Define depth mapping
    depth_mapping = {
        '0-10 m': '0-10m',
        '10-30 m': '10-30m',
        '30+ m': '30m+'
    }
    
    # Apply mapping to the depth column (column index 1)
    df.iloc[:, 1] = df.iloc[:, 1].replace(depth_mapping)
    
    return df

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
    df.iloc[:, 2] = pd.to_datetime(df.iloc[:, 2])
    
    # Sort by date first, then location
    df_sorted = df.sort_values([df.columns[2], df.columns[0]], ascending=[True, True])
    
    return df_sorted

def plot_time_series(datasets, feature, dataset_names=None, output_dir='output'):
    """
    Plot time series data for multiple datasets for a specific feature at each location and depth.
    
    Args:
        datasets (list): List of DataFrames to plot
        feature (str): Name of the feature to plot
        dataset_names (list): List of names for each dataset (for legend)
        output_dir (str): Base directory for saving plots
    """
    # Set default dataset names if none provided
    if dataset_names is None:
        dataset_names = [f'Dataset {i+1}' for i in range(len(datasets))]
    
    # Define colors for different datasets
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    # Convert dates to datetime for all datasets
    for df in datasets:
        df.iloc[:, 2] = pd.to_datetime(df.iloc[:, 2])
    
    # Plot for each location and depth group
    for location in ['SA', 'WG', 'WP']:
        for depth in ['0-10m', '10-30m', '30m+']:
            plt.figure(figsize=(12, 6))
            
            # Plot each dataset
            for df, name, color in zip(datasets, dataset_names, colors):
                # Filter data for current location and depth
                mask = (df.iloc[:, 0] == location) & (df.iloc[:, 1] == depth)
                data = df[mask].sort_values(df.columns[2])  # Sort by date
                
                if not data.empty:
                    plt.plot(data.iloc[:, 2], data[feature], 'o-', 
                            color=color, label=name)
            
            plt.title(f'{feature} Time Series at {location} ({depth})')
            plt.xlabel('Date')
            plt.ylabel(feature)
            plt.grid(True)
            plt.legend()
            
            # Format date axis
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Create output directory if it doesn't exist
            plot_dir = os.path.join(output_dir, location, depth)
            os.makedirs(plot_dir, exist_ok=True)
            
            # Save the plot
            output_path = os.path.join(plot_dir, f'{feature}_time_series_comparison.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

