import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import os

def analyze_chlorophyll_missingness(df, location_col='location', depth_col='depth', 
                                  time_col='date', chlorophyll_col='chlorophyll_a'):
    """
    Analyze and visualize missing chlorophyll-a values for each location.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe containing the measurements
    location_col : str
        Name of the location column
    depth_col : str
        Name of the depth column
    time_col : str
        Name of the time column
    chlorophyll_col : str
        Name of the chlorophyll column
    
    Returns:
    --------
    dict
        Dictionary of location-specific heatmaps
    """
    locations = df[location_col].unique()
    figures = {}
    
    for location in locations:
        df_loc = df[df[location_col] == location].copy()
        
        # Create missingness analysis
        df_loc['missing'] = df_loc[chlorophyll_col].isna().astype(int)
        pivot_df = df_loc.pivot_table(
            values='missing',
            index=depth_col,
            columns=pd.to_datetime(df_loc[time_col]).dt.date,
            aggfunc='mean'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.heatmap(pivot_df, cmap='YlOrRd', cbar_kws={'label': 'Missing Data Ratio'})
        plt.title(f'Chlorophyll-a Missing Values Distribution - {location}')
        plt.xlabel('Date')
        plt.ylabel('Depth (m)')
        
        figures[location] = fig
    
    return figures

def analyze_correlations(df, chlorophyll_col='chlorophyll_a'):
    """
    Analyze correlations between chlorophyll-a and other variables.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe containing the measurements
    chlorophyll_col : str
        Name of the chlorophyll column
    
    Returns:
    --------
    correlations : pandas Series
        Correlation coefficients with chlorophyll-a
    fig : matplotlib figure
        Correlation heatmap
    """
    # Calculate correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[chlorophyll_col].sort_values(ascending=False)
    
    # Create correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Variables')
    
    return correlations, fig

def analyze_chlorophyll_depths(df, depth_col='depth', chlorophyll_col='chlorophyll_a'):
    """
    Count and analyze the number of chlorophyll measurements at each depth.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe containing the measurements
    depth_col : str
        Name of the depth column
    chlorophyll_col : str
        Name of the chlorophyll column
    
    Returns:
    --------
    depths_summary : dict
        Dictionary containing measurement counts by depth
    fig : matplotlib figure
        Bar plot showing number of measurements at each depth
    """
    # Count measurements at each depth (excluding NaN values)
    valid_measurements = df[~df[chlorophyll_col].isna()]
    depth_counts = valid_measurements[depth_col].value_counts().sort_index()
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    depth_counts.plot(kind='bar')
    plt.title('Number of Chlorophyll Measurements by Depth')
    plt.xlabel('Depth (m)')
    plt.ylabel('Number of Measurements')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    
    # Create summary dictionary
    depths_summary = {
        'total_measurements': len(valid_measurements),
        'measurements_by_depth': depth_counts.to_dict(),
        'average_measurements_per_depth': depth_counts.mean(),
        'max_measurements_at_depth': depth_counts.max(),
        'min_measurements_at_depth': depth_counts.min()
    }
    
    return depths_summary, fig

def create_measurement_matrix(df, location_col='location'):
    """
    Create separate measurement matrices for each location.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe containing the measurements
    location_col : str
        Name of the location column
        
    Returns:
    --------
    dict
        Dictionary of location-specific measurement matrices and figures
    """
    # Create output directories for each location
    locations = df[location_col].unique()
    matrices = {}
    figures = {}
    
    for location in locations:
        # Filter data for this location
        df_loc = df[df[location_col] == location].copy()
        
        # Convert dates to datetime
        df_loc['date'] = pd.to_datetime(df_loc['date'])
        
        # Create measurement matrix with actual chlorophyll values
        measurement_matrix = df_loc.pivot_table(
            values='chlorophyll_a',
            index='depth',
            columns=df_loc['date'].dt.date,
            aggfunc='first'
        )
        
        # Create existence matrix
        existence_matrix = df_loc.pivot_table(
            values='chlorophyll_a',
            index='depth',
            columns=df_loc['date'].dt.date,
            aggfunc='count'
        )
        existence_matrix = existence_matrix.notna()
        
        # Create final matrix
        final_matrix = measurement_matrix.copy()
        final_matrix = final_matrix.mask(~existence_matrix, 'N/A')
        
        # Create heatmap visualization
        fig, ax = plt.subplots(figsize=(15, 8))
        plot_matrix = final_matrix.replace('N/A', np.nan).astype(float)
        sns.heatmap(plot_matrix, cmap='viridis', 
                    cbar_kws={'label': 'Chlorophyll-a Measurement'})
        plt.title(f'Chlorophyll-a Measurements Distribution - {location}')
        plt.xlabel('Date')
        plt.ylabel('Depth (m)')
        
        matrices[location] = final_matrix
        figures[location] = fig
    
    return matrices, figures

