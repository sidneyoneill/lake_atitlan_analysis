import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import os

def analyze_feature_missingness(df, feature_name, location_col='location', depth_col='depth', 
                              time_col='date'):
    """
    Analyze and visualize missing values for any feature at each location.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe containing the measurements
    feature_name : str
        Name of the feature column to analyze
    location_col : str
        Name of the location column
    depth_col : str
        Name of the depth column
    time_col : str
        Name of the time column
    
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
        df_loc['missing'] = df_loc[feature_name].isna().astype(int)
        pivot_df = df_loc.pivot_table(
            values='missing',
            index=depth_col,
            columns=pd.to_datetime(df_loc[time_col]).dt.date,
            aggfunc='mean'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.heatmap(pivot_df, cmap='YlOrRd', cbar_kws={'label': 'Missing Data Ratio'})
        plt.title(f'{feature_name} Missing Values Distribution - {location}')
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

def analyze_feature_depths(df, feature_name, depth_col='depth'):
    """
    Count and analyze the number of measurements at each depth for any feature.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe containing the measurements
    feature_name : str
        Name of the feature column to analyze
    depth_col : str
        Name of the depth column
    
    Returns:
    --------
    depths_summary : dict
        Dictionary containing measurement counts by depth
    fig : matplotlib figure
        Bar plot showing number of measurements at each depth
    """
    # Count measurements at each depth (excluding NaN values)
    valid_measurements = df[~df[feature_name].isna()]
    depth_counts = valid_measurements[depth_col].value_counts().sort_index()
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    depth_counts.plot(kind='bar')
    plt.title(f'Number of {feature_name} Measurements by Depth')
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

def create_measurement_matrix(df, feature_name, location_col='location'):
    """
    Create separate measurement matrices for each location.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe containing the measurements
    feature_name : str
        Name of the feature column to analyze
    location_col : str
        Name of the location column
        
    Returns:
    --------
    dict
        Dictionary of location-specific measurement matrices and figures
    """
    locations = df[location_col].unique()
    matrices = {}
    figures = {}
    
    for location in locations:
        # Filter data for this location
        df_loc = df[df[location_col] == location].copy()
        
        # Convert dates to datetime
        df_loc['date'] = pd.to_datetime(df_loc['date'])
        
        # Create measurement matrix with actual feature values
        measurement_matrix = df_loc.pivot_table(
            values=feature_name,
            index='depth',
            columns=df_loc['date'].dt.date,
            aggfunc='first'
        ).fillna('N/A')  # Fill all NaN values with 'N/A' immediately
        
        # Store the matrix and create visualization
        final_matrix = measurement_matrix  # No need for existence matrix anymore
        
        # Create heatmap visualization
        fig, ax = plt.subplots(figsize=(15, 8))
        plot_matrix = final_matrix.replace('N/A', np.nan).astype(float)
        sns.heatmap(plot_matrix, cmap='viridis', 
                    cbar_kws={'label': f'{feature_name} Measurement'})
        plt.title(f'{feature_name} Measurements Distribution - {location}')
        plt.xlabel('Date')
        plt.ylabel('Depth (m)')
        
        matrices[location] = final_matrix
        figures[location] = fig
    
    return matrices, figures

