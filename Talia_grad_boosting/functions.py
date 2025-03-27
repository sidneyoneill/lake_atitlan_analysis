# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:47:54 2025

@author: talia
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    """
    Loads data from a specified Excel file path
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pandas.DataFrame: Loaded data, or None if loading fails
    """
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        print("Data loaded successfully!")
        print(f"Shape of the data: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
    
    
def plot_results(y_test, y_pred, split, dates, feature_name, output_path=None):
    """
    Visualizes the comparison between predicted and actual values over time using line and scatter plots.

    Args:
        y_test: Actual values
        y_pred: Predicted values
        dates: Datetime values for x-axis
        output_path: Optional path to save the plot (e.g., 'outputs/figures/prediction_plot.png')
    """
    plt.figure(figsize=(12, 6))
    
    # Line plots fllor actual and predicted values
    plt.plot(dates[:split], y_test[:split], 'b:', alpha=0.7, label='Train (Actual)', linewidth=1)
    plt.plot(dates[:split], y_pred[:split], 'b-', alpha=0.7, label='Train (Predicted)')
    
    plt.plot(dates[split:], y_test[split:], 'r:', alpha=0.7, label='Test (Actual)', linewidth=1)
    plt.plot(dates[split:], y_pred[split:], 'r-', alpha=0.7, label='Test (Predicted)')
    # Scatter points for better visibility
    #plt.scatter(dates, y_test, color='blue', alpha=0.5, s=20, label='Actual Points')
    #plt.scatter(dates, y_pred, color='red', alpha=0.5, s=20, label='Predicted Points')
    
    # Title and labels
    plt.title(f'Predicting {feature_name} using other variables with Gradient Boosting model', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel(feature_name, fontsize=14)
    
    # Rotate date labels for better readability
    plt.gcf().autofmt_xdate()
    
    # Gridlines
    plt.grid(True, alpha=0.3)
    
    # Legend
    plt.legend()
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    # # Show the plot
    # plt.show()
    
def plot_comp(sid_results, evan_results, location, depth_group, output_path, target_variable, plot_option='sid'):
    """Create comparison plot based on specified option (sid, evan, or both)"""
    
    if plot_option == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        axes = [ax]
    
    def plot_dataset(ax, results, dataset_name):
        ax.plot(results['train_data']['date'], results['train_data']['actual'], 
                'b:', alpha=0.5, label='Train (Actual)', linewidth=1)
        ax.plot(results['train_data']['date'], results['train_data']['predicted'], 
                'b-', alpha=0.5, label='Train (Predicted)')
        ax.plot(results['test_data']['date'], results['test_data']['actual'], 
                'r:', label='Test (Actual)', linewidth=1)
        ax.plot(results['test_data']['date'], results['test_data']['predicted'], 
                'r-', label='Test (Predicted)')
        ax.set_title(f'{dataset_name} Data - {location} at {depth_group}\n(n={results["metrics"]["n_samples"]} samples)')
        ax.legend()
        ax.set_ylabel(target_variable)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    if plot_option in ['both', 'sid']:
        plot_dataset(axes[0], sid_results, 'SID')
    
    if plot_option in ['both', 'evan']:
        plot_idx = 0 if plot_option == 'evan' else 1
        plot_dataset(axes[plot_idx], evan_results, 'EVAN')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()