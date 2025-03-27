from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from functions import load_csv
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def train_rf_model(data, features, target_variable, location, depth_group):
    """Train RF model and return results and predictions"""
    
    # Filter data for current location and depth group
    filtered_data = data[
        (data['location'] == location) & 
        (data['depth_group'] == depth_group)
    ].copy()
    
    print(f"Data shape after filtering: {filtered_data.shape}")
    print(f"Location filter matches: {sum(data['location'] == location)}")
    print(f"Depth group filter matches: {sum(data['depth_group'] == depth_group)}")
    
    if len(filtered_data) < 30:
        print(f"Insufficient data: only {len(filtered_data)} samples")
        return None
    
    # Define features (X) and target variable (y)
    X = filtered_data[features]
    y = filtered_data[target_variable]
    dates = filtered_data['date']
    
    # Split data
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates, test_size=0.3, shuffle=False
    )
    
    # Hyperparameter optimization
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15]
    }
    rf_model = RandomForestRegressor(random_state=42)
    ts_cv = TimeSeriesSplit(n_splits=5)
    
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=ts_cv,
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X_train, y_train)
    
    # Get predictions for both train and test sets
    y_train_pred = grid_search.best_estimator_.predict(X_train)
    y_test_pred = grid_search.best_estimator_.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    # Calculate NSE
    mean_observed = np.mean(y_test)
    numerator = np.sum((y_test - y_test_pred) ** 2)
    denominator = np.sum((y_test - mean_observed) ** 2)
    nse = 1 - (numerator / denominator)
    
    return {
        'train_data': pd.DataFrame({
            'date': dates_train,
            'actual': y_train,
            'predicted': y_train_pred
        }).sort_values('date'),
        'test_data': pd.DataFrame({
            'date': dates_test,
            'actual': y_test,
            'predicted': y_test_pred
        }).sort_values('date'),
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'nse': nse,
            'best_params': grid_search.best_params_,
            'n_samples': len(filtered_data)
        }
    }

def plot_comparison(sid_results, evan_results, location, depth_group, output_path, target_variable, plot_option='both'):
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
        ax.set_title('Predicting Dissolved Oxygen Concentration using other variables with Random Forest model\n', fontsize=16)
        ax.legend()
        ax.set_ylabel('Dissolved Oxygen Concentration (mg/L)', fontsize=14)  # Changed units to micrograms per liter
        ax.set_xlabel('Date', fontsize=14)  # Added 'Date' as the label for the x-axis with font size 14
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(True, alpha=0.3)  # Added gridlines with 0.3 opacity
    
    if plot_option in ['both', 'sid']:
        plot_dataset(axes[0], sid_results, 'SID')
    
    if plot_option in ['both', 'evan']:
        plot_idx = 0 if plot_option == 'evan' else 1
        plot_dataset(axes[plot_idx], evan_results, 'EVAN')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def remove_nan_rows(data, features, target_variable):
    """Remove any rows containing NaN values in any column and return cleaned data"""
    initial_rows = len(data)
    cleaned_data = data.dropna()  # Remove any row with NaN in any column
    removed_rows = initial_rows - len(cleaned_data)
    
    print(f"\nNaN Row Removal Summary:")
    print(f"Initial rows: {initial_rows}")
    print(f"Rows removed: {removed_rows}")
    print(f"Remaining rows: {len(cleaned_data)}")
    print(f"Features used: {features}")
    
    return cleaned_data

def remove_nan_columns(data, features, target_variable, threshold=0.5):
    """Remove features with NaN values above threshold and return cleaned data with remaining features"""
    initial_features = features.copy()
    
    # Calculate NaN percentage for each feature
    nan_percentages = data[features].isna().mean()
    features_to_keep = nan_percentages[nan_percentages < threshold].index.tolist()
    
    print(f"\nNaN Column Removal Summary:")
    print(f"Initial features: {initial_features}")
    print(f"Features removed: {set(initial_features) - set(features_to_keep)}")
    print(f"Remaining features: {features_to_keep}")
    
    return data[features_to_keep + [target_variable, 'date', 'location', 'depth_group']], features_to_keep

def main():
    # Define paths
    sid_data_path = "data/SID_LIMNO_no_outliers_v2.csv"
    evan_data_path = "data/EVAN_LIMNO_processed_v5.csv"
    output_dir = f"outputs/comparison/{cleaning_method}/{target_variable}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load both datasets
    sid_data = load_csv(sid_data_path)
    evan_data = load_csv(evan_data_path)
    
    if sid_data is None or evan_data is None:
        print("Failed to load data. Exiting...")
        exit(1)

    # Define columns to remove for SID data (empty for EVAN)
    sid_columns_to_remove = ['total_dissolved_solids',
                           'turbidity', 
                           'nitrate', 
                           'phosphate', 
                           'ammonium', 
                           'phosphorus']
    evan_columns_to_remove = []  # No columns to remove for EVAN dataset
    
    # Define all possible features for each dataset separately
    sid_all_features = ['temp', 'ph', 'chlorophyll_a', 'dissolved_oxygen', 'secchi', 
                       'biochemical_oxygen_demand', 'total_dissolved_solids', 'turbidity', 
                       'nitrate', 'phosphate', 'ammonium', 'phosphorus']
    
    evan_all_features = ['temp', 'ph', 'chlorophyll_a', 'dissolved_oxygen', 'secchi', 
                        'biochemical_oxygen_demand', 'total_dissolved_solids', 'turbidity', 
                        'nitrate', 'phosphate', 'ammonium', 'phosphorus', 'nitrogen']
    
    # Create initial feature lists (excluding target variable)
    sid_features = [col for col in sid_all_features if col not in sid_columns_to_remove and col != target_variable]
    evan_features = [col for col in evan_all_features if col not in evan_columns_to_remove and col != target_variable]
    
    # Only apply cleaning to SID data
    if cleaning_method == 'columns':
        sid_data, sid_features = remove_nan_columns(sid_data, sid_features, target_variable)
    else:  # rows
        sid_data = remove_nan_rows(sid_data, sid_features, target_variable)
        # When using rows method, keep all features for both datasets
        sid_features = [col for col in sid_all_features if col != target_variable]
        evan_features = [col for col in evan_all_features if col != target_variable]
    
    print(f"\nFinal SID features: {sid_features}")
    print(f"Final EVAN features: {evan_features}")

    # Remove duplicate feature list creation
    print(f"\nUsing SID features: {sid_features}")
    print(f"Using EVAN features: {evan_features}")
    
    # Get available locations and depth groups
    depth_groups = ['0-10m', '10-30m', '30m+'] if specific_depth_group is None else [specific_depth_group]
    locations = sid_data['location'].unique() if specific_location is None else [specific_location]
    
    results = []
    
    # Add diagnostic prints after loading data
    print("\nSID Data Info:")
    print(f"Total samples: {len(sid_data)}")
    print("Samples per location and depth group:")
    print(sid_data.groupby(['location', 'depth_group']).size())
    
    print("\nEVAN Data Info:")
    print(f"Total samples: {len(evan_data)}")
    print("Samples per location and depth group:")
    print(evan_data.groupby(['location', 'depth_group']).size())
    
    # Check depth groups in each dataset
    print("\nUnique depth groups in SID:", sid_data['depth_group'].unique())
    print("Unique depth groups in EVAN:", evan_data['depth_group'].unique())
    
    for location in locations:
        print(f"\nProcessing location: {location}")
        
        for depth_group in depth_groups:
            print(f"\nProcessing depth group: {depth_group}")
            
            # Train models for both datasets with their respective features
            sid_results = train_rf_model(sid_data, sid_features, target_variable, location, depth_group)
            evan_results = train_rf_model(evan_data, evan_features, target_variable, location, depth_group)
            
            if sid_results is None or evan_results is None:
                print(f"Insufficient data for {location} at {depth_group}")
                continue
            
            # Create comparison plot
            output_path = os.path.join(
                output_dir,
                f"comparison_{location}_{depth_group}.png"
            )
            plot_comparison(sid_results, evan_results, location, depth_group, output_path, target_variable, plot_option)
            print(f"Plot saved to: {output_path}")
            
            # Store results with hyperparameters
            results.append({
                'location': location,
                'depth_group': depth_group,
                'sid_rmse': sid_results['metrics']['rmse'],
                'evan_rmse': evan_results['metrics']['rmse'],
                'sid_mae': sid_results['metrics']['mae'],
                'evan_mae': evan_results['metrics']['mae'],
                'sid_nse': sid_results['metrics']['nse'],
                'evan_nse': evan_results['metrics']['nse'],
                'sid_samples': sid_results['metrics']['n_samples'],
                'evan_samples': evan_results['metrics']['n_samples'],
                'sid_best_params': str(sid_results['metrics']['best_params']),
                'evan_best_params': str(evan_results['metrics']['best_params'])
            })
    
    # Save summary results only if we have multiple combinations
    if len(results) > 1:
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{output_dir}/comparison_summary.csv", index=False)
        print("\nSummary results saved to comparison_summary.csv")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(results_df.round(4))
    else:
        # Print single result
        result = results[0]
        print("\nResults:")
        for key, value in result.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    # Set configuration options directly
    target_variable = "chlorophyll_a"  # Change this to your desired target
    cleaning_method = "columns"  # Either "rows" or "columns"
    plot_option = "sid"  # Options: "sid", "evan", or "both"
    
    # Set to None to process all locations/depth groups, or specify for a single combination
    specific_location = 'WG'  # e.g., "Windermere"
    specific_depth_group = None  # e.g., "0-10m"
    
    # Validate inputs
    valid_targets = ['dissolved_oxygen', 'temp', 'ph', 'chlorophyll_a', 'secchi', 
                     'biochemical_oxygen_demand', 'total_dissolved_solids', 'turbidity', 
                     'nitrate', 'phosphate', 'ammonium', 'phosphorus']
    
    if target_variable not in valid_targets:
        print(f"Error: Invalid target variable. Choose from: {', '.join(valid_targets)}")
        exit(1)
    
    if cleaning_method not in ['rows', 'columns']:
        print("Error: Cleaning method must be either 'rows' or 'columns'")
        exit(1)
        
    if plot_option not in ['sid', 'evan', 'both']:
        print("Error: Plot option must be either 'sid', 'evan', or 'both'")
        exit(1)
    
    # Validate location and depth group if specified
    if specific_location is not None and specific_location not in ['SA', 'WG', 'WP']:
        print(f"Error: Invalid location. Choose from: 'SA', 'WG', 'WP'")
        exit(1)
        
    if specific_depth_group is not None and specific_depth_group not in ['0-10m', '10-30m', '30m+']:
        print("Error: Invalid depth group. Choose from: '0-10m', '10-30m', '30m+'")
        exit(1)
    
    main() 