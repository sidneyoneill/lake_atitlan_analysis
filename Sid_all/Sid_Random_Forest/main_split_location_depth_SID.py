from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from functions import load_csv, plot_results
import pandas as pd
import os

def main():

    target_variable = 'temp'
 
    start_date = '14/02/2014' # or '11/01/2018' or '14/02/2014'

    # remove / from start_date
    start_date_file_path = start_date.replace("/", "")

    # Define paths
    data_path = "data/SID_LIMNO_processed_v5.csv"
    output_dir = f"outputs/SID_LIMNO/{start_date_file_path}/{target_variable}"
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    data = load_csv(data_path)
    if data is None:
        print("Failed to load data. Exiting...")
        return

    # Define columns to remove (can be modified as needed)
    columns_to_remove = ['total_dissolved_solids',
                         'turbidity', 
                         'nitrate', 
                         'phosphate', 
                         'ammonium', 
                         'phosphorus']
    
    # Filter data for start date
    data = data[data['date'] >= start_date]

    # Define all possible features
    all_features = ['temp', 'ph', 'chlorophyll_a', 'dissolved_oxygen', 'secchi', 'biochemical_oxygen_demand',
                    'total_dissolved_solids', 'turbidity', 'nitrate', 'phosphate', 'ammonium', 'phosphorus']
    
    # Remove target variable and specified columns from features list
    features = [col for col in all_features if col not in columns_to_remove and col != target_variable]
    print(f"\nUsing features: {features}")

    # Define depth groups
    depth_groups = ['0-10m', '10-30m', '30m+']
    
    # Split data by location
    locations = data['location'].unique()
    results = []

    for location in locations:
        print(f"\nProcessing location: {location}")
        
        for depth_group in depth_groups:
            print(f"\nProcessing depth group: {depth_group}")
            
            # Filter data for current location and depth group
            filtered_data = data[
                (data['location'] == location) & 
                (data['depth_group'] == depth_group)
            ].copy()
            
            # Skip if not enough data
            if len(filtered_data) < 50:
                print(f"Insufficient data for {location} at {depth_group}")
                continue

            # Store dates before splitting
            dates = filtered_data['date']

            # Define features (X) and target variable (y)
            X = filtered_data[features]  # Use modified features list
            y = filtered_data[target_variable]  # Now correctly using turbidity as target

            # Split data for training and testing
            X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
                X, y, dates, test_size=0.2, shuffle=False  # Maintain temporal order
            )

            # Hyperparameter optimization with CV
            print("\nPerforming hyperparameter optimization...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15]
            }
            rf_model = RandomForestRegressor(random_state=42)
            ts_cv = TimeSeriesSplit(n_splits=5)  # Use time series split

            grid_search = GridSearchCV(
                estimator=rf_model, 
                param_grid=param_grid, 
                cv=ts_cv, 
                scoring='neg_mean_squared_error'
            )
            grid_search.fit(X_train, y_train)
            optimized_model = grid_search.best_estimator_
            print("\nBest Hyperparameters:", grid_search.best_params_)

            # Evaluate on test set
            y_pred = optimized_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"\nTest Set MSE: {mse:.4f}, R²: {r2:.4f}")

            # Save results for this location and depth group
            results.append({
                'location': location,
                'depth_group': depth_group,
                'mse': mse,
                'r2': r2,
                'best_params': grid_search.best_params_,
                'n_samples': len(filtered_data)
            })

            # Sort the test data by date for the time series plot
            test_data = pd.DataFrame({
                'date': dates_test,
                'actual': y_test,
                'predicted': y_pred
            }).sort_values('date')

            # Create subdirectory for location
            location_dir = os.path.join(output_dir, location.replace(" ", "_"))
            os.makedirs(location_dir, exist_ok=True)

            # Visualize results
            output_path = os.path.join(
                location_dir, 
                f"rf_predictions_{depth_group.replace(' ', '_')}.png"
            )
            plot_results(
                test_data['actual'],
                test_data['predicted'],
                test_data['date'],
                output_path=output_path
            )
            print(f"Plot saved to: {output_path}")

    # Save summary results
    results_df = pd.DataFrame(results)
    # Round MSE and R2 values to 2 decimal places before saving
    results_df['mse'] = results_df['mse'].round(2)
    results_df['r2'] = results_df['r2'].round(2)
    results_df.to_csv(f"{output_dir}/summary_results_by_depth.csv", index=False)
    print("\nSummary results saved to outputs/figures/summary_results_by_depth.csv")

    # Print summary statistics
    print("\nSummary Statistics:")
    summary = results_df.groupby(['location', 'depth_group']).agg({
        'mse': 'mean',
        'r2': 'mean',
        'n_samples': 'first'
    }).round(4)
    print(summary)

if __name__ == "__main__":
    main()
