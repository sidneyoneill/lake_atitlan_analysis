from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from functions import load_csv, plot_results, standardize_column_names, load_data
import pandas as pd
import os

def main():
    # Define paths
    data_path = "data_ordered.xlsx"
    output_dir = "outputs/EVAN_LIMNO/figures"
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    data = load_data(data_path)
    if data is None:
        print("Failed to load data. Exiting...")
        return
    
    depth_groups = ['0-10 m', '10-30 m', '30+ m']

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

            # Ensure data is sorted by date
            filtered_data = filtered_data.sort_values('date')
            
            # Skip if not enough data
            if len(filtered_data) < 50:
                print(f"Insufficient data for {location} at {depth_group}")
                continue

            # Store dates before splitting
            dates = filtered_data['date']

            # Define features (X) and target variable (y)
            X = filtered_data[['temp', 'chlorophyll_a', 'ph']]
            y = filtered_data['turbidity']

            # Split data for training and testing
            X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
                X, y, dates, test_size=0.3, shuffle=False  # Maintain temporal order
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
