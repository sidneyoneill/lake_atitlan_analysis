import pandas as pd
from functions import load_data, get_column_headers, create_rf_model, optimize_rf_model, plot_results
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import TimeSeriesSplit, train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def main():
    # Define paths
    data_path = "data/processed_LIMNO.xlsx"
    output_dir = "outputs/figures"
    
    # Load the data
    data = load_data(data_path)
    if data is None:
        print("Failed to load data. Exiting...")
        return
    
    # Store dates before splitting
    dates = data['date']
    
    # Define features (X) and target variable (y)
    X = data[['temp', 'turbidity', 'dissolved_oxygen', 'ph', 
              'total_dissolved_solids', 'secchi', 
              'nitrate', 'phosphate', 'ammonium', 
              'phosphorus', 'nitrogen']]
    y = data['chlorophyll_a']
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates, test_size=0.2, shuffle=False  # Maintain temporal order
    )
    
    # Hyperparameter optimization with CV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15]
    }
    rf_model = RandomForestRegressor(random_state=42)
    ts_cv = TimeSeriesSplit(n_splits=5)  # Use time series split
    
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=ts_cv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    optimized_model = grid_search.best_estimator_
    print("\nBest Hyperparameters:", grid_search.best_params_)
    
    # Evaluate on test set
    y_pred = optimized_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nTest Set MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Visualize results (time series plot)
    test_data = pd.DataFrame({'date': dates_test, 'actual': y_test, 'predicted': y_pred}).sort_values('date')
    plot_results(test_data['actual'], test_data['predicted'], test_data['date'], output_path=f"{output_dir}/rf_predictions_timeseries.png")

if __name__ == "__main__":
    main()
