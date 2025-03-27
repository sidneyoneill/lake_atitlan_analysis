import pandas as pd
from functions import load_data, get_column_headers, plot_results, create_optimized_rf_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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
            'phosphorus', 'nitrogen']]  # Input features
    y = data['chlorophyll_a']  # Target variable

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates, test_size=0.2, random_state=42
    )

    # Create and train the Random Forest model with optimized parameters
    rf_model = create_optimized_rf_model()
    rf_model.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Sort the test data by date for the time series plot
    test_data = pd.DataFrame({
        'date': dates_test,
        'actual': y_test,
        'predicted': y_pred
    }).sort_values('date')
    
    # Visualize and save the results
    plot_results(
        test_data['actual'], 
        test_data['predicted'], 
        test_data['date'],
        output_path=f"{output_dir}/rf_predictions_timeseries.png"
    )

if __name__ == "__main__":
    main()
