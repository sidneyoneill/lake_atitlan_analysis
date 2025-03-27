import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def load_csv(file_path):
    """
    Loads data from a specified Excel file path
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pandas.DataFrame: Loaded data, or None if loading fails
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        print(f"Shape of the data: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

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
    
def get_column_headers(df):
    """
    Returns a list of column headers from the DataFrame
    
    Args:
        df (pandas.DataFrame): The input DataFrame
        
    Returns:
        list: List of column headers, or None if df is None
    """
    if df is not None:
        return list(df.columns)
    return None
    
def create_rf_model():
    """
    Creates and returns a Random Forest model with optimized parameters
    
    Returns:
        RandomForestRegressor: Initialized Random Forest model
    """
    return RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

def optimize_rf_model(X_train, y_train):
    """
    Performs grid search to find optimal hyperparameters for the Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training target variable
        
    Returns:
        tuple: (best_model, best_params)
    """
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize base model
    rf_base = RandomForestRegressor(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=2
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def plot_results(y_test, y_pred, dates, output_path=None):
    """
    Visualizes the comparison between predicted and actual values over time using line and scatter plots.

    Args:
        y_test: Actual values
        y_pred: Predicted values
        dates: Datetime values for x-axis
        output_path: Optional path to save the plot (e.g., 'outputs/figures/prediction_plot.png')
    """
    plt.figure(figsize=(12, 6))
    
    # Line plots for actual and predicted values
    plt.plot(dates, y_test, 'b-', label='Actual', alpha=0.7, linewidth=1)
    plt.plot(dates, y_pred, 'r-', label='Predicted', alpha=0.7, linewidth=1)
    
    # Scatter points for better visibility
    plt.scatter(dates, y_test, color='blue', alpha=0.5, s=20, label='Actual Points')
    plt.scatter(dates, y_pred, color='red', alpha=0.5, s=20, label='Predicted Points')
    
    # Title and labels
    plt.title('Chlorophyll-a Predictions vs Actual Values Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Chlorophyll-a Concentration', fontsize=12)
    
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

def create_optimized_rf_model():
    """
    Creates a Random Forest model with the known best hyperparameters
    """
    return RandomForestRegressor(
        max_depth=15,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=300,
        random_state=42
    )

def plot_results_split(y_train, y_train_pred, y_test, y_test_pred, dates_train, dates_test, output_path=None):
    """
    Visualizes the comparison between predicted and actual values for both training and test sets
    
    Args:
        y_train: Actual training values
        y_train_pred: Predicted training values
        y_test: Actual test values
        y_test_pred: Predicted test values
        dates_train: Dates for training data
        dates_test: Dates for test data
        output_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.plot(dates_train, y_train, 'b-', label='Actual (Training)', alpha=0.3, linewidth=1)
    plt.plot(dates_train, y_train_pred, 'r-', label='Predicted (Training)', alpha=0.3, linewidth=1)
    plt.scatter(dates_train, y_train, color='blue', alpha=0.2, s=20)
    plt.scatter(dates_train, y_train_pred, color='red', alpha=0.2, s=20)
    
    # Plot test data with different alpha to distinguish
    plt.plot(dates_test, y_test, 'b-', label='Actual (Test)', alpha=0.7, linewidth=1)
    plt.plot(dates_test, y_test_pred, 'r-', label='Predicted (Test)', alpha=0.7, linewidth=1)
    plt.scatter(dates_test, y_test, color='blue', alpha=0.5, s=20)
    plt.scatter(dates_test, y_test_pred, color='red', alpha=0.5, s=20)
    
    plt.title('Random Forest Predictions vs Actual Values Over Time')
    plt.xlabel('Date')
    plt.ylabel('Chlorophyll-a Concentration')
    
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    # plt.show()

def train_location_model(data, location_name, output_dir):
    """
    Trains and evaluates a model for a specific location
    
    Args:
        data (DataFrame): Data for specific location
        location_name (str): Name of the location
        output_dir (str): Base output directory
    """
    # Store dates before splitting
    dates = data['date']
    
    # Define features and target
    X = data[['temp', 'turbidity', 'dissolved_oxygen', 'ph', 
              'total_dissolved_solids', 'secchi', 
              'nitrate', 'phosphate', 'ammonium', 
              'phosphorus', 'nitrogen']]
    y = data['chlorophyll_a']
    
    # Split data
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates, test_size=0.2, shuffle=False
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
    optimized_model = grid_search.best_estimator_
    
    # Make predictions
    y_train_pred = optimized_model.predict(X_train)
    y_test_pred = optimized_model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print results
    print(f"\nResults for {location_name}:")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Training Set - MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"Test Set     - MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
    
    # Create location-specific output directory
    location_output_dir = os.path.join(output_dir, location_name.replace(" ", "_"))
    os.makedirs(location_output_dir, exist_ok=True)
    
    # Plot and save results
    plot_results_split(
        y_train, y_train_pred, 
        y_test, y_test_pred, 
        dates_train, dates_test,
        output_path=f"{location_output_dir}/predictions_timeseries.png"
    )
    
    return {
        'location': location_name,
        'best_params': grid_search.best_params_,
        'train_mse': train_mse,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_r2': test_r2
    }

def standardize_column_names(df):
    """
    Converts Spanish column names to standardized English names
    
    Args:
        df (pandas.DataFrame): DataFrame with original Spanish column names
        
    Returns:
        pandas.DataFrame: DataFrame with standardized English column names
    """
    column_mapping = {
        'Sitio': 'location',
        'Fecha': 'date',
        'Profuidad (m)': 'depth',
        'Depth Group': 'depth_group',
        'Temp. (¬∞C)': 'temp',
        'Chl a (¬µg/l)': 'chlorophyll_a',
        'Ph (Unidad)': 'ph',
        'Turbidity (NTU)': 'turbidity',
        'DO (mg/L)': 'dissolved_oxygen',
        'TDS (mg/l)': 'total_dissolved_solids',
        'DBO (mg/l)': 'biochemical_oxygen_demand',
        'Sechi  (m)': 'secchi',
        'NO3 (¬µg/L)': 'nitrate',
        'PO4 (¬µg/L)': 'phosphate',
        'NH4 (¬µg/L)': 'ammonium',
        'NT (¬µg/l)': 'nitrogen',
        'PT (¬µg/l)': 'phosphorus'
    }
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_clean = df.copy()
    
    # Rename columns based on the mapping
    df_clean.rename(columns=column_mapping, inplace=True)
    
    # Print the changes made
    print("Column names standardized:")
    for old, new in column_mapping.items():
        if old in df.columns:
            print(f"  {old} → {new}")
            
    return df_clean

def remove_empty_rows(data):
    """
    Remove rows containing any empty values (NaN, None, etc.) from the DataFrame.
    
    Args:
        data (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with empty rows removed
    """
    # Store original length
    original_len = len(data)
    
    # Remove rows with any empty values
    clean_data = data.dropna()
    
    # Calculate number of rows removed
    rows_removed = original_len - len(clean_data)
    
    # Print summary
    print(f"Removed {rows_removed} rows containing empty values")
    print(f"Remaining rows: {len(clean_data)}")
    
    return clean_data
