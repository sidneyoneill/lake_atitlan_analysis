import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')  # Ensure the dataset is sorted by date

        # Handle missing values using linear interpolation
        df = df.interpolate(method='linear', limit_direction='forward', axis=0)

        # Forward fill any remaining NaNs
        df = df.ffill()

        # Backward fill as a last resort
        df = df.bfill()

        df = df[df['date'] >= '2014-01-01']

        return df
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {e}")

# Create sequences for supervised learning
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :-1])  # All features except target
        y.append(data[i + seq_length, -1])    # Target
    return np.array(X), np.array(y)

# Build the LSTM model (with a single LSTM layer)
def create_lstm_model(input_shape, neurons, learning_rate):
    model = Sequential([
        LSTM(neurons, return_sequences=False, input_shape=input_shape),
        Dense(1, activation='linear')  # Single output for turbidity prediction
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=MeanSquaredError(),
                  metrics=[RootMeanSquaredError(), MeanAbsoluteError()])
    return model

# Perform grid search for hyperparameter optimization
def grid_search(X_train, y_train, X_val, y_val, input_shape, grid_params):
    best_model = None
    best_params = None
    best_val_loss = float("inf")

    for neurons in grid_params['neurons']:
        for learning_rate in grid_params['learning_rate']:
            for epochs in grid_params['epochs']:
                print(f"Testing configuration: Neurons={neurons}, Learning Rate={learning_rate}, Epochs={epochs}")
                model = create_lstm_model(input_shape, neurons, learning_rate)

                try:
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=32,
                        verbose=0
                    )

                    # Get the minimum training and validation loss during training
                    train_loss = min(history.history['loss'])  # training validation subset MSE
                    val_loss = min(history.history['val_loss'])  # test MSE

                    print(f"Train MSE: {train_loss:.6f}, Validation MSE: {val_loss:.6f}")

                    # Update the best model if the validation loss improves
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {'neurons': neurons, 'learning_rate': learning_rate, 'epochs': epochs}
                        best_model = model  # Keep the trained model with the best validation loss

                except Exception as e:
                    print(f"Error during training with Neurons={neurons}, Learning Rate={learning_rate}, Epochs={epochs}: {e}")

    if best_model is None:
        raise RuntimeError("No valid model configurations found during grid search.")

    return best_model, best_params


# Main workflow
def lstm_turbidity_workflow(file_path):
    try:
        # Load the dataset
        df = load_dataset(file_path)

        # Filter for the WG location and loop through depth groups
        wg_data = df[df['location'] == 'WG']
        depth_groups = wg_data['depth_group'].unique()

        for depth_group in depth_groups:
            print(f"\nProcessing Depth Group: {depth_group}")

            # Filter data for the specific depth group
            filtered_data = wg_data[wg_data['depth_group'] == depth_group]

            if len(filtered_data) < 30:  # Ensure enough data for sequence creation
                print(f"Not enough data for Depth Group {depth_group}. Skipping.")
                continue

            # Define features and target
            features = ['temp', 'chlorophyll_a', 'ph', 'dissolved_oxygen', 'biochemical_oxygen_demand']#, 'nitrate', 'phosphate', 'ammonium', 'phosphorus']
            target = 'temp'

            # Scale the features and target
            scaler = MinMaxScaler()
            target_scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(filtered_data[features])
            scaled_target = target_scaler.fit_transform(filtered_data[[target]])

            # Combine features and target for sequence preparation
            scaled_data_combined = np.hstack((scaled_features, scaled_target))

            # Create sequences
            seq_length = 12
            X, y = create_sequences(scaled_data_combined, seq_length)

            if len(X) == 0:
                print(f"Insufficient data after sequence creation for Depth Group {depth_group}. Skipping.")
                continue

            # Split data into training, validation, and testing sets
            train_size = int(len(X) * 0.7)
            val_size = int(len(X) * 0.1)

            X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
            y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

            if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
                print(f"Not enough data for training, validation, or testing for Depth Group {depth_group}. Skipping.")
                continue

            # Define grid search parameters
            grid_params = {
                'neurons': [32, 64, 128],
                'learning_rate': [0.001, 0.01],
                'epochs': [100, 200, 300]
            }

            # Perform grid search
            input_shape = X_train.shape[1:]
            best_model, best_params = grid_search(X_train, y_train, X_val, y_val, input_shape, grid_params)
            print(f"Best parameters for Depth Group {depth_group}: {best_params}")

            # Evaluate on test set
            test_loss, test_rmse, test_mae = best_model.evaluate(X_test, y_test, verbose=1)
            print(f"Test MSE: {test_loss:.6f}, Test RMSE: {test_rmse:.6f}, Test MAE: {test_mae:.6f}")

            # Make predictions over the entire time series
            full_sequence_X, _ = create_sequences(scaled_data_combined, seq_length)
            y_pred = best_model.predict(full_sequence_X)

            # Inverse transform the target for actual and predicted values
            full_y_actual = target_scaler.inverse_transform(scaled_target[seq_length:]).flatten()
            y_pred_actual = target_scaler.inverse_transform(y_pred).flatten()

            # Plot the predictions vs actual values
            date_range = filtered_data['date'].iloc[seq_length:].reset_index(drop=True)
            plt.figure(figsize=(12, 4))
            plt.plot(date_range, full_y_actual, label='Actual', color='blue', alpha = 0.7, linestyle='--', linewidth=1)
            plt.plot(date_range, y_pred_actual, label='Predicted', color='orange', linewidth=1)
            plt.axvline(x=date_range.iloc[int(len(date_range) * 0.75)], color='green', linestyle='--', label='Train/Test Split')
            plt.title(f'Actual vs Predicted Turbidity for WG ({depth_group} Depth)', fontsize=16)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Turbidity (NTU)', fontsize=14)
            plt.legend()
            plt.grid(True, alpha = 0.3)
            plt.show()

    except Exception as e:
        print(f"Error during workflow: {e}")

# Execute the workflow
file_path = 'SID_LIMNO_no_outliers_v2.csv'
lstm_turbidity_workflow(file_path)
