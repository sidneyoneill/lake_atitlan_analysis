import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Handle missing values using linear interpolation
        df = df.interpolate(method='linear', limit_direction='forward', axis=0)

        # Forward fill any remaining NaNs
        df = df.ffill()
        df = df.bfill()

        df = df[df['date'] >= '2018-01-01']
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

# Build the LSTM model
def create_lstm_model(input_shape, neurons, learning_rate):
    model = Sequential([
        LSTM(neurons, return_sequences=False, input_shape=input_shape),
        Dense(1, activation='linear')  # Single output for turbidity prediction
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=MeanSquaredError(),
                  metrics=[RootMeanSquaredError()])
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

                    # Get the minimum validation loss
                    val_loss = min(history.history['val_loss'])

                    print(f"Validation MSE: {val_loss:.6f}")

                    # Update the best model if the validation loss improves
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {'neurons': neurons, 'learning_rate': learning_rate, 'epochs': epochs}
                        best_model = model

                except Exception as e:
                    print(f"Error during training: {e}")

    if best_model is None:
        raise RuntimeError("No valid model configurations found.")

    return best_model, best_params

# Function to make rolling forecasts up to 2050
def forecast_future(best_model, last_sequence, num_steps, scaler):
    predictions = []
    input_seq = last_sequence.copy()

    for _ in range(num_steps):
        pred = best_model.predict(input_seq.reshape(1, *input_seq.shape), verbose=0)
        predictions.append(pred[0][0])

        # Update input sequence with the new prediction
        input_seq = np.roll(input_seq, -1, axis=0)
        input_seq[-1, -1] = pred  # Add predicted value in place of target variable

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Main workflow
def lstm_turbidity_workflow(file_path):
    try:
        df = load_dataset(file_path)

        # Filter for the WG location and loop through depth groups
        wg_data = df[df['location'] == 'WG']
        depth_groups = wg_data['depth_group'].unique()

        for depth_group in depth_groups:
            print(f"\nProcessing Depth Group: {depth_group}")

            # Filter data for the specific depth group
            filtered_data = wg_data[wg_data['depth_group'] == depth_group]

            if len(filtered_data) < 30:
                print(f"Not enough data for Depth Group {depth_group}. Skipping.")
                continue

            # Define features and target
            features = ['temp', 'chlorophyll_a', 'ph', 'dissolved_oxygen', 'turbidity', 'total_dissolved_solids', 'biochemical_oxygen_demand', 'nitrate', 'phosphate', 'Ammonium', 'phosphorus']
            target = 'turbidity'

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
            val_size = int(len(X) * 0.2)

            X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
            y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

            if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
                print(f"Not enough data for training, validation, or testing for Depth Group {depth_group}. Skipping.")
                continue

            # Define grid search parameters
            grid_params = {'neurons': [32,64,128], 'learning_rate': [0.001, 0.01], 'epochs': [100, 200, 300, 400]}

            # Perform grid search
            input_shape = X_train.shape[1:]
            best_model, best_params = grid_search(X_train, y_train, X_val, y_val, input_shape, grid_params)
            print(f"Best parameters for Depth Group {depth_group}: {best_params}")

            # Make future predictions until 2050
            last_sequence = X[-1]  # Use the last known sequence
            num_future_steps = (2050 - filtered_data['date'].dt.year.max()) * 12
            future_predictions = forecast_future(best_model, last_sequence, num_future_steps, target_scaler)

            # Generate future dates
            last_date = filtered_data['date'].max()
            future_dates = pd.date_range(start=last_date, periods=num_future_steps + 1, freq='M')[1:]

            # Plot results
            plt.figure(figsize=(12, 6))
            plt.plot(filtered_data['date'].iloc[seq_length:], target_scaler.inverse_transform(scaled_target[seq_length:]).flatten(), label='Actual', color='blue')
            plt.plot(future_dates, future_predictions, label='Predicted (Future)', color='red')
            plt.axvline(x=filtered_data['date'].max(), color='green', linestyle='--', label='Start of Prediction')
            plt.title(f'Actual vs Predicted Turbidity for WG ({depth_group} Depth) up to 2050', fontsize=14)
            plt.xlabel('Date', fontsize=13)
            plt.ylabel('Turbidity (NTU)', fontsize=13)
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error during workflow: {e}")

# Execute the workflow with the provided dataset
file_path = 'EVAN_LIMNO_processed_v5.csv'
lstm_turbidity_workflow(file_path)
