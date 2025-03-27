import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

file_path = 'Lake_data_clean.xlsx'
df = pd.read_excel(file_path)

# Create a dictionary to store the subsets for each Sitio and Depth Group
subset_dict = {}
unique_sites = df['Sitio'].unique()
unique_depths = df['Depth Group'].unique()

# Loop through each unique site and depth group
for site in unique_sites:
    for depth_grp in unique_depths:
        # 1) Filter for site & depth group
        subset = df[(df['Sitio'] == site) & (df['Depth Group'] == depth_grp)].copy()

        # 2) Convert date, sort, set as index
        subset['Fecha'] = pd.to_datetime(subset['Fecha'])
        subset.sort_values('Fecha', inplace=True)
        subset.set_index('Fecha', inplace=True)

        # 3) Separate numeric columns from non-numerics
        numeric_cols = subset.select_dtypes(include=[np.number]).columns
        df_numerics = subset[numeric_cols]  # only the numeric columns

        # 4) Resample to monthly, take mean of multiple readings
        monthly_numeric = df_numerics.resample('MS').mean()
        
        # 5) Interpolate missing months
        monthly_numeric = monthly_numeric.interpolate(method='time')

        # Add 'Sitio' and 'Depth Group' columns for reference
        monthly_numeric.insert(0, 'Sitio', site)
        monthly_numeric.insert(1, 'Depth Group', depth_grp)

        # Store in dictionary
        key = f"{site}_{depth_grp}"
        subset_dict[key] = monthly_numeric

# Process each site and depth group
for key, monthly_df in subset_dict.items():
    site, depth_grp = key.split("_", 1)
    print(f"Processing Site={site}, Depth={depth_grp}")
    
    if 'Temp. (°C)' not in monthly_df.columns:
        print(f" -> No 'Temp. (°C)' column found for {key}; skipping.")
        continue

    temp_series = monthly_df['Temp. (°C)'].dropna()

    # Define exogenous variables (exclude 'Temp. (°C)')
    exog_vars = ['Turbidity (NTU)']
    exog_data = monthly_df[exog_vars].dropna()
    
    # Align temp_series and exog_data
    common_index = temp_series.index.intersection(exog_data.index)
    temp_series = temp_series.loc[common_index]
    exog_data = exog_data.loc[common_index]
    
    if len(temp_series) < 6:
        print(f" -> Not enough data points in {key} to run the model; skipping.")
        continue

    # Train-test split
    train_end_date = '2022-12'
    train = temp_series.loc[:train_end_date]
    test = temp_series.loc['2023-01':]
    exog_train = exog_data.loc[:train_end_date]
    exog_test = exog_data.loc['2023-01':]

    if len(train) < 6 or len(exog_train) < 6:
        print(f" -> Not enough data for {key}; skipping.")
        continue

    # Fit SARIMAX Model
    sarimax_model = SARIMAX(
        train,
        exog=exog_train,  # Exogenous variables
        order=(1, 1, 1),  # Example SARIMAX order
        seasonal_order=(1, 1, 1, 12)  # Example seasonal order
    )
    results = sarimax_model.fit(disp=False)
    print(results.summary())

    # Forecast on the test set
    forecast = results.predict(start=test.index[0], end=test.index[-1], exog=exog_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f"Test RMSE: {rmse:.4f}")

    # Plot actual vs forecast
    plt.figure(figsize=(10, 6))
    plt.plot(train.index, train, label='Training Data', color='blue')
    plt.plot(test.index, test, label='Actual Data (Test)', color='green')
    plt.plot(forecast.index, forecast, label='Forecast', color='orange')
    plt.title(f"Temperature Forecast: Site={site}, Depth={depth_grp}")
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()

print("SARIMAX model processing complete.")
