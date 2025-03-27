import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# SARIMA model from statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Stationarity test
from statsmodels.tsa.stattools import adfuller
# To help find optimal p,d,q using auto_arima (optional)
from pmdarima import auto_arima
# For performance metrics
from sklearn.metrics import mean_squared_error

file_path = 'Lake_data_clean.xlsx'
df = pd.read_excel(file_path)
#print(df.head())
#print(df.info())

# Create a dictionary to store the subsets for each Sitio and Depth Group
subset_dict = {}
unique_sites = df['Sitio'].unique()
unique_depths = df['Depth Group'].unique()

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
        
        # 5) Interpolate if you want no missing months
        #    (This will fill months that had no measurement)
        monthly_numeric = monthly_numeric.interpolate(method='time')

        # 6)  Adding 'Sitio' and 'Depth Group' columns at the start of the DataFrame
        monthly_numeric.insert(0, 'Sitio', site)          # Insert at position 0 (start)
        monthly_numeric.insert(1, 'Depth Group', depth_grp)  # Insert at position 1 (after 'Sitio')

        #print(monthly_numeric)

        # 7) Store in your dictionary
        key = f"{site}_{depth_grp}"
        subset_dict[key] = monthly_numeric

        # Print or inspect the result
        #print(f"Processed subset for {key}:")
        #print(monthly_numeric, "\n")

print(subset_dict)
# Finally, check the keys of the dictionary
#print("Available keys in the subset dictionary:", subset_dict.keys())
#print("Available keys in the subset dictionary:", subset_dict.items())

for key, monthly_df in subset_dict.items():
    # Extract the site/depth from the key if you want to display them separately
    # e.g. if you used f"{site}_{depth_grp}"
    # this split might fail if there's an underscore in the depth name,
    # so adapt as needed if your naming is more complex.
    parts = key.split("_", 1)
    site = parts[0]
    depth_grp = parts[1] if len(parts) > 1 else "Unknown"
    
    print("="*60)
    print(f"PROCESSING: Site={site}, Depth={depth_grp}")
    
    # 1) Make sure we have temperature data
    if 'Temp. (°C)' not in monthly_df.columns:
        print(f" -> No 'Temp. (°C)' column found for {key}; skipping.")
        continue
    
    # 2) Prepare the temperature series
    # Drop any remaining NaNs (after resample/interpolate, there may be none)
    temp_series = monthly_df['Temp. (°C)'].dropna()
    
    if len(temp_series) < 6:
        print(f" -> Not enough data points in {key} to run ADF test or auto_arima; skipping.")
        continue
    
    # 3) ADF Test
    adf_result = adfuller(temp_series)
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]
    
    print(f" -> ADF Statistic = {adf_stat:.4f}, p-value = {adf_pvalue:.4f}")
    if adf_pvalue < 0.05:
        print("    The series appears stationary at 5% significance (no differencing needed).")
    else:
        print("    The series is likely non-stationary (differencing may be needed).")

    #Suppose you want up to 2022-12 as training, 2023+ as test
    train = temp_series.loc[:'2022-12']
    test = temp_series.loc['2023-01':]
        
    print(f" -> Train size: {len(train)}, Test size: {len(test)}")
        
    if len(train) < 6:
        print("    Not enough training data after splitting; skipping auto_arima.")
        continue

    print(" -> Running auto_arima for best model. This may take a moment...")
    auto_model = auto_arima(
        train,
        start_p=0, start_q=0,
        max_p=5, max_q=5,   # Increase if needed, changed from 3 to 5
        start_P=0, start_Q=0,
        max_P=3, max_Q=3,
        seasonal=True,
        m=12,       # monthly seasonality
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    print(f"    Best ARIMA order: {auto_model.order}")
    print(f"    Best Seasonal order: {auto_model.seasonal_order}")

    # 6) Evaluate on Test set
    # Fit the final model on the training set
    # (auto_arima returns a fitted model, but we can call auto_model.predict)
    n_test = len(test)
    future_forecast = auto_model.predict(n_periods=n_test)

    # Compare forecast to actual
    if len(test) > 0:
        test_index = test.index
        # Convert forecast to a Series with the same index as 'test'
        forecast_series = pd.Series(future_forecast, index=test_index, name='Forecast')
        
        rmse = np.sqrt(mean_squared_error(test, forecast_series))
        print(f"    Test RMSE = {rmse:.4f}")
        
        # Plot actual vs forecast
        plt.figure(figsize=(10, 6))
        plt.plot(train.index, train, label='Training Data', color='blue')
        plt.plot(test.index, test, label='Actual Data (Test)', color='green')
        plt.plot(forecast_series.index, forecast_series, label='Forecast', color='orange')
        plt.title(f"Temperature Forecast: Site={site}, Depth={depth_grp}")
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Optionally print or plot the last portion of the time series vs. forecast
        # (uncomment if you want to see a small textual preview)
        # print(test.tail(5))
        # print(forecast_series.tail(5))

        # Forecast until 2050
        '''
        forecast = auto_model.predict(n_periods=336)  # 336 months = 28 years
        future_index = pd.date_range(start=train.index[-1], periods=337, freq='MS')[1:]
        forecast_series = pd.Series(forecast, index=future_index)

        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train, label='Historical Data')
        plt.plot(forecast_series.index, forecast_series, label='Forecast', color='orange')
        plt.fill_between(
            forecast_series.index,
            forecast_series - 1.96 * results.bse[-1],
            forecast_series + 1.96 * results.bse[-1],
            color='orange', alpha=0.3, label='Confidence Interval'
        )
        plt.legend()
        plt.show()
        '''
    
    print("="*60, "\n")