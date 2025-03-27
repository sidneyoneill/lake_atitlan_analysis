import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For SARIMA model and auto-parameter selection
from pmdarima import auto_arima

# For checking stationarity
from statsmodels.tsa.stattools import adfuller

# For ACF, PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# For performance metrics
from sklearn.metrics import mean_absolute_error

# Example file name: "SID_LIMNO_processed_V2.csv"
df = pd.read_csv("SID_LIMNO_processed_V2.csv")

# Get unique combinations of location and depth
location_depth_combinations = df[['location', 'depth_group']].drop_duplicates()

# Loop through each combination
for _, row in location_depth_combinations.iterrows():
    selected_location = row['location']
    selected_depth = row['depth_group']
    
    print(f"\nProcessing {selected_location} - {selected_depth}")
    
    # Filter the DataFrame for the chosen location and depth group
    df_filtered = df[(df['location'] == selected_location) & 
                     (df['depth_group'] == selected_depth)].copy()
    
    # Convert date to datetime and set as index
    df_filtered['date'] = pd.to_datetime(df_filtered['date'], format='%d/%m/%Y')
    df_filtered.set_index('date', inplace=True)
    
    # Sort by date
    df_filtered.sort_index(inplace=True)
    
    # Keep only the 'temp' column for univariate analysis
    df_filtered = df_filtered[['temp']]
    
    # Resample to monthly data using mean temperature
    df_monthly = df_filtered.resample('M').mean()
    
    # Interpolate missing values if needed
    df_monthly['temp'] = df_monthly['temp'].interpolate(method='linear')
    
    # Skip if there's insufficient data
    if len(df_monthly) < 24:  # Skip if less than 2 years of data
        print(f"Skipping {selected_location} - {selected_depth} due to insufficient data")
        continue
        
    try:
        # Time series plot
        plt.figure(figsize=(10, 4))
        plt.plot(df_monthly.index, df_monthly['temp'], marker='o', linestyle='-')
        plt.title(f"Monthly Mean Water Temperature ({selected_location} - {selected_depth})")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.savefig(f'temperature_timeseries_{selected_location}_{selected_depth}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # ACF and PACF plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(df_monthly['temp'], ax=axes[0], lags=24, title='ACF - Temperature')
        plot_pacf(df_monthly['temp'], ax=axes[1], lags=24, title='PACF - Temperature')
        plt.savefig(f'acf_pacf_{selected_location}_{selected_depth}.png', bbox_inches='tight', dpi=300)
        plt.close()

        result = adfuller(df_monthly['temp'].dropna())
        print("ADF Statistic: {:.4f}".format(result[0]))
        print("p-value: {:.4f}".format(result[1]))
        for key, value in result[4].items():
            print('Critical Values:')
            print(f'   {key}, {value:.4f}')

        # Extract the temperature series as a NumPy array
        y = df_monthly['temp'].values

        # Use auto_arima to find the best SARIMA parameters
        # (seasonal=True and m=12 for monthly data)
        model_arima = auto_arima(
            df_monthly['temp'],
            seasonal=True, m=12,
            start_p=1, start_q=1,
            d=0, D=1,  
            start_P=1, max_P=3,  # Increase seasonal autoregression
            start_Q=1, max_Q=3,  
            trace=True
        )

        # Print the model summary
        print(model_arima.summary())

        # Extract residuals from the fitted SARIMA model
        residuals = model_arima.resid()

        # Plot residuals
        plt.figure(figsize=(10, 4))
        plt.plot(residuals, label="SARIMA Residuals")
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title("SARIMA Model Residuals")
        plt.legend()
        plt.savefig(f'sarima_residuals_{selected_location}_{selected_depth}.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Perform Augmented Dickey-Fuller (ADF) test on residuals
        adf_test = adfuller(residuals)

        # Print ADF test results
        print("ADF Statistic:", adf_test[0])
        print("p-value:", adf_test[1])

        # Critical values (for reference)
        for key, value in adf_test[4].items():
            print(f'Critical Value {key}: {value:.4f}')

        # Fit the model
        model_arima.fit(y)

        # The determined order and seasonal_order
        order = model_arima.order
        seasonal_order = model_arima.seasonal_order
        print("Best SARIMA order:", order)
        print("Best seasonal order:", seasonal_order)

        n_periods = 500  # Forecasting 5 years ahead (60 months)
        forecast, conf_int = model_arima.predict(n_periods=n_periods, return_conf_int=True)

        # Generate future time index
        last_date = df_monthly.index[-1]
        forecast_index = pd.date_range(last_date + pd.offsets.MonthEnd(1), periods=n_periods, freq='M')

        # Convert forecast to DataFrame
        forecast_series = pd.Series(forecast, index=forecast_index)
        conf_df = pd.DataFrame(conf_int, index=forecast_index, columns=['lower_temp', 'upper_temp'])

        # Monte Carlo Simulations for Stochastic Forecasting
        n_simulations = 100  # Number of simulated paths
        simulated_forecasts = []

        for _ in range(n_simulations):
            # Generate stochastic forecast by adding noise to the SARIMA prediction
            forecast_simulated = model_arima.predict(n_periods=n_periods) + np.random.normal(
                loc=0, scale=np.std(model_arima.resid()), size=n_periods
            )
            simulated_forecasts.append(forecast_simulated)

        # Convert to NumPy array
        simulated_forecasts = np.array(simulated_forecasts)

        # Compute the mean and confidence intervals across simulations
        forecast_mean = np.mean(simulated_forecasts, axis=0)
        forecast_lower = np.percentile(simulated_forecasts, 2.5, axis=0)
        forecast_upper = np.percentile(simulated_forecasts, 97.5, axis=0)

        # Convert results to Pandas Series
        forecast_mean_series = pd.Series(forecast_mean, index=forecast_index)
        forecast_lower_series = pd.Series(forecast_lower, index=forecast_index)
        forecast_upper_series = pd.Series(forecast_upper, index=forecast_index)

        # Plot Monte Carlo Forecast
        plt.figure(figsize=(12, 5))

        # Plot historical data
        plt.plot(df_monthly.index, df_monthly['temp'], label="Historical")

        # Plot mean forecast from Monte Carlo
        plt.plot(forecast_mean_series.index, forecast_mean_series, label="Mean Forecast", color='orange')

        # Confidence intervals from Monte Carlo simulations
        plt.fill_between(forecast_index, forecast_lower_series, forecast_upper_series, color='pink', alpha=0.3, label="95% Confidence Interval")

        plt.title("Long-Term SARIMA Forecast with Monte Carlo Simulation (60 Months Ahead)")
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.savefig(f'monte_carlo_forecast_{selected_location}_{selected_depth}.png', bbox_inches='tight', dpi=300)
        plt.close()

    except Exception as e:
        print(f"Error processing {selected_location} - {selected_depth}: {str(e)}")
        continue


