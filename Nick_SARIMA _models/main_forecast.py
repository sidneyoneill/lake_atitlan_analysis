# main_forecast.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

file_path = 'Lake_data_clean.xlsx'
df = pd.read_excel(file_path)

# Load best parameters from JSON
with open("best_sarima_models.json", "r") as f:
    best_models = json.load(f)

# We re-create monthly data just like before
subset_dict = {}
unique_sites = df['Sitio'].unique()
unique_depths = df['Depth Group'].unique()

for site in unique_sites:
    for depth_grp in unique_depths:
        # Filter & monthly data
        subset = df[(df['Sitio'] == site) & (df['Depth Group'] == depth_grp)].copy()
        subset['Fecha'] = pd.to_datetime(subset['Fecha'])
        subset.sort_values('Fecha', inplace=True)
        subset.set_index('Fecha', inplace=True)

        numeric_cols = subset.select_dtypes(include=[np.number]).columns
        df_numerics = subset[numeric_cols]

        monthly_numeric = df_numerics.resample('MS').mean()
        monthly_numeric = monthly_numeric.interpolate(method='time')

        key = f"{site}_{depth_grp}"
        subset_dict[key] = monthly_numeric

# Now forecast to 2050 for each site/depth using stored best params
for key, monthly_df in subset_dict.items():
    if key not in best_models:
        # Means we didn't find a valid model in the grid search
        continue

    site, depth_grp = key.split("_", 1)
    print("="*60)
    print(f"FORECASTING: Site={site}, Depth={depth_grp}")

    # Retrieve best parameters from the JSON
    order = best_models[key]["order"]  # e.g. [p,d,q]
    seas_order = best_models[key]["seasonal_order"]  # e.g. [P,D,Q,12]
    # If we stored 'rmse', we can also see it here if needed

    temp_series = monthly_df.get('Temp. (°C)')
    if temp_series is None or temp_series.dropna().empty:
        print(f" -> No temperature data for {key}")
        continue

    # Fit on entire data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = SARIMAX(temp_series.dropna(),
                        order=tuple(order),
                        seasonal_order=tuple(seas_order),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        results = model.fit(disp=False)

    # Forecast steps from the last known date to 2050-12
    last_date = temp_series.dropna().index[-1]
    start_year = last_date.year
    start_month = last_date.month

    # months from after last_date to 2050-12
    # e.g. if last_date is 2023-07, we want from 2023-08 to 2050-12
    forecast_periods = (2050 - start_year)*12 + (12 - start_month)  # up to Dec 2050

    if forecast_periods <= 0:
        print(" -> No future periods to forecast (already past 2050?). Skipping.")
        continue

    # Predict
    forecast_array = results.predict(start=last_date + pd.offsets.MonthBegin(1),
                                     end=last_date + pd.offsets.MonthBegin(forecast_periods))
    
    # Make an index
    future_index = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=forecast_periods,
        freq='MS'
    )

    # Convert to Series
    forecast_series = pd.Series(forecast_array, index=future_index, name='Forecast')

    # Plot entire historical + forecast
    plt.figure(figsize=(10,5))
    plt.plot(temp_series.index, temp_series, label='Historical Data', color='blue')
    plt.plot(forecast_series.index, forecast_series, label='Forecast to 2050', color='red')
    plt.title(f"Final Forecast to 2050 - Site={site}, Depth={depth_grp}")
    plt.xlabel('Date')
    plt.ylabel('Temp (°C)')
    plt.legend()
    plt.grid(True)

    # Save forecast plot
    plot_name = f"{site}_{depth_grp}_forecast_2050.png".replace(" ", "_").replace("+","plus")
    plt.savefig(plot_name, dpi=150, bbox_inches='tight')
    plt.close()

    print(f" -> Saved final forecast plot: {plot_name}")
    print("="*60, "\n")

print("All forecasts to 2050 completed.")
