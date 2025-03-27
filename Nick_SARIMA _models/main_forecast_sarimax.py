# main_forecast_sarimax.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import json
import os

file_path = 'Lake_data_clean.xlsx'
df = pd.read_excel(file_path)

# Load best SARIMAX params
with open("best_sarimax_models.json", "r") as f:
    best_models = json.load(f)

# Create folder for forecast plots
os.makedirs("sarimax_forecast_plots", exist_ok=True)

# Re-create monthly data as before
subset_dict = {}
unique_sites = df['Sitio'].unique()
unique_depths = df['Depth Group'].unique()

for site in unique_sites:
    for depth_grp in unique_depths:
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

###################################################
# Forecast up to 2050 using best SARIMAX + DO as exog
###################################################
for key, monthly_df in subset_dict.items():
    if key not in best_models:
        # Means no valid SARIMAX model was found for this site/depth
        continue

    site, depth_grp = key.split("_", 1)
    print("="*60)
    print(f"[FORECAST SARIMAX] Site={site}, Depth={depth_grp}")

    # Grab best params
    order = best_models[key]["order"]
    seas_order = best_models[key]["seasonal_order"]

    # Prepare target and exog
    if 'Temp. (°C)' not in monthly_df.columns:
        print(" -> No temperature data, skipping.")
        continue
    if 'DO (mg/L)' not in monthly_df.columns:
        print(" -> No DO data, skipping.")
        continue

    temp_series = monthly_df['Temp. (°C)'].dropna()
    do_series = monthly_df['DO (mg/L)'].dropna()

    # Align them
    combined_df = pd.concat([temp_series, do_series], axis=1, join='inner').dropna()
    temp_series = combined_df['Temp. (°C)']
    do_series = combined_df['DO (mg/L)']

    if temp_series.empty:
        print(" -> No data left after alignment, skipping.")
        continue

    # Fit on entire historical period (2018 - 2023)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = SARIMAX(
            temp_series,
            exog=do_series,
            order=tuple(order),
            seasonal_order=tuple(seas_order),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)

    # We'll forecast monthly from the last available date until 2050-12.
    last_date = temp_series.index[-1]
    forecast_start = last_date + pd.offsets.MonthBegin(1)

    # Figure out how many months until 2050-12
    end_year = 2050
    end_month = 12
    forecast_periods = (end_year - last_date.year)*12 + (end_month - last_date.month)
    if forecast_periods <= 0:
        print(" -> Already past 2050 in data or no future months to forecast.")
        continue

    # -- Naive Approach for DO beyond 2023 --
    # We'll extend the exogenous DO data by repeating the last known DO for each future month
    last_do_value = do_series.iloc[-1]
    future_exog = np.full(shape=(forecast_periods,), fill_value=last_do_value)
    # Create date index for these future months
    future_index = pd.date_range(start=forecast_start, periods=forecast_periods, freq='MS')
    future_exog_df = pd.DataFrame({'DO (mg/L)': future_exog}, index=future_index)

    # Forecast
    forecast_array = results.predict(start=forecast_start, end=future_index[-1], exog=future_exog_df)
    forecast_series = pd.Series(forecast_array, index=future_index, name='Forecast')

    # Combine historical + forecast for plotting
    plt.figure(figsize=(10, 5))
    plt.plot(temp_series.index, temp_series, label='Historical (Temp)', color='blue')
    plt.plot(forecast_series.index, forecast_series, label='Forecast (Temp)', color='red')
    plt.title(f"SARIMAX Forecast to 2050\nSite={site}, Depth={depth_grp}")
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)

    out_file = os.path.join("sarimax_forecast_plots", f"{key}_forecast_2050.png")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" -> Saved forecast plot: {out_file}")
    print("="*60, "\n")

print("All SARIMAX forecasts completed.")
