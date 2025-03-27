# main_grid_search_sarimax.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import json
import os

# 1) Read data
file_path = 'Lake_data_clean.xlsx'
df = pd.read_excel(file_path)

# 2) Create folders to save plots, if desired
os.makedirs("sarimax_model_plots", exist_ok=True)

# 3) Create monthly data for each site-depth
subset_dict = {}
unique_sites = df['Sitio'].unique()
unique_depths = df['Depth Group'].unique()

for site in unique_sites:
    for depth_grp in unique_depths:
        # Filter for site & depth group
        subset = df[(df['Sitio'] == site) & (df['Depth Group'] == depth_grp)].copy()
        subset['Fecha'] = pd.to_datetime(subset['Fecha'])
        subset.sort_values('Fecha', inplace=True)
        subset.set_index('Fecha', inplace=True)

        # Numeric columns
        numeric_cols = subset.select_dtypes(include=[np.number]).columns
        df_numerics = subset[numeric_cols]

        # Resample monthly, interpolate
        monthly_numeric = df_numerics.resample('MS').mean()
        monthly_numeric = monthly_numeric.interpolate(method='time')

        # Store in dict
        key = f"{site}_{depth_grp}"
        subset_dict[key] = monthly_numeric

# 4) Define grid search ranges
p_values = range(0, 3)   # e.g. 0..2
d_values = range(0, 2)   # e.g. 0..1
q_values = range(0, 3)   # e.g. 0..2
P_values = range(0, 2)
D_values = range(0, 2)
Q_values = range(0, 2)
m = 12  # monthly seasonality

best_models = {}

for key, monthly_df in subset_dict.items():
    site, depth_grp = key.split("_", 1)
    print("=" * 60)
    print(f"[SARIMAX GRID SEARCH] Site={site}, Depth={depth_grp}")

    # Target = Temperature
    if 'Temp. (°C)' not in monthly_df.columns:
        print(" -> No temperature data, skipping.")
        continue
    temp_series = monthly_df['Temp. (°C)'].dropna()

    # Exogenous = DO
    if 'DO (mg/L)' not in monthly_df.columns:
        print(" -> No DO (mg/L) data, skipping SARIMAX.")
        continue
    do_series = monthly_df['DO (mg/L)'].interpolate(method='time').dropna()

    # Align indexes so we only keep dates with both temperature & DO
    combined_df = pd.concat([temp_series, do_series], axis=1, join='inner').dropna()
    # Now temp_series & do_series have the same timestamps
    temp_series = combined_df['Temp. (°C)']
    do_series = combined_df['DO (mg/L)']

    # Need enough data to do a train/test
    if len(temp_series) < 18:
        print(" -> Not enough data points for a meaningful grid search.")
        continue

    # ADF test (optional)
    adf_stat, adf_pvalue, *_ = adfuller(temp_series)
    print(f" -> ADF: stat={adf_stat:.4f}, p-value={adf_pvalue:.4f}")

    # Train/test split
    train = temp_series.loc[:'2022-12']
    test = temp_series.loc['2023-01':]
    train_exog = do_series.loc[train.index]
    test_exog = do_series.loc[test.index]
    print(f" -> Train size={len(train)}, Test size={len(test)}")

    if len(train) < 12 or len(test) < 1:
        print(" -> Not enough train/test data. Skipping.")
        continue

    best_rmse = float('inf')
    best_params = None
    best_forecast = None

    # 5) Grid search
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", ConvergenceWarning)
                                    model = SARIMAX(
                                        train,
                                        exog=train_exog,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, m),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    results = model.fit(disp=False)

                                # Forecast on test
                                forecast = results.predict(
                                    start=test.index[0],
                                    end=test.index[-1],
                                    exog=test_exog  # exogenous for test
                                )

                                this_rmse = np.sqrt(mean_squared_error(test, forecast))
                                if this_rmse < best_rmse:
                                    best_rmse = this_rmse
                                    best_params = ((p, d, q), (P, D, Q, m))
                                    best_forecast = forecast
                            except (ValueError, OverflowError):
                                continue

    # If no best_params found, skip
    if not best_params:
        print(" -> No valid SARIMAX model found for this subset.")
        continue

    print(f" -> Best params: ARIMA{best_params[0]}, seasonal={best_params[1]}, RMSE={best_rmse:.4f}")

    # 6) Plot best forecast vs. test
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label='Train', color='blue')
    plt.plot(test.index, test, label='Test', color='green')
    plt.plot(best_forecast.index, best_forecast, label='Forecast (Best)', color='red')
    plt.title(f"SARIMAX Best Model - {site}, {depth_grp} (RMSE={best_rmse:.2f})")
    plt.xlabel('Date')
    plt.ylabel('Temp. (°C)')
    plt.legend()
    plt.grid(True)
    plot_file = os.path.join("sarimax_model_plots", f"{key}_best_sarimax.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" -> Saved best model plot to: {plot_file}")

    # 7) Fit final model on entire data (2018-2023)
    full_model = SARIMAX(
        temp_series,
        exog=do_series,
        order=best_params[0],
        seasonal_order=best_params[1],
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    final_results = full_model.fit(disp=False)

    # 8) Save to dictionary
    best_models[key] = {
        "order": best_params[0],
        "seasonal_order": best_params[1],
        "rmse": best_rmse,
        # No need to store entire fitted model here
    }

    print("=" * 60, "\n")

# 9) Save best_models to JSON
with open("best_sarimax_models.json", "w") as f:
    json.dump(best_models, f, indent=2)

print("SARIMAX grid search complete. Best models saved to 'best_sarimax_models.json'.")
