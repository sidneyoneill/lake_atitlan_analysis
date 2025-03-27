# main_grid_search.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import json  # to save best models

file_path = 'Lake_data_clean.xlsx'
df = pd.read_excel(file_path)

# Dictionary storing monthly time-series DataFrames for each site-depth
subset_dict = {}
unique_sites = df['Sitio'].unique()
unique_depths = df['Depth Group'].unique()

########################################
# 1) Create monthly data for each site-depth
########################################
for site in unique_sites:
    for depth_grp in unique_depths:
        # Filter data
        subset = df[(df['Sitio'] == site) & (df['Depth Group'] == depth_grp)].copy()

        # Convert date & index
        subset['Fecha'] = pd.to_datetime(subset['Fecha'])
        subset.sort_values('Fecha', inplace=True)
        subset.set_index('Fecha', inplace=True)

        # Numeric columns only
        numeric_cols = subset.select_dtypes(include=[np.number]).columns
        df_numerics = subset[numeric_cols]

        # Monthly resample + average
        monthly_numeric = df_numerics.resample('MS').mean()
        monthly_numeric = monthly_numeric.interpolate(method='time')

        # Insert site/depth
        monthly_numeric.insert(0, 'Sitio', site)
        monthly_numeric.insert(1, 'Depth Group', depth_grp)

        key = f"{site}_{depth_grp}"
        subset_dict[key] = monthly_numeric

########################################
# 2) SARIMA grid search over (p,d,q)(P,D,Q)m to minimize test RMSE
########################################

p_values = range(0, 3)   # or 0..2
d_values = range(0, 2)   # or 0..1
q_values = range(0, 3)   # or 0..2
P_values = range(0, 2)
D_values = range(0, 2)
Q_values = range(0, 2)
m = 12  # monthly

# We'll store best model info for each site-depth in a dict
best_models = {}

for key, monthly_df in subset_dict.items():
    site, depth_grp = key.split("_", 1)
    print("=" * 60)
    print(f"PROCESSING: Site={site}, Depth={depth_grp}")

    # Ensure we have Temp. (°C)
    if 'Temp. (°C)' not in monthly_df.columns:
        print(f" -> No 'Temp. (°C)' column for {key}; skipping.")
        continue

    temp_series = monthly_df['Temp. (°C)'].dropna()
    if len(temp_series) < 12:
        print(f" -> Not enough data points in {key} for SARIMA grid search.")
        continue

    # ADF test (optional)
    adf_stat, adf_pvalue, *_ = adfuller(temp_series)
    print(f" -> ADF: stat={adf_stat:.4f}, p-value={adf_pvalue:.4f}")

    # Train/test
    train = temp_series.loc[:'2022-12']
    test = temp_series.loc['2023-01':]
    print(f" -> Train size={len(train)}, Test size={len(test)}")

    if len(train) < 6 or len(test) < 1:
        print(" -> Not enough train/test data.")
        continue

    best_rmse = float('inf')
    best_params = None
    best_results = None
    best_forecast = None

    # Grid search
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            try:
                                model = SARIMAX(train,
                                                order=(p,d,q),
                                                seasonal_order=(P,D,Q,m),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", ConvergenceWarning)
                                    results = model.fit(disp=False)

                                forecast = results.predict(start=test.index[0], end=test.index[-1])
                                this_rmse = np.sqrt(mean_squared_error(test, forecast))

                                if this_rmse < best_rmse:
                                    best_rmse = this_rmse
                                    best_params = ((p,d,q),(P,D,Q,m))
                                    best_results = results
                                    best_forecast = forecast
                            except (ConvergenceWarning, ValueError, OverflowError):
                                # Some combos won't converge or produce invalid results
                                continue

    if best_params is None:
        print(" -> No valid model found in grid search!")
        continue

    # Print best model
    print(f" -> Best model by RMSE: order={best_params[0]}, seasonal_order={best_params[1]}")
    print(f" -> Best test RMSE: {best_rmse:.4f}")

    # Plot final forecast on the test set
    plt.figure(figsize=(10,5))
    plt.plot(train.index, train, label='Train', color='blue')
    plt.plot(test.index, test, label='Test', color='green')
    plt.plot(best_forecast.index, best_forecast, label='Forecast (Best)', color='red')
    plt.title(f"Best SARIMA by RMSE - Site={site}, Depth={depth_grp}")
    plt.xlabel('Date')
    plt.ylabel('Temp (°C)')
    plt.legend()
    plt.grid(True)

    # 3) Save the figure
    fig_name = f"{site}_{depth_grp}_best_model.png".replace(" ", "_").replace("+","plus")
    plt.savefig(fig_name, dpi=150, bbox_inches='tight')
    plt.close()  # close to avoid showing all at once

    # Refit on entire data
    full_model = SARIMAX(temp_series, order=best_params[0],
                         seasonal_order=best_params[1],
                         enforce_stationarity=False,
                         enforce_invertibility=False)
    final_results = full_model.fit(disp=False)

    # 4) Save best model parameters + final fitted model's pickled results or just param
    #    We'll store just the parameters + best RMSE in a Python dict
    best_models[key] = {
        "order": best_params[0],
        "seasonal_order": best_params[1],
        "rmse": best_rmse,
        # If you want to forecast in next script, you'll re-fit from scratch using these params
    }

    print("=" * 60, "\n")

# 5) Save best_models dict to JSON so next script can use it
with open("best_sarima_models.json", "w") as f:
    json.dump(best_models, f, indent=2)

print("Grid search complete. Best models saved to 'best_sarima_models.json'.")
