import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
file_path = 'EVAN_LIMNO_processed_v5.xlsx'
df = pd.read_excel(file_path)

# Filter data for site WG
site = 'WG'
df_wg = df[df['location'] == site].copy()

# Convert date column to datetime and set as index
df_wg['date'] = pd.to_datetime(df_wg['date'])
df_wg.set_index('date', inplace=True)

# Variables to analyze
variables = ['turbidity', 'temp']

# Prepare figure
fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Two variables, ACF and PACF for each
axes = axes.flatten()

for i, var in enumerate(variables):
    if var in df_wg.columns:
        # Resample monthly and interpolate missing values
        monthly_series = (df_wg[var].resample('MS')
                                     .mean()
                                     .interpolate(method='time')
                                     .dropna())

        if len(monthly_series) >= 12:
            max_lags = 34
            # --- ACF (no 'method' argument) ---
            plot_acf(monthly_series,
                     ax=axes[2*i],
                     lags=max_lags,
                     title=f"ACF of {var} (Site: {site})")
            axes[2*i].set_xlabel("Lag (Months)")
            axes[2*i].set_ylabel("Autocorrelation")

            # --- PACF (with 'method'='ld') ---
            plot_pacf(monthly_series,
                      ax=axes[2*i + 1],
                      lags=max_lags,
                      method='ld',  # <-- Only valid in PACF
                      title=f"PACF of {var} (Site: {site})")
            axes[2*i + 1].set_xlabel("Lag (Months)")
            axes[2*i + 1].set_ylabel("Partial Autocorrelation")

        else:
            print(f"Not enough data points for {var} at site {site}.")
    else:
        print(f"Variable {var} not found in the dataset for site {site}.")

# Adjust layout
plt.tight_layout()
plt.show()
