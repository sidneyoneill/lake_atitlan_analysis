import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import json

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

##############################################################################
# 1) Load best SARIMA model info (from your grid-search JSON)
##############################################################################
json_file = "evan_sarima_models_temp_turbidity.json"  # Adjust if different
with open(json_file, "r") as f:
    best_models = json.load(f)

##############################################################################
# 2) We'll produce final forecasts (with confidence intervals) up to 2050
#    for site="WG", for these variables and depths
##############################################################################
target_vars = ["temp", "turbidity", "dissolved_oxygen"] 
target_site = "WG"
depth_levels = ["0-10m", "10-30m", "30m+"]  # Adjust if your dataset has these exact names

# Original dataset to build monthly time series
df_path = "EVAN_LIMNO_processed_v5.xlsx"  # or whichever file you have
df = pd.read_excel(df_path)

# Create folder for forecast plots
out_folder = "evan_forecast_WG_confint"
os.makedirs(out_folder, exist_ok=True)

##############################################################################
# 3) Main loop: for each variable, for each depth => forecast with CI
##############################################################################
for var in target_vars:
    print(f"\nForecasting variable: {var}")
    print("=" * 70)

    for depth_grp in depth_levels:
        # Compose the JSON key, e.g. "Temp. (°C)__WG_0-10 m"
        var_key = f"{var}__{target_site}_{depth_grp}"

        if var_key not in best_models:
            print(f"  -> No best-model entry in JSON for key '{var_key}', skipping.")
            continue

        info = best_models[var_key]
        # Extract stored parameters & metrics
        p_d_q       = tuple(info["order"])             # e.g. [0,1,1]
        P_D_Q_m     = tuple(info["seasonal_order"])    # e.g. [1,0,1,12]
        cv_rmse     = info["cv_rmse"]                  # cross-val
        in_mse      = info["in_mse"]
        in_rmse     = info["in_rmse"]

        print(f"[FORECAST] {var_key}")
        print(f" -> Best model: order={p_d_q}, seasonal={P_D_Q_m}")
        print(f" -> CV-RMSE={cv_rmse:.3f}, in-sample MSE={in_mse:.3f}, RMSE={in_rmse:.3f}")

        ######################################################################
        # 3a) Build the monthly series for site=WG, this depth, and var
        ######################################################################
        subset = df[(df["location"] == target_site) & (df["depth_group"] == depth_grp)].copy()
        if subset.empty:
            print("   No rows for this subset in the data. Skipping.")
            continue

        subset["date"] = pd.to_datetime(subset["date"])
        subset.sort_values("date", inplace=True)
        subset.set_index("date", inplace=True)

        if var not in subset.columns:
            print(f"   -> Column {var} not found for this subset; skipping.")
            continue

        monthly_series = (
            subset[var]
            .resample("MS")
            .mean()
            .interpolate("time")
            .dropna()
        )
        if len(monthly_series) < 2:
            print(f"   -> Not enough data for forecasting.")
            continue

        ######################################################################
        # 3b) Fit final model on entire historical series
        ######################################################################
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            final_model = SARIMAX(
                monthly_series,
                order=p_d_q,
                seasonal_order=P_D_Q_m,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            final_fit = final_model.fit(disp=False)

        ######################################################################
        # 3c) Forecast monthly out to 2050-12
        ######################################################################
        last_date = monthly_series.index[-1]
        forecast_periods = (2050 - last_date.year)*12 + (12 - last_date.month)
        if forecast_periods <= 0:
            print("   -> Already beyond 2050, no forecast needed.")
            continue

        future_index = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1),
            periods=forecast_periods,
            freq='MS'
        )

        # get_forecast => includes confidence intervals
        forecast_result = final_fit.get_forecast(steps=forecast_periods)
        mean_forecast = forecast_result.predicted_mean
        conf_int      = forecast_result.conf_int(alpha=0.05)  # 95% CI by default

        # Typically, statsmodels names columns "lower <var>" / "upper <var>"
        # For spaces or parentheses, it might be "lower Temp. (°C)" etc.
        # So we do:
        col_name_lower = f"lower {var}"
        col_name_upper = f"upper {var}"

        if (col_name_lower not in conf_int.columns) or (col_name_upper not in conf_int.columns):
            # If your variable name has special characters, confirm them in conf_int.columns
            print("   -> Could not find matching 'lower'/'upper' columns in conf_int. Check var name.")
            # We'll skip if not found
            continue

        lower_series = conf_int[col_name_lower]
        upper_series = conf_int[col_name_upper]

        # Reindex
        mean_forecast.index = future_index
        lower_series.index  = future_index
        upper_series.index  = future_index

        ######################################################################
        # 3d) Combine + plot (with CI shading)
        ######################################################################
        plt.figure(figsize=(10,5))

        # Plot historical
        plt.plot(monthly_series.index, monthly_series, 'b-o', label='Historical', markersize=3)

        # Plot forecast mean
        plt.plot(mean_forecast.index, mean_forecast, 'r-s', label='Forecast', markersize=3)

        # Shade CI
        plt.fill_between(
            mean_forecast.index,
            lower_series,
            upper_series,
            color='pink',
            alpha=0.3,
            label='95% CI'
        )

        # Title
        title_str = (
            f"{var} Forecast to 2050\n"
            f"Site={target_site}, Depth={depth_grp}\n"
            f"Order={p_d_q}, Seasonal={P_D_Q_m}, "
            f"CV-RMSE={cv_rmse:.3f}, in-sample MSE={in_mse:.3f}, RMSE={in_rmse:.3f}"
        )
        plt.title(title_str)

        plt.xlabel("Date")
        plt.ylabel(var)
        plt.legend()
        plt.grid(True)

        # Save figure
        out_name = f"{target_site}_{depth_grp}_{var}_forecast_2050_ci.png".replace(" ","_").replace(":","")
        out_path = os.path.join(out_folder, out_name)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"   -> Saved forecast with CI: {out_path}")

print("\nDone forecasting Temp & Turbidity for WG with confidence intervals up to 2050.")
