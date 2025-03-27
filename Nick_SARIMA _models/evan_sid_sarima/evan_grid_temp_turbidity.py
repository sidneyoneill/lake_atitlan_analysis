import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
import os

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# For metrics
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    explained_variance_score
)

# For cross-validation
from sklearn.model_selection import TimeSeriesSplit

##############################################################################
# 1) Load dataset and define variables
##############################################################################
file_path = 'EVAN_LIMNO_processed_v5.xlsx'
df = pd.read_excel(file_path)

# Filter for site "WG" only
df = df[df['location'] == 'WG']

target_vars = ["temp", "turbidity"]

unique_sites = df['location'].unique()  # This will only be "WG"
unique_depths = df['depth_group'].unique()

##############################################################################
# 2) Define dictionary to store best models
##############################################################################
best_models = {}

##############################################################################
# 3) Define parameter grid + cross-validation splits
##############################################################################
p_values = range(0, 2)   # e.g. 0..1
d_values = range(0, 2)   # e.g. 0..1
q_values = range(0, 2)   # e.g. 0..1

P_values = range(0, 2)   # e.g. 0..1
D_values = range(0, 2)   # e.g. 0..1
Q_values = range(0, 2)   # e.g. 0..1

m = 12  # monthly seasonality
n_splits = 5  # five-fold TimeSeriesSplit

##############################################################################
# 4) Main loop over each variable, site, depth
##############################################################################
for var in target_vars:
    print(f"\n{'='*80}")
    print(f"Processing variable: {var}")
    print(f"{'='*80}\n")

    # Create a folder for saving plots for this variable
    var_folder = f"plots_{var.replace(' ', '_').replace('(','').replace(')','').replace('/','_')}"
    os.makedirs(var_folder, exist_ok=True)

    for site in unique_sites:  # This will only be "WG"
        for depth_grp in unique_depths:

            # Filter for just this site + depth
            subset = df[(df['location'] == site) & (df['depth_group'] == depth_grp)].copy()
            if subset.empty:
                continue

            # Prepare time index
            subset['date'] = pd.to_datetime(subset['date'])
            subset.sort_values('date', inplace=True)
            subset.set_index('date', inplace=True)

            # Check if the variable is in the columns
            if var not in subset.columns:
                print(f"Skipping {var} for {site}_{depth_grp}: not in dataset columns.")
                continue

            # Resample monthly and interpolate
            monthly_series = subset[var].resample('MS').mean()
            monthly_series = monthly_series.interpolate(method='time').dropna()

            # Need at least 12 points to do anything
            if len(monthly_series) < 12:
                print(f"Skipping {var} for {site}_{depth_grp}, because <12 data points.")
                continue

            # ADF test (optional)
            adf_stat, adf_pvalue, *_ = adfuller(monthly_series)
            print(f"Variable={var}, Site={site}, Depth={depth_grp}, ADF p-value={adf_pvalue:.4f}")

            # TimeSeriesSplit for cross-validation (used for parameter selection)
            tscv = TimeSeriesSplit(n_splits=n_splits)

            best_rmse = float('inf')
            best_params = None
            best_fit = None

            ############################################################################
            # 4a) Grid search across (p,d,q)(P,D,Q,m), measuring average RMSE via tscv
            ############################################################################
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        for P in P_values:
                            for D in D_values:
                                for Q in Q_values:
                                    split_rmses = []
                                    try:
                                        for train_index, test_index in tscv.split(monthly_series):
                                            train_data = monthly_series.iloc[train_index]
                                            test_data  = monthly_series.iloc[test_index]

                                            # Ensure training data is long enough for seasonal modeling
                                            if len(train_data) < m:
                                                raise ValueError("Insufficient training data for seasonal model")

                                            with warnings.catch_warnings():
                                                warnings.simplefilter("ignore", ConvergenceWarning)
                                                warnings.simplefilter("ignore", UserWarning)
                                                warnings.simplefilter("ignore", RuntimeWarning)
                                                model = SARIMAX(
                                                    train_data,
                                                    order=(p, d, q),
                                                    seasonal_order=(P, D, Q, m),
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False
                                                )
                                                fit_res = model.fit(disp=False)

                                            preds = fit_res.predict(start=test_data.index[0],
                                                                    end=test_data.index[-1])
                                            rmse_fold = np.sqrt(mean_squared_error(test_data, preds))
                                            split_rmses.append(rmse_fold)

                                        avg_rmse = np.mean(split_rmses)
                                        if avg_rmse < best_rmse:
                                            best_rmse = avg_rmse
                                            best_params = ((p, d, q), (P, D, Q, m))
                                            best_fit = fit_res

                                    except (ValueError, OverflowError, ConvergenceWarning, IndexError):
                                        continue

            if best_params is None:
                print(f"No valid model found for {var}, {site}_{depth_grp}. Skipping.\n")
                continue

            # -------------------------------
            # NEW: Split data into training (70%) and test (30%) sets
            test_size = int(len(monthly_series) * 0.30)
            if test_size < 1:
                test_size = 1
            train_series = monthly_series.iloc[:-test_size]
            test_series  = monthly_series.iloc[-test_size:]
            split_date = train_series.index[-1]  # This is the last date of training data

            # -------------------------------
            # 4b) Refit on the training data using the best parameters
            final_model = SARIMAX(
                train_series,
                order=best_params[0],
                seasonal_order=best_params[1],
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            final_res = final_model.fit(disp=False)

            # 4c) Generate predictions:
            # In-sample predictions for the training period...
            train_pred = final_res.predict(start=train_series.index[0],
                                           end=train_series.index[-1])
            # Forecast for the test period
            forecast = final_res.predict(start=test_series.index[0],
                                         end=test_series.index[-1])

            # 4d) Evaluate in-sample (training) metrics
            mse_train = mean_squared_error(train_series, train_pred)
            rmse_train = np.sqrt(mse_train)
            mae_train = mean_absolute_error(train_series, train_pred)
            mape_train = mean_absolute_percentage_error(train_series, train_pred) * 100
            r2_train = r2_score(train_series, train_pred)
            evs_train = explained_variance_score(train_series, train_pred)

            # Evaluate test (forecast) metrics
            mse_test = mean_squared_error(test_series, forecast)
            rmse_test = np.sqrt(mse_test)
            mae_test = mean_absolute_error(test_series, forecast)
            mape_test = mean_absolute_percentage_error(test_series, forecast) * 100
            r2_test = r2_score(test_series, forecast)
            evs_test = explained_variance_score(test_series, forecast)

            print(f" Best model: order={best_params[0]}, seasonal={best_params[1]}")
            print(f"   5-fold avg RMSE (CV) = {best_rmse:.4f}")
            print(f"   Training => MSE={mse_train:.3f}, RMSE={rmse_train:.3f}, MAPE={mape_train:.2f}%, "
                  f"MAE={mae_train:.3f}, R2={r2_train:.3f}")
            print(f"   Test     => MSE={mse_test:.3f}, RMSE={rmse_test:.3f}, MAPE={mape_test:.2f}%, "
                  f"MAE={mae_test:.3f}, R2={r2_test:.3f}\n")

            ############################################################################
            # 4e) Plot: Training predictions and Test forecast vs actual data
            ############################################################################
            plt.figure(figsize=(10, 4))
            # Plot training data and its predictions
            plt.plot(train_series.index, train_series.values, '-o', label='Train Actual', markersize=4)
            plt.plot(train_series.index, train_pred, '-s', label='Train Prediction', markersize=4)
            # Plot test data and forecast
            plt.plot(test_series.index, test_series.values, '-o', label='Test Actual', markersize=4)
            plt.plot(test_series.index, forecast, '-s', label='Test Forecast', markersize=4)
            # Draw a vertical line at the train-test split
            plt.axvline(x=split_date, color='gray', linestyle='--', label='Train/Test Split')

            title_str = (
                f"{var} - {site}, {depth_grp}\n"
                f"Order={best_params[0]}, Seas={best_params[1]}\n"
                f"Train RMSE={rmse_train:.3f} | Test RMSE={rmse_test:.3f}"
            )
            plt.title(title_str)
            plt.xlabel('Date')
            plt.ylabel(var)
            plt.legend()
            plt.grid(True)

            combined_for_ylim = pd.concat([train_series, test_series])
            ymin, ymax = combined_for_ylim.min(), combined_for_ylim.max()
            plt.ylim([ymin - 0.1*(ymax-ymin), ymax + 0.1*(ymax-ymin)])

            fig_name = f"{site}_{depth_grp}_{var}_train_test_forecast.png".replace(" ", "_")
            out_path = os.path.join(var_folder, fig_name)
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()

            # 4f) Store final info (you could also store separate train/test metrics)
            var_key = f"{var}__{site}_{depth_grp}"
            best_models[var_key] = {
                "var_name":       var,
                "site":           site,
                "depth":          depth_grp,
                "order":          best_params[0],
                "seasonal_order": best_params[1],
                "cv_rmse":        best_rmse,
                "train_mse":      mse_train,
                "train_rmse":     rmse_train,
                "train_mae":      mae_train,
                "train_mape":     mape_train,
                "train_r2":       r2_train,
                "train_evs":      evs_train,
                "test_mse":       mse_test,
                "test_rmse":      rmse_test,
                "test_mae":       mae_test,
                "test_mape":      mape_test,
                "test_r2":        r2_test,
                "test_evs":       evs_test
            }

##############################################################################
# 5) Save best models to JSON
##############################################################################
out_json = "evan_sarima_models_temp_turbidity.json"
with open(out_json, "w") as f:
    json.dump(best_models, f, indent=2)

print("\nAll variables processed with 5-fold TimeSeriesSplit and a 30% holdout test set. "
      f"Results saved to '{out_json}'.")
