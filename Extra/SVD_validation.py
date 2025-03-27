import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Load the dataset
file_path = "Raw_data_grouped.xlsx"
data = pd.read_excel(file_path)

# Step 1: Select specified columns for the new dataset
metadata_columns = ['Sitio', 'Fecha', 'Profuidad (m)']
numeric_columns = ['Temp. (°C)', 'Chl a (µg/l)', 'Ph (Unidad)', 'Turbidity (NTU)', 'DO (mg/L)', 'TDS (mg/l)',
                   'DBO (mg/l)', 'Sechi  (m)', 'NO3 (µg/L)', 'PO4 (µg/L)', 'NH4 (µg/L)', 'NT (µg/l)', 'PT (µg/l)']
selected_columns = metadata_columns + numeric_columns
selected_data = data[selected_columns].copy()

# Step 2: Filter data for the specified date range (January 2018 to December 2023)
selected_data.loc[:, 'Fecha'] = pd.to_datetime(selected_data['Fecha'])
filtered_data = selected_data[(selected_data['Fecha'] >= '2018-01-01') & (selected_data['Fecha'] <= '2023-12-31')]

# Step 3: Process data (Original SVD Imputation)
def svd_impute(data):
    numeric_data = data[numeric_columns]
    missing_mask = numeric_data.isna().to_numpy()

    # Initial mean imputation
    imputer = SimpleImputer(strategy="mean")
    numeric_data_imputed = imputer.fit_transform(numeric_data)

    # Normalize the data
    scaler = StandardScaler()
    numeric_data_normalized = scaler.fit_transform(numeric_data_imputed)

    # Perform SVD
    svd = TruncatedSVD(n_components=min(numeric_data_normalized.shape[1], 10), random_state=42)
    U = svd.fit_transform(numeric_data_normalized)
    Sigma = np.diag(svd.singular_values_)
    V = svd.components_

    # Reconstruct the matrix
    reconstructed_data_normalized = np.dot(U, np.dot(Sigma, V))

    # Denormalize
    reconstructed_data = scaler.inverse_transform(reconstructed_data_normalized)

    # Replace only missing values
    filled_data = numeric_data.to_numpy().copy()
    filled_data[missing_mask] = reconstructed_data[missing_mask]

    # Clip values to realistic ranges
    for col_idx, col in enumerate(numeric_columns):
        min_val = 0
        max_val = np.percentile(filled_data[:, col_idx], 99)
        filled_data[:, col_idx] = np.clip(filled_data[:, col_idx], min_val, max_val)

    return pd.DataFrame(filled_data, columns=numeric_columns, index=numeric_data.index)

# First run: original imputation
filled_data_original = svd_impute(filtered_data)

# Step 4: Introduce 5% missing values randomly and reapply SVD over 10 loops for mean and variance
mean_rmses = []
mean_pearsons = []
rmse_matrix = np.zeros((10, len(numeric_columns)))

for i in range(10):
    numeric_data_with_nans = filtered_data[numeric_columns].copy()
    mask = np.random.rand(*numeric_data_with_nans.shape) < 0.05  # 5% missing values
    numeric_data_with_nans[mask] = np.nan
    filtered_data_with_nans = filtered_data.copy()
    filtered_data_with_nans[numeric_columns] = numeric_data_with_nans

    # Second run: SVD imputation on data with additional missing values
    filled_data_with_nans = svd_impute(filtered_data_with_nans)

    # Compare the two imputed datasets using RMSE
    rmse_values = [
        np.sqrt(mean_squared_error(filled_data_original[col], filled_data_with_nans[col])) / np.mean(filled_data_original[col])
        for col in numeric_columns
    ]

    rmse_matrix[i, :] = rmse_values

    mean_rmse = np.mean(rmse_values)
    mean_rmses.append(mean_rmse)

    # Calculate Pearson correlation coefficient for each numeric column
    pearson_correlations = [
        pearsonr(filled_data_original[col], filled_data_with_nans[col])[0] for col in numeric_columns
    ]

    mean_pearson_correlation = np.mean(pearson_correlations)
    mean_pearsons.append(mean_pearson_correlation)

# Calculate variance for each column in RMSE matrix
rmse_variances = np.var(rmse_matrix, axis=0)
rmse_variance_df = pd.DataFrame({'Feature': numeric_columns, 'RMSE Variance': rmse_variances})
print("RMSE Variance for each feature:")
print(rmse_variance_df)

# Summary metrics
print("RMSE Matrix:")
print(pd.DataFrame(rmse_matrix, columns=numeric_columns))

mean_mean_rmses = np.mean(mean_rmses)
rmses_var = np.var(mean_rmses)
mean_pearsons = np.mean(mean_pearsons)

summary_metrics_df = pd.DataFrame({
    'Metric': ['Mean Normalized RMSE', 'RMSE Variance', 'Mean Pearson Correlation'],
    'Value': [mean_mean_rmses, rmses_var, mean_pearsons]
})

print(summary_metrics_df)
