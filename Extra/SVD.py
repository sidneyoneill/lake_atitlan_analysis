import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "Raw_data_grouped.xlsx"
data = pd.read_excel(file_path)

# Step 1: Select specified columns for the new dataset
metadata_columns = ['Sitio', 'Fecha', 'Profuidad (m)']
selected_columns = metadata_columns + ['Temp. (\u00b0C)', 'Chl a (\u00b5g/l)', 'Ph (Unidad)', 'Turbidity (NTU)', 'DO (mg/L)', 'TDS (mg/l)', 'DBO (mg/l)', 'Sechi  (m)', 'NO3 (\u00b5g/L)', 'PO4 (\u00b5g/L)',
                                        'NH4 (\u00b5g/L)', 'NT (\u00b5g/l)', 'PT (\u00b5g/l)']
selected_data = data[selected_columns]

# Step 2: Filter data for the specified date range (January 2018 to December 2023)
selected_data['Fecha'] = pd.to_datetime(selected_data['Fecha'])
filtered_data = selected_data[(selected_data['Fecha'] >= '2018-01-01') & (selected_data['Fecha'] <= '2023-12-31')]

# Step 3: Create a mask of missing values
numeric_columns = ['Temp. (\u00b0C)', 'Chl a (\u00b5g/l)', 'Ph (Unidad)', 'Turbidity (NTU)', 'DO (mg/L)', 'TDS (mg/l)', 'DBO (mg/l)', 'Sechi  (m)', 'NO3 (\u00b5g/L)', 'PO4 (\u00b5g/L)',
                                        'NH4 (\u00b5g/L)', 'NT (\u00b5g/l)', 'PT (\u00b5g/l)']
numeric_data = filtered_data[numeric_columns]
missing_mask = numeric_data.isna().to_numpy()

# Step 4: Initial imputation (mean imputation) to enable SVD
imputer = SimpleImputer(strategy="mean")
numeric_data_imputed = imputer.fit_transform(numeric_data)

# Ensure all missing values are handled at this stage
assert not np.any(np.isnan(numeric_data_imputed)), "Initial imputation failed, NaN values still present."

# Step 5: Normalize the data for stability in SVD
scaler = StandardScaler()
numeric_data_normalized = scaler.fit_transform(numeric_data_imputed)

# Step 6: Perform SVD for missing data imputation
svd = TruncatedSVD(n_components=min(numeric_data_normalized.shape[1], 10), random_state=42)
U = svd.fit_transform(numeric_data_normalized)  # U matrix
Sigma = np.diag(svd.singular_values_)           # Sigma matrix
V = svd.components_                             # V^T matrix

# Reconstruct the matrix
reconstructed_data_normalized = np.dot(U, np.dot(Sigma, V))

# Denormalize the data
reconstructed_data = scaler.inverse_transform(reconstructed_data_normalized)

# Step 7: Replace only the missing values in the original dataset
filled_data = numeric_data.to_numpy().copy()
filled_data[missing_mask] = reconstructed_data[missing_mask]

# Validate that non-missing values remain unchanged
assert np.allclose(filled_data[~missing_mask], numeric_data.to_numpy()[~missing_mask]), "Non-missing values were altered!"

# Step 8: Clip values to realistic ranges (no negatives, cap at 99th percentile)
for col_idx, col in enumerate(numeric_columns):
    min_val = 0  # No negative values
    max_val = np.percentile(filled_data[:, col_idx], 99)  # Cap at the 99th percentile
    filled_data[:, col_idx] = np.clip(filled_data[:, col_idx], min_val, max_val)

# Step 9: Convert back to DataFrame
filled_data_df = pd.DataFrame(filled_data, columns=numeric_columns, index=numeric_data.index)
filled_data_df = pd.concat([filtered_data[metadata_columns].reset_index(drop=True), filled_data_df.reset_index(drop=True)], axis=1)

# Step 10: Save the new dataset
output_path = "Lake_data_Selected_SVD2.xlsx"
filled_data_df.to_excel(output_path, index=False)

print(f"Processed dataset saved to {output_path}")
