import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = "Lake data.xlsx"
data = pd.read_excel(file_path)

# Step 1: Define nutrient variables and compute Nutrient Load Index (NLI)
nutrient_columns = ['NO3 (\u00b5g/L)', 'PO4 (\u00b5g/L)', 'NH4 (\u00b5g/L)', 'NT (\u00b5g/l)', 'PT (\u00b5g/l)']
numeric_columns = data.select_dtypes(include="number").columns  # All numeric columns

# Step 2: Select numeric columns for SVD imputation
numeric_data = data[numeric_columns]

# Step 3: Create a mask of missing values
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
filled_data = numeric_data_imputed.copy()
filled_data[missing_mask] = reconstructed_data[missing_mask]

# Step 8: Clip values to realistic ranges (no negatives, cap at 99th percentile)
for col_idx, col in enumerate(numeric_columns):
    min_val = 0  # No negative values
    max_val = np.percentile(filled_data[:, col_idx], 99)  # Cap at the 99th percentile
    filled_data[:, col_idx] = np.clip(filled_data[:, col_idx], min_val, max_val)

# Step 9: Convert back to DataFrame and recompute NLI
filled_data_df = pd.DataFrame(filled_data, columns=numeric_columns, index=numeric_data.index)
filled_data_df['Nutrient Load Index'] = filled_data_df[nutrient_columns].sum(axis=1)

# Step 10: Replace numeric columns in the original dataset
data[numeric_columns] = filled_data_df.drop(columns=['Nutrient Load Index'])
data['Nutrient Load Index'] = filled_data_df['Nutrient Load Index']

# Step 11: Save the updated dataset
output_path = "Lake_data_SVD.xlsx"
data.to_excel(output_path, index=False)

print(f"Processed dataset saved to {output_path}")