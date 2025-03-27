import pandas as pd

file_path = "Lake_data_grp.xlsx"
data = pd.read_excel(file_path)

# Multiply the values in the Turbidity column before 2017 by 0.1 to account for svd inaccuracy
end_date = pd.Timestamp("2017-12-31")
data.loc[data['Fecha'] <= end_date, 'Turbidity (NTU)'] *= 0.1
# Define the Z-score threshold for outlier detection
z_threshold = 2

# Identify numeric columns to process
columns_to_process = data.select_dtypes(include=['float64', 'int64']).columns

# Function to replace outliers with NaN and interpolate
def replace_outliers_with_interpolation(group):
    for col in columns_to_process:
        if col not in ['Fecha']:  # Skip non-numeric columns
            mean = group[col].mean()
            std = group[col].std()
            # Identify outliers
            z_scores = (group[col] - mean) / std
            outliers = z_scores.abs() > z_threshold
            # Replace outliers with NaN
            group.loc[outliers, col] = None
            # Interpolate to fill NaN values
            group[col] = group[col].interpolate(method='linear')
    return group

# Apply the function to the grouped data
cleaned_interpolated_df = data.groupby(['Sitio', 'Depth Group']).apply(replace_outliers_with_interpolation).reset_index(drop=True)

# Save the cleaned and interpolated dataset to a new Excel file
output_path = "Lake_data_final.xlsx"
cleaned_interpolated_df.to_excel(output_path, index=False)

print(f"Cleaned data saved to {output_path}")

