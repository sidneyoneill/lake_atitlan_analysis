import pandas as pd

# Load the dataset
file_path = 'Lake_data_SVD.xlsx'
data = pd.read_excel(file_path)

# Define depth bins and labels
depth_column = 'Profuidad (m)'
depth_bins = [0, 10, 30, float('inf')]
depth_labels = ['0-10 m', '10-30 m', '30+ m']

# Add a depth group column
data['Depth Group'] = pd.cut(data[depth_column], bins=depth_bins, labels=depth_labels, right=False)

# Identify numeric columns
numeric_columns = data.select_dtypes(include='number').columns

# Group by 'Sitio', 'Depth Group', and 'Fecha', calculating the mean for numeric columns
grouped_data = (
    data.groupby(['Sitio', 'Depth Group', 'Fecha'], as_index=False)[numeric_columns]
    .mean()
)

# Filter to ensure only original dates are included
original_dates = data[['Sitio', 'Depth Group', 'Fecha']].drop_duplicates()
grouped_data = grouped_data.merge(original_dates, on=['Sitio', 'Depth Group', 'Fecha'], how='inner')

# Sort the final dataset
grouped_data = grouped_data.sort_values(by=['Sitio', 'Depth Group', 'Fecha'])

# Save the processed dataset to an Excel file
output_path = 'Lake_data_grp.xlsx'
grouped_data.to_excel(output_path, index=False)

