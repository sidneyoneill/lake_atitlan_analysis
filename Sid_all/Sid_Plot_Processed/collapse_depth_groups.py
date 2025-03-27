import pandas as pd
import os

# Load the Excel file
file_path = "data/LIMNO.xlsx"
xls = pd.ExcelFile(file_path)

# Load the sheet into a DataFrame
df = pd.read_excel(xls, sheet_name='Sheet1')

# Define the depth groups
def assign_depth_group(depth):
    if depth <= 10:
        return "0-10m"
    elif depth <= 30:
        return "10-30m"
    else:
        return "30m+"

# Apply the depth group function
df["depth_group"] = df["depth"].apply(assign_depth_group)

# Group by location, date, depth_group, and compute mean for each feature
grouped_df = df.groupby(["location", "date", "depth_group"]).mean(numeric_only=True).reset_index()

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Save the grouped DataFrame to CSV
output_path = "output/grouped_depths.csv"
grouped_df.to_csv(output_path, index=False)