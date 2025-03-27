import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
from functions import read_csv_file, rename_depth_groups_evan, sort_dataset, read_excel_file

# Read an Excel file
df_Evan = read_excel_file('data/Lake_data_clean_final.xlsx', sheet_name='Sheet1')

df_Evan_renamed = rename_depth_groups_evan(df_Evan)

# save df_Evan_renamed to csv
df_Evan_renamed.to_csv('data/Lake_data_clean_final_v2.csv', index=False)

# Example usage:
df_Evan_sorted = sort_dataset(df_Evan_renamed)

# Save sorted dataframe if needed
df_Evan_sorted.to_csv('data/Lake_data_clean_final_v3.csv', index=False)