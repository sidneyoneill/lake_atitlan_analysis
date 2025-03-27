import pandas as pd
from functions import read_csv_file, sort_dataset

# Read a CSV file
df_Sid = read_csv_file('data/SID_LIMNO_no_outliers.csv', encoding='utf-8')

# Example usage:
df_Sid_sorted = sort_dataset(df_Sid)

# Save sorted dataframe if needed
df_Sid_sorted.to_csv('data/SID_LIMNO_no_outliers_v2.csv', index=False)

