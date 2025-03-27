import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
from functions import read_csv_file, read_excel_file, combine_depth_rows, rename_depth_groups, sort_dataset

# Read a CSV file
df_Sid = read_csv_file('data/SID_LIMNO_processed_v2.csv', encoding='utf-8')

df_Sid_combined = combine_depth_rows(df_Sid)

# save df_Sid_combined to csv
df_Sid_combined.to_csv('data/SID_LIMNO_processed_v3.csv', index=False)

# Example usage:
df_Sid_renamed = rename_depth_groups(df_Sid_combined)

# Save renamed dataframe if needed
df_Sid_renamed.to_csv('data/SID_LIMNO_processed_v4.csv', index=False)

# Example usage:
df_Sid_sorted = sort_dataset(df_Sid_renamed)

# Save sorted dataframe if needed
df_Sid_sorted.to_csv('data/SID_LIMNO_processed_v5.csv', index=False)