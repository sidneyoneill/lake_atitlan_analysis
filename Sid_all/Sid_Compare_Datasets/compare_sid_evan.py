from functions import read_csv_file, rename_depth_groups, rename_depth_groups_evan, plot_time_series

def main():
    # Read and process Sid's data
    sid_df = read_csv_file('data/SID_LIMNO_processed_v5.csv')
    if sid_df is None:
        print("Failed to load Sid's data")
        return
    sid_df = rename_depth_groups(sid_df)

    # Read and process Evan's data
    evan_df = read_csv_file('data/EVAN_LIMNO_processed_v5.csv')
    if evan_df is None:
        print("Failed to load Evan's data")
        return
    evan_df = rename_depth_groups_evan(evan_df)

    # Create list of datasets and their names
    datasets = [sid_df, evan_df]
    dataset_names = ["Sid's Data", "Evan's Data"]

    # Plot comparison for temperature
    plot_time_series(datasets, 'secchi', dataset_names, output_dir='comparison_plots')

if __name__ == "__main__":
    main()
