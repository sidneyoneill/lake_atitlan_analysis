import pandas as pd
import numpy as np

def replace_specific_outliers(data_path, anomalies_path, output_path):
    """
    Replace specific outliers listed in anomalies file with mean of adjacent time values.
    
    Args:
        data_path: Path to input CSV file
        anomalies_path: Path to anomalies list CSV
        output_path: Path to save processed CSV file
    """
    # Read the CSV files
    df = pd.read_csv(data_path)
    anomalies_df = pd.read_csv(anomalies_path)
    
    # Convert dates to datetime
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    anomalies_df['date'] = pd.to_datetime(anomalies_df['date'], format='%d/%m/%Y')
    
    # Sort by location, depth_group, and date
    df = df.sort_values(['location', 'depth_group', 'date'])
    
    # Store replacement information
    replacements_info = []
    
    # For each anomaly
    for _, anomaly in anomalies_df.iterrows():
        # Get group data for this location and depth
        group = df[
            (df['location'] == anomaly['location']) & 
            (df['depth_group'] == anomaly['depth_group'])
        ]
        
        # Find the index of the anomaly in the original dataframe
        anomaly_mask = (
            (df['location'] == anomaly['location']) &
            (df['depth_group'] == anomaly['depth_group']) &
            (df['date'] == anomaly['date'])
        )
        
        if not anomaly_mask.any():
            print(f"Warning: Anomaly not found in dataset: {anomaly['location']}, {anomaly['depth_group']}, {anomaly['date']}")
            continue
            
        idx = df[anomaly_mask].index[0]
        
        # Get previous and next values for same location and depth
        prev_value = group[group['date'] < anomaly['date']][anomaly['feature']].iloc[-1] if len(group[group['date'] < anomaly['date']]) > 0 else None
        next_value = group[group['date'] > anomaly['date']][anomaly['feature']].iloc[0] if len(group[group['date'] > anomaly['date']]) > 0 else None
        
        # Calculate replacement value
        if prev_value is not None and next_value is not None:
            replacement = (prev_value + next_value) / 2
        elif prev_value is not None:
            replacement = prev_value
        elif next_value is not None:
            replacement = next_value
        else:
            replacement = group[anomaly['feature']].mean()
        
        # Store replacement information
        replacements_info.append({
            'location': anomaly['location'],
            'depth_group': anomaly['depth_group'],
            'date': anomaly['date'],
            'feature': anomaly['feature'],
            'original_value': anomaly['observed_value'],
            'replacement_value': replacement,
            'prev_value': prev_value,
            'next_value': next_value
        })
        
        # Replace the anomaly
        df.loc[idx, anomaly['feature']] = replacement
    
    # Convert date back to original format before saving
    df['date'] = df['date'].dt.strftime('%d/%m/%Y')
    
    # Create replacements DataFrame
    replacements_df = pd.DataFrame(replacements_info)
    
    # Save processed data and replacements list
    df.to_csv(output_path, index=False)
    replacements_df.to_csv('output_v3/replacements_list.csv', index=False)
    
    return df, replacements_df

# Run the outlier replacement
processed_df, replacements_list = replace_specific_outliers(
    'data/SID_LIMNO_processed_v5.csv',
    'data/final_anomalies_list.csv',
    'output_v3/SID_LIMNO_no_outliers.csv'
)
