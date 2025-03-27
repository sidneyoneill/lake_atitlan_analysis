import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from pathlib import Path
import matplotlib.pyplot as plt

def load_matrices(location, feature):
    """Load the removed and imputed matrices."""
    base_path = f'output/{location}/{feature}'
    removed_path = os.path.join(base_path, 'removed_matrix.csv')
    imputed_path = os.path.join(base_path, 'imputed_matrix.csv')
    
    removed_matrix = pd.read_csv(removed_path, index_col=0)
    imputed_matrix = pd.read_csv(imputed_path, index_col=0)
    
    return removed_matrix, imputed_matrix

def calculate_metrics(original_values, predicted_values):
    """Calculate comparison metrics between original and predicted values."""
    # Remove any NaN values
    mask = ~(np.isnan(original_values) | np.isnan(predicted_values))
    original_values = original_values[mask]
    predicted_values = predicted_values[mask]
    
    if len(original_values) == 0:
        return {
            'RMSE': np.nan,
            'MAE': np.nan,
            'R²': np.nan,
            'n_points': 0
        }
    
    rmse = np.sqrt(mean_squared_error(original_values, predicted_values))
    mae = mean_absolute_error(original_values, predicted_values)
    r2 = r2_score(original_values, predicted_values)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'n_points': len(original_values)
    }

def compare_matrices(removed_matrix, imputed_matrix):
    """Compare values between matrices where removed_matrix has non-empty values."""
    # Convert to numpy arrays for easier handling
    removed_array = removed_matrix.to_numpy()
    imputed_array = imputed_matrix.to_numpy()
    
    # Get indices where removed_matrix has values
    valid_indices = ~np.isnan(removed_array)
    
    # Extract values for comparison
    original_values = removed_array[valid_indices]
    predicted_values = imputed_array[valid_indices]
    
    # Calculate metrics
    metrics = calculate_metrics(original_values, predicted_values)
    
    return metrics, original_values, predicted_values

def plot_validation_results(original_values, predicted_values, feature, location, metrics):
    """Create scatter plot of original vs predicted values."""
    plt.figure(figsize=(10, 8))
    plt.scatter(original_values, predicted_values, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(original_values), min(predicted_values))
    max_val = max(max(original_values), max(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    plt.xlabel('Original Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{feature} Validation: Original vs Predicted Values\nLocation: {location}')
    plt.legend()
    
    # Add metrics to plot
    metrics_text = f'R² = {metrics["R²"]:.3f}\nRMSE = {metrics["RMSE"]:.3f}\nMAE = {metrics["MAE"]:.3f}\nn = {metrics["n_points"]}'
    plt.text(0.05, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    # Save plot
    output_path = f'output/{location}/{feature}/validation_plot.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def calculate_averages(all_metrics):
    """Calculate various average metrics."""
    df = pd.DataFrame(all_metrics).T
    df = df.apply(pd.json_normalize)
    
    # Calculate averages per feature
    feature_averages = df.groupby('feature').agg({
        'RMSE': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'R²': ['mean', 'std'],
        'n_points': 'sum'
    })
    
    # Calculate averages per location
    location_averages = df.groupby('location').agg({
        'RMSE': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'R²': ['mean', 'std'],
        'n_points': 'sum'
    })
    
    # Calculate overall averages
    overall_averages = {
        'RMSE': {'mean': df['RMSE'].mean(), 'std': df['RMSE'].std()},
        'MAE': {'mean': df['MAE'].mean(), 'std': df['MAE'].std()},
        'R²': {'mean': df['R²'].mean(), 'std': df['R²'].std()},
        'n_points': df['n_points'].sum()
    }
    
    return feature_averages, location_averages, overall_averages

def main(features=None):
    """Main function to run the comparison for all locations and features."""
    if features is None:
        features = ["temp"]  # Default feature if none specified
    
    locations = ["SA", "WG", "WP"]
    
    # Store all results for comparison
    all_metrics = {}
    
    for location in locations:
        for feature in features:
            try:
                # Create unique key for this combination
                key = f"{location}_{feature}"
                
                # Load matrices
                removed_matrix, imputed_matrix = load_matrices(location, feature)
                
                # Compare matrices
                metrics, original_values, predicted_values = compare_matrices(removed_matrix, imputed_matrix)
                
                # Add location and feature to metrics
                metrics['location'] = location
                metrics['feature'] = feature
                
                # Store metrics
                all_metrics[key] = metrics
                
                # Create validation plot
                plot_validation_results(original_values, predicted_values, feature, location, metrics)
                
                # Print individual results
                print(f"\nComparison Results for {location} - {feature}:")
                print("-" * 50)
                for metric_name, value in metrics.items():
                    if metric_name not in ['location', 'feature']:
                        print(f"{metric_name}: {value:.4f}" if isinstance(value, float) else f"{metric_name}: {value}")
                
                # Save individual results
                results_dir = f'output/{location}/{feature}'
                os.makedirs(results_dir, exist_ok=True)
                
                # Save metrics to CSV
                pd.DataFrame([metrics]).to_csv(
                    os.path.join(results_dir, 'comparison_metrics.csv'),
                    index=False
                )
                
                # Save original vs predicted values
                comparison_df = pd.DataFrame({
                    'Original': original_values,
                    'Predicted': predicted_values
                })
                comparison_df.to_csv(
                    os.path.join(results_dir, 'value_comparisons.csv'),
                    index=False
                )
                
            except Exception as e:
                print(f"Error processing {location} - {feature}: {str(e)}")
    
    # Calculate averages
    feature_averages, location_averages, overall_averages = calculate_averages(all_metrics)
    
    # Save all results
    output_dir = 'output/validation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save feature averages
    feature_averages.to_csv(os.path.join(output_dir, 'feature_averages.csv'))
    print("\nFeature Averages:")
    print(feature_averages)
    
    # Save location averages
    location_averages.to_csv(os.path.join(output_dir, 'location_averages.csv'))
    print("\nLocation Averages:")
    print(location_averages)
    
    # Save overall averages
    pd.DataFrame(overall_averages).to_csv(os.path.join(output_dir, 'overall_averages.csv'))
    print("\nOverall Averages:")
    print("RMSE: {:.4f} ± {:.4f}".format(overall_averages['RMSE']['mean'], overall_averages['RMSE']['std']))
    print("MAE: {:.4f} ± {:.4f}".format(overall_averages['MAE']['mean'], overall_averages['MAE']['std']))
    print("R²: {:.4f} ± {:.4f}".format(overall_averages['R²']['mean'], overall_averages['R²']['std']))
    print(f"Total validation points: {overall_averages['n_points']}")
    
    # Save combined metrics for all locations and features
    pd.DataFrame(all_metrics).T.to_csv(os.path.join(output_dir, 'all_metrics.csv'))
    
    return all_metrics, feature_averages, location_averages, overall_averages

if __name__ == "__main__":
    # Example usage with multiple features
    features = ["temp", "chlorophyll_a", "nitrate", "phosphate"]  # Add or modify features as needed
    main(features)
