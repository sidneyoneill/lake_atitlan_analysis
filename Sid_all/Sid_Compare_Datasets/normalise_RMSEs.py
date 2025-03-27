import numpy as np
import pandas as pd

# Add measurement counts
measurement_counts = {
    "temp": 61633,
    "chlorophyll_a": 55428,
    "ph": 61613,
    "dissolved_oxygen": 58333,
    "total_dissolved_solids": 44561,
    "biochemical_oxygen_demand": 61633,
    "secchi": 61633,
    "turbidity": 35387,
    "nitrate": 3447,
    "phosphate": 4345,
    "ammonium": 1864,
    "phosphorus": 3242

}

def normalize_rmse(rmse_values, std_values, feature_means, measurement_counts):
    """
    Normalize RMSE values and their standard deviations using mean normalization.
    Results are weighted by measurement counts.
    """
    # Normalize RMSE values by mean
    normalized_rmses = {feature: rmse / feature_means[feature] for feature, rmse in rmse_values.items()}
    
    # Normalize standard deviations by the same means
    normalized_stds = {feature: std_values[feature] / feature_means[feature] for feature in rmse_values.keys()}
    
    # Calculate weights using measurement counts
    total_measurements = sum(measurement_counts.values())
    count_weights = {feature: count/total_measurements for feature, count in measurement_counts.items()}
    
    # Weight both normalized RMSEs and STDs by measurement counts
    count_weighted_rmses = {feature: normalized_rmses[feature] * count_weights[feature] for feature in normalized_rmses.keys()}
    count_weighted_stds = {feature: normalized_stds[feature] * count_weights[feature] for feature in normalized_stds.keys()}
    
    # Calculate weighted averages
    total_weight = sum(count_weights.values())  # Should be 1.0, but include for correctness
    avg_count_weighted_rmse = sum(count_weighted_rmses.values()) / total_weight
    avg_count_weighted_std = sum(count_weighted_stds.values()) / total_weight
    
    return normalized_rmses, normalized_stds, count_weighted_rmses, count_weighted_stds, avg_count_weighted_rmse, avg_count_weighted_std

# RMSE values for features
rmse_mean_values = {
    "temp": 0.0962,
    "chlorophyll_a": 0.6894,
    "ph": 0.0836,
    "dissolved_oxygen": 0.6100,
    "biochemical_oxygen_demand": 1.1954,
    "secchi": 1.9758,
    "turbidity": 0.4293,
    "nitrate": 48.2512,
    "phosphate": 20.0787,
    "ammonium": 12.6111,
    "phosphorus": 15.7372
}

# Update standard deviation values
std_values = {
    "temp": 0.0193,
    "chlorophyll_a": 0.4034,
    "ph": 0.0099,
    "dissolved_oxygen": 0.0415,
    "biochemical_oxygen_demand": 0.0094,
    "secchi": 0.4383,
    "turbidity": 0.3227,
    "nitrate": 21.8166,
    "phosphate": 8.1267,
    "ammonium": 3.9852,
    "phosphorus": 26.7736
}

# Feature means and ranges for SID
sid_means = {
    "temp": 21.83, "chlorophyll_a": 2.07, "ph": 8.22, "dissolved_oxygen": 5.47,
    "biochemical_oxygen_demand": 2.31, "secchi": 6.47, "turbidity": 0.53,
    "nitrate": 63.91, "phosphate": 37.94, "ammonium": 13.95, "phosphorus": 27.41
}

# Normalize for SID
sid_norm_mean, sid_norm_stds, sid_weighted_rmses, sid_weighted_stds, sid_avg_norm_mean, sid_avg_norm_std = normalize_rmse(
    rmse_mean_values, std_values, sid_means, measurement_counts)


# Display results
print("\n=== SID Normalization Results ===")
print("\nFeature-wise values:")
for feature in sid_norm_mean.keys():
    print(f"\n{feature.upper()}:")
    print(f"  Raw RMSE: {rmse_mean_values[feature]:.4f} ± {std_values[feature]:.4f}")
    print(f"  RMSE/Mean: {sid_norm_mean[feature]:.4f} ± {sid_norm_stds[feature]:.4f}")
    print(f"  Count-Weighted RMSE/Mean: {sid_weighted_rmses[feature]:.4f} ± {sid_weighted_stds[feature]:.4f}")
    print(f"  Measurement Count: {measurement_counts[feature]}")

print("\nAverages:")
print(f"Raw Average RMSE: {np.mean(list(rmse_mean_values.values())):.4f} ± {np.mean(list(std_values.values())):.4f}")
print(f"Average RMSE/Mean: {np.mean(list(sid_norm_mean.values())):.4f} ± {np.mean(list(sid_norm_stds.values())):.4f}")
print(f"Count-Weighted Average RMSE/Mean: {sid_avg_norm_mean:.4f} ± {sid_avg_norm_std:.4f}")

# Add combined result using mean RMSE/Mean and count-weighted std
avg_count_weighted_std = sum(sid_weighted_stds.values()) / len(sid_weighted_stds)
print(f"Combined Result (Avg RMSE/Mean ± Avg Count-Weighted STD): {np.mean(list(sid_norm_mean.values())):.4f} ± {avg_count_weighted_std:.4f}")
