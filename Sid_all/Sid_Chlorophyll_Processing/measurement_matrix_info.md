# Chlorophyll Measurement Matrix Explanation

## Overview
The measurement matrix represents chlorophyll-a measurements across different depths and dates, indicating both measurement presence and sampling existence. It is saved as 'measurement_matrix.csv' in the output directory.

## Structure
- **Rows**: Represent different depths (in meters)
- **Columns**: Represent different dates
- **Values**: Three-state indicators
  - **Val**: Represents the chlorophyll measurement amount for that depth-date combination.
  - **0**: Indicates a sample exists but no chlorophyll measurement was taken.
  - **N/A**: Indicates no sample was taken at that depth-date combination.

## Example

date, 2023-01-01, 2023-01-02, 2023-01-03
depth
0m, 2.5, 0, 3.1
5m, 1.8, 4.0, N/A
10m, N/A, 5.2, 6.7

In this example:
- On January 1st:
  - Measurements of 2.5 and 1.8 were taken at 0m and 5m respectively.
  - No sample was collected at 10m (N/A).
- On January 2nd:
  - Measurements of 4.0 and 5.2 were taken at 5m and 10m respectively.
  - A sample exists at 0m but no chlorophyll was measured (0).
- On January 3rd:
  - Measurements of 3.1 and 6.7 were taken at 0m and 10m respectively.
  - No sample was collected at 5m (N/A).

## Implementation Details
The matrix is created using a combination of pivot table operations that:
1. Creates an existence matrix showing where samples exist.
2. Creates a measurement matrix showing where chlorophyll was measured and the measured values.
3. Combines them to create a final matrix distinguishing between:
   - Measured values (chlorophyll amount).
   - Unmeasured but sampled points (0).
   - Unsampled points (N/A).

## Usage
This matrix is useful for:
- Visualizing the temporal and spatial distribution of measurements.
- Identifying patterns in sampling frequency.
- Understanding sampling gaps versus measurement gaps.
- Planning data collection strategies.
- Preparing data for imputation of missing values.

## Summary Statistics
The script provides summary statistics including:
- Total number of dates.
- Total number of depths.
- Total possible measurements (depths × dates).
- Actual number of measurements taken.

These statistics help understand the coverage and sparsity of the measurements across the sampling period and depths, distinguishing between unsampled points and unmeasured samples.
