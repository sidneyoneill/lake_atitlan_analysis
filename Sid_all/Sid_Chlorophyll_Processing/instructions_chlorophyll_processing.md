# Workflow Instructions for Chlorophyll Data Processing

This Markdown file provides detailed step-by-step instructions for implementing the chlorophyll data processing workflow in Python. Follow the steps in order, using the prompts to guide your implementation.

---

## Step 1: Create Grid over Regularly Spaced Values

1. Define a consistent depth grid with fine intervals for the photic zone (e.g. every 1m for 0–30 m) and coarser intervals for deeper layers (e.g., every 10 for 30–250 m).
2. Combine the photic zone and deeper zone grids into a single list of depth values.
3. Ensure this grid will be used as a reference for assigning or interpolating depth measurements.

---

## Step 2: Assign Measurements to Closest Regularly Spaced Value

1. Map each existing depth measurement in your dataset to the closest depth in the predefined grid.
2. For depths outside the range of the grid, assign a `NaN` value.
3. Add a new column to your dataset for the `Assigned_Depth` values.

---

## Step 3: Vertically Interpolate Missing Depths

1. Reshape the dataset into a pivot table where rows represent `Assigned_Depth`, columns represent `Date`, and values represent `Chlorophyll` measurements.
2. Use vertical interpolation, using linear interpolation, to fill missing chlorophyll values at depths within the same date.
3. Clip interpolated values to ensure no negative chlorophyll values.
4. Reshape the pivot table back into a long format.

---

## Step 4: Horizontally Impute Over Time Using KNN

1. Reshape the aggregated data so rows represent `Date` and columns represent `Depth_Group`.
2. Initialize a K-Nearest Neighbors (KNN) imputer, specifying parameters like the number of neighbors (`n_neighbors`) and distance weighting (`weights`).
3. Fit and apply the imputer to fill missing values across dates for each depth group.
4. Reshape the imputed data back into a long format for further analysis.

---

## Step 5: Collapse into Depth Groups

1. Define depth groups based on ecological relevance, such as:
   - Surface: 0–10 m
   - Mid-Depth: 10–20 m
   - Lower Photic: 20–30 m
   - Deep: >30 m
2. Assign each `Assigned_Depth` value to a corresponding depth group.
3. Aggregate chlorophyll measurements by depth group and date, calculating:
   - Mean chlorophyll concentration
   - Standard deviation
   - Count of valid measurements

---

## Validation and Final Steps

1. Inspect the data to ensure that all missing values have been handled appropriately.
2. Visualize the processed chlorophyll concentrations:
   - Create heatmaps of chlorophyll values over depths and dates.
   - Plot time series of chlorophyll levels for each depth group.
3. Save the processed dataset for downstream analysis.

---

Follow these steps sequentially, and prompt your code editor to implement each step one at a time. For further clarification or troubleshooting, refer to this guide as needed.
