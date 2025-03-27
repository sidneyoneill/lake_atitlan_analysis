import pandas as pd

from process_functions import compare_feature_assigned_matrices



def test_compare_matrices():
    """Test the matrix comparison functionality"""
    
    location = 'WG'
    features = ['temp', 'chlorophyll_a']
    
    # Load initial matrices
    temp_df = pd.read_csv(f'output/{location}/temp/assigned_matrix.csv', index_col=0)
    chl_df = pd.read_csv(f'output/{location}/chlorophyll_a/assigned_matrix.csv', index_col=0)
    
    # Store initial shapes
    initial_temp_shape = temp_df.shape
    initial_chl_shape = chl_df.shape
    
    # Run comparison
    result = compare_feature_assigned_matrices(location, features)
    
    # Load updated matrices
    updated_temp = pd.read_csv(f'output/{location}/temp/assigned_matrix.csv', index_col=0)
    updated_chl = pd.read_csv(f'output/{location}/chlorophyll_a/assigned_matrix.csv', index_col=0)
    
    # Verify shapes match after update
    assert updated_temp.shape == updated_chl.shape, f"Shapes still don't match: temp {updated_temp.shape} vs chl {updated_chl.shape}"
    
    # Verify all columns exist in both matrices
    assert set(updated_temp.columns) == set(updated_chl.columns), "Column names don't match between matrices"
    
    # Verify all rows exist in both matrices
    assert set(updated_temp.index) == set(updated_chl.index), "Row indices don't match between matrices"
    
    # Verify the result dictionary contains the correct information
    assert result['matched'] == True, "Result indicates matrices are not matched"
    assert result['Temperature']['shape'] == updated_temp.shape
    assert result['Chlorophyll-a']['shape'] == updated_chl.shape
    
    # Clean up test files
    import shutil
    shutil.rmtree(f'output/{location}')

if __name__ == '__main__':
    test_compare_matrices()