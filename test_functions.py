import pytest
from functions import load_diabetes_df, split_dataframe, train_linear_model, data_integrity_check

# Test load_diabetes_df function
def test_load_diabetes_df():
    diabetes_df = load_diabetes_df()
    assert diabetes_df is not None
    #assert 'data' in diabetes_df
    assert 'target' in diabetes_df
    assert len(diabetes_df) > 0
    # Add more specific assertions as needed

# Test split_dataframe function
def test_split_dataframe():
    train_df, test_df = split_dataframe()
    assert train_df is not None
    assert test_df is not None
    assert len(train_df) > 0
    assert len(test_df) > 0
    # Add more specific assertions as needed

# Test train_linear_model function
def test_train_linear_model():
    train_df, test_df = split_dataframe()
    result = train_linear_model(train_df, test_df)
    assert 'Train-score' in result
    assert 'Test-score' in result
    assert result['Train-score'] >= 0  # R-squared should be non-negative
    assert result['Test-score'] >= 0
    # Add more specific assertions as needed

# Test data_integrity_check function
def test_data_integrity_check():
    data_integrity_check()
    # You can add assertions for the result of data integrity check, e.g., check for specific issues

# Additional custom test cases

# Test if the saved model file exists
def test_saved_model_file_exists():
    import os
    assert os.path.isfile('trained_diabetes_linear_model.joblib')

# Test if the HTML report for data integrity check exists
def test_data_integrity_report_exists():
    import os
    assert os.path.isfile('data_integrity_report.html')

# Test if data integrity check reports any issues
def test_data_integrity_report_issues():
    # Implement assertions to check for specific issues reported in the data integrity report
    pass

# You can add more custom test cases as needed
