import pytest
import pandas as pd
from sklearn.datasets import load_iris

def test_iris_data_shape():
    """Test that IRIS dataset has expected shape."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    assert df.shape[0] == 150, "Dataset should have 150 samples"
    assert df.shape[1] == 5, "Dataset should have 5 columns (4 features + 1 target)"

def test_iris_data_columns():
    """Test that IRIS dataset has expected columns."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    expected_columns = ['sepal length (cm)', 'sepal width (cm)', 
                       'petal length (cm)', 'petal width (cm)', 'target']
    
    assert list(df.columns) == expected_columns, f"Columns mismatch. Expected {expected_columns}"

def test_iris_target_values():
    """Test that target values are within expected range."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    assert df['target'].min() == 0, "Minimum target value should be 0"
    assert df['target'].max() == 2, "Maximum target value should be 2"
    assert set(df['target'].unique()) == {0, 1, 2}, "Target should only contain values 0, 1, 2"

def test_iris_no_missing_values():
    """Test that dataset has no missing values."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    assert df.isnull().sum().sum() == 0, "Dataset should not have any missing values"

def test_iris_feature_ranges():
    """Test that feature values are within reasonable ranges."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    # All feature values should be positive
    assert (df.drop(columns=['target']) >= 0).all().all(), "All feature values should be non-negative"
