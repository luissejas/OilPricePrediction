import pytest
from src.models.xgboost_model import XGBoostForecaster

def test_xgboost_initialization():
    """
    Test 1: Does the model class build successfully with custom parameters?
    This ensures our hyperparameter tuning pipeline won't crash when passing new variables.
    """
    # Setup our test parameters
    test_trees = 55
    test_depth = 8
    
    # Initialize the model
    forecaster = XGBoostForecaster(n_estimators=test_trees, max_depth=test_depth)
    
    # Assertions (If any of these are False, the test fails)
    assert forecaster is not None
    assert forecaster.model.n_estimators == test_trees
    assert forecaster.model.max_depth == test_depth

def test_default_paths():
    """
    Test 2: Does the model default to the correct data path if none is provided?
    """
    forecaster = XGBoostForecaster()
    assert forecaster.data_path == "data/processed_oil_data.csv"