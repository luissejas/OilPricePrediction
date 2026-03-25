import pytest
from src.models.xgboost_model import XGBoostForecaster

@pytest.mark.parametrize("trees, depth", [
    (55, 8),    # The original test case
    (100, 3),   # A shallow learning curve
    (500, 1),   # An extreme gradient boosting curve
    (10, 10)    # Minimal trees, extreme depth
])
def test_xgboost_initialization(trees, depth):
    """
    Test 1: Does the model class build successfully with custom parameters?
    Pytest will treat the single function below as 4 completely separate tests using the matrix above!
    """
    forecaster = XGBoostForecaster(n_estimators=trees, max_depth=depth)
    
    assert forecaster is not None
    assert forecaster.model.n_estimators == trees
    assert forecaster.model.max_depth == depth

def test_default_paths():
    """
    Test 2: Does the model default to the correct data path if none is provided?
    """
    forecaster = XGBoostForecaster()
    assert forecaster.data_path == "data/processed_oil_data.csv"