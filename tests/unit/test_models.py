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

@pytest.mark.slow
def test_xgboost_training_pipeline_is_stable():
    """
    Test 3: Does the ML model actually train and evaluate without crashing?
    Because model training is extremely CPU intensive, we mark this test as "slow".
    You can run `pytest -m "not slow"` to perfectly skip this test locally,
    but your CI/CD server will still run it.
    """
    forecaster = XGBoostForecaster(n_estimators=10, max_depth=3)
    forecaster.load_data()
    mae = forecaster.train_and_evaluate(model_name="Pytest Automated Test")
    
    assert mae > 0  # Mean Absolute Error must be a strictly positive number