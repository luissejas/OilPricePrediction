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

import torch
import torch.nn as nn
from src.models.pytorch_model import OilPriceNN
from src.models.train import train_model

def test_pytorch_architecture_forward_pass():
    """
    Test 4: Does the Neural Network execute matrix math correctly on a standard Tensor block?
    """
    model = OilPriceNN(input_size=5)
    # Arrange: Create exactly 32 rows of fake numerical data (5 features each)
    fake_input = torch.randn(32, 5)
    
    # Act: Thrust the data completely through the Hidden Layers
    predictions = model(fake_input)
    
    # Assert: We gave you 32 rows. Did you safely spit out 32 specific price expectations?
    assert predictions.shape == (32, 1)

def test_pytorch_architecture_backward_pass():
    """
    Test 5: The Single Most Critical Deep Learning Test.
    Does the algorithm successfully calculate mathematical 'Gradients' (blame) for every single layer?
    If not, the dials are frozen, and the model is physically incapable of becoming smarter.
    """
    model = OilPriceNN(input_size=5)
    fake_input = torch.randn(32, 5)
    fake_targets = torch.randn(32, 1) # Fake actual prices
    
    predictions = model(fake_input)
    
    # Calculate the Boss's Anger Error
    criterion = nn.MSELoss()
    loss = criterion(predictions, fake_targets)
    
    # Act: True Backpropagation (Walk backward through the factory)
    loss.backward()
    
    # Assert: Interrogate exactly every single Weight dial in the network
    for param_name, parameter in model.named_parameters():
        # If the parameter.grad == None, the post-it note failed to attach!
        assert parameter.grad is not None, f"FATAL AI BUG: Layer '{param_name}' is frozen and dodged the math!"

@pytest.mark.slow
def test_pytorch_training_pipeline_is_stable():
    """
    Test 6: Does the absolute Full Integration Training Loop successfully operate?
    We run it for exactly 2 epochs to chemically prove stability without wasting 10 minutes of server compute limit.
    """
    try:
        # Ignite the entire architecture locally using your laptop's CPU
        train_model(epochs=2, batch_size=32)
        success = True
    except Exception as e:
        success = False
        print(f"Pipeline exploded with error: {e}")
        
    assert success is True