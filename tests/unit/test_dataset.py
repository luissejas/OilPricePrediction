import pytest
import pandas as pd
import torch
from src.data.dataset import OilDataset

@pytest.fixture
def dummy_dataset():
    """Generates a perfectly predictable Pandas DataFrame offline."""
    df = pd.DataFrame({
        'price_lag_1': [10.0, 11.0, 12.0],
        'sma_7': [50.5, 51.5, 52.5],
        'price': [100.0, 105.0, 110.0]  # This is the target (y)
    })
    # The Dataset extracts features (lags, sma) and target (price)
    return OilDataset(df, target_col='price')

def test_dataset_length(dummy_dataset):
    """Proves the __len__ method correctly counts the Pandas rows."""
    # We fed it 3 rows. Does the warehouse manager know there are exactly 3 boxes?
    assert len(dummy_dataset) == 3

def test_dataset_tensor_conversion(dummy_dataset):
    """Proves the __getitem__ method strictly outputs PyTorch float32 tensors."""
    # Act: Request the very first row (PyTorch index 0)
    x, y = dummy_dataset[0]
    
    # Assert 1: Did it actually build PyTorch tensors?
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    
    # Assert 2: Are they the mathematically required 32-bit floats for Neural Networks?
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    
    # Assert 3: Did it accurately split the features from the target value?
    # x (features) should have 2 columns: [10.0, 50.5]
    assert x.shape == (2,) 
    assert x[0].item() == 10.0
    
    # y (target 'price') should be [100.0]
    assert y.shape == (1,)
    assert y.item() == 100.0
