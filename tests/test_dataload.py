import pytest
from src.data.data_loader import OilDataLoader

def test_data_initialization():
    """
    Ensures the loader defaults to the correct WTI Crude ticker.
    """
    loader = OilDataLoader()
    assert loader.ticker == "CL=F"
    assert loader.data is None

def test_data_fetching():
    """
    Verifies we can hit the Yahoo Finance API and get a usable DataFrame back.
    """
    loader = OilDataLoader()
    df = loader.fetch_data()
    
    # The dataframe should not be empty
    assert not df.empty
    # The column must be flattened and renamed to 'price'
    assert 'price' in df.columns

def test_feature_engineering_quality():
    """
    Ensures our rolling averages and lags are calculated correctly 
    and no missing values (NaNs) are passed to the model.
    """
    loader = OilDataLoader()
    loader.fetch_data()
    df = loader.engineer_features()
    
    expected_columns = [
        'price', 'price_lag_1', 'price_lag_3', 
        'sma_7', 'sma_14', 'volatility_7'
    ]
    
    # 1. Check if all required columns exist
    for col in expected_columns:
        assert col in df.columns
        
    # 2. Check for Data Leakage/Integrity (No NaNs allowed)
    # The first 14 days should have been dropped by the moving average logic
    assert df.isnull().sum().sum() == 0
    
    # 3. Ensure we have enough data to actually train a model
    assert len(df) > 100