import pytest
import pandas as pd
from unittest.mock import patch
from src.data.data_loader import OilDataLoader

@pytest.fixture
def mock_yf_data():
    """Provides an isolated, fake DataFrame preventing the need to call Yahoo Finance."""
    dates = pd.date_range(start="2020-01-01", periods=150)
    # A simple linear curve: 0, 1, 2, 3... guarantees math (like moving averages) works predictably!
    df = pd.DataFrame({'Close': range(150)}, index=dates)
    return df


@pytest.fixture
def loader():
    """Provides a fresh instance of OilDataLoader for any test that requests it."""
    return OilDataLoader()

def test_data_initialization(loader):
    """
    Ensures the loader defaults to the correct WTI Crude ticker.
    """
    assert loader.ticker == "CL=F"
    assert loader.data is None

@patch('src.data.data_loader.yf.download')
def test_data_fetching(mock_download, loader, mock_yf_data):
    """
    Verifies we can hit the Yahoo Finance API (mocked) and get a usable DataFrame back.
    """
    # 1. Arrange: Wiretap the function! Force it to return our fake data instead of calling the internet
    mock_download.return_value = mock_yf_data
    
    # 2. Act: Call fetch_data() -> It will unknowingly trigger our wiretap
    df = loader.fetch_data()
    
    # The dataframe should not be empty
    assert not df.empty
    # The column must be flattened and renamed to 'price'
    assert 'price' in df.columns

@patch('src.data.data_loader.yf.download')
def test_feature_engineering_quality(mock_download, loader, mock_yf_data):
    """
    Ensures our rolling averages and lags are calculated correctly 
    and no missing values (NaNs) are passed to the model.
    """
    # Arrange: Setup the wiretap
    mock_download.return_value = mock_yf_data
    
    # Act
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

@patch('src.data.data_loader.yf.download')
def test_fetch_empty_data_raises_error(mock_download, loader):
    """
    Proves that the system gracefully crashes with a ValueError 
    if Yahoo Finance returns an empty DataFrame (e.g., bad ticker).
    """
    # Arrange: Tell the mock to return an empty DataFrame!
    mock_download.return_value = pd.DataFrame()
    
    # Act & Assert: Call fetch_data and trap the expected explosion
    with pytest.raises(ValueError, match="Error: No data fetched"):
        loader.fetch_data()