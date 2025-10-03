"""Tests for data fetching module."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import DataFetcher, DataPreprocessor
from config import load_config


@pytest.fixture
def config():
    """Create test configuration."""
    return {
        'data': {
            'fred_api_key_env': 'FRED_API_KEY',
            'start_date': '2020-01-01',
            'end_date': '2021-12-31',
            'futures_ticker': 'ES=F',
            'cache_dir': 'test_cache'
        },
        'garch': {
            'p': 1,
            'q': 1,
            'min_window': 60,
            'refit_frequency': 12,
            'max_history': 120,
            'scaling_factor': 100
        }
    }


@pytest.fixture
def sample_permits_data():
    """Create sample building permits data."""
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='M')
    values = np.random.uniform(1000, 2000, len(dates))
    return pd.DataFrame({'permits': values}, index=dates)


@pytest.fixture
def sample_unemployment_data():
    """Create sample unemployment data."""
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='M')
    values = np.random.uniform(3, 8, len(dates))
    return pd.DataFrame({'unemployment': values}, index=dates)


@pytest.fixture
def sample_futures_data():
    """Create sample futures data."""
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
    n = len(dates)
    data = {
        'open': np.random.uniform(4000, 4500, n),
        'high': np.random.uniform(4100, 4600, n),
        'low': np.random.uniform(3900, 4400, n),
        'close': np.random.uniform(4000, 4500, n),
        'volume': np.random.uniform(1000000, 2000000, n)
    }
    return pd.DataFrame(data, index=dates)


class TestDataFetcher:
    """Test DataFetcher class."""

    @patch.dict('os.environ', {'FRED_API_KEY': 'test_key'})
    def test_init(self, config):
        """Test DataFetcher initialization."""
        fetcher = DataFetcher(config)
        assert fetcher.config == config
        assert fetcher.cache_dir.exists()

    @patch.dict('os.environ', {})
    def test_init_no_api_key(self, config):
        """Test initialization without API key."""
        with pytest.raises(ValueError, match="FRED API key not found"):
            DataFetcher(config)

    def test_cache_key_generation(self, config):
        """Test cache key generation."""
        with patch.dict('os.environ', {'FRED_API_KEY': 'test_key'}):
            fetcher = DataFetcher(config)
            key1 = fetcher._get_cache_key(series='PERMIT', start='2020-01-01')
            key2 = fetcher._get_cache_key(series='PERMIT', start='2020-01-01')
            key3 = fetcher._get_cache_key(series='UNRATE', start='2020-01-01')

            assert key1 == key2  # Same parameters
            assert key1 != key3  # Different parameters

    @patch('data.fetchers.Fred')
    def test_get_building_permits(self, mock_fred_class, config, sample_permits_data):
        """Test fetching building permits data."""
        # Mock FRED API
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = sample_permits_data['permits']
        mock_fred_class.return_value = mock_fred

        with patch.dict('os.environ', {'FRED_API_KEY': 'test_key'}):
            fetcher = DataFetcher(config)
            result = fetcher.get_building_permits('2020-01-01', '2021-12-31')

            assert 'permits' in result.columns
            assert len(result) == len(sample_permits_data)
            assert isinstance(result.index, pd.DatetimeIndex)

    @patch('data.fetchers.Fred')
    def test_get_unemployment(self, mock_fred_class, config, sample_unemployment_data):
        """Test fetching unemployment data."""
        mock_fred = MagicMock()
        mock_fred.get_series.return_value = sample_unemployment_data['unemployment']
        mock_fred_class.return_value = mock_fred

        with patch.dict('os.environ', {'FRED_API_KEY': 'test_key'}):
            fetcher = DataFetcher(config)
            result = fetcher.get_unemployment('2020-01-01', '2021-12-31')

            assert 'unemployment' in result.columns
            assert len(result) == len(sample_unemployment_data)

    @patch('yfinance.Ticker')
    def test_get_futures_data(self, mock_ticker_class, config, sample_futures_data):
        """Test fetching futures data."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_futures_data
        mock_ticker_class.return_value = mock_ticker

        with patch.dict('os.environ', {'FRED_API_KEY': 'test_key'}):
            fetcher = DataFetcher(config)
            result = fetcher.get_futures_data('ES=F', '2020-01-01', '2021-12-31')

            assert 'close' in result.columns
            assert len(result) > 0

    def test_compute_permit_growth(self, config, sample_permits_data):
        """Test permit growth calculation."""
        with patch.dict('os.environ', {'FRED_API_KEY': 'test_key'}):
            fetcher = DataFetcher(config)
            result = fetcher.compute_permit_growth(sample_permits_data)

            assert 'permit_growth' in result.columns
            # First value should be 0 (filled NaN)
            assert result['permit_growth'].iloc[0] == 0
            # Should have same length as input
            assert len(result) == len(sample_permits_data)

    def test_align_data(self, config, sample_permits_data, sample_futures_data):
        """Test data alignment."""
        with patch.dict('os.environ', {'FRED_API_KEY': 'test_key'}):
            fetcher = DataFetcher(config)

            # Add signal column to monthly data
            sample_permits_data['signal'] = np.random.choice([-1, 0, 1], len(sample_permits_data))

            result = fetcher.align_data(sample_permits_data, sample_futures_data)

            assert 'close' in result.columns
            assert 'signal' in result.columns
            assert 'signal_change' in result.columns
            # Result should be daily frequency
            assert len(result) <= len(sample_futures_data)


class TestDataPreprocessor:
    """Test DataPreprocessor class."""

    def test_validate_data_valid(self, config):
        """Test data validation with valid data."""
        preprocessor = DataPreprocessor(config)

        # Create valid DataFrame
        dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
        df = pd.DataFrame({'col1': range(len(dates)), 'col2': range(len(dates))}, index=dates)

        assert preprocessor.validate_data(df, ['col1', 'col2']) == True

    def test_validate_data_missing_columns(self, config):
        """Test validation with missing columns."""
        preprocessor = DataPreprocessor(config)

        dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
        df = pd.DataFrame({'col1': range(len(dates))}, index=dates)

        with pytest.raises(ValueError, match="Missing required columns"):
            preprocessor.validate_data(df, ['col1', 'col2'])

    def test_validate_data_empty(self, config):
        """Test validation with empty DataFrame."""
        preprocessor = DataPreprocessor(config)
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="DataFrame is empty"):
            preprocessor.validate_data(df, [])

    def test_handle_outliers(self, config):
        """Test outlier handling."""
        preprocessor = DataPreprocessor(config)

        # Create data with outliers
        data = pd.Series(np.concatenate([
            np.random.normal(0, 1, 100),
            [100, -100]  # Outliers
        ]))

        result = preprocessor.handle_outliers(data, n_std=3)

        # Outliers should be clipped
        assert result.max() < 100
        assert result.min() > -100

    def test_calculate_returns_simple(self, config):
        """Test simple returns calculation."""
        preprocessor = DataPreprocessor(config)

        prices = pd.Series([100, 110, 121, 115])
        returns = preprocessor.calculate_returns(prices, method='simple')

        expected = pd.Series([np.nan, 0.1, 0.1, -0.0495867769])  # Approximate values
        pd.testing.assert_series_equal(returns, prices.pct_change(), check_names=False)

    def test_calculate_returns_log(self, config):
        """Test log returns calculation."""
        preprocessor = DataPreprocessor(config)

        prices = pd.Series([100, 110, 121, 115])
        returns = preprocessor.calculate_returns(prices, method='log')

        # Check first return is NaN
        assert pd.isna(returns.iloc[0])
        # Check other returns are calculated correctly
        assert abs(returns.iloc[1] - np.log(110/100)) < 0.0001

    def test_calculate_realized_volatility(self, config):
        """Test realized volatility calculation."""
        preprocessor = DataPreprocessor(config)

        # Create returns series
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        returns = pd.Series(np.random.normal(0, 0.01, len(dates)), index=dates)

        vol = preprocessor.calculate_realized_volatility(returns, window=60, annualize=True)

        # Should have NaN for first window-1 values
        assert vol.iloc[:59].isna().all()
        # Should have values after window
        assert not vol.iloc[60:].isna().any()
        # Annualized vol should be higher than daily
        assert vol.iloc[60:].mean() > returns.std()

    def test_merge_macro_data(self, config, sample_permits_data, sample_unemployment_data):
        """Test merging macroeconomic data."""
        preprocessor = DataPreprocessor(config)

        result = preprocessor.merge_macro_data(sample_permits_data, sample_unemployment_data)

        assert 'permits' in result.columns
        assert 'unemployment' in result.columns
        # Should handle any misalignment
        assert not result[['permits', 'unemployment']].isna().any().any()

    def test_split_train_test(self, config):
        """Test train/test split."""
        preprocessor = DataPreprocessor(config)

        dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
        df = pd.DataFrame({'value': range(len(dates))}, index=dates)

        train, test = preprocessor.split_train_test(df, '2021-01-01')

        assert len(train) > 0
        assert len(test) > 0
        assert train.index[-1] < test.index[0]
        assert test.index[0] >= pd.to_datetime('2021-01-01')