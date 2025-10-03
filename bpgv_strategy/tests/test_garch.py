"""Tests for GARCH model module."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import GARCHVolatilityForecaster


@pytest.fixture
def config():
    """Create test configuration."""
    return {
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
def sample_returns():
    """Create sample returns data."""
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='M')
    # Create returns with varying volatility
    n = len(dates)
    returns = np.zeros(n)
    for i in range(n):
        if i < n/3:
            returns[i] = np.random.normal(0, 1)  # Low vol period
        elif i < 2*n/3:
            returns[i] = np.random.normal(0, 3)  # High vol period
        else:
            returns[i] = np.random.normal(0, 1.5)  # Medium vol period

    return pd.Series(returns, index=dates)


class TestGARCHVolatilityForecaster:
    """Test GARCH volatility forecasting."""

    def test_init(self, config):
        """Test initialization."""
        forecaster = GARCHVolatilityForecaster(config)
        assert forecaster.garch_params == config['garch']
        assert forecaster.model is None
        assert forecaster.fitted_model is None

    def test_fit_with_sufficient_data(self, config, sample_returns):
        """Test fitting with sufficient data."""
        forecaster = GARCHVolatilityForecaster(config)
        result = forecaster.fit(sample_returns)

        assert result == forecaster  # Should return self
        assert forecaster.fitted_model is not None
        assert forecaster.last_fit_date == sample_returns.index[-1]
        assert len(forecaster.fit_history) == 1

    def test_fit_with_insufficient_data(self, config):
        """Test fitting with insufficient data."""
        forecaster = GARCHVolatilityForecaster(config)
        short_returns = pd.Series([1, 2, 3], index=pd.date_range('2020-01-01', periods=3, freq='M'))

        with pytest.raises(ValueError, match="Insufficient data"):
            forecaster.fit(short_returns)

    def test_forecast_volatility(self, config, sample_returns):
        """Test volatility forecasting."""
        forecaster = GARCHVolatilityForecaster(config)
        forecaster.fit(sample_returns)

        vol_forecast = forecaster.forecast_volatility(horizon=1)

        assert isinstance(vol_forecast, float)
        assert vol_forecast > 0
        assert not np.isnan(vol_forecast)
        assert not np.isinf(vol_forecast)

    def test_forecast_without_fit(self, config):
        """Test forecasting without fitting model first."""
        forecaster = GARCHVolatilityForecaster(config)

        with pytest.raises(ValueError, match="Model must be fitted"):
            forecaster.forecast_volatility()

    def test_rolling_forecast(self, config, sample_returns):
        """Test rolling forecast generation."""
        forecaster = GARCHVolatilityForecaster(config)

        # Use smaller window for testing
        forecasts = forecaster.rolling_forecast(
            sample_returns,
            window=60,
            refit_freq=12,
            min_train_size=60
        )

        assert isinstance(forecasts, pd.DataFrame)
        assert 'volatility_forecast' in forecasts.columns
        assert 'train_window_size' in forecasts.columns
        assert 'model_refitted' in forecasts.columns

        # Check that forecasts are generated
        valid_forecasts = forecasts['volatility_forecast'].dropna()
        assert len(valid_forecasts) > 0
        assert (valid_forecasts > 0).all()

    def test_rolling_forecast_insufficient_data(self, config):
        """Test rolling forecast with insufficient data."""
        forecaster = GARCHVolatilityForecaster(config)
        short_returns = pd.Series(range(30), index=pd.date_range('2020-01-01', periods=30, freq='M'))

        with pytest.raises(ValueError, match="Insufficient data"):
            forecaster.rolling_forecast(short_returns, window=60)

    def test_get_model_diagnostics(self, config, sample_returns):
        """Test getting model diagnostics."""
        forecaster = GARCHVolatilityForecaster(config)

        # Before fitting
        diagnostics = forecaster.get_model_diagnostics()
        assert diagnostics == {'error': 'Model not fitted'}

        # After fitting
        forecaster.fit(sample_returns)
        diagnostics = forecaster.get_model_diagnostics()

        assert 'convergence_flag' in diagnostics
        assert 'log_likelihood' in diagnostics
        assert 'aic' in diagnostics
        assert 'bic' in diagnostics
        assert 'parameters' in diagnostics
        assert 'persistence' in diagnostics
        assert isinstance(diagnostics['parameters'], dict)
        assert 'omega' in diagnostics['parameters']
        assert 'alpha' in diagnostics['parameters']
        assert 'beta' in diagnostics['parameters']

    def test_calculate_volatility_percentiles_expanding(self, config):
        """Test expanding percentile calculation."""
        forecaster = GARCHVolatilityForecaster(config)

        volatilities = pd.Series([1, 2, 3, 4, 5, 3, 2, 1])
        percentiles = forecaster.calculate_volatility_percentiles(volatilities)

        assert len(percentiles) == len(volatilities)
        assert percentiles.iloc[0] == 100  # First value is always 100th percentile
        assert 0 <= percentiles.min() <= 100
        assert 0 <= percentiles.max() <= 100

    def test_calculate_volatility_percentiles_rolling(self, config):
        """Test rolling percentile calculation."""
        forecaster = GARCHVolatilityForecaster(config)

        volatilities = pd.Series(range(1, 11))
        percentiles = forecaster.calculate_volatility_percentiles(volatilities, lookback=3)

        assert len(percentiles) == len(volatilities)
        # Last value in ascending series should be high percentile
        assert percentiles.iloc[-1] == 100

    @patch('models.garch_forecaster.arch_model')
    def test_fit_handles_convergence_issues(self, mock_arch_model, config, sample_returns):
        """Test that fit handles convergence issues gracefully."""
        # Mock failed fit followed by successful fit
        mock_model = MagicMock()
        mock_fit_result = MagicMock()
        mock_fit_result.convergence_flag = 1  # Non-zero indicates convergence issue
        mock_fit_result.params = {'omega': 0.1, 'alpha[1]': 0.1, 'beta[1]': 0.8}
        mock_fit_result.loglikelihood = -100
        mock_fit_result.aic = 200
        mock_fit_result.bic = 210
        mock_model.fit.return_value = mock_fit_result
        mock_arch_model.return_value = mock_model

        forecaster = GARCHVolatilityForecaster(config)
        forecaster.fit(sample_returns)

        assert forecaster.fitted_model is not None
        # Should log warning about convergence

    def test_persistence_calculation(self, config, sample_returns):
        """Test that persistence (alpha + beta) is calculated correctly."""
        forecaster = GARCHVolatilityForecaster(config)
        forecaster.fit(sample_returns)

        diagnostics = forecaster.get_model_diagnostics()
        persistence = diagnostics['persistence']

        # GARCH persistence should be between 0 and 1 for stationarity
        assert 0 <= persistence <= 1

    def test_forecast_edge_cases(self, config, sample_returns):
        """Test forecast handles edge cases."""
        forecaster = GARCHVolatilityForecaster(config)
        forecaster.fit(sample_returns)

        # Test multiple horizon forecasts
        for horizon in [1, 5, 10]:
            forecast = forecaster.forecast_volatility(horizon=horizon)
            assert forecast > 0
            assert not np.isnan(forecast)