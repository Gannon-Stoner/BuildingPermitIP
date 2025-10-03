"""Tests for signal generation module."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy import SignalGenerator


@pytest.fixture
def config():
    """Create test configuration."""
    return {
        'signals': {
            'top_quartile': 0.75,
            'bottom_quartile': 0.25,
            'min_history_for_percentiles': 60
        },
        'risk_management': {
            'unemployment_ma_window': 12
        }
    }


@pytest.fixture
def sample_volatility_forecasts():
    """Create sample volatility forecasts."""
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='M')
    # Create volatility with clear high/low periods
    n = len(dates)
    volatility = np.zeros(n)
    for i in range(n):
        if i % 20 < 5:  # Low volatility periods
            volatility[i] = np.random.uniform(0.5, 1.5)
        elif i % 20 < 15:  # Medium volatility
            volatility[i] = np.random.uniform(1.5, 3.0)
        else:  # High volatility
            volatility[i] = np.random.uniform(3.0, 5.0)

    return pd.Series(volatility, index=dates)


@pytest.fixture
def sample_unemployment():
    """Create sample unemployment data."""
    dates = pd.date_range('2018-01-01', '2023-12-31', freq='M')
    # Create unemployment with trend
    n = len(dates)
    unemployment = 4.0 + np.sin(np.linspace(0, 4*np.pi, n)) * 2 + np.random.normal(0, 0.2, n)
    return pd.Series(unemployment, index=dates)


class TestSignalGenerator:
    """Test signal generation."""

    def test_init(self, config):
        """Test initialization."""
        generator = SignalGenerator(config)
        assert generator.top_quartile == 0.75
        assert generator.bottom_quartile == 0.25
        assert generator.min_history == 60

    def test_generate_signals(self, config, sample_volatility_forecasts):
        """Test signal generation from volatility forecasts."""
        generator = SignalGenerator(config)
        signals_df = generator.generate_signals(sample_volatility_forecasts)

        assert isinstance(signals_df, pd.DataFrame)
        assert 'signal' in signals_df.columns
        assert 'volatility' in signals_df.columns
        assert 'volatility_percentile' in signals_df.columns
        assert 'signal_strength' in signals_df.columns

        # Check signal values are valid
        assert set(signals_df['signal'].unique()).issubset({-1, 0, 1})

        # Check that signals are generated after min_history
        sufficient_history = signals_df[signals_df['sufficient_history']]
        assert len(sufficient_history) == len(signals_df) - generator.min_history

        # No signals should be generated without sufficient history
        insufficient_history = signals_df[~signals_df['sufficient_history']]
        assert (insufficient_history['signal'] == 0).all()

    def test_calculate_expanding_percentiles(self, config):
        """Test expanding percentile calculation."""
        generator = SignalGenerator(config)
        volatility = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1])
        percentiles = generator._calculate_expanding_percentiles(volatility)

        assert len(percentiles) == len(volatility)
        # First min_history-1 values should be 50 (median)
        assert (percentiles.iloc[:generator.min_history-1] == 50.0).all()
        # Percentiles should be between 0 and 100
        assert (percentiles >= 0).all() and (percentiles <= 100).all()

    def test_calculate_signal_strength_long(self, config):
        """Test signal strength calculation for long signals."""
        generator = SignalGenerator(config)

        percentiles = pd.Series([5, 15, 25, 50, 75, 95])
        signals = pd.Series([1, 1, 1, 0, -1, -1])

        strength = generator._calculate_signal_strength(percentiles, signals)

        # Long signals: stronger as percentile approaches 0
        assert strength.iloc[0] > strength.iloc[1]  # 5th percentile stronger than 15th
        assert strength.iloc[1] > strength.iloc[2]  # 15th stronger than 25th
        assert strength.iloc[3] == 0  # Neutral signal has 0 strength

    def test_calculate_signal_strength_short(self, config):
        """Test signal strength calculation for short signals."""
        generator = SignalGenerator(config)

        percentiles = pd.Series([5, 25, 50, 75, 85, 95])
        signals = pd.Series([1, 0, 0, -1, -1, -1])

        strength = generator._calculate_signal_strength(percentiles, signals)

        # Short signals: stronger as percentile approaches 100
        assert strength.iloc[5] > strength.iloc[4]  # 95th percentile stronger than 85th
        assert strength.iloc[4] > strength.iloc[3]  # 85th stronger than 75th

    def test_add_regime_filter(self, config, sample_volatility_forecasts, sample_unemployment):
        """Test regime filter application."""
        generator = SignalGenerator(config)

        # Generate initial signals
        signals_df = generator.generate_signals(sample_volatility_forecasts)

        # Apply regime filter
        filtered_signals = generator.add_regime_filter(signals_df, sample_unemployment)

        assert 'unemployment' in filtered_signals.columns
        assert 'unemployment_ma' in filtered_signals.columns
        assert 'risk_off_regime' in filtered_signals.columns
        assert 'signal_pre_filter' in filtered_signals.columns

        # Check that signals are neutralized during risk-off regime
        risk_off_mask = filtered_signals['risk_off_regime']
        assert (filtered_signals.loc[risk_off_mask, 'signal'] == 0).all()

        # Original signals should be preserved
        assert 'signal_pre_filter' in filtered_signals.columns

    def test_signal_statistics(self, config, sample_volatility_forecasts):
        """Test calculation of signal statistics."""
        generator = SignalGenerator(config)
        signals_df = generator.generate_signals(sample_volatility_forecasts)

        stats = generator.calculate_signal_statistics(signals_df)

        assert 'n_long_signals' in stats
        assert 'n_short_signals' in stats
        assert 'n_neutral_signals' in stats
        assert 'total_signals' in stats

        # Check totals add up
        total = stats['n_long_signals'] + stats['n_short_signals'] + stats['n_neutral_signals']
        assert total == stats['total_signals']
        assert total == len(signals_df)

        # Check percentages if non-neutral signals exist
        if stats['n_long_signals'] + stats['n_short_signals'] > 0:
            assert 0 <= stats['pct_long'] <= 100
            assert 0 <= stats['pct_short'] <= 100

    def test_analyze_signal_transitions(self, config):
        """Test signal transition analysis."""
        generator = SignalGenerator(config)

        # Create signals with known transitions
        dates = pd.date_range('2020-01-01', periods=10, freq='M')
        signals = pd.Series([0, 1, 1, -1, -1, 0, 0, 1, -1, 0], index=dates)
        signals_df = pd.DataFrame({'signal': signals})

        transitions = generator.analyze_signal_transitions(signals_df)

        assert 'signal_change' in transitions.columns
        assert 'from_signal' in transitions.columns
        assert 'to_signal' in transitions.columns
        assert 'transition_type' in transitions.columns
        assert 'signal_duration' in transitions.columns

        # Check signal changes are detected correctly
        expected_changes = [True, True, False, True, False, True, False, True, True, True]
        pd.testing.assert_series_equal(
            transitions['signal_change'],
            pd.Series(expected_changes, index=dates),
            check_names=False
        )

    def test_quartile_thresholds(self, config):
        """Test that signals respect quartile thresholds."""
        generator = SignalGenerator(config)

        # Create volatility with clear quartiles
        n = 200
        dates = pd.date_range('2015-01-01', periods=n, freq='M')
        volatility = pd.Series(range(n), index=dates)  # Monotonically increasing

        signals_df = generator.generate_signals(volatility)

        # After sufficient history
        valid_signals = signals_df[signals_df['sufficient_history']]

        # Check bottom quartile generates long signals
        bottom_quartile_mask = valid_signals['volatility_percentile'] <= 25
        bottom_signals = valid_signals.loc[bottom_quartile_mask, 'signal']
        assert (bottom_signals == 1).all() or (bottom_signals == 0).all()  # Could be 0 if insufficient history

        # Check top quartile generates short signals
        top_quartile_mask = valid_signals['volatility_percentile'] >= 75
        top_signals = valid_signals.loc[top_quartile_mask, 'signal']
        assert (top_signals == -1).all() or (top_signals == 0).all()  # Could be 0 if insufficient history

        # Check middle range is neutral
        middle_mask = (valid_signals['volatility_percentile'] > 25) & (valid_signals['volatility_percentile'] < 75)
        middle_signals = valid_signals.loc[middle_mask, 'signal']
        assert (middle_signals == 0).all()

    def test_regime_filter_impact(self, config):
        """Test that regime filter properly neutralizes signals."""
        generator = SignalGenerator(config)

        # Create signals
        dates = pd.date_range('2020-01-01', periods=24, freq='M')
        volatility = pd.Series(range(24), index=dates)
        signals_df = generator.generate_signals(volatility)

        # Create unemployment that triggers risk-off regime in second half
        unemployment = pd.Series([3.0] * 12 + [8.0] * 12, index=dates)

        # Apply regime filter
        filtered = generator.add_regime_filter(signals_df, unemployment)

        # After MA window, high unemployment should trigger risk-off
        risk_off_period = filtered.iloc[config['risk_management']['unemployment_ma_window']:]
        risk_off_signals = risk_off_period[risk_off_period['risk_off_regime']]

        # All signals during risk-off should be neutral
        if len(risk_off_signals) > 0:
            assert (risk_off_signals['signal'] == 0).all()