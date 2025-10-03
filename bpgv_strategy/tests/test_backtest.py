"""Tests for backtesting module."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest import Backtester, PerformanceAnalyzer


@pytest.fixture
def config():
    """Create test configuration."""
    return {
        'backtest': {
            'initial_capital': 1000000,
            'transaction_cost': 0.0001,
            'min_holding_period_days': 20
        },
        'position_sizing': {
            'max_exposure': 0.10,
            'target_annual_vol': 0.10,
            'min_scale_factor': 0.5,
            'max_scale_factor': 1.5
        },
        'risk_management': {
            'max_drawdown_pct': 0.10,
            'unemployment_ma_window': 12
        },
        'outputs': {
            'results_dir': 'test_outputs',
            'plots_dir': 'test_outputs/plots'
        }
    }


@pytest.fixture
def sample_signals():
    """Create sample signals DataFrame."""
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='M')
    n = len(dates)
    signals = []
    for i in range(n):
        if i % 6 < 2:
            signals.append(1)  # Long
        elif i % 6 < 4:
            signals.append(-1)  # Short
        else:
            signals.append(0)  # Neutral

    df = pd.DataFrame({
        'signal': signals,
        'volatility_percentile': np.random.uniform(0, 100, n),
        'volatility': np.random.uniform(0.1, 0.3, n)
    }, index=dates)

    return df


@pytest.fixture
def sample_price_data():
    """Create sample price data."""
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
    n = len(dates)

    # Create price series with trend and volatility
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, n)  # Daily returns
    price = 4000 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'open': price * np.random.uniform(0.99, 1.01, n),
        'high': price * np.random.uniform(1.00, 1.02, n),
        'low': price * np.random.uniform(0.98, 1.00, n),
        'close': price,
        'volume': np.random.uniform(1e6, 2e6, n)
    }, index=dates)

    return df


class TestBacktester:
    """Test Backtester class."""

    def test_init(self, config):
        """Test backtester initialization."""
        backtester = Backtester(config)
        assert backtester.initial_capital == 1000000
        assert backtester.transaction_cost == 0.0001
        assert backtester.min_holding_period == 20

    def test_run_backtest(self, config, sample_signals, sample_price_data):
        """Test running a complete backtest."""
        backtester = Backtester(config)

        results = backtester.run(sample_signals, sample_price_data)

        assert isinstance(results, pd.DataFrame)
        assert 'portfolio_value' in results.columns
        assert 'position' in results.columns
        assert 'signal' in results.columns
        assert 'returns' in results.columns
        assert 'cumulative_returns' in results.columns
        assert 'drawdown' in results.columns

        # Check portfolio value starts at initial capital
        assert results['portfolio_value'].iloc[0] == pytest.approx(config['backtest']['initial_capital'], rel=0.01)

        # Check returns calculation
        assert len(results['returns'].dropna()) == len(results) - 1

    def test_calculate_asset_volatility(self, config, sample_price_data):
        """Test asset volatility calculation."""
        backtester = Backtester(config)

        test_date = sample_price_data.index[100]
        volatility = backtester.calculate_asset_volatility(
            sample_price_data['close'],
            test_date,
            window=60
        )

        assert isinstance(volatility, float)
        assert 0.05 <= volatility <= 0.50  # Reasonable bounds

    def test_prepare_backtest_data(self, config, sample_signals, sample_price_data):
        """Test data preparation for backtesting."""
        backtester = Backtester(config)

        prepared = backtester._prepare_backtest_data(sample_signals, sample_price_data)

        assert isinstance(prepared, pd.DataFrame)
        assert 'close' in prepared.columns
        assert 'signal' in prepared.columns
        assert 'signal_change' in prepared.columns

        # Should be daily frequency
        assert len(prepared) <= len(sample_price_data)

    def test_calculate_drawdown_series(self, config):
        """Test drawdown calculation."""
        backtester = Backtester(config)

        portfolio_values = pd.Series([100, 110, 105, 95, 100, 90, 100])
        drawdown = backtester._calculate_drawdown_series(portfolio_values)

        expected = pd.Series([0, 0, -0.0454545, -0.136364, -0.090909, -0.181818, -0.090909])
        pd.testing.assert_series_equal(drawdown, expected, rtol=0.001, check_names=False)

    def test_get_trade_statistics(self, config, sample_signals, sample_price_data):
        """Test trade statistics calculation."""
        backtester = Backtester(config)

        # Run backtest to generate trades
        backtester.run(sample_signals, sample_price_data)
        stats = backtester.get_trade_statistics()

        assert 'n_trades' in stats
        if stats['n_trades'] > 0:
            assert 'win_rate' in stats
            assert 'avg_win' in stats
            assert 'avg_loss' in stats
            assert 'profit_factor' in stats

    def test_create_trade_log(self, config, sample_signals, sample_price_data):
        """Test trade log creation."""
        backtester = Backtester(config)

        # Run backtest to generate trades
        backtester.run(sample_signals, sample_price_data)
        trade_log = backtester.create_trade_log()

        if not trade_log.empty:
            assert 'type' in trade_log.columns
            assert 'size' in trade_log.columns
            assert 'price' in trade_log.columns
            assert 'realized_pnl' in trade_log.columns
            assert 'cumulative_pnl' in trade_log.columns

    def test_transaction_costs(self, config, sample_signals, sample_price_data):
        """Test that transaction costs are applied."""
        # Run with and without transaction costs
        config_no_cost = config.copy()
        config_no_cost['backtest']['transaction_cost'] = 0

        backtester_with_cost = Backtester(config)
        backtester_no_cost = Backtester(config_no_cost)

        results_with_cost = backtester_with_cost.run(sample_signals, sample_price_data)
        results_no_cost = backtester_no_cost.run(sample_signals, sample_price_data)

        # Final value should be lower with transaction costs (if trades were made)
        if len(backtester_with_cost.trades) > 0:
            assert results_with_cost['portfolio_value'].iloc[-1] < results_no_cost['portfolio_value'].iloc[-1]

    def test_drawdown_breach_handling(self, config):
        """Test drawdown breach detection and handling."""
        backtester = Backtester(config)

        # Create signals and prices that will trigger drawdown
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')

        # Create a price series with a large drop
        prices = pd.Series(100, index=dates)
        prices[100:150] = 80  # 20% drop
        price_df = pd.DataFrame({
            'close': prices,
            'volume': 1000000
        })

        # Create signals
        signal_dates = pd.date_range('2020-01-01', '2020-12-31', freq='M')
        signals_df = pd.DataFrame({
            'signal': 1,  # Always long
            'volatility_percentile': 20,
            'volatility': 0.15
        }, index=signal_dates)

        results = backtester.run(signals_df, price_df)

        # Check that drawdown is detected
        assert 'drawdown_breach' in results.columns
        assert results['drawdown_breach'].any()  # Should have at least one breach


class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer class."""

    def test_init(self, config):
        """Test performance analyzer initialization."""
        analyzer = PerformanceAnalyzer(config)
        assert analyzer.outputs_dir.exists()
        assert analyzer.plots_dir.exists()

    def test_calculate_metrics(self, config):
        """Test performance metrics calculation."""
        analyzer = PerformanceAnalyzer(config)

        # Create sample equity curve
        dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
        n = len(dates)
        returns = np.random.normal(0.0003, 0.01, n)
        equity_curve = pd.Series(1000000 * np.exp(np.cumsum(returns)), index=dates)

        metrics = analyzer.calculate_metrics(equity_curve)

        assert 'annual_return' in metrics
        assert 'annual_volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'calmar_ratio' in metrics
        assert 'win_rate' in metrics

        # Check metrics are reasonable
        assert -100 <= metrics['max_drawdown'] <= 0
        assert 0 <= metrics['win_rate'] <= 100
        assert metrics['annual_volatility'] > 0

    def test_calculate_monthly_returns(self, config):
        """Test monthly returns calculation."""
        analyzer = PerformanceAnalyzer(config)

        # Create daily equity curve
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        equity_curve = pd.Series(range(1000000, 1000000 + len(dates)), index=dates)

        monthly_returns = analyzer.calculate_monthly_returns(equity_curve)

        assert isinstance(monthly_returns, pd.Series)
        assert len(monthly_returns) <= 12  # At most 12 months
        assert monthly_returns.index.freq == 'M' or monthly_returns.index.freq is None

    def test_create_summary_report(self, config):
        """Test summary report generation."""
        analyzer = PerformanceAnalyzer(config)

        metrics = {
            'total_return': 25.5,
            'annual_return': 12.3,
            'sharpe_ratio': 1.5,
            'max_drawdown': -10.2
        }

        trade_stats = {
            'n_trades': 50,
            'win_rate': 55.0,
            'avg_win': 1000,
            'avg_loss': -800
        }

        report = analyzer.create_summary_report(metrics, trade_stats)

        assert isinstance(report, str)
        assert 'PERFORMANCE METRICS' in report
        assert 'TRADING STATISTICS' in report
        assert '25.5' in report  # Total return
        assert '50' in report  # Number of trades

    def test_generate_equity_curve(self, config):
        """Test equity curve extraction."""
        analyzer = PerformanceAnalyzer(config)

        results_df = pd.DataFrame({
            'portfolio_value': [1000000, 1010000, 1020000, 1015000],
            'other_column': [1, 2, 3, 4]
        })

        equity_curve = analyzer.generate_equity_curve(results_df)

        assert isinstance(equity_curve, pd.Series)
        assert len(equity_curve) == len(results_df)
        assert equity_curve.equals(results_df['portfolio_value'])

    def test_generate_equity_curve_missing_column(self, config):
        """Test error handling when portfolio_value column is missing."""
        analyzer = PerformanceAnalyzer(config)

        results_df = pd.DataFrame({
            'other_column': [1, 2, 3, 4]
        })

        with pytest.raises(ValueError, match="must contain 'portfolio_value'"):
            analyzer.generate_equity_curve(results_df)