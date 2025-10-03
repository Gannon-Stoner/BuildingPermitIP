import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

from config import load_config
from data import DataFetcher, DataPreprocessor
from models import GARCHVolatilityForecaster
from strategy import SignalGenerator
from backtest.engine import Backtester

# Quick test with limited date range
config = load_config()

print('Starting quick test...')

# Initialize components
fetcher = DataFetcher(config)
preprocessor = DataPreprocessor(config)
garch = GARCHVolatilityForecaster(config)
signal_gen = SignalGenerator(config)
backtest_engine = Backtester(config)

# Fetch limited data (need more history for GARCH)
permits = fetcher.get_building_permits('2015-01-01', '2021-12-31')
permits_with_growth = fetcher.compute_permit_growth(permits)
futures = fetcher.get_futures_data('ES=F', '2020-01-01', '2021-12-31')

print(f'Data loaded - Permits: {len(permits)}, Futures: {len(futures)}')

# Prepare for GARCH  
permit_growth_clean = preprocessor.prepare_garch_data(permits_with_growth['permit_growth'])

# Run GARCH with appropriate window
vol_forecasts = garch.rolling_forecast(permit_growth_clean, window=60, refit_freq=12)
print(f'Volatility forecasts: {len(vol_forecasts)}')

# Generate signals
signals = signal_gen.generate_signals(vol_forecasts['volatility_forecast'])
print(f'Signals generated: Long={sum(signals["signal"]==1)}, Short={sum(signals["signal"]==-1)}')

# Align data
aligned_data = fetcher.align_data(signals, futures, 'signal')

# Run backtest (split aligned_data into signals and price data)
results = backtest_engine.run(signals, futures, initial_capital=1000000)

# Check results
final_value = results['portfolio_value'].iloc[-1]
total_return = (final_value / 1000000 - 1) * 100
max_position = results['position'].abs().max()

# Debug: Check if any trades were made
n_trades = sum(results['position'] != results['position'].shift(1))
n_long = sum(signals['signal'] == 1)
n_short = sum(signals['signal'] == -1)

print(f'Final portfolio value: ${final_value:,.0f}')
print(f'Total return: {total_return:.2f}%')
print(f'Max position size: ${max_position:,.0f}')
print(f'Max leverage: {max_position/1000000:.1f}x')
print(f'Number of trades: {n_trades}')
print(f'Signals - Long: {n_long}, Short: {n_short}')

# Check aligned data
if 'signal' in aligned_data.columns:
    print(f'Aligned signals - Non-zero: {sum(aligned_data["signal"] != 0)}')
    print(f'Signal changes in aligned data: {sum(aligned_data["signal_change"])}')

# Check what signals were passed to backtest
print(f'\\nSignals passed to backtest:')
print(f'  Shape: {signals.shape}')
print(f'  Non-zero signals: {sum(signals["signal"] != 0)}')
if 'volatility_percentile' in signals.columns:
    print(f'  Volatility percentiles: min={signals["volatility_percentile"].min():.1f}, max={signals["volatility_percentile"].max():.1f}')

# Check for reasonable returns
if abs(total_return) > 100:
    print('WARNING: Returns still seem unrealistic!')
else:
    print('Returns look more reasonable now.')
