#!/usr/bin/env python3
"""
Diagnostic script to identify issues with signal generation and volatility forecasting.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import logging
from config import load_config
from data import DataFetcher, DataPreprocessor
from models import GARCHVolatilityForecaster
from strategy import SignalGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_issues():
    """Run diagnostics on the strategy components."""

    logger.info("="*60)
    logger.info("STRATEGY DIAGNOSTICS")
    logger.info("="*60)

    # Load config
    config = load_config()

    # Initialize components
    fetcher = DataFetcher(config)
    preprocessor = DataPreprocessor(config)

    # 1. Check permit data and growth
    logger.info("\n1. CHECKING PERMIT DATA")
    logger.info("-"*40)

    permits = fetcher.get_building_permits("2015-01-01", "2021-12-31")
    permits_with_growth = fetcher.compute_permit_growth(permits)

    logger.info(f"Permits loaded: {len(permits)} months")
    logger.info(f"Permit values - Mean: {permits['permits'].mean():.0f}, Std: {permits['permits'].std():.0f}")
    logger.info(f"Growth rates - Mean: {permits_with_growth['permit_growth'].mean():.2f}%, Std: {permits_with_growth['permit_growth'].std():.2f}%")

    # Check for extreme values
    logger.info(f"Growth range: {permits_with_growth['permit_growth'].min():.2f}% to {permits_with_growth['permit_growth'].max():.2f}%")

    # 2. Check GARCH volatility forecasting
    logger.info("\n2. CHECKING GARCH VOLATILITY FORECASTING")
    logger.info("-"*40)

    # Prepare data for GARCH
    permit_growth_clean = preprocessor.prepare_garch_data(permits_with_growth['permit_growth'])
    logger.info(f"Clean growth data: {len(permit_growth_clean)} observations")
    logger.info(f"Clean growth - Mean: {permit_growth_clean.mean():.2f}%, Std: {permit_growth_clean.std():.2f}%")

    # Run GARCH
    garch = GARCHVolatilityForecaster(config)

    # Try rolling forecast with smaller window
    try:
        vol_forecasts = garch.rolling_forecast(
            permit_growth_clean,
            window=config['garch']['max_history'],
            refit_freq=config['garch']['refit_frequency']
        )

        logger.info(f"Volatility forecasts generated: {len(vol_forecasts)} observations")

        if 'volatility_forecast' in vol_forecasts.columns:
            vol_series = vol_forecasts['volatility_forecast'].dropna()
            logger.info(f"Volatility forecast stats:")
            logger.info(f"  Mean: {vol_series.mean():.4f}")
            logger.info(f"  Std: {vol_series.std():.4f}")
            logger.info(f"  Min: {vol_series.min():.4f}")
            logger.info(f"  Max: {vol_series.max():.4f}")

            # Check for constant values
            if vol_series.std() < 0.0001:
                logger.warning("WARNING: Volatility forecasts are nearly constant!")

    except Exception as e:
        logger.error(f"GARCH forecasting failed: {e}")
        return

    # 3. Check signal generation
    logger.info("\n3. CHECKING SIGNAL GENERATION")
    logger.info("-"*40)

    signal_gen = SignalGenerator(config)
    signals = signal_gen.generate_signals(vol_forecasts['volatility_forecast'])

    logger.info(f"Signals generated: {len(signals)} observations")

    # Check percentiles
    if 'volatility_percentile' in signals.columns:
        percentiles = signals['volatility_percentile'].dropna()
        logger.info(f"Percentile distribution:")
        logger.info(f"  Min: {percentiles.min():.1f}")
        logger.info(f"  25th: {percentiles.quantile(0.25):.1f}")
        logger.info(f"  50th: {percentiles.quantile(0.50):.1f}")
        logger.info(f"  75th: {percentiles.quantile(0.75):.1f}")
        logger.info(f"  Max: {percentiles.max():.1f}")

        # Check signal thresholds
        bottom_quartile = config['signals']['bottom_quartile'] * 100
        top_quartile = config['signals']['top_quartile'] * 100

        logger.info(f"\nSignal thresholds:")
        logger.info(f"  Long when percentile <= {bottom_quartile:.0f}")
        logger.info(f"  Short when percentile >= {top_quartile:.0f}")

        # Count how many would trigger
        would_be_long = (percentiles <= bottom_quartile).sum()
        would_be_short = (percentiles >= top_quartile).sum()
        logger.info(f"\nPotential signals (before min history filter):")
        logger.info(f"  Would be long: {would_be_long}")
        logger.info(f"  Would be short: {would_be_short}")

    # Check actual signals
    signal_counts = signals['signal'].value_counts()
    logger.info(f"\nActual signal distribution:")
    for signal_val, count in signal_counts.items():
        signal_name = {1: 'Long', 0: 'Neutral', -1: 'Short'}.get(signal_val, signal_val)
        logger.info(f"  {signal_name}: {count}")

    # Check sufficient history flag
    if 'sufficient_history' in signals.columns:
        sufficient = signals['sufficient_history'].sum()
        logger.info(f"\nSufficient history periods: {sufficient} out of {len(signals)}")

        # Check signals with sufficient history
        signals_with_history = signals[signals['sufficient_history']]
        if not signals_with_history.empty:
            signal_counts_history = signals_with_history['signal'].value_counts()
            logger.info(f"\nSignals with sufficient history:")
            for signal_val, count in signal_counts_history.items():
                signal_name = {1: 'Long', 0: 'Neutral', -1: 'Short'}.get(signal_val, signal_val)
                logger.info(f"  {signal_name}: {count}")

    # 4. Debug expanding percentile calculation
    logger.info("\n4. DEBUGGING PERCENTILE CALCULATION")
    logger.info("-"*40)

    # Manually calculate percentiles for a few points
    vol_array = vol_forecasts['volatility_forecast'].dropna().values
    logger.info(f"Volatility array length: {len(vol_array)}")

    if len(vol_array) > 65:
        for i in [60, 65, 70, len(vol_array)-1]:
            historical = vol_array[:i+1]
            current = vol_array[i]
            rank = (historical <= current).sum()
            percentile = (rank / len(historical)) * 100
            logger.info(f"Point {i}: Value={current:.4f}, Percentile={percentile:.1f}")
            logger.info(f"  Historical range: {historical.min():.4f} to {historical.max():.4f}")

    # 5. Check for data issues
    logger.info("\n5. DATA QUALITY CHECKS")
    logger.info("-"*40)

    # Check if volatility values are all similar
    if len(vol_array) > 0:
        unique_vals = len(np.unique(np.round(vol_array, 4)))
        logger.info(f"Unique volatility values (rounded to 4 decimals): {unique_vals}")

        if unique_vals < 10:
            logger.warning("WARNING: Very few unique volatility values - may indicate GARCH issues")

    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTICS COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    diagnose_issues()