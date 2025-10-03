"""
Main execution script for BPGV trading strategy.

Usage:
    python main.py --start-date 2010-01-01 --end-date 2025-09-30
    python main.py --config config/config.yaml
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from config import load_config
from utils import setup_logger
from data import DataFetcher, DataPreprocessor
from models import GARCHVolatilityForecaster, RegimeDetector
from strategy import SignalGenerator
from backtest import Backtester, PerformanceAnalyzer

# Suppress warnings
warnings.filterwarnings('ignore')


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run BPGV Trading Strategy')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Setup logging
    log_file = f"outputs/logs/bpgv_strategy_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = setup_logger(log_level=args.log_level, log_file=log_file)

    logger.info("=" * 60)
    logger.info("BPGV TRADING STRATEGY - EXECUTION STARTED")
    logger.info("=" * 60)

    try:
        # 1. Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)

        # Override dates if provided
        if args.start_date:
            config['data']['start_date'] = args.start_date
        if args.end_date:
            config['data']['end_date'] = args.end_date

        logger.info(f"Configuration loaded: {args.config}")
        logger.info(f"Date range: {config['data']['start_date']} to {config['data'].get('end_date', 'present')}")

        # 2. Initialize data fetcher and fetch data
        logger.info("\n" + "=" * 40)
        logger.info("STEP 1: FETCHING DATA")
        logger.info("=" * 40)

        data_fetcher = DataFetcher(config)
        data_preprocessor = DataPreprocessor(config)

        # Fetch all required data
        all_data = data_fetcher.fetch_all_data(
            start_date=config['data']['start_date'],
            end_date=config['data'].get('end_date')
        )

        # Merge macroeconomic data
        macro_data = data_preprocessor.merge_macro_data(
            all_data['permits'],
            all_data['unemployment']
        )

        logger.info(f"Data fetched successfully:")
        logger.info(f"  - Building permits: {len(all_data['permits'])} months")
        logger.info(f"  - Unemployment: {len(all_data['unemployment'])} months")
        logger.info(f"  - Futures data: {len(all_data['futures'])} days")

        # 3. Train GARCH model and generate volatility forecasts
        logger.info("\n" + "=" * 40)
        logger.info("STEP 2: GARCH MODEL & VOLATILITY FORECASTING")
        logger.info("=" * 40)

        # Prepare data for GARCH
        permit_growth = data_preprocessor.prepare_garch_data(macro_data['permit_growth'])

        # Initialize and run GARCH forecaster
        garch_forecaster = GARCHVolatilityForecaster(config)

        # Generate rolling volatility forecasts
        logger.info("Generating volatility forecasts using walk-forward analysis...")
        volatility_forecasts = garch_forecaster.rolling_forecast(
            permit_growth,
            window=config['garch']['max_history'],
            refit_freq=config['garch']['refit_frequency']
        )

        # Get model diagnostics
        diagnostics = garch_forecaster.get_model_diagnostics()
        logger.info(f"GARCH model diagnostics:")
        logger.info(f"  - Persistence: {diagnostics.get('persistence', 'N/A'):.3f}")
        logger.info(f"  - Log-likelihood: {diagnostics.get('log_likelihood', 'N/A'):.2f}")
        logger.info(f"  - AIC: {diagnostics.get('aic', 'N/A'):.2f}")

        # 4. Generate trading signals
        logger.info("\n" + "=" * 40)
        logger.info("STEP 3: SIGNAL GENERATION")
        logger.info("=" * 40)

        signal_generator = SignalGenerator(config)

        # Generate signals from volatility forecasts
        signals = signal_generator.generate_signals(
            volatility_forecasts['volatility_forecast']
        )

        # Apply regime filter
        signals = signal_generator.add_regime_filter(
            signals,
            macro_data['unemployment']
        )

        # Get signal statistics
        signal_stats = signal_generator.calculate_signal_statistics(signals)
        logger.info(f"Signal statistics:")
        logger.info(f"  - Long signals: {signal_stats['n_long_signals']}")
        logger.info(f"  - Short signals: {signal_stats['n_short_signals']}")
        logger.info(f"  - Neutral periods: {signal_stats['n_neutral_signals']}")
        logger.info(f"  - Filtered by regime: {signal_stats.get('n_filtered_by_regime', 0)}")

        # 5. Run backtest
        logger.info("\n" + "=" * 40)
        logger.info("STEP 4: BACKTESTING")
        logger.info("=" * 40)

        backtester = Backtester(config)

        # Run backtest
        logger.info(f"Running backtest with initial capital: ${config['backtest']['initial_capital']:,.0f}")
        backtest_results = backtester.run(
            signals_df=signals,
            price_data=all_data['futures']
        )

        # Get trade statistics
        trade_stats = backtester.get_trade_statistics()
        logger.info(f"Trade statistics:")
        logger.info(f"  - Total trades: {trade_stats.get('n_trades', 0)}")
        logger.info(f"  - Win rate: {trade_stats.get('win_rate', 0):.1%}")
        logger.info(f"  - Profit factor: {trade_stats.get('profit_factor', 0):.2f}")

        # 6. Calculate performance metrics
        logger.info("\n" + "=" * 40)
        logger.info("STEP 5: PERFORMANCE ANALYSIS")
        logger.info("=" * 40)

        performance_analyzer = PerformanceAnalyzer(config)

        # Extract equity curve
        equity_curve = performance_analyzer.generate_equity_curve(backtest_results)

        # Calculate metrics
        metrics = performance_analyzer.calculate_metrics(equity_curve)

        logger.info(f"Performance metrics:")
        logger.info(f"  - Total Return: {metrics.get('total_return', 0):.2f}%")
        logger.info(f"  - Annual Return: {metrics.get('annual_return', 0):.2f}%")
        logger.info(f"  - Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  - Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        logger.info(f"  - Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")

        # 7. Generate visualizations
        logger.info("\n" + "=" * 40)
        logger.info("STEP 6: GENERATING VISUALIZATIONS")
        logger.info("=" * 40)

        performance_analyzer.plot_performance(
            backtest_results,
            signals,
            volatility_forecasts
        )

        # 8. Save results
        logger.info("\n" + "=" * 40)
        logger.info("STEP 7: SAVING RESULTS")
        logger.info("=" * 40)

        performance_analyzer.save_results(
            backtest_results,
            metrics,
            trade_stats
        )

        # Save trade log
        trade_log = backtester.create_trade_log()
        if not trade_log.empty:
            trade_log_path = Path(config['outputs']['results_dir']) / 'trade_log.csv'
            trade_log.to_csv(trade_log_path)
            logger.info(f"Trade log saved to {trade_log_path}")

        # 9. Print summary report
        logger.info("\n" + "=" * 40)
        logger.info("EXECUTION COMPLETE")
        logger.info("=" * 40)

        report = performance_analyzer.create_summary_report(metrics, trade_stats)
        print("\n" + report)

        # Final summary
        logger.info(f"\nResults saved to: {config['outputs']['results_dir']}")
        logger.info(f"Plots saved to: {config['outputs']['plots_dir']}")
        logger.info(f"Log file: {log_file}")

        logger.info("\nStrategy execution completed successfully!")

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        print(f"\nError: {e}")
        print(f"Check log file for details: {log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()