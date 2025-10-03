# Building Permit Growth Volatility (BPGV) Trading Strategy

## Overview

The BPGV Trading Strategy is a quantitative trading system that uses Building Permit Growth Volatility as a macroeconomic signal to trade E-mini S&P 500 futures. The strategy employs GARCH(1,1) modeling to forecast housing permit volatility and generates long/short signals based on volatility quartiles, with additional regime filtering based on unemployment trends.

**Developed by:** AlgoGators Team (Gannon Stoner, Christian Cardenas, Dean Lucas)
**Date:** September 2025

## Core Thesis

Building permit volatility serves as a leading indicator of economic uncertainty and market volatility. The strategy capitalizes on the relationship between housing market volatility and broader market movements:
- **Low permit volatility** → Economic stability → Long equity exposure
- **High permit volatility** → Economic uncertainty → Short/risk-off positioning

## Features

- **GARCH(1,1) Volatility Forecasting**: State-of-the-art econometric modeling for volatility prediction
- **Walk-Forward Analysis**: Out-of-sample testing with periodic model refitting
- **Dynamic Position Sizing**: Volatility-targeted positioning with signal strength scaling
- **Risk Management**: Drawdown controls and regime-based exposure reduction
- **Comprehensive Backtesting**: Monthly rebalancing with daily P&L tracking
- **Performance Analytics**: Detailed metrics including Sharpe, Sortino, and Calmar ratios

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Housing Permit data repository (cloned locally)

### Setup

1. Clone the Housing Permit data repository:
```bash
cd ~
git clone https://github.com/idkh0wtocode/Housing_Permit_IP.git
```

2. Navigate to the strategy directory:
```bash
cd /Users/gannonstoner/PycharmProjects/BuildingPermitProject/bpgv_strategy
```

3. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Verify data access:
```bash
python test_local_data.py
```

## Usage

### Basic Execution

Run the strategy with default parameters:
```bash
python main.py
```

### Custom Date Range

Specify custom start and end dates:
```bash
python main.py --start-date 2010-01-01 --end-date 2025-09-30
```

### Custom Configuration

Use a different configuration file:
```bash
python main.py --config config/custom_config.yaml
```

### Advanced Options

```bash
python main.py \
    --start-date 2015-01-01 \
    --end-date 2025-09-30 \
    --log-level DEBUG
```

## Configuration

The strategy is configured via `config/config.yaml`:

```yaml
data:
  housing_data_path: "/Users/gannonstoner/Housing_Permit_IP"
  permit_file: "PERMIT.csv"
  start_date: "2000-01-01"
  end_date: "2025-09-30"
  futures_ticker: "ES=F"

garch:
  p: 1                    # GARCH p parameter
  q: 1                    # GARCH q parameter
  min_window: 60          # Minimum months for fitting
  refit_frequency: 12     # Refit every 12 months
  max_history: 120        # Maximum lookback (10 years)

signals:
  top_quartile: 0.75      # Short signal threshold
  bottom_quartile: 0.25   # Long signal threshold

position_sizing:
  max_exposure: 0.10      # 10% max position
  target_annual_vol: 0.10 # 10% volatility target

risk_management:
  max_drawdown_pct: 0.10  # 10% drawdown limit
  unemployment_ma_window: 12

backtest:
  initial_capital: 1000000
  transaction_cost: 0.0001  # 1 basis point
```

## Strategy Details

### Signal Generation Process

1. **Data Collection**: Monthly building permits (PERMIT) and unemployment (UNRATE) from FRED
2. **Volatility Forecasting**: GARCH(1,1) model generates one-step-ahead volatility forecasts
3. **Percentile Calculation**: Expanding window percentile ranks of volatility
4. **Signal Mapping**:
   - Volatility < 25th percentile → **Long signal** (low uncertainty)
   - Volatility > 75th percentile → **Short signal** (high uncertainty)
   - Middle range → **Neutral** (no position)
5. **Regime Filter**: Force neutral when unemployment > 12-month MA (risk-off)

### Position Sizing

Positions are sized using volatility targeting with signal strength scaling:
```
base_size = (target_vol / asset_vol) * portfolio_value
scale_factor = 0.5 + signal_strength  # Range: 0.5x to 1.5x
position = base_size * scale_factor * signal_direction
```

### Risk Management

- **Maximum Exposure**: 10% of portfolio value
- **Drawdown Control**: Force close all positions if drawdown > 10%
- **Regime Filter**: No positions during elevated unemployment periods
- **Minimum Holding**: Positions held until next monthly signal

## Performance Metrics

The strategy calculates comprehensive performance metrics:

- **Returns**: Total, Annual (CAGR)
- **Risk**: Volatility, Maximum Drawdown, VaR, CVaR
- **Risk-Adjusted**: Sharpe, Sortino, Calmar ratios
- **Trading**: Win rate, Profit factor, Trade statistics

## Output Files

After execution, the following files are generated:

```
outputs/
├── backtest_results.csv      # Daily portfolio values and positions
├── performance_metrics.json  # All performance metrics
├── trade_log.csv             # Detailed trade history
├── report.txt               # Formatted summary report
└── plots/
    └── performance_report.png # 4-panel visualization
```

## Testing

Run the test suite:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=. --cov-report=html

# Run specific test module
pytest tests/test_garch.py -v
```

## Project Structure

```
bpgv_strategy/
├── config/               # Configuration files
├── data/                # Data fetching and preprocessing
├── models/              # GARCH and regime detection models
├── strategy/            # Signal generation and risk management
├── backtest/           # Backtesting engine and performance
├── utils/              # Logging and utilities
├── tests/              # Comprehensive test suite
├── notebooks/          # Jupyter notebooks for analysis
├── main.py            # Main execution script
└── requirements.txt   # Python dependencies
```

## Performance Expectations

Based on the strategy design and historical analysis:

- **Expected Sharpe Ratio**: 0.5 - 1.2
- **Maximum Drawdown**: Limited to 10% by risk controls
- **Win Rate**: 45-55% (typical for trend strategies)
- **Trade Frequency**: 12-20 trades per year (monthly rebalancing)

## Data Sources

This implementation uses:
- **Housing Permit Data**: Local repository from https://github.com/idkh0wtocode/Housing_Permit_IP
  - National permits: `PERMIT.csv` (1960-2025)
  - State-level data: `state_permits/` directory
  - City-level data: `city_permits/` directory
- **Futures Data**: Downloaded from Yahoo Finance (ES=F)
- **Unemployment Data**: Currently using synthetic data for backtesting

## Known Limitations

1. **Data Lag**: Building permit data has ~2 week publication lag
2. **Futures Data**: Limited to 2010-present for E-mini S&P 500
3. **Transaction Costs**: Simplified model (fixed 1bp)
4. **Single Asset**: Currently trades only ES futures
5. **No Slippage Model**: Assumes execution at close prices
6. **Unemployment Data**: Using synthetic data (real data integration pending)

## Future Enhancements

- [ ] Multi-asset support (Real Estate futures, sector ETFs)
- [ ] Dynamic regime detection (beyond unemployment)
- [ ] Machine learning signal enhancement
- [ ] Intraday execution optimization
- [ ] Live trading integration
- [ ] Monte Carlo simulation for robustness testing
- [ ] Walk-forward parameter optimization

## References

1. FRED Economic Data: [https://fred.stlouisfed.org/](https://fred.stlouisfed.org/)
2. GARCH Models: Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity"
3. Building Permits as Leading Indicator: Conference Board Leading Economic Index methodology
4. Original Strategy Proposal: AlgoGators Team (September 17, 2025)

## Support

For questions or issues:
- Create an issue in the project repository
- Contact the development team

## License

This project is proprietary software developed by the AlgoGators team.

## Disclaimer

This trading strategy is for educational and research purposes only. Past performance does not guarantee future results. Trading futures involves substantial risk of loss and is not suitable for all investors. Always conduct your own research and consider your financial situation before trading.

---

*Last updated: September 2025*