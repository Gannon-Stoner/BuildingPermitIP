"""Backtesting engine for BPGV trading strategy."""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from strategy import PositionSizer, RiskManager

logger = logging.getLogger(__name__)


class Backtester:
    """Backtesting engine with monthly rebalancing and daily P&L tracking."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backtester with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.backtest_params = config['backtest']
        self.initial_capital = self.backtest_params['initial_capital']
        self.transaction_cost = self.backtest_params['transaction_cost']
        self.min_holding_period = self.backtest_params.get('min_holding_period_days', 20)

        # Initialize components
        self.position_sizer = PositionSizer(config)
        self.risk_manager = RiskManager(config)

        # Track state
        self.trades = []
        self.daily_results = []

    def run(self, signals_df: pd.DataFrame,
            price_data: pd.DataFrame,
            initial_capital: Optional[float] = None,
            transaction_cost: Optional[float] = None) -> pd.DataFrame:
        """
        Run backtest with monthly rebalancing and daily P&L tracking.

        Args:
            signals_df: DataFrame with trading signals (monthly)
            price_data: DataFrame with daily price data
            initial_capital: Starting capital (uses config default if None)
            transaction_cost: Transaction cost rate (uses config default if None)

        Returns:
            DataFrame with comprehensive backtest results
        """
        # Use provided parameters or defaults
        initial_capital = initial_capital or self.initial_capital
        transaction_cost = transaction_cost or self.transaction_cost

        logger.info(f"Starting backtest - Capital: ${initial_capital:,.0f}, "
                   f"Transaction cost: {transaction_cost:.2%}")

        # Prepare data
        backtest_data = self._prepare_backtest_data(signals_df, price_data)

        if backtest_data.empty:
            logger.error("No valid data for backtesting")
            return pd.DataFrame()

        # Initialize tracking variables
        portfolio_value = initial_capital
        cash = initial_capital
        position = 0.0  # Position value in dollars
        shares = 0.0  # Number of shares/contracts
        position_entry_price = 0.0
        margin_requirement = 0.0  # Margin posted for futures
        last_signal_date = None
        days_since_signal = 0

        # Reset risk manager
        self.risk_manager.reset_risk_state()

        # Track results
        results = []

        # Iterate through each trading day
        for date, row in backtest_data.iterrows():
            # Update days since last signal
            if row['signal_change'] and not pd.isna(row['signal']):
                last_signal_date = date
                days_since_signal = 0
            else:
                days_since_signal += 1

            # Get current price
            current_price = row['close']

            # Calculate asset volatility
            asset_volatility = self.calculate_asset_volatility(
                price_data['close'],
                date
            )

            # Mark-to-market current position
            if shares != 0:
                unrealized_pnl = shares * (current_price - position_entry_price)
                portfolio_value = cash + unrealized_pnl
            else:
                unrealized_pnl = 0.0
                portfolio_value = cash

            # Update risk manager
            self.risk_manager.update_peak(portfolio_value)

            # Check for drawdown breach
            drawdown_breach = self.risk_manager.check_drawdown_breach(portfolio_value)

            # Calculate target position based on current signal
            target_position = 0.0

            if drawdown_breach:
                # Force close position if in drawdown breach
                target_position = 0.0
                if position != 0:
                    logger.info(f"{date}: Closing position due to drawdown breach")

            elif row['signal_change'] and days_since_signal >= 0:
                # Calculate target position on signal change
                logger.debug(f"{date}: Signal change detected. Signal = {row['signal']}")
                if not pd.isna(row['signal']) and row['signal'] != 0:
                    target_position = self.position_sizer.calculate_position_size(
                        signal=int(row['signal']),
                        volatility_percentile=row.get('volatility_percentile', 50),
                        portfolio_value=portfolio_value,
                        asset_volatility=asset_volatility,
                        regime_scale=row.get('regime_scale', 1.0)
                    )

                    # Check position limits
                    is_valid, reason = self.risk_manager.check_position_limits(
                        position,
                        target_position,
                        portfolio_value
                    )

                    if not is_valid:
                        logger.debug(f"{date}: Position rejected - {reason}")
                        target_position = 0.0
                else:
                    # Neutral signal - close position
                    target_position = 0.0

            # Execute trade if position needs to change
            if target_position != position and (row['signal_change'] or drawdown_breach):
                logger.info(f"{date}: Executing trade. Current position: ${position:,.0f}, Target: ${target_position:,.0f}")
                # Calculate trade details (trade_size is already in dollars)
                trade_size = target_position - position
                trade_value = abs(trade_size)  # Already in dollar terms
                trade_cost = trade_value * transaction_cost

                # Calculate margin requirements for futures
                new_margin_required = abs(target_position) * 0.10 if target_position != 0 else 0

                if position == 0:  # Opening new position
                    required_capital = new_margin_required + trade_cost
                else:  # Adjusting or closing position
                    if target_position == 0:
                        # Closing position - only need trade cost
                        required_capital = trade_cost
                    else:
                        # Adjusting position - calculate margin change
                        margin_change = new_margin_required - margin_requirement
                        required_capital = max(0, margin_change) + trade_cost

                if cash >= required_capital or position != 0:
                    # Close existing position
                    if position != 0 and shares != 0:
                        # Realize P&L based on shares
                        realized_pnl = shares * (current_price - position_entry_price)
                        # Return margin to cash plus P&L
                        cash += margin_requirement + realized_pnl - trade_cost / 2

                        # Record trade
                        self.trades.append({
                            'date': date,
                            'type': 'close',
                            'size': -position,
                            'shares': -shares,
                            'price': current_price,
                            'value': abs(position),
                            'realized_pnl': realized_pnl,
                            'cost': trade_cost / 2
                        })

                        # Reset position tracking
                        position = 0
                        shares = 0
                        position_entry_price = 0
                        margin_requirement = 0

                    # Open new position
                    if target_position != 0:
                        # Check if we have enough cash for margin (already calculated above)
                        if cash >= new_margin_required + trade_cost / 2:
                            # Calculate shares based on target position value
                            shares = target_position / current_price
                            position = target_position
                            position_entry_price = current_price
                            margin_requirement = new_margin_required
                            cash -= margin_requirement + trade_cost / 2  # Post margin + costs

                            # Record trade
                            self.trades.append({
                                'date': date,
                                'type': 'open',
                                'size': position,
                                'shares': shares,
                                'price': current_price,
                                'value': abs(position),
                                'margin_posted': margin_requirement,
                                'realized_pnl': 0,
                                'cost': trade_cost / 2
                            })

                            logger.debug(f"{date}: {'Long' if position > 0 else 'Short'} "
                                       f"position opened - Shares: {abs(shares):.2f}, "
                                       f"Value: ${abs(position):,.0f}, Margin: ${margin_requirement:,.0f}")
                        else:
                            # Not enough cash for margin, skip trade
                            logger.warning(f"{date}: Insufficient margin. Required: ${new_margin_required:,.0f}, Available: ${cash:,.0f}")
                            shares = 0
                            position = 0
                            position_entry_price = 0
                            margin_requirement = 0
                            target_position = 0  # Don't try to open position
                    else:
                        position = 0
                        shares = 0
                        position_entry_price = 0
                        margin_requirement = 0

            # Record daily results
            results.append({
                'date': date,
                'close_price': current_price,
                'signal': row['signal'],
                'position': position,
                'cash': cash,
                'portfolio_value': portfolio_value,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': self.trades[-1]['realized_pnl'] if self.trades and self.trades[-1]['date'] == date else 0,
                'volatility_percentile': row.get('volatility_percentile', np.nan),
                'asset_volatility': asset_volatility,
                'drawdown_breach': drawdown_breach,
                'days_since_signal': days_since_signal
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)

        # Calculate additional metrics
        results_df['returns'] = results_df['portfolio_value'].pct_change()
        results_df['cumulative_returns'] = (1 + results_df['returns']).cumprod() - 1
        results_df['drawdown'] = self._calculate_drawdown_series(results_df['portfolio_value'])

        # Log summary
        total_return = (results_df['portfolio_value'].iloc[-1] / initial_capital - 1) * 100
        n_trades = len(self.trades)
        logger.info(f"Backtest complete - Total return: {total_return:.1f}%, "
                   f"Final value: ${results_df['portfolio_value'].iloc[-1]:,.0f}, "
                   f"Trades: {n_trades}")

        return results_df

    def calculate_asset_volatility(self, price_series: pd.Series,
                                  current_date: pd.Timestamp,
                                  window: int = 60) -> float:
        """
        Calculate rolling realized volatility of the asset.

        Args:
            price_series: Series of asset prices
            current_date: Current date
            window: Look-back window (days)

        Returns:
            Annualized volatility
        """
        # Get historical prices up to current date
        historical_prices = price_series[price_series.index <= current_date].tail(window + 1)

        if len(historical_prices) < 2:
            return 0.15  # Default volatility

        # Calculate returns
        returns = historical_prices.pct_change().dropna()

        if len(returns) < window / 2:
            return 0.15  # Default if insufficient data

        # Calculate and annualize volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        # Cap volatility at reasonable levels
        annual_vol = np.clip(annual_vol, 0.05, 0.50)

        return annual_vol

    def _prepare_backtest_data(self, signals_df: pd.DataFrame,
                              price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and align data for backtesting.

        Args:
            signals_df: Monthly signals DataFrame
            price_data: Daily price DataFrame

        Returns:
            Aligned DataFrame for backtesting
        """
        # Ensure datetime indices and remove timezone info for comparison
        signals_df.index = pd.to_datetime(signals_df.index)
        price_data.index = pd.to_datetime(price_data.index)

        # Remove timezone info if present
        if signals_df.index.tz is not None:
            signals_df.index = signals_df.index.tz_localize(None)
        if price_data.index.tz is not None:
            price_data.index = price_data.index.tz_localize(None)

        # Get overlapping date range
        start_date = max(signals_df.index.min(), price_data.index.min())
        end_date = min(signals_df.index.max(), price_data.index.max())

        # Filter to overlapping period
        signals_filtered = signals_df[start_date:end_date]
        prices_filtered = price_data[start_date:end_date]

        # Reindex signals to daily frequency (forward fill)
        daily_index = pd.date_range(start=start_date, end=end_date, freq='D')

        # Forward fill signals
        signals_daily = signals_filtered.reindex(daily_index, method='ffill')

        # Merge with price data
        backtest_data = pd.merge(
            prices_filtered[['close']],
            signals_daily,
            left_index=True,
            right_index=True,
            how='inner'
        )

        # Mark signal changes
        backtest_data['signal_change'] = (
            backtest_data['signal'] != backtest_data['signal'].shift(1)
        )

        # Drop any remaining NaN values in critical columns
        backtest_data = backtest_data.dropna(subset=['close', 'signal'])

        logger.info(f"Prepared backtest data: {len(backtest_data)} days, "
                   f"from {backtest_data.index[0]} to {backtest_data.index[-1]}")

        return backtest_data

    def _calculate_drawdown_series(self, portfolio_values: pd.Series) -> pd.Series:
        """
        Calculate drawdown series from portfolio values.

        Args:
            portfolio_values: Series of portfolio values

        Returns:
            Series of drawdown percentages
        """
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        return drawdown

    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics from executed trades.

        Returns:
            Dictionary of trade statistics
        """
        if not self.trades:
            return {'n_trades': 0}

        trades_df = pd.DataFrame(self.trades)

        # Separate opens and closes
        opens = trades_df[trades_df['type'] == 'open']
        closes = trades_df[trades_df['type'] == 'close']

        stats = {
            'n_trades': len(closes),
            'n_long_trades': len(closes[closes['size'] > 0]),
            'n_short_trades': len(closes[closes['size'] < 0]),
            'total_realized_pnl': closes['realized_pnl'].sum() if not closes.empty else 0,
            'total_costs': trades_df['cost'].sum(),
            'avg_trade_size': opens['value'].mean() if not opens.empty else 0
        }

        # Win/loss statistics
        if not closes.empty:
            winning_trades = closes[closes['realized_pnl'] > 0]
            losing_trades = closes[closes['realized_pnl'] < 0]

            stats['n_winners'] = len(winning_trades)
            stats['n_losers'] = len(losing_trades)
            stats['win_rate'] = len(winning_trades) / len(closes) if len(closes) > 0 else 0

            stats['avg_win'] = winning_trades['realized_pnl'].mean() if not winning_trades.empty else 0
            stats['avg_loss'] = losing_trades['realized_pnl'].mean() if not losing_trades.empty else 0
            stats['largest_win'] = winning_trades['realized_pnl'].max() if not winning_trades.empty else 0
            stats['largest_loss'] = losing_trades['realized_pnl'].min() if not losing_trades.empty else 0

            # Profit factor
            total_wins = winning_trades['realized_pnl'].sum() if not winning_trades.empty else 0
            total_losses = abs(losing_trades['realized_pnl'].sum()) if not losing_trades.empty else 0
            stats['profit_factor'] = total_wins / total_losses if total_losses > 0 else np.inf if total_wins > 0 else 0

        return stats

    def create_trade_log(self) -> pd.DataFrame:
        """
        Create detailed trade log DataFrame.

        Returns:
            DataFrame with all trade details
        """
        if not self.trades:
            return pd.DataFrame()

        trade_log = pd.DataFrame(self.trades)
        trade_log.set_index('date', inplace=True)

        # Add cumulative P&L
        trade_log['cumulative_pnl'] = trade_log['realized_pnl'].cumsum()
        trade_log['cumulative_cost'] = trade_log['cost'].cumsum()
        trade_log['net_pnl'] = trade_log['cumulative_pnl'] - trade_log['cumulative_cost']

        return trade_log