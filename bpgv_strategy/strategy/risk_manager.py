"""Risk management with drawdown controls."""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk through drawdown monitoring and exposure controls."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.risk_params = config['risk_management']
        self.max_drawdown_pct = self.risk_params['max_drawdown_pct']
        self.max_position_loss_pct = self.risk_params.get('max_position_loss_pct', 0.05)
        self.max_exposure = config['position_sizing']['max_exposure']

        # Track state
        self.peak_value = 0.0
        self.in_drawdown_breach = False
        self.drawdown_breach_date = None
        self.drawdown_history = []

        # Track position entry for stop loss
        self.position_entry_value = None
        self.position_entry_date = None

    def check_drawdown_breach(self, current_value: float,
                             peak_value: Optional[float] = None) -> bool:
        """
        Check if drawdown exceeds maximum allowed threshold.

        Args:
            current_value: Current portfolio value
            peak_value: Peak portfolio value (uses internal if not provided)

        Returns:
            True if drawdown breached, False otherwise
        """
        # Update peak if not provided
        if peak_value is None:
            peak_value = self.peak_value

        # Ensure peak is updated
        if current_value > peak_value:
            peak_value = current_value

        # Calculate drawdown
        if peak_value > 0:
            drawdown = (peak_value - current_value) / peak_value
        else:
            drawdown = 0.0

        # Check breach
        breach = drawdown > self.max_drawdown_pct

        # Log if breach status changed
        if breach and not self.in_drawdown_breach:
            logger.warning(f"Drawdown breach detected: {drawdown:.1%} > {self.max_drawdown_pct:.1%}")
            self.in_drawdown_breach = True
            self.drawdown_breach_date = pd.Timestamp.now()
        elif not breach and self.in_drawdown_breach:
            logger.info(f"Recovered from drawdown breach. Current drawdown: {drawdown:.1%}")
            self.in_drawdown_breach = False

        # Store drawdown for history
        self.drawdown_history.append({
            'value': current_value,
            'peak': peak_value,
            'drawdown': drawdown,
            'breached': breach
        })

        return breach

    def update_peak(self, current_value: float) -> float:
        """
        Update peak portfolio value.

        Args:
            current_value: Current portfolio value

        Returns:
            Updated peak value
        """
        if current_value > self.peak_value:
            self.peak_value = current_value
            logger.debug(f"New portfolio peak: ${self.peak_value:,.0f}")

        return self.peak_value

    def should_reduce_exposure(self, signals_df: pd.DataFrame,
                              portfolio_values: pd.Series) -> pd.Series:
        """
        Determine exposure reduction factor based on risk conditions.

        Args:
            signals_df: DataFrame with signals and regime indicators
            portfolio_values: Series of portfolio values

        Returns:
            Series of exposure multipliers (0.0 to 1.0)
        """
        exposure_multiplier = pd.Series(1.0, index=signals_df.index)

        # Check drawdown for each period
        peak_value = 0.0
        for i, (date, value) in enumerate(portfolio_values.items()):
            # Update peak
            peak_value = max(peak_value, value)

            # Check drawdown breach
            if self.check_drawdown_breach(value, peak_value):
                # Set exposure to 0 during drawdown breach
                exposure_multiplier.iloc[i] = 0.0

            # Check regime filter if available
            if 'risk_off_regime' in signals_df.columns:
                if signals_df.loc[date, 'risk_off_regime']:
                    # Reduce exposure during risk-off regime
                    exposure_multiplier.iloc[i] *= 0.5

        logger.info(f"Exposure adjustments - Periods with reduction: {(exposure_multiplier < 1.0).sum()}")

        return exposure_multiplier

    def calculate_portfolio_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio risk metrics.

        Args:
            portfolio_values: Series of portfolio values

        Returns:
            Dictionary of risk metrics
        """
        metrics = {}

        # Returns
        returns = portfolio_values.pct_change().dropna()

        if len(returns) == 0:
            return metrics

        # Volatility
        metrics['volatility'] = returns.std()
        metrics['annualized_volatility'] = metrics['volatility'] * np.sqrt(252)

        # Drawdown metrics
        rolling_max = portfolio_values.expanding().max()
        drawdown_series = (portfolio_values - rolling_max) / rolling_max

        metrics['max_drawdown'] = drawdown_series.min()
        metrics['avg_drawdown'] = drawdown_series[drawdown_series < 0].mean() if any(drawdown_series < 0) else 0
        metrics['current_drawdown'] = drawdown_series.iloc[-1]

        # Drawdown duration
        in_drawdown = drawdown_series < 0
        drawdown_periods = in_drawdown.astype(int).groupby((~in_drawdown).cumsum()).sum()
        metrics['max_drawdown_duration'] = drawdown_periods.max() if len(drawdown_periods) > 0 else 0
        metrics['avg_drawdown_duration'] = drawdown_periods[drawdown_periods > 0].mean() if any(drawdown_periods > 0) else 0

        # Value at Risk (95% confidence)
        metrics['var_95'] = returns.quantile(0.05)

        # Conditional Value at Risk (CVaR)
        var_threshold = metrics['var_95']
        tail_losses = returns[returns <= var_threshold]
        metrics['cvar_95'] = tail_losses.mean() if len(tail_losses) > 0 else var_threshold

        # Downside deviation
        negative_returns = returns[returns < 0]
        metrics['downside_deviation'] = negative_returns.std() if len(negative_returns) > 0 else 0

        # Ulcer Index (measure of drawdown severity and duration)
        squared_drawdowns = drawdown_series ** 2
        metrics['ulcer_index'] = np.sqrt(squared_drawdowns.mean())

        return metrics

    def generate_risk_report(self, portfolio_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive risk report.

        Args:
            portfolio_data: DataFrame with portfolio values and positions

        Returns:
            DataFrame with risk metrics over time
        """
        report = pd.DataFrame(index=portfolio_data.index)

        # Portfolio value and returns
        report['portfolio_value'] = portfolio_data['portfolio_value']
        report['returns'] = report['portfolio_value'].pct_change()

        # Running peak and drawdown
        report['peak_value'] = report['portfolio_value'].expanding().max()
        report['drawdown'] = (report['portfolio_value'] - report['peak_value']) / report['peak_value']

        # Rolling volatility (20-day)
        report['rolling_volatility'] = report['returns'].rolling(window=20).std() * np.sqrt(252)

        # Drawdown breach indicator
        report['drawdown_breach'] = report['drawdown'] < -self.max_drawdown_pct

        # Position exposure if available
        if 'position_size' in portfolio_data.columns and 'portfolio_value' in portfolio_data.columns:
            report['exposure'] = abs(portfolio_data['position_size']) / portfolio_data['portfolio_value']
        else:
            report['exposure'] = 0

        # Risk-adjusted metrics
        if 'exposure' in report.columns:
            report['risk_adjusted_return'] = report['returns'] / (report['exposure'] + 0.001)  # Avoid division by zero

        # Mark risk events
        report['risk_event'] = (
            (report['drawdown_breach']) |
            (report['rolling_volatility'] > self.target_annual_vol * 1.5) if hasattr(self, 'target_annual_vol') else False
        )

        # Summary statistics
        logger.info(f"Risk report summary - "
                   f"Max drawdown: {report['drawdown'].min():.1%}, "
                   f"Breach days: {report['drawdown_breach'].sum()}, "
                   f"Avg exposure: {report['exposure'].mean():.1%}")

        return report

    def check_position_stop_loss(self, current_position: float,
                                current_value: float,
                                portfolio_value: float) -> bool:
        """
        Check if position has hit stop loss.

        Args:
            current_position: Current position size
            current_value: Current position value (unrealized P&L)
            portfolio_value: Current portfolio value

        Returns:
            True if stop loss hit, False otherwise
        """
        if current_position == 0 or self.position_entry_value is None:
            return False

        # Calculate position P&L
        position_pnl = current_value - self.position_entry_value
        position_loss_pct = position_pnl / portfolio_value

        # Check if loss exceeds threshold
        if position_loss_pct < -self.max_position_loss_pct:
            logger.warning(f"Stop loss triggered: Position loss {position_loss_pct:.1%} exceeds max {-self.max_position_loss_pct:.1%}")
            return True

        return False

    def update_position_entry(self, portfolio_value: float, position_size: float, date: Optional[pd.Timestamp] = None):
        """
        Update position entry tracking for stop loss calculation.

        Args:
            portfolio_value: Portfolio value when position entered
            position_size: Size of position
            date: Entry date
        """
        if position_size != 0:
            self.position_entry_value = portfolio_value
            self.position_entry_date = date or pd.Timestamp.now()
            logger.debug(f"Position entry tracked: ${portfolio_value:,.0f} at {self.position_entry_date}")
        else:
            # Clear position entry when flat
            self.position_entry_value = None
            self.position_entry_date = None

    def check_position_limits(self, current_position: float,
                            new_position: float,
                            portfolio_value: float) -> Tuple[bool, str]:
        """
        Check if new position respects risk limits.

        Args:
            current_position: Current position size
            new_position: Proposed new position size
            portfolio_value: Current portfolio value

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Check absolute exposure limit (use max_exposure, not drawdown limit)
        max_position = self.max_exposure * portfolio_value
        if abs(new_position) > max_position:
            return False, f"Position size ${abs(new_position):,.0f} exceeds max ${max_position:,.0f}"

        # Check if in drawdown breach
        if self.in_drawdown_breach and new_position != 0:
            return False, "Cannot open positions during drawdown breach"

        # Check position flip limit (avoid whipsaws)
        if current_position != 0 and new_position != 0:
            if np.sign(current_position) != np.sign(new_position):
                position_change = abs(new_position - current_position)
                if position_change > 2 * max_position:
                    return False, f"Position flip too large: ${position_change:,.0f}"

        return True, "Position within limits"

    def calculate_kelly_fraction(self, win_rate: float,
                                avg_win: float,
                                avg_loss: float) -> float:
        """
        Calculate Kelly criterion for position sizing.

        Args:
            win_rate: Probability of winning
            avg_win: Average win amount
            avg_loss: Average loss amount (positive number)

        Returns:
            Kelly fraction (capped at reasonable levels)
        """
        if avg_loss == 0 or win_rate == 0 or win_rate == 1:
            return 0.0

        # Kelly formula: f = (p*b - q) / b
        # where p = win_rate, q = 1-p, b = avg_win/avg_loss
        loss_rate = 1 - win_rate
        win_loss_ratio = avg_win / avg_loss

        kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio

        # Cap Kelly fraction for safety (typically 25% of full Kelly)
        kelly_capped = np.clip(kelly * 0.25, 0, self.max_drawdown_pct)

        return kelly_capped

    def reset_risk_state(self):
        """Reset internal risk management state."""
        self.peak_value = 0.0
        self.in_drawdown_breach = False
        self.drawdown_breach_date = None
        self.drawdown_history = []
        logger.info("Risk manager state reset")