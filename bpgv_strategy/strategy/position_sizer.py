"""Position sizing logic with volatility targeting."""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculates position sizes using volatility targeting approach."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize position sizer with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.sizing_params = config['position_sizing']
        self.max_exposure = self.sizing_params['max_exposure']
        self.target_annual_vol = self.sizing_params['target_annual_vol']
        self.min_scale_factor = self.sizing_params['min_scale_factor']
        self.max_scale_factor = self.sizing_params['max_scale_factor']

        # Get contract specifications
        self.contract_specs = config.get('futures_contracts', {})

    def get_contract_multiplier(self, ticker: str) -> float:
        """
        Get contract multiplier for a futures contract.

        Args:
            ticker: Futures ticker symbol

        Returns:
            Contract multiplier (default 50 if not found)
        """
        if ticker in self.contract_specs:
            return self.contract_specs[ticker].get('multiplier', 50)

        # Default multipliers if not in config
        logger.warning(f"Contract spec not found for {ticker}, using default multiplier of 50")
        return 50

    def calculate_position_size(self, signal: int,
                               volatility_percentile: float,
                               portfolio_value: float,
                               asset_volatility: float,
                               regime_scale: float = 1.0) -> float:
        """
        Calculate position size using volatility targeting with signal strength scaling.

        Base sizing: Volatility targeting at 10% annualized portfolio volatility
        Signal strength scaling:
            - For long signals: More aggressive as volatility approaches 5th percentile
            - For short signals: More aggressive as volatility approaches 95th percentile
            - Scale factor range: 0.5x to 1.5x of base size

        Args:
            signal: Trading signal (-1, 0, 1)
            volatility_percentile: Percentile rank of current volatility (0-100)
            portfolio_value: Current portfolio value
            asset_volatility: Annualized asset volatility
            regime_scale: Scaling factor from regime filter (0-1)

        Returns:
            Signed position size (negative for shorts)
        """
        # No position if neutral signal
        if signal == 0:
            return 0.0

        # Validate inputs
        if asset_volatility <= 0:
            logger.warning("Invalid asset volatility, returning zero position")
            return 0.0

        if portfolio_value <= 0:
            logger.warning("Invalid portfolio value, returning zero position")
            return 0.0

        # Calculate base position size using volatility targeting
        base_size = self._calculate_base_size(portfolio_value, asset_volatility)

        # Calculate signal strength based on percentile
        signal_strength = self._calculate_signal_strength(signal, volatility_percentile)

        # Calculate scale factor (0.5 to 1.5x)
        scale_factor = self.min_scale_factor + signal_strength * (self.max_scale_factor - self.min_scale_factor)

        # Apply scale factor to base size
        position_size = base_size * scale_factor * signal

        # Apply regime scaling (for unemployment filter)
        position_size *= regime_scale

        # Apply maximum exposure cap
        position_size = self._apply_exposure_cap(position_size, portfolio_value)

        logger.debug(f"Position size calculated - Signal: {signal}, "
                    f"Vol percentile: {volatility_percentile:.1f}, "
                    f"Base size: ${base_size:,.0f}, "
                    f"Scale: {scale_factor:.2f}, "
                    f"Regime scale: {regime_scale:.2f}, "
                    f"Final: ${position_size:,.0f}")

        return position_size

    def _calculate_base_size(self, portfolio_value: float,
                            asset_volatility: float) -> float:
        """
        Calculate base position size using volatility targeting.

        Formula: position_size = (target_vol / asset_vol) * portfolio_value

        Args:
            portfolio_value: Current portfolio value
            asset_volatility: Annualized asset volatility

        Returns:
            Base position size (unsigned)
        """
        base_size = (self.target_annual_vol / asset_volatility) * portfolio_value

        return base_size

    def _calculate_signal_strength(self, signal: int,
                                  volatility_percentile: float) -> float:
        """
        Calculate signal strength based on volatility percentile.

        For long signals: Stronger as volatility approaches 0th percentile
        For short signals: Stronger as volatility approaches 100th percentile

        Args:
            signal: Trading signal (-1 or 1)
            volatility_percentile: Percentile rank (0-100)

        Returns:
            Signal strength (0-1)
        """
        if signal == 1:  # Long signal
            # Strongest at 0th percentile, weakest at 25th percentile
            if volatility_percentile <= 5:
                strength = 1.0
            elif volatility_percentile >= 25:
                strength = 0.0
            else:
                # Linear interpolation between 5th and 25th percentile
                strength = 1.0 - ((volatility_percentile - 5) / 20)

        elif signal == -1:  # Short signal
            # Strongest at 100th percentile, weakest at 75th percentile
            if volatility_percentile >= 95:
                strength = 1.0
            elif volatility_percentile <= 75:
                strength = 0.0
            else:
                # Linear interpolation between 75th and 95th percentile
                strength = (volatility_percentile - 75) / 20

        else:
            strength = 0.0

        return np.clip(strength, 0.0, 1.0)

    def _apply_exposure_cap(self, position_size: float,
                           portfolio_value: float) -> float:
        """
        Apply maximum exposure cap to position size.

        Args:
            position_size: Calculated position size (signed)
            portfolio_value: Current portfolio value

        Returns:
            Capped position size
        """
        max_absolute_position = self.max_exposure * portfolio_value

        # Apply cap while preserving sign
        if abs(position_size) > max_absolute_position:
            logger.debug(f"Position capped from ${abs(position_size):,.0f} to ${max_absolute_position:,.0f}")
            position_size = np.sign(position_size) * max_absolute_position

        return position_size

    def calculate_leverage(self, position_size: float,
                          portfolio_value: float) -> float:
        """
        Calculate implied leverage from position size.

        Args:
            position_size: Position size (signed)
            portfolio_value: Portfolio value

        Returns:
            Leverage ratio
        """
        if portfolio_value == 0:
            return 0.0

        return abs(position_size) / portfolio_value

    def calculate_position_volatility(self, position_size: float,
                                     asset_volatility: float,
                                     portfolio_value: float) -> float:
        """
        Calculate expected portfolio volatility from position.

        Args:
            position_size: Position size (signed)
            asset_volatility: Asset volatility
            portfolio_value: Portfolio value

        Returns:
            Expected portfolio volatility
        """
        if portfolio_value == 0:
            return 0.0

        position_weight = abs(position_size) / portfolio_value
        portfolio_vol = position_weight * asset_volatility

        return portfolio_vol

    def create_position_sizing_report(self, signals_df: pd.DataFrame,
                                     portfolio_value: float,
                                     asset_volatility_series: pd.Series) -> pd.DataFrame:
        """
        Create detailed position sizing report for all signals.

        Args:
            signals_df: DataFrame with signals and volatility percentiles
            portfolio_value: Starting portfolio value
            asset_volatility_series: Series of asset volatilities

        Returns:
            DataFrame with position sizing details
        """
        report = pd.DataFrame(index=signals_df.index)

        # Copy relevant columns
        report['signal'] = signals_df['signal']
        report['volatility_percentile'] = signals_df.get('volatility_percentile', 50)

        # Align asset volatility
        report['asset_volatility'] = asset_volatility_series.reindex(report.index, method='ffill')

        # Calculate position sizes
        position_sizes = []
        leverages = []
        portfolio_vols = []

        current_portfolio_value = portfolio_value

        for idx, row in report.iterrows():
            # Calculate position size
            position_size = self.calculate_position_size(
                signal=row['signal'],
                volatility_percentile=row['volatility_percentile'],
                portfolio_value=current_portfolio_value,
                asset_volatility=row['asset_volatility']
            )

            # Calculate metrics
            leverage = self.calculate_leverage(position_size, current_portfolio_value)
            portfolio_vol = self.calculate_position_volatility(
                position_size,
                row['asset_volatility'],
                current_portfolio_value
            )

            position_sizes.append(position_size)
            leverages.append(leverage)
            portfolio_vols.append(portfolio_vol)

            # Note: Portfolio value would be updated in actual backtest

        report['position_size'] = position_sizes
        report['leverage'] = leverages
        report['portfolio_volatility'] = portfolio_vols

        # Add signal strength
        report['signal_strength'] = report.apply(
            lambda row: self._calculate_signal_strength(
                row['signal'],
                row['volatility_percentile']
            ),
            axis=1
        )

        # Calculate statistics
        active_positions = report[report['signal'] != 0]
        if not active_positions.empty:
            logger.info(f"Position sizing report - "
                       f"Avg leverage: {active_positions['leverage'].mean():.2f}x, "
                       f"Max leverage: {active_positions['leverage'].max():.2f}x, "
                       f"Avg portfolio vol: {active_positions['portfolio_volatility'].mean():.1%}")

        return report

    def adjust_for_correlation(self, position_sizes: Dict[str, float],
                              correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Adjust position sizes based on asset correlations (for multi-asset strategies).

        Args:
            position_sizes: Dictionary of asset positions
            correlation_matrix: Correlation matrix between assets

        Returns:
            Adjusted position sizes
        """
        # This is a placeholder for potential multi-asset extension
        # For single asset (ES futures), just return original sizes
        return position_sizes