"""Regime detection based on unemployment data."""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detects economic regimes based on unemployment trends."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize regime detector with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.unemployment_ma_window = config['risk_management']['unemployment_ma_window']

    def detect_unemployment_regime(self, unemployment_data: pd.Series) -> pd.DataFrame:
        """
        Detect risk-off regime based on unemployment rate trend.

        When unemployment > 12-month moving average, it signals economic weakness.

        Args:
            unemployment_data: Series of unemployment rates

        Returns:
            DataFrame with regime indicators
        """
        df = pd.DataFrame({'unemployment': unemployment_data})

        # Calculate moving average
        df['unemployment_ma'] = df['unemployment'].rolling(
            window=self.unemployment_ma_window,
            min_periods=1
        ).mean()

        # Detect risk-off regime
        df['risk_off_regime'] = df['unemployment'] > df['unemployment_ma']

        # Calculate regime strength (how far above/below MA)
        df['regime_strength'] = (df['unemployment'] - df['unemployment_ma']) / df['unemployment_ma']

        # Add regime duration counter
        df['regime_change'] = df['risk_off_regime'] != df['risk_off_regime'].shift(1)
        df['regime_duration'] = df.groupby(df['regime_change'].cumsum()).cumcount() + 1

        logger.info(f"Detected {df['risk_off_regime'].sum()} risk-off periods out of {len(df)} observations")

        return df

    def apply_regime_filter(self, signals: pd.DataFrame,
                           unemployment_regime: pd.DataFrame) -> pd.DataFrame:
        """
        Apply regime filter to trading signals.

        Force signal = 0 (neutral) during risk-off regimes.

        Args:
            signals: DataFrame with trading signals
            unemployment_regime: DataFrame with regime indicators

        Returns:
            DataFrame with filtered signals
        """
        # Merge signals with regime data
        filtered = pd.merge(
            signals,
            unemployment_regime[['risk_off_regime', 'regime_strength']],
            left_index=True,
            right_index=True,
            how='left'
        )

        # Store original signals for analysis
        filtered['signal_original'] = filtered['signal'].copy()

        # Apply filter: force neutral during risk-off regime
        filtered.loc[filtered['risk_off_regime'] == True, 'signal'] = 0

        # Log filtering impact
        n_filtered = (filtered['signal'] != filtered['signal_original']).sum()
        if n_filtered > 0:
            logger.info(f"Regime filter neutralized {n_filtered} signals")

        return filtered

    def calculate_regime_statistics(self, regime_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate statistics about detected regimes.

        Args:
            regime_data: DataFrame with regime indicators

        Returns:
            Dictionary of regime statistics
        """
        stats = {}

        if 'risk_off_regime' in regime_data.columns:
            # Basic statistics
            stats['pct_risk_off'] = (regime_data['risk_off_regime'].sum() / len(regime_data)) * 100
            stats['n_regime_changes'] = regime_data['regime_change'].sum() if 'regime_change' in regime_data else 0

            # Average regime duration
            if 'regime_duration' in regime_data.columns:
                risk_off_periods = regime_data[regime_data['risk_off_regime']]
                if not risk_off_periods.empty:
                    # Get max duration for each regime period
                    regime_groups = risk_off_periods.groupby(
                        (risk_off_periods['regime_change']).cumsum()
                    )
                    stats['avg_risk_off_duration'] = regime_groups['regime_duration'].max().mean()
                else:
                    stats['avg_risk_off_duration'] = 0

            # Regime strength statistics
            if 'regime_strength' in regime_data.columns:
                stats['avg_regime_strength'] = regime_data['regime_strength'].mean()
                stats['max_regime_strength'] = regime_data['regime_strength'].max()
                stats['min_regime_strength'] = regime_data['regime_strength'].min()

        logger.info(f"Regime statistics: {stats}")
        return stats

    def identify_recession_periods(self, unemployment_data: pd.Series,
                                  threshold: float = 5.0) -> pd.Series:
        """
        Identify potential recession periods based on unemployment levels.

        Args:
            unemployment_data: Series of unemployment rates
            threshold: Unemployment rate threshold for recession (default 5%)

        Returns:
            Series of boolean values indicating recession periods
        """
        # Simple recession indicator: unemployment > threshold and rising
        unemployment_change = unemployment_data.diff()
        recession = (unemployment_data > threshold) & (unemployment_change > 0)

        # Smooth with 3-month window to avoid noise
        recession_smoothed = recession.rolling(window=3, min_periods=1).mean() > 0.5

        n_recession_periods = recession_smoothed.sum()
        logger.info(f"Identified {n_recession_periods} potential recession periods")

        return recession_smoothed

    def combine_regime_indicators(self, unemployment_regime: pd.DataFrame,
                                 additional_indicators: Optional[Dict[str, pd.Series]] = None) -> pd.DataFrame:
        """
        Combine multiple regime indicators for comprehensive risk assessment.

        Args:
            unemployment_regime: DataFrame with unemployment regime indicators
            additional_indicators: Optional dict of additional indicators

        Returns:
            DataFrame with combined regime assessment
        """
        combined = unemployment_regime.copy()

        # Add additional indicators if provided
        if additional_indicators:
            for name, indicator in additional_indicators.items():
                combined[name] = indicator

        # Create composite risk score (0-100)
        risk_components = []

        # Unemployment regime contributes 50%
        if 'risk_off_regime' in combined.columns:
            risk_components.append(combined['risk_off_regime'].astype(float) * 50)

        # Regime strength contributes 25%
        if 'regime_strength' in combined.columns:
            # Normalize regime strength to 0-25 scale
            strength_normalized = combined['regime_strength'].clip(-1, 1)
            risk_components.append((strength_normalized + 1) * 12.5)

        # Regime duration contributes 25%
        if 'regime_duration' in combined.columns:
            # Longer regimes indicate more persistent risk
            duration_normalized = combined['regime_duration'].clip(0, 12) / 12
            risk_components.append(duration_normalized * 25)

        if risk_components:
            combined['composite_risk_score'] = sum(risk_components)
            combined['high_risk'] = combined['composite_risk_score'] > 50

            logger.info(f"Composite risk score - Mean: {combined['composite_risk_score'].mean():.1f}, "
                       f"High risk periods: {combined['high_risk'].sum()}")

        return combined