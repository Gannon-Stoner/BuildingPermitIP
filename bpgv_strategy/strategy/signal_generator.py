"""Signal generation logic for BPGV trading strategy."""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates trading signals based on volatility forecasts and percentiles."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize signal generator with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.signal_params = config['signals']
        self.top_quartile = self.signal_params['top_quartile']
        self.bottom_quartile = self.signal_params['bottom_quartile']
        self.min_history = self.signal_params.get('min_history_for_percentiles', 60)

    def generate_signals(self, volatility_forecasts: pd.Series) -> pd.DataFrame:
        """
        Generate trading signals based on volatility percentiles.

        Top quartile (75th percentile) → Signal = -1 (short/risk-off)
        Bottom quartile (25th percentile) → Signal = 1 (long)
        Middle range → Signal = 0 (neutral)

        Args:
            volatility_forecasts: Series of volatility forecasts

        Returns:
            DataFrame with signals and related metrics
        """
        # Initialize results DataFrame
        signals_df = pd.DataFrame(index=volatility_forecasts.index)
        signals_df['volatility'] = volatility_forecasts

        # Calculate expanding percentile ranks
        signals_df['volatility_percentile'] = self._calculate_expanding_percentiles(
            volatility_forecasts
        )

        # Generate signals based on percentiles
        signals_df['signal'] = 0  # Initialize as neutral

        # Long signal when volatility is in bottom quartile (low volatility = bullish)
        signals_df.loc[
            signals_df['volatility_percentile'] <= (self.bottom_quartile * 100),
            'signal'
        ] = 1

        # Short signal when volatility is in top quartile (high volatility = bearish)
        signals_df.loc[
            signals_df['volatility_percentile'] >= (self.top_quartile * 100),
            'signal'
        ] = -1

        # Add signal strength (0-1 scale based on distance from quartile boundaries)
        signals_df['signal_strength'] = self._calculate_signal_strength(
            signals_df['volatility_percentile'],
            signals_df['signal']
        )

        # Add quartile indicators
        signals_df['in_top_quartile'] = signals_df['volatility_percentile'] >= (self.top_quartile * 100)
        signals_df['in_bottom_quartile'] = signals_df['volatility_percentile'] <= (self.bottom_quartile * 100)

        # Mark periods with insufficient history
        # For GARCH forecasts, we need at least a few observations to calculate percentiles meaningfully
        min_obs_for_percentiles = min(10, len(signals_df) // 4)  # At least 10 obs or 25% of data
        signals_df['sufficient_history'] = False
        if len(signals_df) > min_obs_for_percentiles:
            signals_df.loc[signals_df.index[min_obs_for_percentiles:], 'sufficient_history'] = True

        # Don't generate signals without sufficient history
        signals_df.loc[~signals_df['sufficient_history'], 'signal'] = 0
        signals_df.loc[~signals_df['sufficient_history'], 'signal_strength'] = 0

        # Log signal distribution
        signal_counts = signals_df['signal'].value_counts()
        logger.info(f"Generated signals - Long: {signal_counts.get(1, 0)}, "
                   f"Short: {signal_counts.get(-1, 0)}, "
                   f"Neutral: {signal_counts.get(0, 0)}")

        return signals_df

    def _calculate_expanding_percentiles(self, volatility: pd.Series) -> pd.Series:
        """
        Calculate expanding percentile ranks of volatility.

        Args:
            volatility: Series of volatility values

        Returns:
            Series of percentile ranks (0-100)
        """
        # Use pandas expanding window for proper percentile calculation
        percentiles = volatility.expanding(min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )

        return percentiles

    def _calculate_signal_strength(self, percentiles: pd.Series,
                                  signals: pd.Series) -> pd.Series:
        """
        Calculate signal strength based on distance from quartile boundaries.

        For long signals: Stronger as percentile approaches 0
        For short signals: Stronger as percentile approaches 100

        Args:
            percentiles: Series of percentile values (0-100)
            signals: Series of signal values (-1, 0, 1)

        Returns:
            Series of signal strengths (0-1)
        """
        strength = pd.Series(index=percentiles.index, dtype=float)
        strength[:] = 0.0

        # Long signal strength (percentile 0-25)
        long_mask = signals == 1
        strength[long_mask] = 1.0 - (percentiles[long_mask] / (self.bottom_quartile * 100))

        # Short signal strength (percentile 75-100)
        short_mask = signals == -1
        strength[short_mask] = (percentiles[short_mask] - (self.top_quartile * 100)) / (100 - self.top_quartile * 100)

        # Clip to [0, 1] range
        strength = strength.clip(0, 1)

        return strength

    def add_regime_filter(self, signals_df: pd.DataFrame,
                         unemployment_data: pd.Series) -> pd.DataFrame:
        """
        Add regime filter based on unemployment trend.

        If current unemployment > 12-month MA → Force signal = 0 (risk-off)

        Args:
            signals_df: DataFrame with trading signals
            unemployment_data: Series of unemployment rates

        Returns:
            DataFrame with regime-filtered signals
        """
        # Calculate 12-month moving average
        unemployment_ma = unemployment_data.rolling(
            window=self.config['risk_management']['unemployment_ma_window'],
            min_periods=1
        ).mean()

        # Align unemployment data with signals
        aligned_unemployment = unemployment_data.reindex(signals_df.index, method='ffill')
        aligned_unemployment_ma = unemployment_ma.reindex(signals_df.index, method='ffill')

        # Add to signals DataFrame
        signals_df['unemployment'] = aligned_unemployment
        signals_df['unemployment_ma'] = aligned_unemployment_ma

        # Identify risk-off regime
        signals_df['risk_off_regime'] = signals_df['unemployment'] > signals_df['unemployment_ma']

        # Store original signals
        signals_df['signal_pre_filter'] = signals_df['signal'].copy()

        # Apply regime-based position scaling (not signal modification)
        # Get filter strength from config (0 = no filter, 0.5 = half strength, 1 = neutralize)
        filter_strength = self.config['risk_management'].get('unemployment_filter_strength', 0.5)
        risk_off_mask = signals_df['risk_off_regime'] == True

        # Add regime scaling factor for position sizing
        signals_df['regime_scale'] = 1.0
        scaling_factor = 1.0 - filter_strength  # 0.5 strength = 0.5 scaling (50% position)

        # During risk-off, scale DOWN positions (but keep signal direction)
        signals_df.loc[risk_off_mask, 'regime_scale'] = scaling_factor

        # For complete neutralization (backward compatibility)
        if filter_strength >= 1.0:
            signals_df.loc[risk_off_mask, 'signal'] = 0
            signals_df.loc[risk_off_mask, 'signal_strength'] = 0
        else:
            # Scale the signal strength but NOT the signal itself
            if 'signal_strength' in signals_df.columns:
                signals_df.loc[risk_off_mask, 'signal_strength'] *= scaling_factor

        # Log filter impact
        n_affected = risk_off_mask.sum()
        n_modified = (signals_df['signal'] != signals_df['signal_pre_filter']).sum()
        if n_affected > 0:
            logger.info(f"Regime filter scaled {n_affected} signals (modified {n_modified}) due to elevated unemployment")

        return signals_df

    def calculate_signal_statistics(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistics about generated signals.

        Args:
            signals_df: DataFrame with signals

        Returns:
            Dictionary of signal statistics
        """
        stats = {}

        # Basic counts
        signal_counts = signals_df['signal'].value_counts()
        stats['n_long_signals'] = int(signal_counts.get(1, 0))
        stats['n_short_signals'] = int(signal_counts.get(-1, 0))
        stats['n_neutral_signals'] = int(signal_counts.get(0, 0))
        stats['total_signals'] = len(signals_df)

        # Percentages
        total_non_neutral = stats['n_long_signals'] + stats['n_short_signals']
        if total_non_neutral > 0:
            stats['pct_long'] = (stats['n_long_signals'] / total_non_neutral) * 100
            stats['pct_short'] = (stats['n_short_signals'] / total_non_neutral) * 100
        else:
            stats['pct_long'] = 0
            stats['pct_short'] = 0

        # Signal strength statistics
        if 'signal_strength' in signals_df.columns:
            non_zero_strength = signals_df[signals_df['signal_strength'] > 0]['signal_strength']
            if len(non_zero_strength) > 0:
                stats['avg_signal_strength'] = float(non_zero_strength.mean())
                stats['max_signal_strength'] = float(non_zero_strength.max())
                stats['min_signal_strength'] = float(non_zero_strength.min())

        # Volatility statistics
        if 'volatility' in signals_df.columns:
            stats['avg_volatility'] = float(signals_df['volatility'].mean())
            stats['max_volatility'] = float(signals_df['volatility'].max())
            stats['min_volatility'] = float(signals_df['volatility'].min())

        # Regime filter impact
        if 'signal_pre_filter' in signals_df.columns:
            stats['n_filtered_by_regime'] = int((signals_df['signal'] != signals_df['signal_pre_filter']).sum())
            stats['pct_filtered'] = (stats['n_filtered_by_regime'] / len(signals_df)) * 100

        # Risk-off regime statistics
        if 'risk_off_regime' in signals_df.columns:
            stats['pct_risk_off_regime'] = (signals_df['risk_off_regime'].sum() / len(signals_df)) * 100

        logger.info(f"Signal statistics: {stats}")
        return stats

    def analyze_signal_transitions(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze transitions between different signal states.

        Args:
            signals_df: DataFrame with signals

        Returns:
            DataFrame with transition analysis
        """
        transitions = pd.DataFrame(index=signals_df.index)
        transitions['signal'] = signals_df['signal']

        # Identify signal changes
        transitions['signal_change'] = transitions['signal'] != transitions['signal'].shift(1)

        # Categorize transitions
        transitions['from_signal'] = transitions['signal'].shift(1)
        transitions['to_signal'] = transitions['signal']

        # Create transition labels
        def get_transition_label(row):
            if pd.isna(row['from_signal']):
                return 'initial'
            if not row['signal_change']:
                return 'no_change'

            from_label = {-1: 'short', 0: 'neutral', 1: 'long'}[row['from_signal']]
            to_label = {-1: 'short', 0: 'neutral', 1: 'long'}[row['to_signal']]
            return f"{from_label}_to_{to_label}"

        transitions['transition_type'] = transitions.apply(get_transition_label, axis=1)

        # Count signal duration
        signal_groups = transitions.groupby((transitions['signal_change']).cumsum())
        transitions['signal_duration'] = signal_groups.cumcount() + 1

        # Log transition statistics
        transition_counts = transitions['transition_type'].value_counts()
        logger.info(f"Signal transitions: {transition_counts.to_dict()}")

        return transitions