"""Data preprocessing utilities for BPGV trading strategy."""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses and validates data for the trading strategy."""

    def __init__(self, config: Dict):
        """
        Initialize DataPreprocessor with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def validate_data(self, data: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate that DataFrame contains required columns and has no critical issues.

        Args:
            data: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check for required columns
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for empty DataFrame
        if data.empty:
            raise ValueError("DataFrame is empty")

        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Check for sufficient data
        if len(data) < 60:  # Minimum for GARCH
            raise ValueError(f"Insufficient data: {len(data)} rows (minimum 60 required)")

        return True

    def handle_outliers(self, data: pd.Series, n_std: float = 5.0) -> pd.Series:
        """
        Handle outliers using winsorization.

        Args:
            data: Series to process
            n_std: Number of standard deviations for outlier threshold

        Returns:
            Series with outliers handled
        """
        mean = data.mean()
        std = data.std()

        # Define outlier bounds
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std

        # Winsorize outliers
        data_clean = data.clip(lower=lower_bound, upper=upper_bound)

        n_outliers = (data != data_clean).sum()
        if n_outliers > 0:
            logger.info(f"Handled {n_outliers} outliers using {n_std}-sigma winsorization")

        return data_clean

    def calculate_returns(self, prices: pd.Series, method: str = 'simple') -> pd.Series:
        """
        Calculate returns from price series.

        Args:
            prices: Price series
            method: 'simple' or 'log' returns

        Returns:
            Returns series
        """
        if method == 'simple':
            returns = prices.pct_change()
        elif method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            raise ValueError(f"Unknown returns method: {method}")

        return returns

    def calculate_realized_volatility(self, returns: pd.Series,
                                     window: int = 60,
                                     annualize: bool = True) -> pd.Series:
        """
        Calculate rolling realized volatility.

        Args:
            returns: Returns series
            window: Rolling window size
            annualize: Whether to annualize volatility

        Returns:
            Volatility series
        """
        # Calculate rolling standard deviation
        vol = returns.rolling(window=window, min_periods=window).std()

        # Annualize if requested
        if annualize:
            # Determine frequency
            if pd.infer_freq(returns.index) in ['D', 'B']:
                vol = vol * np.sqrt(252)  # Daily data
            elif pd.infer_freq(returns.index) == 'M':
                vol = vol * np.sqrt(12)  # Monthly data

        return vol

    def prepare_garch_data(self, permit_growth: pd.Series) -> pd.Series:
        """
        Prepare building permit growth data for GARCH modeling.

        Args:
            permit_growth: Permit growth series (percentage)

        Returns:
            Cleaned and scaled series ready for GARCH
        """
        # Remove NaN values
        clean_growth = permit_growth.dropna()

        # Handle outliers
        clean_growth = self.handle_outliers(clean_growth, n_std=4.0)

        # Note: Scaling by 100 is already done in compute_permit_growth
        # GARCH models work better with percentage returns

        logger.info(f"Prepared GARCH data: {len(clean_growth)} observations")
        return clean_growth

    def merge_macro_data(self, permits_df: pd.DataFrame,
                        unemployment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge macroeconomic datasets.

        Args:
            permits_df: DataFrame with permits data
            unemployment_df: DataFrame with unemployment data

        Returns:
            Merged DataFrame
        """
        # Merge on date index
        merged = pd.merge(
            permits_df,
            unemployment_df,
            left_index=True,
            right_index=True,
            how='outer'
        )

        # Forward fill unemployment (it updates less frequently)
        merged['unemployment'] = merged['unemployment'].fillna(method='ffill')

        # Drop rows where we don't have both datasets
        merged = merged.dropna(subset=['permits', 'unemployment'])

        logger.info(f"Merged macro data: {len(merged)} observations")
        return merged

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for analysis.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with additional features
        """
        df = data.copy()

        # Moving averages for permits
        if 'permits' in df.columns:
            df['permits_ma_3'] = df['permits'].rolling(3).mean()
            df['permits_ma_12'] = df['permits'].rolling(12).mean()

        # Unemployment trend
        if 'unemployment' in df.columns:
            df['unemployment_ma_12'] = df['unemployment'].rolling(12).mean()
            df['unemployment_trend'] = df['unemployment'] - df['unemployment_ma_12']

        # Year-over-year changes
        if 'permits' in df.columns:
            df['permits_yoy'] = df['permits'].pct_change(12) * 100

        return df

    def split_train_test(self, data: pd.DataFrame,
                        test_start_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets.

        Args:
            data: Full dataset
            test_start_date: Start date for test set

        Returns:
            Tuple of (train_data, test_data)
        """
        test_start = pd.to_datetime(test_start_date)

        train_data = data[data.index < test_start]
        test_data = data[data.index >= test_start]

        logger.info(f"Split data: Train {len(train_data)} rows, Test {len(test_data)} rows")

        return train_data, test_data

    def prepare_backtest_data(self, macro_data: pd.DataFrame,
                             futures_data: pd.DataFrame,
                             signals: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare final dataset for backtesting.

        Args:
            macro_data: Macroeconomic data with signals
            futures_data: Futures price data
            signals: Trading signals DataFrame

        Returns:
            Combined DataFrame ready for backtesting
        """
        # Ensure all DataFrames have datetime index
        macro_data.index = pd.to_datetime(macro_data.index)
        futures_data.index = pd.to_datetime(futures_data.index)
        signals.index = pd.to_datetime(signals.index)

        # Get date range
        start_date = max(
            macro_data.index.min(),
            futures_data.index.min(),
            signals.index.min()
        )
        end_date = min(
            macro_data.index.max(),
            futures_data.index.max(),
            signals.index.max()
        )

        # Create daily date range
        daily_index = pd.date_range(start=start_date, end=end_date, freq='D')

        # Reindex futures data to daily
        futures_daily = futures_data.reindex(daily_index)
        futures_daily = futures_daily.fillna(method='ffill')  # Forward fill prices

        # Reindex signals to daily (forward fill from monthly)
        signals_daily = signals.reindex(daily_index)
        signals_daily = signals_daily.fillna(method='ffill')

        # Reindex macro data to daily
        macro_daily = macro_data.reindex(daily_index)
        macro_daily = macro_daily.fillna(method='ffill')

        # Combine all data
        backtest_data = pd.concat([
            futures_daily[['close', 'volume']],
            signals_daily,
            macro_daily[['unemployment', 'permit_growth']]
        ], axis=1)

        # Mark signal change dates
        backtest_data['signal_change'] = (
            backtest_data['signal'] != backtest_data['signal'].shift(1)
        )

        # Drop rows with missing critical data
        backtest_data = backtest_data.dropna(subset=['close', 'signal'])

        logger.info(f"Prepared backtest data: {len(backtest_data)} daily observations")
        return backtest_data