"""Databento data fetcher for CME housing futures and other futures data."""
import os
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False
    logger.warning("Databento library not installed. Install with: pip install databento")


class DatabentoFetcher:
    """Fetches futures data from Databento API."""

    def __init__(self, config: Dict[str, Any], cache_dir: Optional[str] = None):
        """
        Initialize Databento fetcher.

        Args:
            config: Configuration dictionary
            cache_dir: Directory for caching data
        """
        if not DATABENTO_AVAILABLE:
            raise ImportError("Databento library is required. Install with: pip install databento")

        self.config = config
        self.cache_dir = Path(cache_dir or config['data'].get('cache_dir', 'outputs/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Get API key from environment variable
        api_key_env = config['data'].get('databento_api_key_env', 'DATABENTO_API_KEY')
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(f"Databento API key not found in environment variable '{api_key_env}'. "
                           f"Sign up at https://databento.com and set your API key.")

        # Initialize Databento client
        try:
            self.client = db.Historical(self.api_key)
            logger.info("Databento client initialized successfully")
        except Exception as e:
            raise ValueError(f"Failed to initialize Databento client: {e}")

    def _get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_str = str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and fresh."""
        cache_file = self.cache_dir / f"databento_{cache_key}.pkl"
        if cache_file.exists():
            # Check if cache is less than 24 hours old
            if (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)) < timedelta(hours=24):
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Loading from Databento cache: {cache_key}")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None

    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        """Save data to cache."""
        cache_file = self.cache_dir / f"databento_{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                logger.info(f"Saved to Databento cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def get_housing_futures(self, ticker: str = "CUS.FUT",
                          start_date: str = "2006-01-01",
                          end_date: Optional[str] = None,
                          dataset: str = "GLBX.MDP3") -> pd.DataFrame:
        """
        Fetch CME Case-Shiller housing index futures data.

        Available tickers:
        - CUS.FUT: Composite (10-city) Index
        - BOS.FUT: Boston
        - CHI.FUT: Chicago
        - DEN.FUT: Denver
        - LAV.FUT: Las Vegas
        - LAX.FUT: Los Angeles
        - MIA.FUT: Miami
        - NYM.FUT: New York
        - SDG.FUT: San Diego
        - SFR.FUT: San Francisco
        - WDC.FUT: Washington DC

        Args:
            ticker: Databento symbol (e.g., "CUS.FUT")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (defaults to today)
            dataset: Databento dataset (default: GLBX.MDP3 for CME Globex)

        Returns:
            DataFrame with OHLCV data and datetime index
        """
        # Check cache first
        cache_key = self._get_cache_key(ticker=ticker, start=start_date, end=end_date, dataset=dataset)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            logger.info(f"Fetching housing futures from Databento: {ticker} ({start_date} to {end_date})")

            # Fetch data from Databento
            data = self.client.timeseries.get_range(
                dataset=dataset,
                symbols=ticker,
                stype_in='parent',  # Use parent symbol for continuous contract
                schema='ohlcv-1d',  # Daily OHLCV data
                start=start_date,
                end=end_date,
            )

            # Convert to pandas DataFrame
            df = data.to_df()

            if df.empty:
                raise ValueError(f"No data retrieved for {ticker}")

            # Process DataFrame
            df = self._process_databento_ohlcv(df)

            # Cache the data
            self._save_to_cache(df, cache_key)

            logger.info(f"Retrieved {len(df)} days of housing futures data for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching housing futures data: {e}")
            raise

    def get_futures_data(self, ticker: str,
                        start_date: str = "2010-01-01",
                        end_date: Optional[str] = None,
                        dataset: str = "GLBX.MDP3") -> pd.DataFrame:
        """
        Generic futures data fetcher for any CME futures.

        Args:
            ticker: Databento symbol (e.g., "ES.FUT", "CUS.FUT")
            start_date: Start date
            end_date: End date
            dataset: Databento dataset

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache
        cache_key = self._get_cache_key(ticker=ticker, start=start_date, end=end_date, dataset=dataset)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            logger.info(f"Fetching futures data from Databento: {ticker}")

            # Fetch data
            data = self.client.timeseries.get_range(
                dataset=dataset,
                symbols=ticker,
                stype_in='parent',
                schema='ohlcv-1d',
                start=start_date,
                end=end_date,
            )

            # Convert and process
            df = data.to_df()
            if df.empty:
                raise ValueError(f"No data retrieved for {ticker}")

            df = self._process_databento_ohlcv(df)

            # Cache
            self._save_to_cache(df, cache_key)

            logger.info(f"Retrieved {len(df)} days of futures data for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching futures data: {e}")
            raise

    def _process_databento_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Databento OHLCV data into standard format.

        Args:
            df: Raw Databento DataFrame

        Returns:
            Processed DataFrame with standard OHLCV columns
        """
        # Databento returns columns like 'open', 'high', 'low', 'close', 'volume'
        # Ensure datetime index
        if 'ts_event' in df.columns:
            df.index = pd.to_datetime(df['ts_event'])
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Standardize column names
        column_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }

        df_clean = pd.DataFrame()
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df_clean[new_col] = df[old_col]

        # Remove timezone if present (for consistency with yfinance data)
        if df_clean.index.tz is not None:
            df_clean.index = df_clean.index.tz_localize(None)

        # Sort by date
        df_clean = df_clean.sort_index()

        # Check for duplicate dates (data quality issue)
        duplicate_dates = df_clean.index.duplicated(keep=False)
        if duplicate_dates.any():
            n_duplicates = duplicate_dates.sum()
            logger.warning(f"Found {n_duplicates} duplicate date entries in Databento data. Deduplicating...")

            # Convert index to date (remove time component) and group by date
            df_clean['date'] = df_clean.index.date

            # For daily OHLCV, take the last entry for each date
            # (assuming last = end of day values)
            df_clean = df_clean.groupby('date').agg({
                'open': 'first',    # First open of the day
                'high': 'max',      # Highest high
                'low': 'min',       # Lowest low
                'close': 'last',    # Last close of the day
                'volume': 'sum'     # Total volume
            })

            # Convert date back to datetime index
            df_clean.index = pd.to_datetime(df_clean.index)

        # Validate price data - remove negative or zero prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df_clean.columns:
                invalid_mask = (df_clean[col] <= 0)
                if invalid_mask.any():
                    n_invalid = invalid_mask.sum()
                    logger.warning(f"Found {n_invalid} invalid (<=0) values in {col}. Removing affected rows...")
                    df_clean = df_clean[~invalid_mask]

        # Additional validation: Remove unrealistically low prices for housing futures
        # Case-Shiller housing index futures should be in range 100-500 based on historical data
        if 'close' in df_clean.columns and len(df_clean) > 0:
            # Calculate median price to determine expected range
            median_price = df_clean['close'].median()

            # If median suggests housing futures (typically 150-400), apply stricter validation
            if median_price > 100:
                min_valid_price = 100  # Housing futures shouldn't be below 100
                low_price_mask = df_clean['close'] < min_valid_price

                if low_price_mask.any():
                    n_low = low_price_mask.sum()
                    low_prices = df_clean.loc[low_price_mask, 'close'].values
                    logger.warning(f"Found {n_low} unrealistically low prices (< {min_valid_price}). "
                                 f"Values: {low_prices[:10]}. Removing affected rows...")
                    df_clean = df_clean[~low_price_mask]

        # Statistical outlier detection using IQR method
        if 'close' in df_clean.columns and len(df_clean) > 10:
            Q1 = df_clean['close'].quantile(0.25)
            Q3 = df_clean['close'].quantile(0.75)
            IQR = Q3 - Q1

            # Define outliers as values outside 3 * IQR from Q1/Q3
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outlier_mask = (df_clean['close'] < lower_bound) | (df_clean['close'] > upper_bound)
            if outlier_mask.any():
                n_outliers = outlier_mask.sum()
                outlier_prices = df_clean.loc[outlier_mask, 'close'].values
                logger.warning(f"Found {n_outliers} statistical outliers (outside 3*IQR range [{lower_bound:.2f}, {upper_bound:.2f}]). "
                             f"Outlier values: {outlier_prices[:10]}. Removing affected rows...")
                df_clean = df_clean[~outlier_mask]

        # Check for extreme price changes (>50% in one day) which might indicate data errors
        if 'close' in df_clean.columns and len(df_clean) > 1:
            price_changes = df_clean['close'].pct_change().abs()
            extreme_changes = price_changes > 0.5
            if extreme_changes.any():
                n_extreme = extreme_changes.sum()
                logger.warning(f"Found {n_extreme} extreme price changes (>50%). This may indicate data quality issues.")
                # Log the dates with extreme changes for investigation
                extreme_dates = df_clean.index[extreme_changes].tolist()
                logger.warning(f"Dates with extreme changes: {extreme_dates[:5]}")  # Show first 5
                # Remove rows with extreme changes
                df_clean = df_clean[~extreme_changes]

        # Forward fill any missing data
        df_clean = df_clean.ffill()

        # Drop any remaining NaNs
        df_clean = df_clean.dropna()

        # Final validation
        if df_clean.empty:
            logger.error("No valid data remaining after cleaning")
        else:
            logger.info(f"Cleaned data: {len(df_clean)} days, "
                       f"price range: ${df_clean['close'].min():.2f} - ${df_clean['close'].max():.2f}")

        return df_clean

    def get_available_symbols(self, search: str = "housing") -> list:
        """
        Search for available symbols in Databento.

        Args:
            search: Search term

        Returns:
            List of available symbols
        """
        try:
            # Note: This is a placeholder - actual implementation depends on Databento API
            logger.info(f"Searching for symbols matching: {search}")

            # Common CME housing futures symbols
            housing_symbols = [
                "CUS.FUT",  # Composite
                "BOS.FUT",  # Boston
                "CHI.FUT",  # Chicago
                "DEN.FUT",  # Denver
                "LAV.FUT",  # Las Vegas
                "LAX.FUT",  # Los Angeles
                "MIA.FUT",  # Miami
                "NYM.FUT",  # New York
                "SDG.FUT",  # San Diego
                "SFR.FUT",  # San Francisco
                "WDC.FUT",  # Washington DC
            ]

            return housing_symbols

        except Exception as e:
            logger.error(f"Error searching symbols: {e}")
            return []
