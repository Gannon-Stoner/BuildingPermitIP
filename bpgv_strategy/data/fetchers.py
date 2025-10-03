"""Data fetching classes for BPGV trading strategy using local repository data."""
import os
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and caches housing permit and futures data."""

    def __init__(self, config: Dict[str, Any], cache_dir: Optional[str] = None):
        """
        Initialize DataFetcher with configuration.

        Args:
            config: Configuration dictionary
            cache_dir: Directory for caching data
        """
        self.config = config
        self.cache_dir = Path(cache_dir or config['data'].get('cache_dir', 'outputs/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set path to housing permit data repository
        self.housing_data_path = Path(config['data']['housing_data_path'])
        if not self.housing_data_path.exists():
            raise ValueError(f"Housing data repository not found at {self.housing_data_path}. "
                           f"Please clone https://github.com/idkh0wtocode/Housing_Permit_IP.git")

    def _get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_str = str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available and fresh."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            # Check if cache is less than 24 hours old
            if (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)) < timedelta(hours=24):
                try:
                    with open(cache_file, 'rb') as f:
                        logger.info(f"Loading from cache: {cache_key}")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None

    def _save_to_cache(self, data: pd.DataFrame, cache_key: str) -> None:
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                logger.info(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def get_building_permits(self, start_date: str = "1960-01-01",
                           end_date: Optional[str] = None,
                           permit_file: Optional[str] = None) -> pd.DataFrame:
        """
        Load building permit data from local repository.

        Args:
            start_date: Start date for data
            end_date: End date for data (defaults to today)
            permit_file: Name of permit file (defaults to config value)

        Returns:
            DataFrame with datetime index and 'permits' column
        """
        # Use provided file or default from config
        if permit_file is None:
            permit_file = self.config['data'].get('permit_file', 'PERMIT.csv')

        csv_path = self.housing_data_path / permit_file

        if not csv_path.exists():
            raise FileNotFoundError(f"Permit data file not found: {csv_path}")

        try:
            logger.info(f"Loading building permits from: {csv_path}")

            # Read CSV file
            df = pd.read_csv(csv_path)

            # The CSV has 'observation_date' and 'PERMIT' columns
            if 'observation_date' in df.columns and 'PERMIT' in df.columns:
                df_clean = pd.DataFrame({
                    'permits': df['PERMIT'].values
                }, index=pd.to_datetime(df['observation_date']))
            else:
                # Try to infer columns
                date_col = df.columns[0] if 'date' in df.columns[0].lower() else None
                value_col = df.columns[1] if len(df.columns) > 1 else None

                if not date_col or not value_col:
                    raise ValueError(f"Cannot identify date and value columns in {csv_path}")

                df_clean = pd.DataFrame({
                    'permits': df[value_col].values
                }, index=pd.to_datetime(df[date_col]))

            # Drop NaN values
            df_clean = df_clean.dropna()

            logger.info(f"Loaded {len(df_clean)} rows. Date range: {df_clean.index.min()} to {df_clean.index.max()}")

            # Filter to date range
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df_clean = df_clean[df_clean.index >= start_dt]
                logger.info(f"After start date filter: {len(df_clean)} rows")
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df_clean = df_clean[df_clean.index <= end_dt]
                logger.info(f"After end date filter: {len(df_clean)} rows")

            if df_clean.empty:
                raise ValueError(f"No data available for date range {start_date} to {end_date}")

            logger.info(f"Loaded {len(df_clean)} months of permit data from {df_clean.index.min()} to {df_clean.index.max()}")
            return df_clean

        except Exception as e:
            logger.error(f"Error loading building permits from {csv_path}: {e}")
            raise

    def get_state_permits(self, state_code: str, start_date: str = "1960-01-01",
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load state-level building permit data.

        Args:
            state_code: State code (e.g., 'CA', 'FL', 'TX')
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with state permit data
        """
        state_dir = self.housing_data_path / 'state_permits'

        # Look for state file (could be various naming patterns)
        possible_files = [
            f"{state_code}BPPRIVSA.csv",
            f"{state_code}BPPRIV.csv",
            f"{state_code.lower()}bpprivsa.csv",
            f"{state_code.lower()}_permits.csv"
        ]

        csv_path = None
        for filename in possible_files:
            test_path = state_dir / filename
            if test_path.exists():
                csv_path = test_path
                break

        if not csv_path:
            # List available state files
            available = [f.name for f in state_dir.glob("*.csv")]
            logger.warning(f"No data found for state {state_code}. Available: {available[:5]}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(csv_path)
            # Process similar to national data
            df_clean = pd.DataFrame()
            df_clean.index = pd.to_datetime(df.iloc[:, 0])  # First column is date
            df_clean['permits'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')  # Second column is value
            df_clean = df_clean.dropna()

            # Filter dates
            if start_date:
                df_clean = df_clean[df_clean.index >= pd.to_datetime(start_date)]
            if end_date:
                df_clean = df_clean[df_clean.index <= pd.to_datetime(end_date)]

            logger.info(f"Loaded {len(df_clean)} months of {state_code} permit data")
            return df_clean

        except Exception as e:
            logger.error(f"Error loading state permits for {state_code}: {e}")
            return pd.DataFrame()

    def get_city_permits(self, city_name: str, start_date: str = "1960-01-01",
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load city-level building permit data.

        Args:
            city_name: City name or identifier
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with city permit data
        """
        city_dir = self.housing_data_path / 'city_permits'

        # Look for city file
        csv_path = city_dir / f"{city_name}.csv"
        if not csv_path.exists():
            # Try alternative naming
            for file in city_dir.glob("*.csv"):
                if city_name.lower() in file.name.lower():
                    csv_path = file
                    break

        if not csv_path.exists():
            available = [f.name for f in city_dir.glob("*.csv")]
            logger.warning(f"No data found for city {city_name}. Available: {available[:5]}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(csv_path)
            # Process similar to national data
            df_clean = pd.DataFrame()
            df_clean.index = pd.to_datetime(df.iloc[:, 0])  # First column is date
            df_clean['permits'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')  # Second column is value
            df_clean = df_clean.dropna()

            # Filter dates
            if start_date:
                df_clean = df_clean[df_clean.index >= pd.to_datetime(start_date)]
            if end_date:
                df_clean = df_clean[df_clean.index <= pd.to_datetime(end_date)]

            logger.info(f"Loaded {len(df_clean)} months of {city_name} permit data")
            return df_clean

        except Exception as e:
            logger.error(f"Error loading city permits for {city_name}: {e}")
            return pd.DataFrame()

    def get_unemployment(self, start_date: str = "1960-01-01",
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Generate synthetic unemployment data for backtesting.
        In production, this should be replaced with actual unemployment data.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch (defaults to today)

        Returns:
            DataFrame with datetime index and 'unemployment' column
        """
        logger.info("Generating synthetic unemployment data for backtesting")

        # Create monthly date range
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        dates = pd.date_range(start=start_date, end=end_date, freq='MS')

        # Generate realistic unemployment rates with cycles
        n = len(dates)
        trend = 5.0  # Base unemployment rate
        cycle = 2.0 * np.sin(2 * np.pi * np.arange(n) / 48)  # 4-year cycle
        noise = np.random.normal(0, 0.3, n)

        unemployment = trend + cycle + noise
        unemployment = np.clip(unemployment, 2.5, 10.0)  # Keep within realistic bounds

        df = pd.DataFrame({'unemployment': unemployment}, index=dates)

        logger.info(f"Generated {len(df)} months of unemployment data")
        return df

    def get_futures_data(self, ticker: Optional[str] = None,
                        start_date: str = "2010-01-01",
                        end_date: Optional[str] = None,
                        source: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch futures data from configured source (yfinance or Databento).

        Args:
            ticker: Futures ticker symbol (uses config default if None)
            start_date: Start date for data fetch
            end_date: End date for data fetch (defaults to today)
            source: Data source override ("yfinance" or "databento")

        Returns:
            DataFrame with OHLCV data and datetime index
        """
        # Determine data source
        if source is None:
            source = self.config['data'].get('futures_source', 'yfinance')

        # Get ticker from config if not provided
        if ticker is None:
            ticker = self.config['data'].get('futures_ticker', 'ES=F')

        logger.info(f"Fetching futures data using {source}: {ticker}")

        # Route to appropriate fetcher
        if source.lower() == 'databento':
            return self._get_futures_databento(ticker, start_date, end_date)
        else:
            return self._get_futures_yfinance(ticker, start_date, end_date)

    def _get_futures_yfinance(self, ticker: str, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
        """
        Fetch futures data from yfinance.

        Args:
            ticker: yfinance ticker (e.g., "ES=F")
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = self._get_cache_key(ticker=ticker, start=start_date, end=end_date, source='yfinance')
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            logger.info(f"Fetching from yfinance: {ticker}")

            # Download futures data
            futures = yf.Ticker(ticker)
            df = futures.history(start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                logger.warning(f"No futures data retrieved for {ticker}, trying alternate method")
                # Try downloading with different method
                df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

            if df.empty:
                raise ValueError(f"No futures data retrieved for {ticker}")

            # Clean column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]

            # Ensure datetime index
            df.index = pd.to_datetime(df.index)

            # Handle missing data
            df = df.dropna(subset=['close'])

            self._save_to_cache(df, cache_key)
            logger.info(f"Retrieved {len(df)} days from yfinance")
            return df

        except Exception as e:
            logger.error(f"Error fetching from yfinance: {e}")
            raise

    def _get_futures_databento(self, ticker: str, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
        """
        Fetch futures data from Databento.

        Args:
            ticker: Databento ticker (e.g., "ES.FUT", "CUS.FUT")
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Import DatabentoFetcher
            from .databento_fetcher import DatabentoFetcher

            # Initialize Databento fetcher
            databento_fetcher = DatabentoFetcher(self.config, self.cache_dir)

            # Get dataset from config
            dataset = self.config['data'].get('databento_dataset', 'GLBX.MDP3')

            # Fetch data
            df = databento_fetcher.get_futures_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                dataset=dataset
            )

            logger.info(f"Retrieved {len(df)} days from Databento")
            return df

        except ImportError as e:
            logger.error("Databento library not installed. Install with: pip install databento")
            raise
        except Exception as e:
            logger.error(f"Error fetching from Databento: {e}")
            raise

    def compute_permit_growth(self, permits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate month-over-month building permit growth rate.

        Args:
            permits_df: DataFrame with 'permits' column

        Returns:
            DataFrame with additional 'permit_growth' column (percentage)
        """
        df = permits_df.copy()

        # Calculate month-over-month percentage change
        df['permit_growth'] = df['permits'].pct_change() * 100  # Convert to percentage

        # Handle edge cases
        df['permit_growth'] = df['permit_growth'].replace([np.inf, -np.inf], np.nan)

        # Fill initial NaN with 0
        df['permit_growth'] = df['permit_growth'].fillna(0)

        logger.info(f"Computed permit growth: {len(df)} observations")
        return df

    def align_data(self, monthly_data: pd.DataFrame, daily_data: pd.DataFrame,
                  signal_column: str = 'signal') -> pd.DataFrame:
        """
        Align monthly macroeconomic data with daily futures data.
        Forward-fills monthly data to daily frequency.

        Args:
            monthly_data: Monthly frequency DataFrame with signals
            daily_data: Daily frequency DataFrame with prices
            signal_column: Name of signal column to forward-fill

        Returns:
            Aligned DataFrame with both signals and prices at daily frequency
        """
        try:
            # Ensure datetime indices
            monthly_data.index = pd.to_datetime(monthly_data.index)
            daily_data.index = pd.to_datetime(daily_data.index)

            # Remove timezone info if present for comparison
            if monthly_data.index.tz is not None:
                monthly_data.index = monthly_data.index.tz_localize(None)
            if daily_data.index.tz is not None:
                daily_data.index = daily_data.index.tz_localize(None)

            # Get the overlapping date range
            start_date = max(monthly_data.index.min(), daily_data.index.min())
            end_date = min(monthly_data.index.max(), daily_data.index.max())

            # Filter both datasets to overlapping period
            monthly_data = monthly_data[start_date:end_date]
            daily_data = daily_data[start_date:end_date]

            # Create daily index
            daily_index = pd.date_range(start=start_date, end=end_date, freq='D')

            # Reindex monthly data to daily frequency and forward-fill
            monthly_reindexed = monthly_data.reindex(daily_index, method='ffill')

            # Merge with daily futures data
            aligned_df = pd.merge(
                daily_data,
                monthly_reindexed,
                left_index=True,
                right_index=True,
                how='inner'
            )

            # Mark signal change dates (for rebalancing)
            if signal_column in aligned_df.columns:
                aligned_df['signal_change'] = (
                    aligned_df[signal_column] != aligned_df[signal_column].shift(1)
                )
            else:
                aligned_df['signal_change'] = False

            # Add days since last signal change
            aligned_df['days_since_signal'] = aligned_df.groupby(
                (aligned_df['signal_change']).cumsum()
            ).cumcount()

            logger.info(f"Aligned data: {len(aligned_df)} daily observations")
            return aligned_df

        except Exception as e:
            logger.error(f"Error aligning data: {e}")
            raise

    def fetch_all_data(self, start_date: str = "2000-01-01",
                      end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch all required data for the strategy.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            Dictionary containing all fetched datasets
        """
        data = {}

        logger.info("Fetching all required data...")

        try:
            # Building permits from local repository
            permits = self.get_building_permits(start_date, end_date)
            permits_with_growth = self.compute_permit_growth(permits)
            data['permits'] = permits_with_growth

            # Unemployment (synthetic for now)
            data['unemployment'] = self.get_unemployment(start_date, end_date)

            # Futures data (adjust start date based on source)
            futures_start = max(start_date, "2010-01-01")

            # Adjust for Databento limitations if using Databento
            futures_source = self.config['data'].get('futures_source', 'yfinance')
            if futures_source.lower() == 'databento':
                # Databento GLBX.MDP3 dataset starts 2010-06-06
                databento_min_date = "2010-06-06"
                if futures_start < databento_min_date:
                    logger.warning(f"Adjusting futures start date from {futures_start} to {databento_min_date} "
                                 f"(Databento GLBX.MDP3 dataset limitation)")
                    futures_start = databento_min_date

            data['futures'] = self.get_futures_data(
                self.config['data']['futures_ticker'],
                futures_start,
                end_date
            )

            logger.info("Successfully fetched all data")
            return data

        except Exception as e:
            logger.error(f"Error fetching all data: {e}")
            raise