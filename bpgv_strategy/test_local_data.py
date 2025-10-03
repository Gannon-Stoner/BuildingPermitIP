#!/usr/bin/env python3
"""
Test script to verify the strategy works with local repository data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from data import DataFetcher
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_local_data():
    """Test loading data from local repository."""

    logger.info("="*60)
    logger.info("Testing Local Repository Data Loading")
    logger.info("="*60)

    # Load configuration
    config = load_config()
    logger.info(f"Housing data path: {config['data']['housing_data_path']}")

    # Initialize DataFetcher
    try:
        fetcher = DataFetcher(config)
        logger.info("✓ DataFetcher initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize DataFetcher: {e}")
        return False

    # Test 1: Load national permits
    logger.info("\nTest 1: Loading national building permits...")
    try:
        permits_df = fetcher.get_building_permits(
            start_date="2000-01-01",
            end_date="2025-08-31"
        )

        logger.info(f"✓ Loaded {len(permits_df)} months of national permit data")
        logger.info(f"  Date range: {permits_df.index.min()} to {permits_df.index.max()}")
        logger.info(f"  Sample statistics:")
        logger.info(f"    Mean: {permits_df['permits'].mean():.0f} permits/month")
        logger.info(f"    Std: {permits_df['permits'].std():.0f}")
        logger.info(f"    Min: {permits_df['permits'].min():.0f}")
        logger.info(f"    Max: {permits_df['permits'].max():.0f}")

        # Compute growth
        permits_with_growth = fetcher.compute_permit_growth(permits_df)
        logger.info(f"  Growth rate statistics:")
        logger.info(f"    Mean: {permits_with_growth['permit_growth'].mean():.2f}%")
        logger.info(f"    Std: {permits_with_growth['permit_growth'].std():.2f}%")

    except Exception as e:
        logger.error(f"✗ Failed to load national permits: {e}")
        return False

    # Test 2: Load state permits (example: California)
    logger.info("\nTest 2: Loading state permits (California)...")
    try:
        ca_permits = fetcher.get_state_permits("CA", start_date="2010-01-01")
        if not ca_permits.empty:
            logger.info(f"✓ Loaded {len(ca_permits)} months of CA permit data")
            logger.info(f"  Mean: {ca_permits['permits'].mean():.0f} permits/month")
        else:
            logger.info("  California data not available or empty")
    except Exception as e:
        logger.warning(f"  State data test skipped: {e}")

    # Test 3: Load city permits (example: Austin)
    logger.info("\nTest 3: Loading city permits (Austin)...")
    try:
        austin_permits = fetcher.get_city_permits("austin", start_date="2010-01-01")
        if not austin_permits.empty:
            logger.info(f"✓ Loaded {len(austin_permits)} months of Austin permit data")
            logger.info(f"  Mean: {austin_permits['permits'].mean():.0f} permits/month")
        else:
            logger.info("  Austin data not available or empty")
    except Exception as e:
        logger.warning(f"  City data test skipped: {e}")

    # Test 4: Generate unemployment data
    logger.info("\nTest 4: Generating synthetic unemployment data...")
    try:
        unemployment = fetcher.get_unemployment("2000-01-01", "2024-12-31")
        logger.info(f"✓ Generated {len(unemployment)} months of unemployment data")
        logger.info(f"  Mean rate: {unemployment['unemployment'].mean():.1f}%")
        logger.info(f"  Range: {unemployment['unemployment'].min():.1f}% - {unemployment['unemployment'].max():.1f}%")
    except Exception as e:
        logger.error(f"✗ Failed to generate unemployment data: {e}")
        return False

    # Test 5: Fetch futures data
    logger.info("\nTest 5: Fetching futures data (may take a moment)...")
    try:
        futures = fetcher.get_futures_data("ES=F", "2020-01-01", "2024-01-01")
        logger.info(f"✓ Fetched {len(futures)} days of futures data")
        logger.info(f"  Date range: {futures.index.min()} to {futures.index.max()}")
        logger.info(f"  Latest close: ${futures['close'].iloc[-1]:.2f}")
    except Exception as e:
        logger.warning(f"  Futures data fetch failed (may need internet): {e}")

    # Test 6: Full data pipeline
    logger.info("\nTest 6: Testing full data pipeline...")
    try:
        all_data = fetcher.fetch_all_data("2015-01-01", "2024-01-01")
        logger.info("✓ Full data pipeline successful")
        logger.info(f"  Permits: {len(all_data['permits'])} months")
        logger.info(f"  Unemployment: {len(all_data['unemployment'])} months")
        logger.info(f"  Futures: {len(all_data['futures'])} days")
    except Exception as e:
        logger.error(f"✗ Full pipeline failed: {e}")
        return False

    # Summary
    logger.info("\n" + "="*60)
    logger.info("Local Data Test Complete!")
    logger.info("="*60)
    logger.info("\nThe strategy is ready to use with local housing permit data.")
    logger.info("No FRED API key required.")
    logger.info("\nTo run the full strategy:")
    logger.info("  python main.py --start-date 2010-01-01 --end-date 2024-12-31")

    return True


if __name__ == "__main__":
    success = test_local_data()
    sys.exit(0 if success else 1)