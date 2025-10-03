"""Data module for BPGV trading strategy."""
from .fetchers import DataFetcher
from .preprocessor import DataPreprocessor

# Optional Databento import (only if installed)
try:
    from .databento_fetcher import DatabentoFetcher
    __all__ = ["DataFetcher", "DataPreprocessor", "DatabentoFetcher"]
except ImportError:
    __all__ = ["DataFetcher", "DataPreprocessor"]