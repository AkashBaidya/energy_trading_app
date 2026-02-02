"""
Data Loading Module for Energy Trading Application.
Handles loading data from files and uploaded sources using Polars.
"""

import polars as pl
from pathlib import Path
from typing import Optional, Union, Dict, BinaryIO
from dataclasses import dataclass
import logging

from .config import config, ColumnConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataLoadResult:
    """Container for loaded data results."""
    auction_data: Optional[pl.DataFrame] = None
    forecast_inputs: Optional[pl.DataFrame] = None
    system_prices: Optional[pl.DataFrame] = None
    errors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = {}
    
    @property
    def is_valid(self) -> bool:
        """Check if all required data is loaded."""
        return (
            self.auction_data is not None and 
            len(self.errors) == 0
        )
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary statistics of loaded data."""
        summary = {}
        if self.auction_data is not None:
            summary["auction_data"] = {
                "rows": self.auction_data.height,
                "columns": self.auction_data.width,
                "column_names": self.auction_data.columns
            }
        if self.forecast_inputs is not None:
            summary["forecast_inputs"] = {
                "rows": self.forecast_inputs.height,
                "columns": self.forecast_inputs.width,
                "column_names": self.forecast_inputs.columns
            }
        if self.system_prices is not None:
            summary["system_prices"] = {
                "rows": self.system_prices.height,
                "columns": self.system_prices.width,
                "column_names": self.system_prices.columns
            }
        return summary


class DataLoader:
    """
    Data loader class for energy trading datasets.
    Supports loading from files and uploaded data using Polars.
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            base_path: Base directory for data files
        """
        self.base_path = base_path or config.data_paths.base_dir
        self.column_config = config.columns
    
    def load_csv(
        self,
        source: Union[str, Path, BinaryIO],
        delimiter: str = ";",
        skip_rows: int = 0,
        **kwargs
    ) -> pl.DataFrame:
        """
        Load a CSV file using Polars.
        
        Args:
            source: File path or file-like object
            delimiter: CSV delimiter
            skip_rows: Number of rows to skip
            **kwargs: Additional arguments for pl.read_csv
            
        Returns:
            Polars DataFrame
        """
        try:
            if isinstance(source, (str, Path)):
                df = pl.read_csv(
                    source,
                    separator=delimiter,
                    skip_rows=skip_rows,
                    infer_schema_length=10000,
                    **kwargs
                )
            else:
                # Handle file-like objects (uploads)
                content = source.read()
                if isinstance(content, str):
                    content = content.encode('utf-8')
                df = pl.read_csv(
                    content,
                    separator=delimiter,
                    skip_rows=skip_rows,
                    infer_schema_length=10000,
                    **kwargs
                )
            
            logger.info(f"Successfully loaded data with {df.height} rows and {df.width} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    def load_auction_data(
        self,
        source: Optional[Union[str, Path, BinaryIO]] = None
    ) -> pl.DataFrame:
        """
        Load auction data from file or upload.
        
        Args:
            source: File path or uploaded file object
            
        Returns:
            Polars DataFrame with auction data
        """
        if source is None:
            source = self.base_path / config.data_paths.auction_data
        
        df = self.load_csv(source, delimiter=";")
        
        # Check if first row contains units (skip it)
        if df.height > 0:
            first_row = df.row(0)
            if any("GBP" in str(v) or "MW" in str(v) for v in first_row if v):
                df = df.slice(1)
                logger.info("Skipped units row in auction data")
        
        return df
    
    def load_forecast_inputs(
        self,
        source: Optional[Union[str, Path, BinaryIO]] = None
    ) -> pl.DataFrame:
        """
        Load forecast input variables.
        
        Args:
            source: File path or uploaded file object
            
        Returns:
            Polars DataFrame with forecast inputs
        """
        if source is None:
            source = self.base_path / config.data_paths.forecast_inputs
        
        df = self.load_csv(source, delimiter=";")
        
        # Skip units row if present
        if df.height > 0:
            first_row = df.row(0)
            if any("GBP" in str(v) or "MW" in str(v) or "%" in str(v) for v in first_row if v):
                df = df.slice(1)
                logger.info("Skipped units row in forecast inputs")
        
        return df
    
    def load_system_prices(
        self,
        source: Optional[Union[str, Path, BinaryIO]] = None
    ) -> pl.DataFrame:
        """
        Load system prices data.
        
        Args:
            source: File path or uploaded file object
            
        Returns:
            Polars DataFrame with system prices
        """
        if source is None:
            source = self.base_path / config.data_paths.system_prices
        
        df = self.load_csv(source, delimiter=";")
        
        # Skip units row if present
        if df.height > 0:
            first_row = df.row(0)
            if any("GBP" in str(v) for v in first_row if v):
                df = df.slice(1)
                logger.info("Skipped units row in system prices")
        
        return df
    
    def load_all_from_directory(
        self,
        directory: Optional[Path] = None
    ) -> DataLoadResult:
        """
        Load all data files from a directory.
        
        Args:
            directory: Directory containing data files
            
        Returns:
            DataLoadResult with all loaded data
        """
        if directory is not None:
            self.base_path = Path(directory)
        
        result = DataLoadResult()
        
        # Load auction data
        try:
            auction_path = self.base_path / config.data_paths.auction_data
            if auction_path.exists():
                result.auction_data = self.load_auction_data(auction_path)
            else:
                result.errors["auction_data"] = f"File not found: {auction_path}"
        except Exception as e:
            result.errors["auction_data"] = str(e)
        
        # Load forecast inputs
        try:
            forecast_path = self.base_path / config.data_paths.forecast_inputs
            if forecast_path.exists():
                result.forecast_inputs = self.load_forecast_inputs(forecast_path)
            else:
                logger.warning(f"Forecast inputs not found: {forecast_path}")
        except Exception as e:
            result.errors["forecast_inputs"] = str(e)
        
        # Load system prices
        try:
            system_path = self.base_path / config.data_paths.system_prices
            if system_path.exists():
                result.system_prices = self.load_system_prices(system_path)
            else:
                logger.warning(f"System prices not found: {system_path}")
        except Exception as e:
            result.errors["system_prices"] = str(e)
        
        return result
    
    def load_from_uploads(
        self,
        auction_file: Optional[BinaryIO] = None,
        forecast_file: Optional[BinaryIO] = None,
        system_file: Optional[BinaryIO] = None
    ) -> DataLoadResult:
        """
        Load data from uploaded files.
        
        Args:
            auction_file: Uploaded auction data file
            forecast_file: Uploaded forecast inputs file
            system_file: Uploaded system prices file
            
        Returns:
            DataLoadResult with all loaded data
        """
        result = DataLoadResult()
        
        if auction_file is not None:
            try:
                result.auction_data = self.load_auction_data(auction_file)
            except Exception as e:
                result.errors["auction_data"] = str(e)
        
        if forecast_file is not None:
            try:
                result.forecast_inputs = self.load_forecast_inputs(forecast_file)
            except Exception as e:
                result.errors["forecast_inputs"] = str(e)
        
        if system_file is not None:
            try:
                result.system_prices = self.load_system_prices(system_file)
            except Exception as e:
                result.errors["system_prices"] = str(e)
        
        return result


def validate_dataframe(
    df: pl.DataFrame,
    required_columns: list[str],
    name: str = "DataFrame"
) -> tuple[bool, list[str]]:
    """
    Validate that a DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        name: Name for error messages
        
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing
