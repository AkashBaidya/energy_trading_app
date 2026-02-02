"""
Data Cleaning Module for Energy Trading Application.
Handles data preprocessing, cleaning, and imputation using Polars.
"""

import polars as pl
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from .config import config, ImputationMethod, CleaningConfig

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Report of cleaning operations performed."""
    original_rows: int
    final_rows: int
    columns_converted: List[str]
    missing_values_filled: Dict[str, int]
    outliers_handled: Dict[str, int]
    duplicates_removed: int
    errors: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "original_rows": self.original_rows,
            "final_rows": self.final_rows,
            "rows_removed": self.original_rows - self.final_rows,
            "columns_converted": self.columns_converted,
            "missing_values_filled": self.missing_values_filled,
            "outliers_handled": self.outliers_handled,
            "duplicates_removed": self.duplicates_removed,
            "errors": self.errors
        }


class DataCleaner:
    """
    Data cleaner class for energy trading datasets.
    Provides comprehensive data cleaning and preprocessing using Polars.
    """
    
    def __init__(self, cleaning_config: Optional[CleaningConfig] = None):
        """
        Initialize the data cleaner.
        
        Args:
            cleaning_config: Configuration for cleaning operations
        """
        self.config = cleaning_config or config.cleaning
        self.column_config = config.columns
    
    def parse_datetime(
        self,
        df: pl.DataFrame,
        column: str,
        date_format: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Parse datetime column with custom format.
        
        Args:
            df: Input DataFrame
            column: Column name to parse
            date_format: Date format string (e.g., "[%d/%m/%Y %H:%M]")
            
        Returns:
            DataFrame with parsed datetime column
        """
        format_str = date_format or self.config.date_format
        
        try:
            # Remove brackets if present and parse
            df = df.with_columns(
                pl.col(column)
                .str.replace_all(r"[\[\]]", "")
                .str.strptime(pl.Datetime, format="%d/%m/%Y %H:%M", strict=False)
                .alias(column)
            )
            logger.info(f"Successfully parsed datetime column: {column}")
        except Exception as e:
            logger.warning(f"Could not parse datetime with format {format_str}: {e}")
            # Try alternative parsing
            try:
                df = df.with_columns(
                    pl.col(column)
                    .str.replace_all(r"[\[\]]", "")
                    .str.to_datetime(strict=False)
                    .alias(column)
                )
            except Exception as e2:
                logger.error(f"Failed to parse datetime: {e2}")
        
        return df
    
    def convert_to_numeric(
        self,
        df: pl.DataFrame,
        columns: Optional[List[str]] = None,
        errors: str = "null"
    ) -> Tuple[pl.DataFrame, List[str]]:
        """
        Convert columns to numeric types.
        
        Args:
            df: Input DataFrame
            columns: List of columns to convert (None = auto-detect)
            errors: How to handle errors ("null" or "raise")
            
        Returns:
            Tuple of (DataFrame, list of converted columns)
        """
        converted = []
        
        if columns is None:
            # Auto-detect columns that should be numeric
            columns = [
                col for col in df.columns 
                if df[col].dtype == pl.Utf8 and col != self.column_config.DATE_COLUMN
            ]
        
        for col in columns:
            if col in df.columns:
                try:
                    # Try to convert string to float
                    df = df.with_columns(
                        pl.col(col)
                        .str.replace(",", ".")  # Handle European decimals
                        .cast(pl.Float64, strict=False)
                        .alias(col)
                    )
                    converted.append(col)
                except Exception as e:
                    if errors == "raise":
                        raise
                    logger.warning(f"Could not convert column {col}: {e}")
        
        logger.info(f"Converted {len(converted)} columns to numeric")
        return df, converted
    
    def get_missing_value_stats(self, df: pl.DataFrame) -> Dict[str, Dict]:
        """
        Get statistics about missing values in the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with missing value statistics per column
        """
        stats = {}
        for col in df.columns:
            null_count = df[col].null_count()
            total = df.height
            stats[col] = {
                "null_count": null_count,
                "null_percentage": round(null_count / total * 100, 2) if total > 0 else 0,
                "non_null_count": total - null_count
            }
        return stats
    
    def impute_rolling(
        self,
        df: pl.DataFrame,
        column: str,
        method: ImputationMethod = ImputationMethod.ROLLING_MEDIAN,
        window_size: Optional[int] = None
    ) -> pl.DataFrame:
        """
        Impute missing values using rolling statistics.
        
        Args:
            df: Input DataFrame
            column: Column to impute
            method: Imputation method
            window_size: Rolling window size
            
        Returns:
            DataFrame with imputed values
        """
        window = window_size or self.config.rolling_window
        
        if method == ImputationMethod.ROLLING_MEDIAN:
            fill_values = df[column].rolling_median(window_size=window, min_periods=1)
        elif method == ImputationMethod.ROLLING_MEAN:
            fill_values = df[column].rolling_mean(window_size=window, min_periods=1)
        elif method == ImputationMethod.FORWARD_FILL:
            fill_values = df[column].forward_fill()
        elif method == ImputationMethod.BACKWARD_FILL:
            fill_values = df[column].backward_fill()
        elif method == ImputationMethod.INTERPOLATE_LINEAR:
            fill_values = df[column].interpolate()
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        df = df.with_columns(
            pl.when(pl.col(column).is_null())
            .then(fill_values)
            .otherwise(pl.col(column))
            .alias(column)
        )
        
        return df
    
    def find_best_imputation_method(
        self,
        df: pl.DataFrame,
        column: str,
        test_size: int = 24
    ) -> Tuple[ImputationMethod, float]:
        """
        Find the best imputation method by testing on a subset.
        
        Args:
            df: Input DataFrame
            column: Column to test
            test_size: Number of values to use for testing
            
        Returns:
            Tuple of (best method, RMSE)
        """
        # Get non-null values for testing
        non_null_df = df.filter(pl.col(column).is_not_null())
        
        if non_null_df.height < test_size * 2:
            return self.config.default_imputation, float('inf')
        
        # Create test set by artificially removing values
        test_indices = list(range(non_null_df.height - test_size, non_null_df.height))
        actual_values = non_null_df[column].to_numpy()[test_indices]
        
        # Create a copy with test values set to null
        test_df = non_null_df.with_row_index()
        test_df = test_df.with_columns(
            pl.when(pl.col("index").is_in(test_indices))
            .then(None)
            .otherwise(pl.col(column))
            .alias(column)
        )
        
        best_method = self.config.default_imputation
        best_rmse = float('inf')
        
        methods = [
            ImputationMethod.ROLLING_MEAN,
            ImputationMethod.ROLLING_MEDIAN,
            ImputationMethod.FORWARD_FILL,
            ImputationMethod.INTERPOLATE_LINEAR
        ]
        
        for method in methods:
            try:
                imputed_df = self.impute_rolling(test_df.clone(), column, method)
                predicted = imputed_df[column].to_numpy()[test_indices]
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((actual_values - predicted) ** 2))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_method = method
                    
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
                continue
        
        logger.info(f"Best imputation method for {column}: {best_method.value} (RMSE: {best_rmse:.4f})")
        return best_method, best_rmse
    
    def fill_missing_values(
        self,
        df: pl.DataFrame,
        columns: Optional[List[str]] = None,
        auto_select_method: bool = True
    ) -> Tuple[pl.DataFrame, Dict[str, int]]:
        """
        Fill missing values in specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to fill (None = all numeric columns)
            auto_select_method: Whether to automatically select best method
            
        Returns:
            Tuple of (DataFrame, dict of filled counts per column)
        """
        filled_counts = {}
        
        if columns is None:
            columns = [
                col for col in df.columns 
                if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
        
        for col in columns:
            null_count_before = df[col].null_count()
            
            if null_count_before == 0:
                continue
            
            if auto_select_method:
                method, _ = self.find_best_imputation_method(df, col)
            else:
                method = self.config.default_imputation
            
            df = self.impute_rolling(df, col, method)
            
            null_count_after = df[col].null_count()
            filled_counts[col] = null_count_before - null_count_after
            
            logger.info(f"Filled {filled_counts[col]} missing values in {col} using {method.value}")
        
        return df, filled_counts
    
    def detect_outliers(
        self,
        df: pl.DataFrame,
        column: str,
        method: str = "zscore",
        threshold: Optional[float] = None
    ) -> pl.Series:
        """
        Detect outliers in a column.
        
        Args:
            df: Input DataFrame
            column: Column to check
            method: Detection method ("zscore" or "iqr")
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean Series indicating outliers
        """
        threshold = threshold or self.config.outlier_std_threshold
        
        if method == "zscore":
            mean = df[column].mean()
            std = df[column].std()
            z_scores = (df[column] - mean) / std
            outliers = z_scores.abs() > threshold
        elif method == "iqr":
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers
    
    def remove_duplicates(
        self,
        df: pl.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first"
    ) -> Tuple[pl.DataFrame, int]:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to consider for duplicates
            keep: Which duplicate to keep ("first", "last", or "none")
            
        Returns:
            Tuple of (DataFrame, number of duplicates removed)
        """
        original_count = df.height
        
        if keep == "first":
            df = df.unique(subset=subset, keep="first")
        elif keep == "last":
            df = df.unique(subset=subset, keep="last")
        else:
            df = df.unique(subset=subset, keep="none")
        
        removed = original_count - df.height
        logger.info(f"Removed {removed} duplicate rows")
        
        return df, removed
    
    def clean_auction_data(
        self,
        df: pl.DataFrame,
        auto_impute: bool = True
    ) -> Tuple[pl.DataFrame, CleaningReport]:
        """
        Clean auction data with all preprocessing steps.
        
        Args:
            df: Raw auction DataFrame
            auto_impute: Whether to automatically impute missing values
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning report)
        """
        original_rows = df.height
        errors = []
        
        # Parse datetime
        date_col = self.column_config.DATE_COLUMN
        if date_col in df.columns:
            df = self.parse_datetime(df, date_col)
        
        # Convert numeric columns
        df, converted_cols = self.convert_to_numeric(df)
        
        # Remove duplicates
        df, duplicates_removed = self.remove_duplicates(
            df, 
            subset=[date_col] if date_col in df.columns else None
        )
        
        # Fill missing values
        filled_counts = {}
        if auto_impute:
            df, filled_counts = self.fill_missing_values(df, auto_select_method=True)
        
        # Sort by date if available
        if date_col in df.columns:
            df = df.sort(date_col)
        
        report = CleaningReport(
            original_rows=original_rows,
            final_rows=df.height,
            columns_converted=converted_cols,
            missing_values_filled=filled_counts,
            outliers_handled={},
            duplicates_removed=duplicates_removed,
            errors=errors
        )
        
        return df, report
    
    def clean_forecast_data(
        self,
        df: pl.DataFrame,
        auto_impute: bool = True
    ) -> Tuple[pl.DataFrame, CleaningReport]:
        """
        Clean forecast input data.
        
        Args:
            df: Raw forecast DataFrame
            auto_impute: Whether to automatically impute missing values
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning report)
        """
        return self.clean_auction_data(df, auto_impute)
    
    def clean_system_prices(
        self,
        df: pl.DataFrame,
        auto_impute: bool = True
    ) -> Tuple[pl.DataFrame, CleaningReport]:
        """
        Clean system prices data.
        
        Args:
            df: Raw system prices DataFrame
            auto_impute: Whether to automatically impute missing values
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning report)
        """
        return self.clean_auction_data(df, auto_impute)


def calculate_data_quality_score(df: pl.DataFrame) -> Dict[str, float]:
    """
    Calculate a data quality score for the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    total_cells = df.height * df.width
    null_cells = sum(df[col].null_count() for col in df.columns)
    
    completeness = 1 - (null_cells / total_cells) if total_cells > 0 else 0
    
    # Count numeric columns
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Int64]]
    
    return {
        "completeness": round(completeness * 100, 2),
        "total_rows": df.height,
        "total_columns": df.width,
        "null_cells": null_cells,
        "numeric_columns": len(numeric_cols)
    }
