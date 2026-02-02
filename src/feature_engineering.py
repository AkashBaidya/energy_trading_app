"""
Feature Engineering Module for Energy Trading Application.
Handles feature creation and transformation using Polars.
"""

import polars as pl
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import logging

from .config import config

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for engineered features."""
    features: pl.DataFrame
    feature_names: List[str]
    target_column: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FeatureEngineer:
    """
    Feature engineering class for energy trading data.
    Creates technical indicators and derived features.
    """
    
    def __init__(self):
        self.column_config = config.columns
    
    def add_time_features(self, df: pl.DataFrame, date_column: str) -> pl.DataFrame:
        """
        Add time-based features from datetime column.
        
        Args:
            df: Input DataFrame
            date_column: Name of datetime column
            
        Returns:
            DataFrame with added time features
        """
        if date_column not in df.columns:
            logger.warning(f"Date column {date_column} not found")
            return df
        
        df = df.with_columns([
            pl.col(date_column).dt.hour().alias("hour"),
            pl.col(date_column).dt.weekday().alias("day_of_week"),
            pl.col(date_column).dt.month().alias("month"),
            pl.col(date_column).dt.day().alias("day_of_month"),
            pl.col(date_column).dt.ordinal_day().alias("day_of_year"),
            pl.col(date_column).dt.week().alias("week_of_year"),
        ])
        
        # Add cyclical features for hour and month
        df = df.with_columns([
            (np.pi * 2 * pl.col("hour") / 24).sin().alias("hour_sin"),
            (np.pi * 2 * pl.col("hour") / 24).cos().alias("hour_cos"),
            (np.pi * 2 * pl.col("month") / 12).sin().alias("month_sin"),
            (np.pi * 2 * pl.col("month") / 12).cos().alias("month_cos"),
        ])
        
        # Add weekend indicator
        df = df.with_columns([
            (pl.col("day_of_week") >= 5).cast(pl.Int8).alias("is_weekend")
        ])
        
        logger.info("Added time-based features")
        return df
    
    def add_lag_features(
        self,
        df: pl.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 24, 48, 168]
    ) -> pl.DataFrame:
        """
        Add lag features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                df = df.with_columns(
                    pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")
                )
        
        logger.info(f"Added lag features for {len(columns)} columns")
        return df
    
    def add_rolling_features(
        self,
        df: pl.DataFrame,
        columns: List[str],
        windows: List[int] = [3, 6, 12, 24]
    ) -> pl.DataFrame:
        """
        Add rolling statistics features.
        
        Args:
            df: Input DataFrame
            columns: Columns to calculate rolling stats for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                df = df.with_columns([
                    pl.col(col).rolling_mean(window_size=window).alias(f"{col}_rolling_mean_{window}"),
                    pl.col(col).rolling_std(window_size=window).alias(f"{col}_rolling_std_{window}"),
                    pl.col(col).rolling_min(window_size=window).alias(f"{col}_rolling_min_{window}"),
                    pl.col(col).rolling_max(window_size=window).alias(f"{col}_rolling_max_{window}"),
                ])
        
        logger.info(f"Added rolling features for {len(columns)} columns")
        return df
    
    def add_price_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add price-related derived features.
        
        Args:
            df: Input DataFrame with price columns
            
        Returns:
            DataFrame with price features
        """
        price_first = self.column_config.PRICE_FIRST_AUCTION
        price_second = self.column_config.PRICE_SECOND_AUCTION
        forecast_first = self.column_config.FORECAST_FIRST_AUCTION
        
        # Price difference
        if price_first in df.columns and price_second in df.columns:
            df = df.with_columns([
                (pl.col(price_second) - pl.col(price_first)).alias("price_diff"),
                ((pl.col(price_second) - pl.col(price_first)) / pl.col(price_first) * 100).alias("price_diff_pct"),
            ])
        
        # Forecast error
        if price_first in df.columns and forecast_first in df.columns:
            df = df.with_columns([
                (pl.col(price_first) - pl.col(forecast_first)).alias("forecast_error_first"),
                ((pl.col(price_first) - pl.col(forecast_first)).abs() / pl.col(price_first) * 100).alias("forecast_error_pct"),
            ])
        
        # Price volatility (using rolling std)
        if price_first in df.columns:
            df = df.with_columns([
                pl.col(price_first).rolling_std(window_size=24).alias("price_volatility_24h"),
            ])
        
        logger.info("Added price-related features")
        return df
    
    def add_volume_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add volume-related features.
        
        Args:
            df: Input DataFrame with volume columns
            
        Returns:
            DataFrame with volume features
        """
        vol_first = self.column_config.VOLUME_FIRST_AUCTION
        vol_second = self.column_config.VOLUME_SECOND_AUCTION
        
        if vol_first in df.columns and vol_second in df.columns:
            df = df.with_columns([
                (pl.col(vol_first) + pl.col(vol_second)).alias("total_volume"),
                (pl.col(vol_second) / pl.col(vol_first)).alias("volume_ratio"),
            ])
        
        if vol_first in df.columns:
            df = df.with_columns([
                pl.col(vol_first).rolling_mean(window_size=24).alias("volume_rolling_mean_24h"),
            ])
        
        logger.info("Added volume-related features")
        return df
    
    def add_system_price_features(
        self,
        df: pl.DataFrame,
        system_df: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """
        Add system price features.
        
        Args:
            df: Main DataFrame
            system_df: System prices DataFrame (optional)
            
        Returns:
            DataFrame with system price features
        """
        sys_price = self.column_config.SYSTEM_PRICE
        sys_low = self.column_config.FORECAST_SYSTEM_LOW
        sys_high = self.column_config.FORECAST_SYSTEM_HIGH
        
        # Calculate forecast system average if both bounds exist
        if sys_low in df.columns and sys_high in df.columns:
            df = df.with_columns([
                ((pl.col(sys_low) + pl.col(sys_high)) / 2).alias("forecast_system_avg"),
            ])
        
        # System price spread
        if sys_low in df.columns and sys_high in df.columns:
            df = df.with_columns([
                (pl.col(sys_high) - pl.col(sys_low)).alias("system_price_spread"),
            ])
        
        logger.info("Added system price features")
        return df
    
    def create_feature_set(
        self,
        df: pl.DataFrame,
        target_column: Optional[str] = None,
        include_lags: bool = True,
        include_rolling: bool = True,
        include_time: bool = True
    ) -> FeatureSet:
        """
        Create a complete feature set for modeling.
        
        Args:
            df: Input DataFrame
            target_column: Target variable column name
            include_lags: Whether to include lag features
            include_rolling: Whether to include rolling features
            include_time: Whether to include time features
            
        Returns:
            FeatureSet with all engineered features
        """
        date_col = self.column_config.DATE_COLUMN
        
        # Add time features
        if include_time and date_col in df.columns:
            df = self.add_time_features(df, date_col)
        
        # Add price features
        df = self.add_price_features(df)
        
        # Add volume features
        df = self.add_volume_features(df)
        
        # Add system price features
        df = self.add_system_price_features(df)
        
        # Add lag features for key columns
        if include_lags:
            price_cols = [
                self.column_config.PRICE_FIRST_AUCTION,
                self.column_config.PRICE_SECOND_AUCTION
            ]
            df = self.add_lag_features(df, [c for c in price_cols if c in df.columns])
        
        # Add rolling features
        if include_rolling:
            price_cols = [self.column_config.PRICE_FIRST_AUCTION]
            df = self.add_rolling_features(df, [c for c in price_cols if c in df.columns])
        
        # Get feature names (exclude date and target)
        exclude_cols = {date_col, target_column} if target_column else {date_col}
        feature_names = [col for col in df.columns if col not in exclude_cols]
        
        return FeatureSet(
            features=df,
            feature_names=feature_names,
            target_column=target_column,
            metadata={
                "include_lags": include_lags,
                "include_rolling": include_rolling,
                "include_time": include_time
            }
        )


def merge_datasets(
    auction_df: pl.DataFrame,
    forecast_df: Optional[pl.DataFrame] = None,
    system_df: Optional[pl.DataFrame] = None,
    date_column: str = "Date (WET)"
) -> pl.DataFrame:
    """
    Merge all datasets on the date column.
    
    Args:
        auction_df: Auction data DataFrame
        forecast_df: Forecast inputs DataFrame
        system_df: System prices DataFrame
        date_column: Column to merge on
        
    Returns:
        Merged DataFrame
    """
    result = auction_df
    
    if forecast_df is not None and date_column in forecast_df.columns:
        # Get columns to add (exclude date column)
        cols_to_add = [col for col in forecast_df.columns if col != date_column]
        forecast_subset = forecast_df.select([date_column] + cols_to_add)
        result = result.join(forecast_subset, on=date_column, how="left")
        logger.info(f"Merged forecast data: added {len(cols_to_add)} columns")
    
    if system_df is not None and date_column in system_df.columns:
        cols_to_add = [col for col in system_df.columns if col != date_column]
        system_subset = system_df.select([date_column] + cols_to_add)
        result = result.join(system_subset, on=date_column, how="left")
        logger.info(f"Merged system data: added {len(cols_to_add)} columns")
    
    return result
