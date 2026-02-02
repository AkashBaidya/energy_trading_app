"""
Configuration settings for Energy Trading Application.
Contains all constants, paths, and configuration parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
from enum import Enum


class ImputationMethod(Enum):
    """Supported missing value imputation methods."""
    ROLLING_MEAN = "rolling_mean"
    ROLLING_MEDIAN = "rolling_median"
    INTERPOLATE_LINEAR = "interpolate_linear"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"


class ModelType(Enum):
    """Supported model types for price forecasting."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    SVR = "svr"
    ELASTIC_NET = "elastic_net"
    MLP = "mlp"
    LSTM = "lstm"


@dataclass
class DataPaths:
    """Data file paths configuration."""
    base_dir: Path = field(default_factory=lambda: Path("data"))
    auction_data: str = "auction_data.csv"
    forecast_inputs: str = "forecast_inputs.csv"
    system_prices: str = "system_prices.csv"
    
    def get_full_path(self, filename: str) -> Path:
        return self.base_dir / filename


@dataclass
class ColumnConfig:
    """Column name mappings and configurations."""
    
    # Auction data columns
    DATE_COLUMN: str = "Date (WET)"
    PRICE_FIRST_AUCTION: str = "price_first_auction"
    PRICE_SECOND_AUCTION: str = "price_second_auction"
    VOLUME_FIRST_AUCTION: str = "traded_volume_first_auction"
    VOLUME_SECOND_AUCTION: str = "traded_volume_second_auction"
    FORECAST_FIRST_AUCTION: str = "price_forecast_first_auction"
    
    # System prices columns
    SYSTEM_PRICE: str = "system_price"
    FORECAST_SYSTEM_LOW: str = "forecast_system_price_low"
    FORECAST_SYSTEM_HIGH: str = "forecast_system_price_high"
    
    # Feature columns for prediction
    FEATURE_COLUMNS: List[str] = field(default_factory=lambda: [
        "demand",
        "within_day_availability",
        "within_day_margin",
        "margin",
        "long_term_wind",
        "long_term_wind_over_demand",
        "long_term_wind_over_margin",
        "long_term_solar",
        "long_term_solar_over_demand",
        "long_term_solar_over_margin",
        "margin_over_demand",
        "snsp_forecast",
        "stack_price",
        "within_day_potential_stack_price",
        "previous_day_ahead_price",
        "previous_continuous_half_hour_vwap",
        "inertia_forecast",
    ])
    
    # Numeric columns that need type conversion
    NUMERIC_COLUMNS: List[str] = field(default_factory=lambda: [
        "price_first_auction",
        "price_second_auction",
        "traded_volume_first_auction",
        "traded_volume_second_auction",
        "price_forecast_first_auction",
    ])


@dataclass
class CleaningConfig:
    """Data cleaning configuration parameters."""
    rolling_window: int = 24  # Hours for rolling calculations
    min_periods: int = 1
    default_imputation: ImputationMethod = ImputationMethod.ROLLING_MEDIAN
    outlier_std_threshold: float = 3.0
    date_format: str = "[%d/%m/%Y %H:%M]"
    skip_first_row: bool = True  # Skip units row


@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    transaction_cost: float = 5.0  # GBP per MW
    max_volume_per_trade: float = 10.0  # MW
    price_diff_threshold: float = 10.0  # GBP/MWh for naive strategy
    train_test_split: float = 0.8


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    figure_width: int = 12
    figure_height: int = 6
    color_palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
    ])
    theme: str = "plotly_white"


@dataclass
class AppConfig:
    """Main application configuration."""
    data_paths: DataPaths = field(default_factory=DataPaths)
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # App settings
    app_title: str = "Energy Trading Analytics Platform"
    app_icon: str = "⚡"
    page_layout: str = "wide"


# Global configuration instance
config = AppConfig()
