"""
Model Training Module for Energy Trading Application.
Handles price forecasting and trading strategy models.
"""

import polars as pl
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Try to import optional deep learning libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
from pathlib import Path
import logging

from .config import config, ModelType, TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    rmse: float
    mae: float
    r2: float
    smape: float
    mape: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "RMSE": round(self.rmse, 4),
            "MAE": round(self.mae, 4),
            "R²": round(self.r2, 4),
            "SMAPE (%)": round(self.smape, 2),
            "MAPE (%)": round(self.mape, 2)
        }


@dataclass
class TrainingResult:
    """Container for training results."""
    model: Any
    model_type: ModelType
    metrics: ModelMetrics
    feature_importance: Optional[Dict[str, float]] = None
    scaler: Optional[StandardScaler] = None
    feature_names: List[str] = field(default_factory=list)
    
    def save(self, path: Path) -> None:
        """Save model and scaler to disk."""
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type.value
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "TrainingResult":
        """Load model from disk."""
        data = joblib.load(path)
        return cls(
            model=data["model"],
            model_type=ModelType(data["model_type"]),
            metrics=None,
            scaler=data.get("scaler"),
            feature_names=data.get("feature_names", [])
        )


class PriceForecaster:
    """
    Price forecasting model for energy trading.
    Supports multiple model types and cross-validation.
    """

    def __init__(self, model_type: ModelType = ModelType.LINEAR_REGRESSION):
        """
        Initialize the forecaster.

        Args:
            model_type: Type of model to use
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False
        self.lstm_lookback = 10  # Lookback window for LSTM

    def _build_lstm_model(self, input_shape: Tuple[int, int]) -> Any:
        """Build LSTM model architecture."""
        if not TENSORFLOW_AVAILABLE:
            raise ValueError("TensorFlow not installed. Install with: pip install tensorflow")

        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _prepare_lstm_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM by creating sequences."""
        X_seq, y_seq = [], []
        for i in range(self.lstm_lookback, len(X)):
            X_seq.append(X[i-self.lstm_lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def _create_model(self, model_type: ModelType) -> Any:
        """Create a model instance based on type."""
        if model_type == ModelType.LINEAR_REGRESSION:
            return LinearRegression()
        elif model_type == ModelType.RANDOM_FOREST:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == ModelType.XGBOOST:
            if not XGBOOST_AVAILABLE:
                logger.warning("XGBoost not installed, falling back to GradientBoosting")
                return GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == ModelType.SVR:
            return SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
        elif model_type == ModelType.ELASTIC_NET:
            return ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
        elif model_type == ModelType.MLP:
            return MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        elif model_type == ModelType.LSTM:
            # LSTM returns None here; handled specially in train method
            return None
        else:
            return LinearRegression()
    
    @staticmethod
    def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        return 100 * np.mean(
            2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted) + 1e-8)
        )
    
    @staticmethod
    def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = actual != 0
        return 100 * np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        scale_features: bool = True,
        test_size: float = 0.2
    ) -> TrainingResult:
        """
        Train the forecasting model.

        Args:
            X: Feature matrix
            y: Target values
            feature_names: Names of features
            scale_features: Whether to scale features
            test_size: Proportion of data for testing

        Returns:
            TrainingResult with model and metrics
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Time series: don't shuffle
        )

        # Scale features
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # Handle LSTM specially
        if self.model_type == ModelType.LSTM:
            if not TENSORFLOW_AVAILABLE:
                raise ValueError("TensorFlow not installed. Install with: pip install tensorflow")

            # Prepare sequences
            X_train_seq, y_train_seq = self._prepare_lstm_data(X_train_scaled, y_train)
            X_test_seq, y_test_seq = self._prepare_lstm_data(X_test_scaled, y_test)

            if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                raise ValueError(f"Not enough data for LSTM. Need at least {self.lstm_lookback + 1} samples.")

            # Build and train LSTM
            self.model = self._build_lstm_model((self.lstm_lookback, X_train.shape[1]))
            self.model.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=32,
                validation_split=0.1,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ]
            )
            self.is_fitted = True

            # Predictions
            y_pred = self.model.predict(X_test_seq, verbose=0).flatten()
            y_test_final = y_test_seq  # Use the adjusted test set

            # Feature importance not available for LSTM
            feature_importance = None

        else:
            # Standard sklearn models
            self.model = self._create_model(self.model_type)
            self.model.fit(X_train_scaled, y_train)
            self.is_fitted = True

            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            y_test_final = y_test

            # Feature importance (if available)
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            elif hasattr(self.model, 'coef_'):
                coef = self.model.coef_
                if np.ndim(coef) > 1:
                    coef = coef.flatten()
                feature_importance = dict(zip(self.feature_names, np.abs(coef)))

        # Calculate metrics
        metrics = ModelMetrics(
            rmse=np.sqrt(mean_squared_error(y_test_final, y_pred)),
            mae=mean_absolute_error(y_test_final, y_pred),
            r2=r2_score(y_test_final, y_pred),
            smape=self.calculate_smape(y_test_final, y_pred),
            mape=self.calculate_mape(y_test_final, y_pred)
        )

        logger.info(f"Model trained - RMSE: {metrics.rmse:.4f}, R²: {metrics.r2:.4f}")

        return TrainingResult(
            model=self.model,
            model_type=self.model_type,
            metrics=metrics,
            feature_importance=feature_importance,
            scaler=self.scaler if scale_features else None,
            feature_names=self.feature_names
        )
    
    def predict(self, X: np.ndarray, scale: bool = True) -> np.ndarray:
        """
        Make predictions using trained model.

        Args:
            X: Feature matrix
            scale: Whether to scale features

        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        if scale and self.scaler is not None:
            X = self.scaler.transform(X)

        if self.model_type == ModelType.LSTM:
            # Need to prepare sequences for LSTM
            if len(X) < self.lstm_lookback:
                raise ValueError(f"Need at least {self.lstm_lookback} samples for LSTM prediction")
            X_seq = []
            for i in range(self.lstm_lookback, len(X)):
                X_seq.append(X[i-self.lstm_lookback:i])
            X_seq = np.array(X_seq)
            return self.model.predict(X_seq, verbose=0).flatten()

        return self.model.predict(X)
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target values
            cv: Number of folds
            
        Returns:
            Dictionary with CV scores
        """
        model = self._create_model(self.model_type)
        scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        
        return {
            "cv_rmse_mean": np.sqrt(-scores.mean()),
            "cv_rmse_std": np.sqrt(-scores).std(),
            "cv_scores": np.sqrt(-scores).tolist()
        }


@dataclass
class TradingSignal:
    """Container for trading signal."""
    action: str  # "buy", "sell", "hold"
    confidence: float
    volume: float
    price: float
    expected_profit: float
    reason: str


class TradingStrategy:
    """
    Trading strategy implementation for energy markets.
    """
    
    def __init__(self, trading_config: Optional[TradingConfig] = None):
        """
        Initialize trading strategy.
        
        Args:
            trading_config: Trading configuration parameters
        """
        self.config = trading_config or config.trading
    
    def naive_strategy(
        self,
        forecast_price_first: float,
        forecast_price_second: float,
        actual_price_first: Optional[float] = None,
        actual_price_second: Optional[float] = None
    ) -> TradingSignal:
        """
        Naive trading strategy based on price difference threshold.
        
        Args:
            forecast_price_first: Forecasted first auction price
            forecast_price_second: Forecasted second auction price
            actual_price_first: Actual first auction price (for validation)
            actual_price_second: Actual second auction price (for validation)
            
        Returns:
            TradingSignal with action and details
        """
        price_diff = forecast_price_second - forecast_price_first
        threshold = self.config.price_diff_threshold
        volume = self.config.max_volume_per_trade
        cost = self.config.transaction_cost * 2 * volume  # Buy and sell
        
        if price_diff > threshold:
            # Buy in first, sell in second
            expected_profit = price_diff * volume - cost
            return TradingSignal(
                action="buy_first_sell_second",
                confidence=min(abs(price_diff) / threshold, 1.0),
                volume=volume,
                price=forecast_price_first,
                expected_profit=expected_profit,
                reason=f"Price expected to increase by {price_diff:.2f} GBP/MWh"
            )
        elif price_diff < -threshold:
            # Sell in first, buy in second
            expected_profit = -price_diff * volume - cost
            return TradingSignal(
                action="sell_first_buy_second",
                confidence=min(abs(price_diff) / threshold, 1.0),
                volume=volume,
                price=forecast_price_first,
                expected_profit=expected_profit,
                reason=f"Price expected to decrease by {-price_diff:.2f} GBP/MWh"
            )
        else:
            return TradingSignal(
                action="hold",
                confidence=1 - abs(price_diff) / threshold,
                volume=0,
                price=0,
                expected_profit=0,
                reason=f"Price difference ({price_diff:.2f}) below threshold ({threshold})"
            )
    
    def calculate_profit(
        self,
        signals: List[TradingSignal],
        actual_prices_first: List[float],
        actual_prices_second: List[float]
    ) -> Dict[str, float]:
        """
        Calculate actual profits from trading signals.
        
        Args:
            signals: List of trading signals
            actual_prices_first: Actual first auction prices
            actual_prices_second: Actual second auction prices
            
        Returns:
            Dictionary with profit statistics
        """
        total_profit = 0
        total_trades = 0
        winning_trades = 0
        
        for signal, price_first, price_second in zip(signals, actual_prices_first, actual_prices_second):
            if signal.action == "hold":
                continue
            
            total_trades += 1
            cost = self.config.transaction_cost * 2 * signal.volume
            
            if signal.action == "buy_first_sell_second":
                profit = (price_second - price_first) * signal.volume - cost
            else:  # sell_first_buy_second
                profit = (price_first - price_second) * signal.volume - cost
            
            total_profit += profit
            if profit > 0:
                winning_trades += 1
        
        return {
            "total_profit": total_profit,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "average_profit_per_trade": total_profit / total_trades if total_trades > 0 else 0
        }


def prepare_training_data(
    df: pl.DataFrame,
    feature_columns: List[str],
    target_column: str,
    drop_na: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for model training.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
        target_column: Target column name
        drop_na: Whether to drop rows with missing values
        
    Returns:
        Tuple of (X, y, valid_feature_names)
    """
    # Filter to columns that exist
    valid_features = [col for col in feature_columns if col in df.columns]
    
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column} not found in DataFrame")
    
    # Select columns
    subset = df.select(valid_features + [target_column])
    
    # Drop NaN if requested
    if drop_na:
        subset = subset.drop_nulls()
    
    # Convert to numpy
    X = subset.select(valid_features).to_numpy()
    y = subset.select(target_column).to_numpy().ravel()
    
    logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
    
    return X, y, valid_features
