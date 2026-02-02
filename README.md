# ⚡ Energy Trading Analytics Platform

A comprehensive data engineering application for analyzing UK day-ahead electricity market data, built with **Polars**, clean Python code, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Polars](https://img.shields.io/badge/Polars-0.20+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Data Engineering Best Practices](#-data-engineering-best-practices)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)

## 🎯 Features

### Data Loading
- **Multiple Sources**: Upload CSV files or load from a folder
- **Automatic Schema Detection**: Smart data type inference
- **Validation**: Built-in data validation and error handling

### Data Cleaning
- **Smart Imputation**: Auto-selects the best method (rolling mean/median, interpolation)
- **Type Conversion**: Automatic numeric type conversion
- **Duplicate Handling**: Configurable duplicate removal
- **Missing Value Analysis**: Visual reports of data quality

### Visualization
- **Interactive Charts**: Built with Plotly for full interactivity
- **Price Time Series**: Track price movements over time
- **Distribution Analysis**: Understand price distributions
- **Correlation Heatmaps**: Identify feature relationships
- **Hourly Patterns**: Discover time-based patterns

### Model Training
- **8 Model Types**:
  - Traditional ML: Linear Regression, Random Forest, Gradient Boosting, XGBoost
  - Regularized: ElasticNet, SVR (Support Vector Regression)
  - Deep Learning: MLP Neural Network, LSTM (Long Short-Term Memory)
- **Train All Models**: One-click training of all available models
- **Feature Engineering**: Automatic lag, rolling, and time features
- **Cross-Validation**: Robust model evaluation
- **Feature Importance**: Understand what drives predictions

### Model Comparison
- **Side-by-Side Metrics**: Compare RMSE, MAE, R², SMAPE, MAPE across models
- **Visual Comparison**: Bar charts and radar plots for multi-metric analysis
- **Prediction Overlay**: Compare model predictions against actual values
- **Residual Analysis**: Box plots and error statistics for each model
- **Best Model Selection**: Automatic highlighting of best performers

### Trading Strategy
- **Naive Strategy**: Rule-based trading signals
- **Full Backtesting**: Complete historical strategy evaluation
- **Profit Analysis**: Track cumulative returns and trade statistics
- **Win Rate Tracking**: Monitor winning vs losing trades
- **Price Spread Analysis**: Visualize threshold optimization

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                        │
├─────────────────────────────────────────────────────────────────┤
│  Pages: Home | Data Loading | Cleaning | Visualization          │
│         Model Training | Predictions | Model Comparison          │
│         Trading Strategy                                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌─────────────────┐  │
│  │  Data     │ │  Data     │ │  Feature  │ │  ML Models      │  │
│  │  Loader   │ │  Cleaner  │ │  Engineer │ │  (8 types)      │  │
│  └───────────┘ └───────────┘ └───────────┘ └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Models: LinearReg | RandomForest | GradientBoosting | XGBoost   │
│          ElasticNet | SVR | MLP Neural Net | LSTM                │
├─────────────────────────────────────────────────────────────────┤
│                      Polars DataFrames                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐    │
│  │  auction_data   │  │ forecast_inputs │  │ system_prices │    │
│  │     .csv        │  │     .csv        │  │     .csv      │    │
│  └─────────────────┘  └─────────────────┘  └───────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Quick Start with uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. Install it first:

```bash
# Install uv (Windows)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project:

```bash
# Clone or download the project
cd energy_trading_app

# Create virtual environment and install dependencies
uv sync

# Optional: Install deep learning support (for LSTM model)
uv sync --extra deep-learning

# Run the application
uv run streamlit run app.py
```

### Quick Start with pip

```bash
# Clone or download the project
cd energy_trading_app

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install deep learning support (for LSTM model)
pip install tensorflow>=2.15.0

# Run the application
streamlit run app.py
```

### Optional Dependencies

| Package | Purpose | uv Command | pip Command |
|---------|---------|------------|-------------|
| tensorflow | LSTM deep learning | `uv sync --extra deep-learning` | `pip install tensorflow` |
| dev tools | Testing, linting | `uv sync --extra dev` | `pip install pytest black ruff` |
| all extras | Everything | `uv sync --all-extras` | - |

**Note**: XGBoost is included by default. TensorFlow is optional and only needed for the LSTM model.

### Docker (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

## 📖 Usage

### 1. Start the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 2. Load Data

**Option A: Upload Files**
- Navigate to "Data Loading" page
- Upload your CSV files (auction_data.csv required)

**Option B: Load from Folder**
- Set the folder path in sidebar settings
- Click "Load from Folder"

### 3. Clean Data

- Navigate to "Data Cleaning" page
- Configure cleaning options
- Click "Clean Data"
- Optionally merge all datasets

### 4. Explore Visualizations

- Navigate to "Visualization" page
- Select chart type
- Customize parameters

### 5. Train Models

- Navigate to "Model Training" page
- Select features and model type
- **Single Model**: Click "Train Selected Model"
- **All Models**: Click "Train All Models" to train all 8 model types at once
- Review performance metrics and feature importance

### 6. Compare Models

- Navigate to "Model Comparison" page
- View side-by-side metrics comparison table
- Explore visual comparisons (bar charts, radar plots)
- Compare predictions across models
- Analyze residuals and error distributions
- Get recommendations for best model selection

### 7. Evaluate Trading Strategy

- Navigate to "Trading Strategy" page
- Configure strategy parameters
- Analyze results

## 📁 Project Structure

```
energy_trading_app/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration and constants
│   ├── data_loader.py       # Data loading utilities
│   ├── data_cleaner.py      # Data cleaning and preprocessing
│   ├── feature_engineering.py # Feature creation
│   ├── models.py            # ML models and trading strategy
│   └── visualizations.py    # Plotly visualizations
├── data/                    # Data directory (create if needed)
│   ├── auction_data.csv
│   ├── forecast_inputs.csv
│   └── system_prices.csv
├── models/                  # Saved models directory
└── tests/                   # Unit tests
```

## 🛠 Data Engineering Best Practices

This project follows industry best practices:

### 1. **Separation of Concerns**
Each module has a single responsibility:
- `data_loader.py`: Only handles data ingestion
- `data_cleaner.py`: Only handles data cleaning
- `models.py`: Only handles ML operations

### 2. **Configuration Management**
All settings in `config.py`:
```python
from src.config import config

# Access settings
config.trading.transaction_cost  # 5.0
config.columns.PRICE_FIRST_AUCTION  # "price_first_auction"
```

### 3. **Type Hints**
Full type annotations for better IDE support and documentation:
```python
def load_csv(
    self,
    source: Union[str, Path, BinaryIO],
    delimiter: str = ";",
) -> pl.DataFrame:
```

### 4. **Dataclasses for Data Containers**
Structured data with automatic methods:
```python
@dataclass
class CleaningReport:
    original_rows: int
    final_rows: int
    columns_converted: List[str]
```

### 5. **Polars for Performance**
Using Polars instead of Pandas for:
- Faster processing (10-100x)
- Lower memory usage
- Lazy evaluation support
- Better type system

### 6. **Logging**
Consistent logging throughout:
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Data loaded successfully")
```

### 7. **Error Handling**
Graceful error handling with informative messages:
```python
try:
    df = loader.load_auction_data(file)
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
```

## 📚 API Reference

### DataLoader

```python
from src import DataLoader

loader = DataLoader(base_path=Path("data"))

# Load single file
df = loader.load_auction_data("auction_data.csv")

# Load all from directory
result = loader.load_all_from_directory()
print(result.auction_data)
```

### DataCleaner

```python
from src import DataCleaner, ImputationMethod

cleaner = DataCleaner()

# Clean data
cleaned_df, report = cleaner.clean_auction_data(df, auto_impute=True)

# Check report
print(f"Filled {sum(report.missing_values_filled.values())} values")
```

### FeatureEngineer

```python
from src import FeatureEngineer

fe = FeatureEngineer()

# Create features
feature_set = fe.create_feature_set(
    df,
    target_column="price_second_auction",
    include_lags=True,
    include_rolling=True
)

print(f"Created {len(feature_set.feature_names)} features")
```

### PriceForecaster

```python
from src import PriceForecaster, ModelType

# Available model types:
# ModelType.LINEAR_REGRESSION - Fast, interpretable baseline
# ModelType.RANDOM_FOREST     - Ensemble, handles non-linearity
# ModelType.GRADIENT_BOOSTING - Sequential boosting
# ModelType.XGBOOST           - Optimized gradient boosting
# ModelType.SVR               - Support Vector Regression
# ModelType.ELASTIC_NET       - L1+L2 regularization
# ModelType.MLP               - Neural Network
# ModelType.LSTM              - Deep learning for sequences

forecaster = PriceForecaster(ModelType.RANDOM_FOREST)

# Train
result = forecaster.train(X, y, feature_names)
print(f"RMSE: {result.metrics.rmse}")
print(f"R²: {result.metrics.r2}")

# Predict
predictions = forecaster.predict(X_new)

# Train all models for comparison
from src.config import ModelType
results = {}
for model_type in ModelType:
    forecaster = PriceForecaster(model_type)
    results[model_type.value] = forecaster.train(X, y, feature_names)
```

### TradingStrategy

```python
from src import TradingStrategy

strategy = TradingStrategy()

# Generate signal
signal = strategy.naive_strategy(
    forecast_price_first=100.0,
    forecast_price_second=115.0
)

print(f"Action: {signal.action}, Expected Profit: £{signal.expected_profit:.2f}")
```

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
@dataclass
class TradingConfig:
    transaction_cost: float = 5.0      # GBP per MW
    max_volume_per_trade: float = 10.0 # MW
    price_diff_threshold: float = 10.0 # GBP/MWh
```

## 📊 Data Format

### auction_data.csv

| Column | Type | Description |
|--------|------|-------------|
| Date (WET) | datetime | Timestamp in Western European Time |
| price_first_auction | float | First auction clearing price (GBP/MWh) |
| price_second_auction | float | Second auction clearing price (GBP/MWh) |
| traded_volume_first_auction | float | Volume traded in first auction (MW) |
| traded_volume_second_auction | float | Volume traded in second auction (MW) |
| price_forecast_first_auction | float | Forecasted first auction price |

### forecast_inputs.csv

Contains variables for price forecasting:
- demand, margin, availability
- wind/solar forecasts
- system indicators

### system_prices.csv

| Column | Type | Description |
|--------|------|-------------|
| Date (WET) | datetime | Timestamp |
| system_price | float | Grid balancing price |
| forecast_system_price_low | float | Low forecast |
| forecast_system_price_high | float | High forecast |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- UK Energy Trading Dataset from Kaggle
- Polars team for the amazing DataFrame library
- Streamlit team for the web framework
