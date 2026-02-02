"""
Energy Trading Analytics Platform - Streamlit Application

A comprehensive data engineering application for:
- Loading and cleaning energy trading data
- Feature engineering and visualization
- Price forecasting and trading strategy evaluation
"""

import streamlit as st
import polars as pl
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
from typing import Optional
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config, ModelType, ImputationMethod
from src.data_loader import DataLoader, DataLoadResult
from src.data_cleaner import DataCleaner, CleaningReport, calculate_data_quality_score
from src.feature_engineering import FeatureEngineer, merge_datasets
from src.models import PriceForecaster, TradingStrategy, prepare_training_data
from src.visualizations import EnergyVisualizer, create_summary_dashboard


# Page configuration
st.set_page_config(
    page_title=config.app_title,
    page_icon=config.app_icon,
    layout=config.page_layout,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def get_first_valid_df(*dataframes):
    """Return the first non-None DataFrame from the arguments."""
    for df in dataframes:
        if df is not None:
            return df
    return None


def init_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'auction_data' not in st.session_state:
        st.session_state.auction_data = None
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = None
    if 'system_data' not in st.session_state:
        st.session_state.system_data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'merged_data' not in st.session_state:
        st.session_state.merged_data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'training_result' not in st.session_state:
        st.session_state.training_result = None
    # Multi-model comparison storage
    if 'all_model_results' not in st.session_state:
        st.session_state.all_model_results = {}
    if 'model_predictions' not in st.session_state:
        st.session_state.model_predictions = {}


def render_sidebar():
    """Render sidebar navigation and options."""
    st.sidebar.markdown(f"## {config.app_icon} Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "📁 Data Loading", "🧹 Data Cleaning", "📊 Visualization",
         "🤖 Model Training", "📈 Predictions", "🔬 Model Comparison", "💹 Trading Strategy"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    
    # Data source option
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Upload Files", "Load from Folder"]
    )
    
    if data_source == "Load from Folder":
        folder_path = st.sidebar.text_input(
            "Data Folder Path",
            value="data"
        )
        st.session_state.folder_path = folder_path
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Data Status")
    
    if st.session_state.data_loaded:
        st.sidebar.success("✅ Data Loaded")
        if st.session_state.auction_data is not None:
            st.sidebar.info(f"Auction: {st.session_state.auction_data.height:,} rows")
        if st.session_state.forecast_data is not None:
            st.sidebar.info(f"Forecast: {st.session_state.forecast_data.height:,} rows")
    else:
        st.sidebar.warning("⚠️ No Data Loaded")
    
    if st.session_state.model_trained:
        st.sidebar.success("✅ Model Trained")
    
    return page, data_source


def render_home():
    """Render home page."""
    st.markdown('<p class="main-header">⚡ Energy Trading Analytics Platform</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the **Energy Trading Analytics Platform**! This application provides 
    comprehensive tools for analyzing UK day-ahead electricity market data.
    
    ### 🎯 Key Features
    
    - **📁 Data Loading**: Upload CSV files or load from a folder
    - **🧹 Data Cleaning**: Automated preprocessing with smart imputation
    - **📊 Visualization**: Interactive charts and dashboards
    - **🤖 Model Training**: Multiple forecasting algorithms
    - **📈 Predictions**: Price forecasting with confidence intervals
    - **💹 Trading Strategy**: Evaluate and backtest strategies
    
    ### 🚀 Getting Started
    
    1. Navigate to **Data Loading** to upload your data
    2. Use **Data Cleaning** to preprocess and validate
    3. Explore your data in **Visualization**
    4. Train models in **Model Training**
    5. Make predictions and evaluate strategies
    
    ### 📚 Data Requirements
    
    The application expects the following CSV files:
    - `auction_data.csv` - Auction prices and volumes
    - `forecast_inputs.csv` - Input variables for forecasting
    - `system_prices.csv` - Grid balancing prices (optional)
    """)
    
    # Quick stats if data is loaded
    if st.session_state.data_loaded and st.session_state.auction_data is not None:
        st.markdown('<p class="section-header">📊 Quick Overview</p>', unsafe_allow_html=True)
        
        df = st.session_state.auction_data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{df.height:,}")
        with col2:
            st.metric("Features", f"{df.width}")
        with col3:
            quality = calculate_data_quality_score(df)
            st.metric("Data Completeness", f"{quality['completeness']}%")
        with col4:
            if config.columns.PRICE_FIRST_AUCTION in df.columns:
                avg_price = df[config.columns.PRICE_FIRST_AUCTION].mean()
                st.metric("Avg Price", f"£{avg_price:.2f}")


def render_data_loading(data_source: str):
    """Render data loading page."""
    st.markdown('<p class="section-header">📁 Data Loading</p>', unsafe_allow_html=True)
    
    loader = DataLoader()
    
    if data_source == "Upload Files":
        st.markdown("### Upload Your Data Files")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            auction_file = st.file_uploader(
                "Auction Data (CSV)",
                type=['csv'],
                key="auction_upload",
                help="Upload auction_data.csv"
            )
        
        with col2:
            forecast_file = st.file_uploader(
                "Forecast Inputs (CSV)",
                type=['csv'],
                key="forecast_upload",
                help="Upload forecast_inputs.csv (optional)"
            )
        
        with col3:
            system_file = st.file_uploader(
                "System Prices (CSV)",
                type=['csv'],
                key="system_upload",
                help="Upload system_prices.csv (optional)"
            )
        
        if st.button("🔄 Load Data", type="primary"):
            with st.spinner("Loading data..."):
                try:
                    if auction_file is not None:
                        # Reset file position
                        auction_file.seek(0)
                        content = auction_file.read().decode('utf-8')
                        st.session_state.auction_data = loader.load_auction_data(
                            io.StringIO(content)
                        )
                        st.session_state.data_loaded = True
                    
                    if forecast_file is not None:
                        forecast_file.seek(0)
                        content = forecast_file.read().decode('utf-8')
                        st.session_state.forecast_data = loader.load_forecast_inputs(
                            io.StringIO(content)
                        )
                    
                    if system_file is not None:
                        system_file.seek(0)
                        content = system_file.read().decode('utf-8')
                        st.session_state.system_data = loader.load_system_prices(
                            io.StringIO(content)
                        )
                    
                    if st.session_state.data_loaded:
                        st.success("✅ Data loaded successfully!")
                    else:
                        st.warning("⚠️ Please upload at least the auction data file.")
                        
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
    
    else:  # Load from folder
        st.markdown("### Load from Folder")
        
        folder_path = st.session_state.get('folder_path', 'data')
        st.info(f"📂 Looking for files in: `{folder_path}`")
        
        if st.button("🔄 Load from Folder", type="primary"):
            with st.spinner("Loading data from folder..."):
                try:
                    loader = DataLoader(base_path=Path(folder_path))
                    result = loader.load_all_from_directory()
                    
                    if result.auction_data is not None:
                        st.session_state.auction_data = result.auction_data
                        st.session_state.data_loaded = True
                    
                    if result.forecast_inputs is not None:
                        st.session_state.forecast_data = result.forecast_inputs
                    
                    if result.system_prices is not None:
                        st.session_state.system_data = result.system_prices
                    
                    if result.errors:
                        for name, error in result.errors.items():
                            st.warning(f"⚠️ {name}: {error}")
                    
                    if st.session_state.data_loaded:
                        st.success("✅ Data loaded successfully!")
                        
                except Exception as e:
                    st.error(f"❌ Error loading data: {str(e)}")
    
    # Display loaded data preview
    if st.session_state.data_loaded and st.session_state.auction_data is not None:
        st.markdown("### 📋 Data Preview")
        
        tabs = st.tabs(["Auction Data", "Forecast Inputs", "System Prices"])
        
        with tabs[0]:
            if st.session_state.auction_data is not None:
                st.dataframe(
                    st.session_state.auction_data.head(100).to_pandas(),
                    use_container_width=True
                )
                st.caption(f"Showing 100 of {st.session_state.auction_data.height:,} rows")
        
        with tabs[1]:
            if st.session_state.forecast_data is not None:
                st.dataframe(
                    st.session_state.forecast_data.head(100).to_pandas(),
                    use_container_width=True
                )
            else:
                st.info("No forecast data loaded")
        
        with tabs[2]:
            if st.session_state.system_data is not None:
                st.dataframe(
                    st.session_state.system_data.head(100).to_pandas(),
                    use_container_width=True
                )
            else:
                st.info("No system price data loaded")


def render_data_cleaning():
    """Render data cleaning page."""
    st.markdown('<p class="section-header">🧹 Data Cleaning</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
        return
    
    cleaner = DataCleaner()
    
    # Cleaning options
    st.markdown("### ⚙️ Cleaning Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_impute = st.checkbox("Auto-select imputation method", value=True)
        imputation_method = st.selectbox(
            "Manual Imputation Method",
            [m.value for m in ImputationMethod],
            disabled=auto_impute
        )
    
    with col2:
        remove_duplicates = st.checkbox("Remove duplicates", value=True)
        handle_outliers = st.checkbox("Detect outliers", value=False)
    
    if st.button("🧹 Clean Data", type="primary"):
        with st.spinner("Cleaning data..."):
            try:
                # Clean auction data
                cleaned_auction, report = cleaner.clean_auction_data(
                    st.session_state.auction_data,
                    auto_impute=auto_impute
                )
                st.session_state.cleaned_data = cleaned_auction
                
                # Display cleaning report
                st.markdown("### 📊 Cleaning Report")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Rows", f"{report.original_rows:,}")
                with col2:
                    st.metric("Final Rows", f"{report.final_rows:,}")
                with col3:
                    st.metric("Duplicates Removed", f"{report.duplicates_removed:,}")
                with col4:
                    total_filled = sum(report.missing_values_filled.values())
                    st.metric("Values Imputed", f"{total_filled:,}")
                
                if report.missing_values_filled:
                    st.markdown("#### Missing Values Filled")
                    st.json(report.missing_values_filled)
                
                if report.columns_converted:
                    st.markdown("#### Columns Converted to Numeric")
                    st.write(", ".join(report.columns_converted))
                
                st.success("✅ Data cleaned successfully!")
                
            except Exception as e:
                st.error(f"❌ Error cleaning data: {str(e)}")
    
    # Merge datasets
    st.markdown("### 🔗 Merge Datasets")
    
    if st.button("Merge All Datasets"):
        with st.spinner("Merging datasets..."):
            try:
                base_df = get_first_valid_df(st.session_state.cleaned_data, st.session_state.auction_data)
                
                merged = merge_datasets(
                    base_df,
                    st.session_state.forecast_data,
                    st.session_state.system_data
                )
                
                st.session_state.merged_data = merged
                st.success(f"✅ Merged dataset: {merged.height:,} rows, {merged.width} columns")
                
            except Exception as e:
                st.error(f"❌ Error merging data: {str(e)}")
    
    # Data quality report
    if st.session_state.cleaned_data is not None:
        st.markdown("### 📈 Data Quality Score")
        
        quality = calculate_data_quality_score(st.session_state.cleaned_data)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Completeness", f"{quality['completeness']}%")
        with col2:
            st.metric("Total Cells", f"{quality['total_rows'] * quality['total_columns']:,}")
        with col3:
            st.metric("Null Cells", f"{quality['null_cells']:,}")
    
    # Download cleaned data
    if st.session_state.cleaned_data is not None:
        st.markdown("### 💾 Export Cleaned Data")
        
        csv_data = st.session_state.cleaned_data.write_csv()
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv_data,
            file_name="cleaned_auction_data.csv",
            mime="text/csv"
        )


def render_visualization():
    """Render visualization page."""
    st.markdown('<p class="section-header">📊 Data Visualization</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
        return
    
    viz = EnergyVisualizer()
    df = get_first_valid_df(st.session_state.merged_data, st.session_state.cleaned_data, st.session_state.auction_data)
    
    # Visualization options
    viz_type = st.selectbox(
        "Select Visualization",
        ["Price Time Series", "Price Distribution", "Missing Values", 
         "Correlation Heatmap", "Hourly Patterns", "Custom Chart"]
    )
    
    if viz_type == "Price Time Series":
        fig = viz.plot_price_timeseries(df)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Price Distribution":
        fig = viz.plot_price_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Missing Values":
        fig = viz.plot_missing_values(df)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Correlation Heatmap":
        numeric_cols = [col for col in df.columns 
                       if df[col].dtype in [pl.Float64, pl.Int64]]
        selected_cols = st.multiselect(
            "Select columns for correlation",
            numeric_cols,
            default=numeric_cols[:10]
        )
        if selected_cols:
            fig = viz.plot_correlation_heatmap(df, selected_cols)
            st.plotly_chart(fig, use_container_width=True)
            
    elif viz_type == "Hourly Patterns":
        # Add time features if not present
        fe = FeatureEngineer()
        date_col = config.columns.DATE_COLUMN
        if date_col in df.columns and 'hour' not in df.columns:
            df = fe.add_time_features(df, date_col)
        
        numeric_cols = [col for col in df.columns 
                       if df[col].dtype in [pl.Float64, pl.Int64] and col != 'hour']
        selected_col = st.selectbox("Select column", numeric_cols)
        
        if selected_col and 'hour' in df.columns:
            fig = viz.plot_hourly_patterns(df, selected_col)
            st.plotly_chart(fig, use_container_width=True)
            
    elif viz_type == "Custom Chart":
        st.markdown("### Build Your Own Chart")
        
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis", df.columns)
        with col2:
            y_col = st.selectbox("Y-axis", [c for c in df.columns if c != x_col])
        
        chart_type = st.radio("Chart Type", ["Line", "Scatter", "Bar"], horizontal=True)
        
        pdf = df.select([x_col, y_col]).to_pandas()
        
        if chart_type == "Line":
            st.line_chart(pdf, x=x_col, y=y_col)
        elif chart_type == "Scatter":
            st.scatter_chart(pdf, x=x_col, y=y_col)
        else:
            st.bar_chart(pdf, x=x_col, y=y_col)


def render_model_training():
    """Render model training page."""
    st.markdown('<p class="section-header">🤖 Model Training</p>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
        return

    df = get_first_valid_df(st.session_state.merged_data, st.session_state.cleaned_data, st.session_state.auction_data)

    # Feature engineering
    st.markdown("### 🔧 Feature Engineering")

    col1, col2 = st.columns(2)
    with col1:
        include_lags = st.checkbox("Include lag features", value=True)
        include_rolling = st.checkbox("Include rolling features", value=True)
    with col2:
        include_time = st.checkbox("Include time features", value=True)

    # Model selection
    st.markdown("### 🎯 Model Configuration")

    # Model type descriptions
    model_descriptions = {
        "linear_regression": "Simple, fast, interpretable baseline",
        "random_forest": "Ensemble of decision trees, handles non-linearity",
        "gradient_boosting": "Sequential boosting, good for structured data",
        "xgboost": "Optimized gradient boosting (requires xgboost package)",
        "svr": "Support Vector Regression, good for small datasets",
        "elastic_net": "L1+L2 regularization, handles multicollinearity",
        "mlp": "Neural Network, captures complex patterns",
        "lstm": "Deep learning for sequences (requires tensorflow)"
    }

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Model Type",
            [m.value for m in ModelType],
            help="Select a model type to train"
        )
        st.caption(f"*{model_descriptions.get(model_type, '')}*")
    with col2:
        target_column = st.selectbox(
            "Target Variable",
            [config.columns.PRICE_SECOND_AUCTION, config.columns.PRICE_FIRST_AUCTION]
        )

    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)

    # Training buttons
    col1, col2 = st.columns(2)

    with col1:
        train_single = st.button("🚀 Train Selected Model", type="primary")

    with col2:
        train_all = st.button("🔄 Train All Models", type="secondary")

    # Helper function to train a single model
    def train_single_model(model_type_str, X, y, valid_features, test_size):
        try:
            forecaster = PriceForecaster(ModelType(model_type_str))
            result = forecaster.train(X, y, valid_features, test_size=test_size)

            # Get predictions
            split_idx = int(len(y) * (1 - test_size))
            X_test = X[split_idx:]
            y_test = y[split_idx:]

            if result.scaler is not None:
                X_test_scaled = result.scaler.transform(X_test)
            else:
                X_test_scaled = X_test

            # Handle LSTM prediction differently
            if ModelType(model_type_str) == ModelType.LSTM:
                y_pred = forecaster.predict(X_test_scaled, scale=False)
                # Adjust y_test for LSTM lookback
                y_test = y_test[forecaster.lstm_lookback:]
            else:
                y_pred = result.model.predict(X_test_scaled)

            return result, y_test, y_pred, None
        except Exception as e:
            return None, None, None, str(e)

    # Feature preparation (shared between single and all)
    def prepare_features():
        fe = FeatureEngineer()
        feature_set = fe.create_feature_set(
            df,
            target_column=target_column,
            include_lags=include_lags,
            include_rolling=include_rolling,
            include_time=include_time
        )

        feature_cols = [col for col in config.columns.FEATURE_COLUMNS
                        if col in feature_set.features.columns]

        for col in feature_set.features.columns:
            if any(x in col for x in ['_lag_', '_rolling_', 'hour', 'month', 'is_weekend']):
                feature_cols.append(col)

        feature_cols = list(set(feature_cols))

        X, y, valid_features = prepare_training_data(
            feature_set.features,
            feature_cols,
            target_column
        )
        return X, y, valid_features

    if train_single:
        with st.spinner(f"Training {model_type} model..."):
            try:
                X, y, valid_features = prepare_features()
                st.info(f"Training with {len(valid_features)} features on {len(y):,} samples")

                result, y_test, y_pred, error = train_single_model(
                    model_type, X, y, valid_features, test_size
                )

                if error:
                    st.error(f"❌ Error: {error}")
                    return

                # Store results
                st.session_state.training_result = result
                st.session_state.model_trained = True
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.target_column = target_column

                # Also store in comparison dict
                st.session_state.all_model_results[model_type] = result
                st.session_state.model_predictions[model_type] = {
                    'y_test': y_test, 'y_pred': y_pred
                }

                # Display results
                st.markdown("### 📊 Training Results")

                col1, col2, col3, col4, col5 = st.columns(5)
                metrics = result.metrics.to_dict()

                with col1:
                    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                with col2:
                    st.metric("MAE", f"{metrics['MAE']:.4f}")
                with col3:
                    st.metric("R²", f"{metrics['R²']:.4f}")
                with col4:
                    st.metric("SMAPE", f"{metrics['SMAPE (%)']}%")
                with col5:
                    st.metric("MAPE", f"{metrics['MAPE (%)']}%")

                if result.feature_importance:
                    st.markdown("### 📈 Feature Importance")
                    viz = EnergyVisualizer()
                    fig = viz.plot_feature_importance(result.feature_importance)
                    st.plotly_chart(fig, use_container_width=True)

                st.success("✅ Model trained successfully!")

            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    if train_all:
        st.markdown("### 🔄 Training All Models")

        try:
            X, y, valid_features = prepare_features()
            st.info(f"Training with {len(valid_features)} features on {len(y):,} samples")

            all_models = [m.value for m in ModelType]
            progress_bar = st.progress(0)
            status_text = st.empty()

            results_summary = []
            successful_models = 0

            for i, mt in enumerate(all_models):
                status_text.text(f"Training {mt}...")
                progress_bar.progress((i + 1) / len(all_models))

                result, y_test, y_pred, error = train_single_model(
                    mt, X, y, valid_features, test_size
                )

                if error:
                    results_summary.append({
                        'Model': mt,
                        'Status': f'❌ {error[:50]}...' if len(error) > 50 else f'❌ {error}',
                        'RMSE': '-', 'MAE': '-', 'R²': '-', 'SMAPE': '-', 'MAPE': '-'
                    })
                else:
                    st.session_state.all_model_results[mt] = result
                    st.session_state.model_predictions[mt] = {
                        'y_test': y_test, 'y_pred': y_pred
                    }
                    successful_models += 1

                    metrics = result.metrics.to_dict()
                    results_summary.append({
                        'Model': mt,
                        'Status': '✅',
                        'RMSE': f"{metrics['RMSE']:.4f}",
                        'MAE': f"{metrics['MAE']:.4f}",
                        'R²': f"{metrics['R²']:.4f}",
                        'SMAPE': f"{metrics['SMAPE (%)']}%",
                        'MAPE': f"{metrics['MAPE (%)']}%"
                    })

            progress_bar.progress(1.0)
            status_text.text("Training complete!")

            st.session_state.model_trained = True

            # Display summary table
            st.markdown("### 📊 Training Summary")
            import pandas as pd
            summary_df = pd.DataFrame(results_summary)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.success(f"✅ Successfully trained {successful_models}/{len(all_models)} models!")
            st.info("📊 Go to **Model Comparison** tab to compare all models in detail.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    # Show already trained models
    if st.session_state.all_model_results:
        st.markdown("---")
        st.markdown("### 📋 Trained Models")
        trained_models = list(st.session_state.all_model_results.keys())
        st.write(f"**{len(trained_models)} models trained:** {', '.join(trained_models)}")


def render_predictions():
    """Render predictions page."""
    st.markdown('<p class="section-header">📈 Price Predictions</p>', unsafe_allow_html=True)

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first!")
        return

    # Check if we have multiple models trained
    if st.session_state.all_model_results:
        trained_models = list(st.session_state.all_model_results.keys())

        st.markdown("### 🎯 Select Model")
        selected_model = st.selectbox(
            "Choose a trained model to view predictions",
            trained_models,
            help="Select which model's predictions to analyze"
        )

        result = st.session_state.all_model_results[selected_model]
        predictions = st.session_state.model_predictions.get(selected_model, {})
        y_test = predictions.get('y_test')
        y_pred = predictions.get('y_pred')
    else:
        # Fallback to single model
        result = st.session_state.training_result
        y_test = st.session_state.get('y_test')
        y_pred = st.session_state.get('y_pred')
        selected_model = result.model_type.value if result else "Unknown"

    if y_test is None or y_pred is None:
        st.warning("⚠️ No prediction data available. Please retrain the model.")
        return

    st.info(f"📊 Showing predictions for **{selected_model}** model")

    # Model Performance Metrics
    st.markdown("### 🎯 Model Performance")
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = result.metrics.to_dict()

    with col1:
        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
    with col2:
        st.metric("MAE", f"{metrics['MAE']:.4f}")
    with col3:
        st.metric("R²", f"{metrics['R²']:.4f}")
    with col4:
        st.metric("SMAPE", f"{metrics['SMAPE (%)']}%")
    with col5:
        st.metric("MAPE", f"{metrics['MAPE (%)']}%")

    st.markdown("### 📊 Prediction Analysis")

    # Actual vs Predicted Scatter Plot
    st.markdown("#### Actual vs Predicted Values")
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(color='#1f77b4', opacity=0.6)
    ))
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    fig_scatter.update_layout(
        xaxis_title="Actual Price (GBP/MWh)",
        yaxis_title="Predicted Price (GBP/MWh)",
        height=450
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Residuals Analysis
    residuals = y_test - y_pred

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Residual Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residuals,
            nbinsx=50,
            marker_color='#1f77b4',
            opacity=0.7
        ))
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        fig_hist.update_layout(
            xaxis_title="Residual (Actual - Predicted)",
            yaxis_title="Frequency",
            height=350
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.markdown("#### Residuals vs Predicted")
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(color='#1f77b4', opacity=0.5)
        ))
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        fig_resid.update_layout(
            xaxis_title="Predicted Price (GBP/MWh)",
            yaxis_title="Residual",
            height=350
        )
        st.plotly_chart(fig_resid, use_container_width=True)

    # Prediction over time
    st.markdown("#### Predictions Over Time (Test Set)")
    fig_time = go.Figure()
    x_range = list(range(len(y_test)))
    fig_time.add_trace(go.Scatter(
        x=x_range,
        y=y_test,
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4')
    ))
    fig_time.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name='Predicted',
        line=dict(color='#ff7f0e', dash='dash')
    ))
    fig_time.update_layout(
        xaxis_title="Sample Index",
        yaxis_title="Price (GBP/MWh)",
        height=400
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # Error Statistics
    st.markdown("### 📉 Error Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Error", f"{residuals.mean():.4f}")
    with col2:
        st.metric("Std Error", f"{residuals.std():.4f}")
    with col3:
        st.metric("Max Overestimate", f"{residuals.min():.4f}")
    with col4:
        st.metric("Max Underestimate", f"{residuals.max():.4f}")


def render_model_comparison():
    """Render model comparison page."""
    st.markdown('<p class="section-header">🔬 Model Comparison</p>', unsafe_allow_html=True)

    if not st.session_state.all_model_results:
        st.warning("⚠️ No models trained yet! Go to **Model Training** and train models first.")
        st.info("💡 Tip: Use **Train All Models** to train all available models at once for comparison.")
        return

    results = st.session_state.all_model_results
    predictions = st.session_state.model_predictions

    st.markdown(f"### 📊 Comparing {len(results)} Models")

    # Metrics comparison table
    st.markdown("### 📈 Performance Metrics")

    import pandas as pd

    metrics_data = []
    for model_name, result in results.items():
        m = result.metrics.to_dict()
        metrics_data.append({
            'Model': model_name,
            'RMSE': m['RMSE'],
            'MAE': m['MAE'],
            'R²': m['R²'],
            'SMAPE (%)': m['SMAPE (%)'],
            'MAPE (%)': m['MAPE (%)']
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Highlight best values
    st.dataframe(
        metrics_df.style.highlight_min(
            subset=['RMSE', 'MAE', 'SMAPE (%)', 'MAPE (%)'],
            color='lightgreen'
        ).highlight_max(
            subset=['R²'],
            color='lightgreen'
        ),
        use_container_width=True,
        hide_index=True
    )

    # Find best model for each metric
    st.markdown("### 🏆 Best Models by Metric")
    col1, col2, col3 = st.columns(3)

    with col1:
        best_rmse_idx = metrics_df['RMSE'].idxmin()
        st.metric("Best RMSE", metrics_df.loc[best_rmse_idx, 'Model'],
                  f"{metrics_df.loc[best_rmse_idx, 'RMSE']:.4f}")

    with col2:
        best_r2_idx = metrics_df['R²'].idxmax()
        st.metric("Best R²", metrics_df.loc[best_r2_idx, 'Model'],
                  f"{metrics_df.loc[best_r2_idx, 'R²']:.4f}")

    with col3:
        best_mae_idx = metrics_df['MAE'].idxmin()
        st.metric("Best MAE", metrics_df.loc[best_mae_idx, 'Model'],
                  f"{metrics_df.loc[best_mae_idx, 'MAE']:.4f}")

    # Bar chart comparison
    st.markdown("### 📊 Visual Comparison")

    metric_to_plot = st.selectbox(
        "Select Metric to Compare",
        ['RMSE', 'MAE', 'R²', 'SMAPE (%)', 'MAPE (%)']
    )

    fig_bar = go.Figure()

    # Color based on whether higher or lower is better
    if metric_to_plot == 'R²':
        colors = ['#2ecc71' if v == metrics_df[metric_to_plot].max() else '#3498db'
                  for v in metrics_df[metric_to_plot]]
    else:
        colors = ['#2ecc71' if v == metrics_df[metric_to_plot].min() else '#3498db'
                  for v in metrics_df[metric_to_plot]]

    fig_bar.add_trace(go.Bar(
        x=metrics_df['Model'],
        y=metrics_df[metric_to_plot],
        marker_color=colors,
        text=[f"{v:.4f}" for v in metrics_df[metric_to_plot]],
        textposition='outside'
    ))

    fig_bar.update_layout(
        title=f"{metric_to_plot} by Model",
        xaxis_title="Model",
        yaxis_title=metric_to_plot,
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Radar chart for multi-metric comparison
    st.markdown("### 🎯 Multi-Metric Radar Chart")

    # Normalize metrics for radar chart (0-1 scale, inverted for error metrics)
    radar_df = metrics_df.copy()

    for col in ['RMSE', 'MAE', 'SMAPE (%)', 'MAPE (%)']:
        max_val = radar_df[col].max()
        min_val = radar_df[col].min()
        if max_val != min_val:
            radar_df[col + '_norm'] = 1 - (radar_df[col] - min_val) / (max_val - min_val)
        else:
            radar_df[col + '_norm'] = 1.0

    # R² is already 0-1 scale (higher is better)
    radar_df['R²_norm'] = (radar_df['R²'] - radar_df['R²'].min()) / (radar_df['R²'].max() - radar_df['R²'].min() + 1e-8)

    categories = ['RMSE', 'MAE', 'R²', 'SMAPE', 'MAPE']
    norm_cols = ['RMSE_norm', 'MAE_norm', 'R²_norm', 'SMAPE (%)_norm', 'MAPE (%)_norm']

    fig_radar = go.Figure()

    colors_radar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for i, (_, row) in enumerate(radar_df.iterrows()):
        values = [row[col] for col in norm_cols]
        values.append(values[0])  # Close the polygon

        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            name=row['Model'],
            line_color=colors_radar[i % len(colors_radar)]
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Normalized Performance (Higher = Better)",
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Predictions comparison
    st.markdown("### 📉 Prediction Comparison")

    if predictions:
        # Select models to compare
        models_to_compare = st.multiselect(
            "Select Models to Compare (max 4)",
            list(predictions.keys()),
            default=list(predictions.keys())[:min(4, len(predictions))]
        )

        if models_to_compare:
            fig_pred = go.Figure()

            # Add actual values (use first model's y_test as they should be similar)
            ref_y_test = predictions[models_to_compare[0]]['y_test']
            x_range = list(range(len(ref_y_test)))

            fig_pred.add_trace(go.Scatter(
                x=x_range,
                y=ref_y_test,
                mode='lines',
                name='Actual',
                line=dict(color='black', width=2)
            ))

            # Add predictions for each model
            colors_pred = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for i, model_name in enumerate(models_to_compare[:4]):
                y_pred = predictions[model_name]['y_pred']
                # Adjust x_range if different lengths (LSTM)
                x_pred = list(range(len(y_pred)))
                fig_pred.add_trace(go.Scatter(
                    x=x_pred,
                    y=y_pred,
                    mode='lines',
                    name=model_name,
                    line=dict(color=colors_pred[i], dash='dash')
                ))

            fig_pred.update_layout(
                title="Actual vs Predicted Values",
                xaxis_title="Sample Index",
                yaxis_title="Price (GBP/MWh)",
                height=450
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Residual comparison
            st.markdown("### 📊 Residual Analysis")

            col1, col2 = st.columns(2)

            with col1:
                fig_resid_box = go.Figure()
                for model_name in models_to_compare[:4]:
                    y_test = predictions[model_name]['y_test']
                    y_pred = predictions[model_name]['y_pred']
                    residuals = y_test - y_pred
                    fig_resid_box.add_trace(go.Box(
                        y=residuals,
                        name=model_name
                    ))

                fig_resid_box.update_layout(
                    title="Residual Distribution by Model",
                    yaxis_title="Residual (Actual - Predicted)",
                    height=400
                )
                st.plotly_chart(fig_resid_box, use_container_width=True)

            with col2:
                # Error statistics table
                error_stats = []
                for model_name in models_to_compare[:4]:
                    y_test = predictions[model_name]['y_test']
                    y_pred = predictions[model_name]['y_pred']
                    residuals = y_test - y_pred
                    error_stats.append({
                        'Model': model_name,
                        'Mean Error': f"{residuals.mean():.4f}",
                        'Std Error': f"{residuals.std():.4f}",
                        'Max Overest.': f"{residuals.min():.4f}",
                        'Max Underest.': f"{residuals.max():.4f}"
                    })

                st.markdown("#### Error Statistics")
                st.dataframe(pd.DataFrame(error_stats), use_container_width=True, hide_index=True)

    # Model recommendations
    st.markdown("### 💡 Recommendations")

    best_overall_model = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']

    st.info(f"""
    **Best Overall Model:** `{best_overall_model}` (lowest RMSE)

    **Interpretation Guide:**
    - **RMSE/MAE**: Lower is better - measures average prediction error
    - **R²**: Higher is better (max 1.0) - measures variance explained
    - **SMAPE/MAPE**: Lower is better - percentage-based error metrics

    **Model Selection Tips:**
    - For **production use**, prefer simpler models (Linear, ElasticNet) if performance is similar
    - **Random Forest/XGBoost** are good for capturing non-linear patterns
    - **LSTM** may need more data and tuning for optimal performance
    - Consider **training time** and **interpretability** alongside accuracy
    """)


def render_trading_strategy():
    """Render trading strategy page."""
    st.markdown('<p class="section-header">💹 Trading Strategy</p>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first!")
        return

    df = get_first_valid_df(st.session_state.merged_data, st.session_state.cleaned_data, st.session_state.auction_data)

    strategy = TradingStrategy()

    st.markdown("### ⚙️ Strategy Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        threshold = st.number_input(
            "Price Difference Threshold (GBP)",
            min_value=1.0, max_value=50.0, value=10.0, step=1.0
        )
    with col2:
        volume = st.number_input(
            "Trade Volume (MW)",
            min_value=1.0, max_value=100.0, value=10.0, step=1.0
        )
    with col3:
        transaction_cost = st.number_input(
            "Transaction Cost (GBP/MW)",
            min_value=0.0, max_value=20.0, value=5.0, step=0.5
        )

    # Update strategy config
    strategy.config.price_diff_threshold = threshold
    strategy.config.max_volume_per_trade = volume
    strategy.config.transaction_cost = transaction_cost

    st.markdown("### 📊 Strategy Overview")

    st.markdown("""
    **Naive Strategy Rules:**
    - **Buy First, Sell Second**: When forecasted price increase > threshold
    - **Sell First, Buy Second**: When forecasted price decrease > threshold
    - **Hold**: When price difference is below threshold
    """)

    # Check for required columns
    price_first_col = config.columns.PRICE_FIRST_AUCTION
    price_second_col = config.columns.PRICE_SECOND_AUCTION

    if price_first_col not in df.columns or price_second_col not in df.columns:
        st.warning("⚠️ Required price columns not found in data. Need both first and second auction prices.")
        return

    # Backtest mode selection
    st.markdown("### 🎯 Backtest Mode")

    has_trained_models = bool(st.session_state.all_model_results)

    backtest_mode = st.radio(
        "Select backtest approach",
        ["Oracle (Perfect Foresight)", "Model-Based Predictions", "Compare All Models"],
        help="Oracle uses actual future prices (upper bound). Model-based uses trained model predictions.",
        horizontal=True,
        disabled=not has_trained_models and True  # Disable model options if no models
    )

    if not has_trained_models and backtest_mode != "Oracle (Perfect Foresight)":
        st.warning("⚠️ No trained models available. Please train models first or use Oracle mode.")
        backtest_mode = "Oracle (Perfect Foresight)"

    selected_model = None
    if backtest_mode == "Model-Based Predictions" and has_trained_models:
        selected_model = st.selectbox(
            "Select model for predictions",
            list(st.session_state.all_model_results.keys())
        )

    # Helper function to run backtest
    def run_backtest(forecast_first, forecast_second, actual_first, actual_second, label=""):
        signals = []
        actions = []
        profits = []
        cumulative_profits = []
        running_profit = 0

        min_len = min(len(forecast_first), len(forecast_second), len(actual_first), len(actual_second))

        for i in range(min_len):
            signal = strategy.naive_strategy(
                forecast_price_first=forecast_first[i],
                forecast_price_second=forecast_second[i]
            )
            signals.append(signal)
            actions.append(signal.action)

            # Calculate profit using ACTUAL prices
            if signal.action == "hold":
                profit = 0
            else:
                cost = transaction_cost * 2 * volume
                if signal.action == "buy_first_sell_second":
                    profit = (actual_second[i] - actual_first[i]) * volume - cost
                else:
                    profit = (actual_first[i] - actual_second[i]) * volume - cost

            profits.append(profit)
            running_profit += profit
            cumulative_profits.append(running_profit)

        return {
            'label': label,
            'signals': signals,
            'actions': actions,
            'profits': profits,
            'cumulative': cumulative_profits,
            'total_profit': sum(profits),
            'total_trades': sum(1 for a in actions if a != "hold"),
            'winning_trades': sum(1 for p in profits if p > 0),
            'win_rate': sum(1 for p in profits if p > 0) / max(1, sum(1 for a in actions if a != "hold")) * 100
        }

    # Run Backtest Button
    col1, col2 = st.columns(2)
    with col1:
        run_backtest_btn = st.button("🚀 Run Backtest", type="primary")
    with col2:
        if has_trained_models:
            compare_all_btn = st.button("📊 Compare All Models", type="secondary")
        else:
            compare_all_btn = False

    if run_backtest_btn:
        with st.spinner("Running backtest..."):
            try:
                prices_first = df[price_first_col].drop_nulls().to_numpy()
                prices_second = df[price_second_col].drop_nulls().to_numpy()
                min_len = min(len(prices_first), len(prices_second))
                prices_first = prices_first[:min_len]
                prices_second = prices_second[:min_len]

                if backtest_mode == "Oracle (Perfect Foresight)":
                    # Use actual prices as "forecast"
                    result = run_backtest(prices_first, prices_second, prices_first, prices_second, "Oracle")
                    st.session_state.backtest_results = {'Oracle': result}
                    st.session_state.current_backtest = 'Oracle'

                elif backtest_mode == "Model-Based Predictions" and selected_model:
                    # Use model predictions
                    preds = st.session_state.model_predictions.get(selected_model, {})
                    y_pred = preds.get('y_pred')
                    y_test = preds.get('y_test')

                    if y_pred is None:
                        st.error("No predictions available for this model")
                        return

                    # For simplicity, use predictions as second auction forecast
                    forecast_second = y_pred
                    actual_second = y_test
                    # Assume first auction is actual (or use a ratio)
                    actual_first = actual_second * 0.95  # Rough approximation
                    forecast_first = forecast_second * 0.95

                    result = run_backtest(forecast_first, forecast_second, actual_first, actual_second, selected_model)
                    st.session_state.backtest_results = {selected_model: result}
                    st.session_state.current_backtest = selected_model

                st.success("✅ Backtest completed!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    if compare_all_btn and has_trained_models:
        with st.spinner("Running backtests for all models..."):
            try:
                prices_first = df[price_first_col].drop_nulls().to_numpy()
                prices_second = df[price_second_col].drop_nulls().to_numpy()
                min_len = min(len(prices_first), len(prices_second))
                prices_first = prices_first[:min_len]
                prices_second = prices_second[:min_len]

                all_results = {}

                # Oracle baseline
                oracle_result = run_backtest(prices_first, prices_second, prices_first, prices_second, "Oracle (Perfect)")
                all_results['Oracle (Perfect)'] = oracle_result

                # Each trained model
                progress = st.progress(0)
                models = list(st.session_state.all_model_results.keys())

                for i, model_name in enumerate(models):
                    preds = st.session_state.model_predictions.get(model_name, {})
                    y_pred = preds.get('y_pred')
                    y_test = preds.get('y_test')

                    if y_pred is not None and y_test is not None:
                        forecast_second = y_pred
                        actual_second = y_test
                        actual_first = actual_second * 0.95
                        forecast_first = forecast_second * 0.95

                        result = run_backtest(forecast_first, forecast_second, actual_first, actual_second, model_name)
                        all_results[model_name] = result

                    progress.progress((i + 1) / len(models))

                st.session_state.backtest_results = all_results
                st.session_state.current_backtest = 'comparison'
                st.success(f"✅ Compared {len(all_results)} strategies!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Display results
    if 'backtest_results' in st.session_state and st.session_state.backtest_results:
        results = st.session_state.backtest_results

        if len(results) == 1:
            # Single model results
            result = list(results.values())[0]
            st.markdown(f"### 📈 Backtest Results: {result['label']}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Profit", f"£{result['total_profit']:,.2f}")
            with col2:
                st.metric("Total Trades", f"{result['total_trades']:,}")
            with col3:
                st.metric("Win Rate", f"{result['win_rate']:.1f}%")
            with col4:
                avg_profit = result['total_profit'] / max(1, result['total_trades'])
                st.metric("Avg Profit/Trade", f"£{avg_profit:.2f}")

            # Cumulative profit chart
            viz = EnergyVisualizer()
            fig_profit = viz.plot_profit_curve(result['cumulative'])
            st.plotly_chart(fig_profit, use_container_width=True)

        else:
            # Comparison mode
            st.markdown("### 📊 Model Comparison - Backtest Results")

            import pandas as pd
            comparison_data = []
            for name, res in results.items():
                comparison_data.append({
                    'Strategy': name,
                    'Total Profit (£)': f"{res['total_profit']:,.2f}",
                    'Total Trades': res['total_trades'],
                    'Winning Trades': res['winning_trades'],
                    'Win Rate (%)': f"{res['win_rate']:.1f}",
                    'Avg Profit/Trade (£)': f"{res['total_profit'] / max(1, res['total_trades']):.2f}"
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            # Find best performer
            best_model = max(results.items(), key=lambda x: x[1]['total_profit'])
            st.success(f"🏆 **Best Performer:** {best_model[0]} with £{best_model[1]['total_profit']:,.2f} profit")

            # Cumulative profit comparison chart
            st.markdown("### 📈 Cumulative Profit Comparison")
            fig_compare = go.Figure()

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
            for i, (name, res) in enumerate(results.items()):
                fig_compare.add_trace(go.Scatter(
                    x=list(range(len(res['cumulative']))),
                    y=res['cumulative'],
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=2 if 'Oracle' in name else 1)
                ))

            fig_compare.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_compare.update_layout(
                title="Cumulative Profit by Strategy",
                xaxis_title="Trade Number",
                yaxis_title="Cumulative Profit (£)",
                height=500
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            # Profit distribution comparison
            st.markdown("### 📉 Profit Distribution by Model")
            fig_box = go.Figure()
            for name, res in results.items():
                non_zero = [p for p in res['profits'] if p != 0]
                if non_zero:
                    fig_box.add_trace(go.Box(y=non_zero, name=name))

            fig_box.update_layout(
                title="Profit per Trade Distribution",
                yaxis_title="Profit (£)",
                height=400
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # Price Spread Analysis (always show)
    st.markdown("### 📊 Price Spread Analysis")
    prices_first = df[price_first_col].drop_nulls().to_numpy()
    prices_second = df[price_second_col].drop_nulls().to_numpy()
    min_len = min(len(prices_first), len(prices_second))
    spreads = prices_second[:min_len] - prices_first[:min_len]

    fig_spread = go.Figure()
    fig_spread.add_trace(go.Histogram(x=spreads, nbinsx=50, marker_color='#3498db', opacity=0.7))
    fig_spread.add_vline(x=threshold, line_dash="dash", line_color="green", annotation_text=f"+{threshold}")
    fig_spread.add_vline(x=-threshold, line_dash="dash", line_color="red", annotation_text=f"-{threshold}")
    fig_spread.update_layout(
        title="Price Spread Distribution (Second - First Auction)",
        xaxis_title="Price Spread (GBP/MWh)",
        yaxis_title="Frequency",
        height=400
    )
    st.plotly_chart(fig_spread, use_container_width=True)


def main():
    """Main application entry point."""
    init_session_state()
    
    page, data_source = render_sidebar()
    
    if page == "🏠 Home":
        render_home()
    elif page == "📁 Data Loading":
        render_data_loading(data_source)
    elif page == "🧹 Data Cleaning":
        render_data_cleaning()
    elif page == "📊 Visualization":
        render_visualization()
    elif page == "🤖 Model Training":
        render_model_training()
    elif page == "📈 Predictions":
        render_predictions()
    elif page == "🔬 Model Comparison":
        render_model_comparison()
    elif page == "💹 Trading Strategy":
        render_trading_strategy()


if __name__ == "__main__":
    main()
