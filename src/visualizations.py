"""
Visualization Module for Energy Trading Application.
Creates charts and plots using Plotly.
"""

import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict, Any
import logging

from .config import config

logger = logging.getLogger(__name__)


class EnergyVisualizer:
    """
    Visualization class for energy trading data.
    Creates interactive charts using Plotly.
    """
    
    def __init__(self):
        self.config = config.visualization
        self.colors = self.config.color_palette
    
    def _apply_theme(self, fig: go.Figure) -> go.Figure:
        """Apply consistent theme to figure."""
        fig.update_layout(
            template=self.config.theme,
            font=dict(family="Arial, sans-serif", size=12),
            title_font=dict(size=16),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        return fig
    
    def plot_price_timeseries(
        self,
        df: pl.DataFrame,
        date_column: str = "Date (WET)",
        price_columns: Optional[List[str]] = None,
        title: str = "Electricity Prices Over Time"
    ) -> go.Figure:
        """
        Plot price time series.
        
        Args:
            df: Input DataFrame
            date_column: Date column name
            price_columns: List of price columns to plot
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if price_columns is None:
            price_columns = [
                config.columns.PRICE_FIRST_AUCTION,
                config.columns.PRICE_SECOND_AUCTION
            ]
        
        # Filter to columns that exist
        price_columns = [col for col in price_columns if col in df.columns]
        
        if not price_columns:
            logger.warning("No price columns found in DataFrame")
            return go.Figure()
        
        # Convert to pandas for Plotly (temporary)
        pdf = df.select([date_column] + price_columns).to_pandas()
        
        fig = go.Figure()
        
        for i, col in enumerate(price_columns):
            fig.add_trace(go.Scatter(
                x=pdf[date_column],
                y=pdf[col],
                mode='lines',
                name=col.replace("_", " ").title(),
                line=dict(color=self.colors[i % len(self.colors)])
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price (GBP/MWh)",
            height=500
        )
        
        return self._apply_theme(fig)
    
    def plot_price_distribution(
        self,
        df: pl.DataFrame,
        price_columns: Optional[List[str]] = None,
        title: str = "Price Distribution"
    ) -> go.Figure:
        """
        Plot price distribution histograms.
        
        Args:
            df: Input DataFrame
            price_columns: List of price columns
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if price_columns is None:
            price_columns = [
                config.columns.PRICE_FIRST_AUCTION,
                config.columns.PRICE_SECOND_AUCTION
            ]
        
        price_columns = [col for col in price_columns if col in df.columns]
        
        if not price_columns:
            return go.Figure()
        
        fig = make_subplots(
            rows=1, cols=len(price_columns),
            subplot_titles=[col.replace("_", " ").title() for col in price_columns]
        )
        
        for i, col in enumerate(price_columns):
            values = df[col].drop_nulls().to_numpy()
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=col,
                    marker_color=self.colors[i % len(self.colors)],
                    opacity=0.7
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            height=400
        )
        
        return self._apply_theme(fig)
    
    def plot_price_comparison(
        self,
        df: pl.DataFrame,
        date_column: str = "Date (WET)",
        actual_col: str = "price_second_auction",
        forecast_col: str = "price_forecast_second_auction",
        title: str = "Actual vs Forecasted Prices"
    ) -> go.Figure:
        """
        Plot actual vs forecasted prices.
        
        Args:
            df: Input DataFrame
            date_column: Date column name
            actual_col: Actual price column
            forecast_col: Forecasted price column
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if actual_col not in df.columns or forecast_col not in df.columns:
            logger.warning("Required columns not found")
            return go.Figure()
        
        pdf = df.select([date_column, actual_col, forecast_col]).to_pandas()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pdf[date_column],
            y=pdf[actual_col],
            mode='lines',
            name='Actual',
            line=dict(color=self.colors[0])
        ))
        
        fig.add_trace(go.Scatter(
            x=pdf[date_column],
            y=pdf[forecast_col],
            mode='lines',
            name='Forecasted',
            line=dict(color=self.colors[1], dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price (GBP/MWh)",
            height=500
        )
        
        return self._apply_theme(fig)
    
    def plot_correlation_heatmap(
        self,
        df: pl.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Feature Correlation Matrix"
    ) -> go.Figure:
        """
        Plot correlation heatmap.
        
        Args:
            df: Input DataFrame
            columns: Columns to include (None = all numeric)
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if columns is None:
            columns = [col for col in df.columns 
                      if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        
        columns = [col for col in columns if col in df.columns][:15]  # Limit for readability
        
        if len(columns) < 2:
            return go.Figure()
        
        # Calculate correlation matrix
        corr_data = df.select(columns).to_pandas().corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_data.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            width=800
        )
        
        return self._apply_theme(fig)
    
    def plot_missing_values(
        self,
        df: pl.DataFrame,
        title: str = "Missing Values by Column"
    ) -> go.Figure:
        """
        Plot missing values summary.
        
        Args:
            df: Input DataFrame
            title: Chart title
            
        Returns:
            Plotly figure
        """
        missing_data = []
        for col in df.columns:
            null_count = df[col].null_count()
            total = df.height
            missing_data.append({
                "column": col,
                "missing": null_count,
                "missing_pct": round(null_count / total * 100, 2) if total > 0 else 0
            })
        
        missing_df = pl.DataFrame(missing_data).sort("missing", descending=True)
        
        # Filter to columns with missing values
        missing_df = missing_df.filter(pl.col("missing") > 0)
        
        if missing_df.height == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No missing values found!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        fig = go.Figure(go.Bar(
            x=missing_df["column"].to_list(),
            y=missing_df["missing_pct"].to_list(),
            marker_color=self.colors[0],
            text=[f"{v}%" for v in missing_df["missing_pct"].to_list()],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Column",
            yaxis_title="Missing (%)",
            xaxis_tickangle=-45,
            height=500
        )
        
        return self._apply_theme(fig)
    
    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 15,
        title: str = "Feature Importance"
    ) -> go.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            feature_importance: Dictionary of feature names to importance
            top_n: Number of top features to show
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Sort and get top N
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        features = [f[0] for f in sorted_features]
        importance = [f[1] for f in sorted_features]
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color=self.colors[0]
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, len(features) * 25),
            yaxis=dict(autorange="reversed")
        )
        
        return self._apply_theme(fig)
    
    def plot_trading_signals(
        self,
        df: pl.DataFrame,
        date_column: str,
        price_column: str,
        signal_column: str,
        title: str = "Trading Signals"
    ) -> go.Figure:
        """
        Plot prices with trading signals overlay.
        
        Args:
            df: Input DataFrame
            date_column: Date column name
            price_column: Price column name
            signal_column: Signal column name
            title: Chart title
            
        Returns:
            Plotly figure
        """
        pdf = df.to_pandas()
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=pdf[date_column],
            y=pdf[price_column],
            mode='lines',
            name='Price',
            line=dict(color=self.colors[0])
        ))
        
        # Buy signals
        buy_mask = pdf[signal_column].str.contains('buy', case=False, na=False)
        if buy_mask.any():
            fig.add_trace(go.Scatter(
                x=pdf.loc[buy_mask, date_column],
                y=pdf.loc[buy_mask, price_column],
                mode='markers',
                name='Buy',
                marker=dict(symbol='triangle-up', size=12, color='green')
            ))
        
        # Sell signals
        sell_mask = pdf[signal_column].str.contains('sell', case=False, na=False)
        if sell_mask.any():
            fig.add_trace(go.Scatter(
                x=pdf.loc[sell_mask, date_column],
                y=pdf.loc[sell_mask, price_column],
                mode='markers',
                name='Sell',
                marker=dict(symbol='triangle-down', size=12, color='red')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price (GBP/MWh)",
            height=500
        )
        
        return self._apply_theme(fig)
    
    def plot_profit_curve(
        self,
        cumulative_profits: List[float],
        dates: Optional[List] = None,
        title: str = "Cumulative Profit"
    ) -> go.Figure:
        """
        Plot cumulative profit curve.
        
        Args:
            cumulative_profits: List of cumulative profits
            dates: Optional list of dates
            title: Chart title
            
        Returns:
            Plotly figure
        """
        x_vals = dates if dates else list(range(len(cumulative_profits)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=cumulative_profits,
            mode='lines',
            fill='tozeroy',
            name='Cumulative Profit',
            line=dict(color=self.colors[0])
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title=title,
            xaxis_title="Trade Number" if dates is None else "Date",
            yaxis_title="Cumulative Profit (GBP)",
            height=400
        )
        
        return self._apply_theme(fig)
    
    def plot_hourly_patterns(
        self,
        df: pl.DataFrame,
        value_column: str,
        title: str = "Hourly Patterns"
    ) -> go.Figure:
        """
        Plot hourly patterns (average by hour of day).
        
        Args:
            df: Input DataFrame with 'hour' column
            value_column: Column to aggregate
            title: Chart title
            
        Returns:
            Plotly figure
        """
        if 'hour' not in df.columns or value_column not in df.columns:
            return go.Figure()
        
        hourly_avg = (
            df.group_by('hour')
            .agg(pl.col(value_column).mean().alias('avg'))
            .sort('hour')
        )
        
        fig = go.Figure(go.Bar(
            x=hourly_avg['hour'].to_list(),
            y=hourly_avg['avg'].to_list(),
            marker_color=self.colors[0]
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Hour of Day",
            yaxis_title=f"Average {value_column}",
            height=400
        )
        
        return self._apply_theme(fig)


def create_summary_dashboard(
    df: pl.DataFrame,
    date_column: str = "Date (WET)"
) -> Dict[str, go.Figure]:
    """
    Create a dashboard of summary visualizations.
    
    Args:
        df: Input DataFrame
        date_column: Date column name
        
    Returns:
        Dictionary of figure names to figures
    """
    viz = EnergyVisualizer()
    
    figures = {}
    
    # Price time series
    figures["price_timeseries"] = viz.plot_price_timeseries(df, date_column)
    
    # Price distribution
    figures["price_distribution"] = viz.plot_price_distribution(df)
    
    # Missing values
    figures["missing_values"] = viz.plot_missing_values(df)
    
    # Correlation (if enough numeric columns)
    numeric_cols = [col for col in df.columns 
                   if df[col].dtype in [pl.Float64, pl.Int64]]
    if len(numeric_cols) >= 2:
        figures["correlation"] = viz.plot_correlation_heatmap(df, numeric_cols[:10])
    
    return figures
