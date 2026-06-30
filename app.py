"""
Simplified Streamlit App for Zero-to-Forecast
Clean and user-friendly interface for time series forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import base64
from typing import List, Optional, Dict, Tuple
import time
from datetime import datetime

import sys
from pathlib import Path

# Add zero_to_forecast to path
sys.path.append(str(Path(__file__).parent / "zero_to_forecast"))

from ztf.baselines import (
    flat_baseline, baseline_generate, ensemble_best_baselines,
    improved_baseline_generate, final_optimal_baseline
)
from ztf.eval import compute_all_metrics
from ztf.dataset import load_jsonl

# Page configuration
st.set_page_config(
    page_title="Zero-to-Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS for clean styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = {}

@st.cache_resource
def get_baseline_models():
    """Get available baseline models with caching"""
    return {
        "Ensemble Best": ensemble_best_baselines,
        "Improved Baseline": improved_baseline_generate,
        "Final Optimal": final_optimal_baseline,
        "Symbolic Baseline": baseline_generate,
        "Flat Baseline": lambda text, length, freq, domain: flat_baseline(length, domain)
    }

@st.cache_resource
def load_dataset():
    """Load the NL2TS-675 dataset with caching"""
    try:
        dataset_path = "zero_to_forecast/data/nl2ts_675.jsonl"
        if os.path.exists(dataset_path):
            return load_jsonl(dataset_path)
        else:
            st.warning("Dataset not found, using sample data")
            return []
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return []

def create_simple_time_series_plot(values: List[float], title: str = "Time Series Forecast") -> go.Figure:
    """Create simple time series plot"""
    
    x_values = list(range(1, len(values) + 1))
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6, color='#1f77b4')
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        yaxis_title="Value",
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        showlegend=False,
        height=400
    )
    
    fig.update_xaxes(gridcolor='#f0f0f0')
    fig.update_yaxes(gridcolor='#f0f0f0')
    
    return fig

def create_comparison_plot(all_predictions: Dict[str, List[float]], title: str = "Model Comparison") -> go.Figure:
    """Create comparison plot for multiple models"""
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (model_name, values) in enumerate(all_predictions.items()):
        x_values = list(range(1, len(values) + 1))
        
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=values,
                mode='lines',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2)
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        yaxis_title="Value",
        plot_bgcolor='white',
        font=dict(family="Arial, sans-serif"),
        showlegend=True,
        height=400
    )
    
    fig.update_xaxes(gridcolor='#f0f0f0')
    fig.update_yaxes(gridcolor='#f0f0f0')
    
    return fig

def calculate_basic_metrics(values: List[float]) -> Dict[str, float]:
    """Calculate basic performance metrics"""
    if not values:
        return {}
    
    metrics = {
        "Mean": np.mean(values),
        "Std": np.std(values),
        "Min": np.min(values),
        "Max": np.max(values),
        "Range": np.max(values) - np.min(values)
    }
    
    # Trend analysis
    if len(values) > 1:
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        metrics["Trend Slope"] = slope
        metrics["Trend Direction"] = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
    
    return metrics

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        📈 Zero-to-Forecast
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Transform natural language descriptions into time series forecasts**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        models = get_baseline_models()
        selected_model_name = st.selectbox(
            "Forecasting Method:",
            list(models.keys()),
            index=0
        )
        selected_model = models[selected_model_name]
        
        # Parameters
        st.subheader("Parameters")
        forecast_length = st.slider(
            "Forecast Length:",
            min_value=6,
            max_value=100,
            value=24,
            step=6
        )
        
        domain = st.selectbox(
            "Domain:",
            ["Finance", "Healthcare", "IoT", "Retail", "Weather", "Technology"],
            index=0
        )
        
        frequency = st.selectbox(
            "Frequency:",
            ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly"],
            index=1
        )
        
        # Options
        st.subheader("Options")
        enable_comparison = st.checkbox("Compare all models", value=False)
    
    # Main content
    st.header("🎯 Generate Forecast")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dynamic examples based on domain
        domain_examples = {
            "Finance": "Stock price increased steadily over the quarter with moderate volatility",
            "Healthcare": "Patient admissions showed seasonal patterns with peak during flu season",
            "IoT": "Sensor readings showed daily cycles with peak activity during business hours",
            "Retail": "Sales increased steadily over the month with weekend peaks",
            "Weather": "Temperature increased gradually over the week with daily fluctuations",
            "Technology": "Server load increased steadily with peak usage during business hours"
        }
        
        example = domain_examples.get(domain, "Enter your description here...")
        
        description = st.text_area(
            f"Describe your {domain.lower()} time series:",
            value=example,
            height=100,
            placeholder=f"Example: {example}"
        )
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <strong>💡 Tips:</strong>
            <ul>
                <li>Be specific about trends</li>
                <li>Mention time patterns</li>
                <li>Include intensity levels</li>
                <li>Add context details</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Generate forecast
    if st.button("🚀 Generate Forecast", type="primary"):
        if description.strip():
            with st.spinner("Generating your forecast..."):
                try:
                    # Generate prediction
                    predicted_values = selected_model(description, forecast_length, frequency, domain)
                    predicted_values = [float(x) for x in predicted_values]
                    
                    # Store in session state
                    st.session_state.predictions[selected_model_name] = predicted_values
                    
                    # Calculate metrics
                    metrics = calculate_basic_metrics(predicted_values)
                    
                    # Display results
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create and display plot
                        fig = create_simple_time_series_plot(
                            predicted_values,
                            f"{domain} Forecast"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Display metrics
                        st.subheader("📊 Metrics")
                        
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, (int, float)):
                                st.markdown(f"""
                                <div class="metric-box">
                                    <div class="metric-value">{metric_value:.2f}</div>
                                    <div class="metric-label">{metric_name}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="metric-box">
                                    <div class="metric-value">{metric_value}</div>
                                    <div class="metric-label">{metric_name}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Model comparison if enabled
                    if enable_comparison:
                        st.header("📊 Model Comparison")
                        
                        # Generate predictions for all models
                        all_predictions = {}
                        for model_name, model_func in models.items():
                            try:
                                pred = model_func(description, forecast_length, frequency, domain)
                                all_predictions[model_name] = [float(x) for x in pred]
                            except Exception as e:
                                st.warning(f"Failed to generate prediction for {model_name}: {e}")
                        
                        if all_predictions:
                            comparison_fig = create_comparison_plot(
                                all_predictions,
                                "Model Comparison"
                            )
                            st.plotly_chart(comparison_fig, use_container_width=True)
                            
                            # Store comparison results
                            st.session_state.comparison_results = all_predictions
                            
                            # Simple comparison table
                            st.subheader("📋 Performance Summary")
                            comparison_data = []
                            for model_name, values in all_predictions.items():
                                model_metrics = calculate_basic_metrics(values)
                                comparison_data.append({
                                    'Model': model_name,
                                    'Mean': model_metrics.get('Mean', 0),
                                    'Std': model_metrics.get('Std', 0),
                                    'Trend': model_metrics.get('Trend Direction', 'Unknown')
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")
                    st.info("Please try a different description or check your input parameters.")
        else:
            st.warning("Please enter a description to generate a forecast.")
    
    # Export section
    if st.session_state.predictions:
        st.header("📤 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export data as CSV
            for model_name, values in st.session_state.predictions.items():
                df = pd.DataFrame({
                    'Period': range(1, len(values) + 1),
                    'Value': values
                })
                
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {model_name} CSV",
                    data=csv_data,
                    file_name=f"{model_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Export comparison if available
            if st.session_state.comparison_results:
                comparison_data = []
                for model_name, values in st.session_state.comparison_results.items():
                    for i, value in enumerate(values):
                        comparison_data.append({
                            'Model': model_name,
                            'Period': i + 1,
                            'Value': value
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                csv_data = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comparison CSV",
                    data=csv_data,
                    file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Zero-to-Forecast</strong> - Natural Language to Time Series Prediction</p>
        <p>Accepted at NeurIPS BERT2S Workshop 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
