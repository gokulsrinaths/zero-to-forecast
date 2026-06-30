"""
Demo Examples for Zero-to-Forecast
Showcase various use cases and capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_api import LLaMAForecaster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def demo_basic_forecasting():
    """Demonstrate basic forecasting capabilities"""
    print("🚀 Zero-to-Forecast Demo: Basic Forecasting")
    print("=" * 50)
    
    # Initialize forecaster (will use mock if no API key)
    try:
        forecaster = LLaMAForecaster()
        print("✅ LLaMA forecaster initialized successfully")
    except Exception as e:
        print(f"⚠️  Using mock forecaster: {e}")
        return
    
    # Example descriptions from different domains
    examples = [
        {
            "domain": "Finance",
            "description": "Sales dropped in Q1 and Q2, then surged in Q3 after a campaign.",
            "expected_pattern": "Decline then sharp increase"
        },
        {
            "domain": "IoT",
            "description": "Energy usage spiked at the top of each hour due to backup jobs, then dropped after.",
            "expected_pattern": "Periodic spikes"
        },
        {
            "domain": "Healthcare",
            "description": "Blood pressure readings showed a gradual decline over the week of medication.",
            "expected_pattern": "Steady decline"
        },
        {
            "domain": "Weather",
            "description": "Temperature increased steadily from morning to afternoon, then dropped in the evening.",
            "expected_pattern": "Rise then fall"
        },
        {
            "domain": "Technology",
            "description": "Website traffic peaked during lunch hours and decreased during off-peak times.",
            "expected_pattern": "Peak during specific hours"
        }
    ]
    
    for example in examples:
        print(f"\n📊 {example['domain']} Example:")
        print(f"Description: {example['description']}")
        print(f"Expected Pattern: {example['expected_pattern']}")
        
        try:
            forecast = forecaster.forecast_from_text(example['description'], length=7)
            print(f"Generated Forecast: {forecast}")
            
            # Basic statistics
            mean_val = np.mean(forecast)
            std_val = np.std(forecast)
            trend = "Increasing" if forecast[-1] > forecast[0] else "Decreasing" if forecast[-1] < forecast[0] else "Stable"
            
            print(f"Statistics: Mean={mean_val:.2f}, Std={std_val:.2f}, Trend={trend}")
            
        except Exception as e:
            print(f"❌ Error generating forecast: {e}")
        
        print("-" * 50)

def demo_batch_forecasting():
    """Demonstrate batch forecasting capabilities"""
    print("\n🔄 Zero-to-Forecast Demo: Batch Forecasting")
    print("=" * 50)
    
    try:
        forecaster = LLaMAForecaster()
    except Exception as e:
        print(f"⚠️  Using mock forecaster: {e}")
        return
    
    # Multiple descriptions for batch processing
    descriptions = [
        "Revenue growth was slow in January, accelerated in February, then stabilized in March.",
        "Server CPU usage remained high during business hours and low overnight.",
        "Customer satisfaction scores improved gradually after the service improvements.",
        "Product demand increased sharply after the marketing campaign launch.",
        "Network latency spiked during peak hours and returned to normal during off-peak."
    ]
    
    print("Processing batch of 5 descriptions...")
    
    try:
        forecasts = forecaster.batch_forecast(descriptions, length=6)
        
        for i, (desc, forecast) in enumerate(zip(descriptions, forecasts)):
            print(f"\n{i+1}. {desc[:60]}...")
            print(f"   Forecast: {forecast}")
            
    except Exception as e:
        print(f"❌ Error in batch forecasting: {e}")

def demo_domain_specific():
    """Demonstrate domain-specific forecasting"""
    print("\n🎯 Zero-to-Forecast Demo: Domain-Specific Forecasting")
    print("=" * 50)
    
    try:
        forecaster = LLaMAForecaster()
    except Exception as e:
        print(f"⚠️  Using mock forecaster: {e}")
        return
    
    # Domain-specific examples
    domain_examples = {
        "Finance": [
            "Stock price dipped after earnings report, then recovered gradually over the week.",
            "Quarterly revenue showed consistent growth with seasonal spikes in Q4."
        ],
        "Healthcare": [
            "Patient heart rate increased during exercise and gradually returned to baseline.",
            "Blood glucose levels remained stable throughout the monitoring period."
        ],
        "IoT": [
            "Sensor readings fluctuated randomly with occasional calibration spikes.",
            "Device temperature rose steadily during operation and cooled during idle periods."
        ]
    }
    
    for domain, examples in domain_examples.items():
        print(f"\n🏥 {domain} Domain:")
        for i, description in enumerate(examples):
            try:
                forecast = forecaster.forecast_from_text(description, length=8)
                print(f"  {i+1}. {description[:50]}...")
                print(f"     Forecast: {forecast}")
            except Exception as e:
                print(f"  {i+1}. Error: {e}")

def demo_error_analysis():
    """Demonstrate error analysis capabilities"""
    print("\n🔍 Zero-to-Forecast Demo: Error Analysis")
    print("=" * 50)
    
    # Load sample data
    try:
        df = pd.read_csv('../data/synthetic_pairs.csv')
        print(f"✅ Loaded {len(df)} samples from dataset")
    except Exception as e:
        print(f"❌ Could not load dataset: {e}")
        return
    
    # Sample a few examples for analysis
    sample = df.sample(n=3, random_state=42)
    
    try:
        forecaster = LLaMAForecaster()
    except Exception as e:
        print(f"⚠️  Using mock forecaster: {e}")
        return
    
    for _, row in sample.iterrows():
        description = row['description']
        actual = eval(row['time_series'])  # Parse the string representation
        
        print(f"\n📊 Example: {description[:60]}...")
        print(f"Actual: {actual}")
        
        try:
            predicted = forecaster.forecast_from_text(description, len(actual))
            print(f"Predicted: {predicted}")
            
            # Calculate error metrics
            mae = np.mean(np.abs(np.array(predicted) - np.array(actual)))
            mse = np.mean((np.array(predicted) - np.array(actual)) ** 2)
            rmse = np.sqrt(mse)
            correlation = np.corrcoef(predicted, actual)[0, 1] if len(predicted) > 1 else 0
            
            print(f"Error Metrics: MAE={mae:.3f}, MSE={mse:.3f}, RMSE={rmse:.3f}, Corr={correlation:.3f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

def demo_visualization():
    """Demonstrate visualization capabilities"""
    print("\n📈 Zero-to-Forecast Demo: Visualization")
    print("=" * 50)
    
    try:
        forecaster = LLaMAForecaster()
    except Exception as e:
        print(f"⚠️  Using mock forecaster: {e}")
        return
    
    # Create a complex example
    description = "Sales started low, increased steadily for three periods, peaked, then declined sharply before stabilizing."
    
    try:
        forecast = forecaster.forecast_from_text(description, length=10)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Main forecast plot
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(forecast) + 1), forecast, 'o-', linewidth=2, markersize=8, color='#1f77b4')
        plt.title('Zero-to-Forecast: Generated Time Series', fontsize=14, fontweight='bold')
        plt.xlabel('Time Period')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        
        # Trend analysis
        plt.subplot(1, 2, 2)
        changes = np.diff(forecast)
        plt.bar(range(1, len(changes) + 1), changes, 
               color=['green' if x > 0 else 'red' if x < 0 else 'gray' for x in changes],
               alpha=0.7)
        plt.title('Period-over-Period Changes', fontsize=14, fontweight='bold')
        plt.xlabel('Period')
        plt.ylabel('Change')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('demo_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualization saved as 'demo_visualization.png'")
        print(f"Generated forecast: {forecast}")
        
    except Exception as e:
        print(f"❌ Error in visualization: {e}")

def main():
    """Run all demo examples"""
    print("🎉 Welcome to Zero-to-Forecast Demo!")
    print("This demo showcases the capabilities of our natural language to time series forecasting system.")
    print("Built for the NeurIPS BERT2S Workshop submission.\n")
    
    # Run all demos
    demo_basic_forecasting()
    demo_batch_forecasting()
    demo_domain_specific()
    demo_error_analysis()
    demo_visualization()
    
    print("\n🎊 Demo completed!")
    print("For more information, visit the project repository or run the Streamlit app:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()
