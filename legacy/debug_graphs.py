#!/usr/bin/env python3
"""
Debug script to test baseline functions and identify graph issues.
"""

import sys
from pathlib import Path

# Add zero_to_forecast to path
sys.path.append(str(Path(__file__).parent / "zero_to_forecast"))

from ztf.baselines import (
    flat_baseline, baseline_generate, ensemble_best_baselines,
    improved_baseline_generate, final_optimal_baseline
)

def test_baseline_functions():
    """Test all baseline functions to ensure they return correct data types."""
    
    test_text = "Stock price showing steady upward trend with seasonal fluctuations"
    test_length = 20
    test_freq = "daily"
    test_domain = "finance"
    
    print("Testing baseline functions...")
    print("=" * 50)
    
    # Test flat_baseline
    print("\n1. Testing flat_baseline:")
    try:
        result = flat_baseline(test_length, test_domain)
        print(f"   Type: {type(result)}")
        print(f"   Length: {len(result)}")
        print(f"   First 5 values: {result[:5]}")
        print(f"   All values numeric: {all(isinstance(x, (int, float)) for x in result)}")
        print(f"   Min: {min(result):.2f}, Max: {max(result):.2f}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test baseline_generate
    print("\n2. Testing baseline_generate:")
    try:
        result = baseline_generate(test_text, test_length, test_freq, test_domain)
        print(f"   Type: {type(result)}")
        print(f"   Length: {len(result)}")
        print(f"   First 5 values: {result[:5]}")
        print(f"   All values numeric: {all(isinstance(x, (int, float)) for x in result)}")
        print(f"   Min: {min(result):.2f}, Max: {max(result):.2f}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test ensemble_best_baselines
    print("\n3. Testing ensemble_best_baselines:")
    try:
        result = ensemble_best_baselines(test_text, test_length, test_freq, test_domain)
        print(f"   Type: {type(result)}")
        print(f"   Length: {len(result)}")
        print(f"   First 5 values: {result[:5]}")
        print(f"   All values numeric: {all(isinstance(x, (int, float)) for x in result)}")
        print(f"   Min: {min(result):.2f}, Max: {max(result):.2f}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test improved_baseline_generate
    print("\n4. Testing improved_baseline_generate:")
    try:
        result = improved_baseline_generate(test_text, test_length, test_freq, test_domain)
        print(f"   Type: {type(result)}")
        print(f"   Length: {len(result)}")
        print(f"   First 5 values: {result[:5]}")
        print(f"   All values numeric: {all(isinstance(x, (int, float)) for x in result)}")
        print(f"   Min: {min(result):.2f}, Max: {max(result):.2f}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test final_optimal_baseline
    print("\n5. Testing final_optimal_baseline:")
    try:
        result = final_optimal_baseline(test_text, test_length, test_freq, test_domain)
        print(f"   Type: {type(result)}")
        print(f"   Length: {len(result)}")
        print(f"   First 5 values: {result[:5]}")
        print(f"   All values numeric: {all(isinstance(x, (int, float)) for x in result)}")
        print(f"   Min: {min(result):.2f}, Max: {max(result):.2f}")
    except Exception as e:
        print(f"   ERROR: {e}")

def test_plotting():
    """Test plotting with sample data."""
    import plotly.graph_objects as go
    import numpy as np
    
    print("\n" + "=" * 50)
    print("Testing plotting functionality...")
    
    # Test with sample data
    sample_data = [100, 105, 110, 108, 115, 120, 118, 125, 130, 128]
    
    print(f"Sample data: {sample_data}")
    print(f"Data type: {type(sample_data)}")
    print(f"All numeric: {all(isinstance(x, (int, float)) for x in sample_data)}")
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(sample_data) + 1)),
        y=sample_data,
        mode='lines+markers',
        name='Test Data',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, color='#1f77b4')
    ))
    
    fig.update_layout(
        title="Test Plot",
        xaxis_title="Time Period",
        yaxis_title="Value",
        template="plotly_white",
        height=400
    )
    
    print("Plot created successfully!")
    print("If you see this message, plotting should work fine.")

if __name__ == "__main__":
    test_baseline_functions()
    test_plotting()
