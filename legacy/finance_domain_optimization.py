import numpy as np
import re
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import json

class FinanceDomainOptimizer:
    def __init__(self):
        # Financial terminology patterns
        self.financial_keywords = {
            'market': ['stock', 'market', 'trading', 'price', 'volume', 'index', 's&p', 'nasdaq', 'dow'],
            'volatility': ['volatility', 'vix', 'volatile', 'swing', 'fluctuation', 'uncertainty'],
            'trend': ['trend', 'uptrend', 'downtrend', 'bullish', 'bearish', 'rally', 'correction'],
            'momentum': ['momentum', 'rsi', 'macd', 'oscillator', 'overbought', 'oversold'],
            'support_resistance': ['support', 'resistance', 'breakout', 'breakdown', 'level'],
            'timeframe': ['daily', 'weekly', 'monthly', 'intraday', 'hourly', 'minute'],
            'sector': ['tech', 'finance', 'healthcare', 'energy', 'consumer', 'industrial'],
            'event': ['earnings', 'fomc', 'fed', 'inflation', 'gdp', 'unemployment', 'jobs']
        }
        
        # Market volatility patterns
        self.volatility_patterns = {
            'high_volatility': ['spike', 'surge', 'jump', 'crash', 'rally', 'panic', 'frenzy'],
            'low_volatility': ['stable', 'steady', 'calm', 'quiet', 'rangebound', 'sideways'],
            'increasing_volatility': ['rising', 'increasing', 'growing', 'escalating', 'building'],
            'decreasing_volatility': ['falling', 'declining', 'easing', 'subsiding', 'calming']
        }
        
        # Financial time series characteristics
        self.financial_characteristics = {
            'mean_reversion': ['bounce', 'recovery', 'pullback', 'retracement', 'correction'],
            'momentum': ['breakout', 'continuation', 'extension', 'acceleration'],
            'seasonality': ['earnings', 'quarterly', 'monthly', 'yearly', 'fiscal'],
            'cyclical': ['business cycle', 'economic cycle', 'market cycle']
        }

    def extract_financial_features(self, text, length, freq):
        """Extract finance-specific features from text"""
        text_lower = text.lower()
        
        features = {
            'market_terms': 0,
            'volatility_terms': 0,
            'trend_terms': 0,
            'momentum_terms': 0,
            'support_resistance_terms': 0,
            'timeframe_terms': 0,
            'sector_terms': 0,
            'event_terms': 0,
            'high_volatility': 0,
            'low_volatility': 0,
            'increasing_volatility': 0,
            'decreasing_volatility': 0,
            'mean_reversion': 0,
            'momentum_pattern': 0,
            'seasonality': 0,
            'cyclical': 0
        }
        
        # Count financial terminology
        for category, keywords in self.financial_keywords.items():
            for keyword in keywords:
                features[f'{category}_terms'] += text_lower.count(keyword)
        
        # Count volatility patterns
        for pattern, keywords in self.volatility_patterns.items():
            for keyword in keywords:
                features[pattern] += text_lower.count(keyword)
        
        # Count financial characteristics
        for characteristic, keywords in self.financial_characteristics.items():
            for keyword in keywords:
                if characteristic in features:
                    features[characteristic] += text_lower.count(keyword)
        
        # Normalize by text length
        text_length = len(text.split())
        for key in features:
            features[key] = features[key] / max(text_length, 1)
        
        return features

    def detect_financial_patterns(self, text, length, freq):
        """Detect specific financial patterns in the text"""
        text_lower = text.lower()
        
        patterns = {
            'is_market_data': False,
            'is_volatility_focus': False,
            'is_trend_analysis': False,
            'is_momentum_analysis': False,
            'is_support_resistance': False,
            'is_earnings_focus': False,
            'is_economic_event': False,
            'is_sector_analysis': False,
            'volatility_level': 'medium',  # low, medium, high
            'trend_direction': 'neutral',  # bullish, bearish, neutral
            'timeframe': 'daily',
            'complexity_score': 0.5
        }
        
        # Market data detection
        if any(term in text_lower for term in ['price', 'volume', 'index', 'stock', 'market']):
            patterns['is_market_data'] = True
        
        # Volatility focus
        if any(term in text_lower for term in ['volatility', 'vix', 'volatile', 'swing']):
            patterns['is_volatility_focus'] = True
        
        # Trend analysis
        if any(term in text_lower for term in ['trend', 'uptrend', 'downtrend', 'bullish', 'bearish']):
            patterns['is_trend_analysis'] = True
        
        # Momentum analysis
        if any(term in text_lower for term in ['momentum', 'rsi', 'macd', 'oscillator']):
            patterns['is_momentum_analysis'] = True
        
        # Support/Resistance
        if any(term in text_lower for term in ['support', 'resistance', 'breakout', 'breakdown']):
            patterns['is_support_resistance'] = True
        
        # Earnings focus
        if any(term in text_lower for term in ['earnings', 'quarterly', 'revenue', 'profit']):
            patterns['is_earnings_focus'] = True
        
        # Economic events
        if any(term in text_lower for term in ['fomc', 'fed', 'inflation', 'gdp', 'jobs']):
            patterns['is_economic_event'] = True
        
        # Sector analysis
        if any(term in text_lower for term in ['sector', 'industry', 'tech', 'finance', 'healthcare']):
            patterns['is_sector_analysis'] = True
        
        # Volatility level detection
        high_vol_terms = ['spike', 'surge', 'jump', 'crash', 'rally', 'panic']
        low_vol_terms = ['stable', 'steady', 'calm', 'quiet', 'rangebound']
        
        if any(term in text_lower for term in high_vol_terms):
            patterns['volatility_level'] = 'high'
        elif any(term in text_lower for term in low_vol_terms):
            patterns['volatility_level'] = 'low'
        
        # Trend direction
        bullish_terms = ['bullish', 'uptrend', 'rally', 'breakout', 'higher']
        bearish_terms = ['bearish', 'downtrend', 'correction', 'breakdown', 'lower']
        
        if any(term in text_lower for term in bullish_terms):
            patterns['trend_direction'] = 'bullish'
        elif any(term in text_lower for term in bearish_terms):
            patterns['trend_direction'] = 'bearish'
        
        # Timeframe detection
        if any(term in text_lower for term in ['intraday', 'hourly', 'minute']):
            patterns['timeframe'] = 'intraday'
        elif any(term in text_lower for term in ['weekly', 'monthly', 'quarterly']):
            patterns['timeframe'] = 'longer_term'
        
        # Complexity score based on financial terms
        financial_terms = sum(1 for category in self.financial_keywords.values() 
                            for keyword in category if keyword in text_lower)
        patterns['complexity_score'] = min(financial_terms / 10.0, 1.0)
        
        return patterns

    def generate_finance_optimized_series(self, text, length, freq, domain):
        """Generate finance-optimized time series based on detected patterns"""
        if domain != 'finance':
            return self.generate_fallback_series(length, freq)
        
        # Extract features and patterns
        features = self.extract_financial_features(text, length, freq)
        patterns = self.detect_financial_patterns(text, length, freq)
        
        # Base series generation with finance-specific adjustments
        base_series = self.generate_base_financial_series(length, freq, patterns)
        
        # Apply pattern-specific modifications
        modified_series = self.apply_financial_pattern_modifications(base_series, features, patterns)
        
        # Apply volatility adjustments
        volatility_adjusted = self.apply_volatility_adjustments(modified_series, patterns)
        
        # Apply trend adjustments
        trend_adjusted = self.apply_trend_adjustments(volatility_adjusted, patterns)
        
        # Final smoothing and normalization
        final_series = self.apply_financial_smoothing(trend_adjusted, patterns)
        
        return final_series.tolist()

    def generate_base_financial_series(self, length, freq, patterns):
        """Generate base financial time series"""
        # Start with a random walk with mean reversion
        np.random.seed(42)  # For reproducibility
        
        # Base parameters based on patterns
        if patterns['volatility_level'] == 'high':
            volatility = 0.3
            mean_reversion = 0.1
        elif patterns['volatility_level'] == 'low':
            volatility = 0.1
            mean_reversion = 0.3
        else:  # medium
            volatility = 0.2
            mean_reversion = 0.2
        
        # Generate base series
        series = np.zeros(length)
        series[0] = np.random.normal(100, 10)  # Start around 100
        
        for i in range(1, length):
            # Mean reversion component
            mean_reversion_component = mean_reversion * (100 - series[i-1])
            
            # Random walk component
            random_component = np.random.normal(0, volatility * 10)
            
            # Update series
            series[i] = series[i-1] + mean_reversion_component + random_component
        
        return series

    def apply_financial_pattern_modifications(self, series, features, patterns):
        """Apply finance-specific pattern modifications"""
        modified = series.copy()
        
        # Market data patterns
        if patterns['is_market_data']:
            # Add more realistic market movements
            for i in range(1, len(modified)):
                if np.random.random() < 0.1:  # 10% chance of significant move
                    modified[i] += np.random.normal(0, 5)
        
        # Volatility focus patterns
        if patterns['is_volatility_focus']:
            # Increase volatility
            volatility_multiplier = 1.5 if patterns['volatility_level'] == 'high' else 1.2
            for i in range(1, len(modified)):
                modified[i] += np.random.normal(0, volatility_multiplier * 3)
        
        # Trend analysis patterns
        if patterns['is_trend_analysis']:
            # Add trend component
            trend_strength = 0.5 if patterns['trend_direction'] != 'neutral' else 0.1
            trend_direction = 1 if patterns['trend_direction'] == 'bullish' else -1
            for i in range(len(modified)):
                modified[i] += trend_direction * trend_strength * i
        
        # Momentum patterns
        if patterns['is_momentum_analysis']:
            # Add momentum effects
            for i in range(2, len(modified)):
                momentum = (modified[i-1] - modified[i-2]) * 0.3
                modified[i] += momentum
        
        # Support/Resistance patterns
        if patterns['is_support_resistance']:
            # Add support/resistance levels
            support_level = np.min(modified) * 0.95
            resistance_level = np.max(modified) * 1.05
            for i in range(len(modified)):
                if modified[i] < support_level:
                    modified[i] = support_level + np.random.normal(0, 1)
                elif modified[i] > resistance_level:
                    modified[i] = resistance_level - np.random.normal(0, 1)
        
        return modified

    def apply_volatility_adjustments(self, series, patterns):
        """Apply volatility-specific adjustments"""
        adjusted = series.copy()
        
        if patterns['volatility_level'] == 'high':
            # Add volatility clustering
            for i in range(1, len(adjusted)):
                if np.random.random() < 0.2:  # Volatility clustering
                    adjusted[i] += np.random.normal(0, 8)
                else:
                    adjusted[i] += np.random.normal(0, 2)
        
        elif patterns['volatility_level'] == 'low':
            # Reduce volatility
            for i in range(1, len(adjusted)):
                adjusted[i] += np.random.normal(0, 0.5)
        
        return adjusted

    def apply_trend_adjustments(self, series, patterns):
        """Apply trend-specific adjustments"""
        adjusted = series.copy()
        
        if patterns['trend_direction'] == 'bullish':
            # Add upward trend
            trend_component = np.linspace(0, 20, len(adjusted))
            adjusted += trend_component
        
        elif patterns['trend_direction'] == 'bearish':
            # Add downward trend
            trend_component = np.linspace(0, -20, len(adjusted))
            adjusted += trend_component
        
        return adjusted

    def apply_financial_smoothing(self, series, patterns):
        """Apply finance-specific smoothing"""
        # Use Savitzky-Golay filter for financial data
        window_length = min(11, len(series) // 3)
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 3:
            smoothed = savgol_filter(series, window_length, 3)
        else:
            smoothed = series
        
        # Add some noise back to maintain realism
        noise = np.random.normal(0, 0.5, len(smoothed))
        final = smoothed + noise
        
        return final

    def generate_fallback_series(self, length, freq):
        """Generate fallback series for non-finance domains"""
        np.random.seed(42)
        series = np.cumsum(np.random.normal(0, 1, length)) + 100
        return series.tolist()

def finance_optimized_baseline(text, length, freq, domain):
    """Finance-optimized baseline function"""
    optimizer = FinanceDomainOptimizer()
    return optimizer.generate_finance_optimized_series(text, length, freq, domain)

def test_finance_optimization():
    """Test the finance domain optimization"""
    # Load test data
    with open('nl2ts_200.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Filter finance domain data
    finance_data = [item for item in data if item.get('domain') == 'finance']
    
    if not finance_data:
        print("No finance domain data found")
        return
    
    print(f"Testing Finance Domain Optimization on {len(finance_data)} samples")
    print("=" * 60)
    
    total_mae = 0
    optimizer = FinanceDomainOptimizer()
    
    for i, item in enumerate(finance_data[:50]):  # Test first 50 finance samples
        text = item['text']
        target = item['target']
        length = len(target)
        freq = item.get('freq', 'D')
        domain = item.get('domain', 'finance')
        
        # Generate prediction
        prediction = optimizer.generate_finance_optimized_series(text, length, freq, domain)
        
        # Calculate MAE
        mae = np.mean(np.abs(np.array(prediction) - np.array(target)))
        total_mae += mae
        
        if i < 5:  # Show first 5 examples
            print(f"Sample {i+1}:")
            print(f"  Text: {text[:100]}...")
            print(f"  MAE: {mae:.2f}")
            print(f"  Target range: [{min(target):.2f}, {max(target):.2f}]")
            print(f"  Prediction range: [{min(prediction):.2f}, {max(prediction):.2f}]")
            print()
    
    mean_mae = total_mae / min(50, len(finance_data))
    print(f"Finance Domain Mean MAE: {mean_mae:.2f}")
    print(f"Target MAE: < 25.0")
    print(f"Improvement needed: {max(0, mean_mae - 25.0):.2f}")
    
    return mean_mae

if __name__ == "__main__":
    test_finance_optimization()
