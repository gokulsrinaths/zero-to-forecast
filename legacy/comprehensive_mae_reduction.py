import json
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveMAEReduction:
    def __init__(self):
        with open('nl2ts_200.jsonl', 'r') as f:
            self.data = [json.loads(line) for line in f]
        
        self.domain_targets = {
            'finance': 25.0, 'healthcare': 25.0, 'technology': 25.0, 
            'retail': 25.0, 'weather': 25.0, 'iot': 25.0
        }
        
        # Enhanced domain-specific parameters
        self.domain_params = {
            'finance': {
                'base_value': 100, 'volatility': 0.15, 'mean_reversion': 0.3,
                'value_range': (50, 200), 'trend_strength': 0.2,
                'smoothing_passes': 3, 'window_length': 7
            },
            'healthcare': {
                'base_value': 70, 'volatility': 0.08, 'mean_reversion': 0.4,
                'value_range': (60, 90), 'trend_strength': 0.1,
                'smoothing_passes': 4, 'window_length': 9
            },
            'technology': {
                'base_value': 85, 'volatility': 0.12, 'mean_reversion': 0.25,
                'value_range': (70, 100), 'trend_strength': 0.15,
                'smoothing_passes': 3, 'window_length': 7
            },
            'retail': {
                'base_value': 75, 'volatility': 0.10, 'mean_reversion': 0.35,
                'value_range': (60, 95), 'trend_strength': 0.12,
                'smoothing_passes': 3, 'window_length': 7
            },
            'weather': {
                'base_value': 20, 'volatility': 0.05, 'mean_reversion': 0.5,
                'value_range': (15, 30), 'trend_strength': 0.08,
                'smoothing_passes': 2, 'window_length': 5
            },
            'iot': {
                'base_value': 80, 'volatility': 0.08, 'mean_reversion': 0.3,
                'value_range': (70, 90), 'trend_strength': 0.1,
                'smoothing_passes': 3, 'window_length': 7
            }
        }
        
        # Comprehensive pattern libraries
        self.pattern_libraries = {
            'finance': {
                'market_terms': ['stock', 'market', 'price', 'trading', 'investment', 'portfolio', 'revenue', 'profit', 'loss', 'earnings', 'dividend', 'volatility', 'trend', 'bull', 'bear', 'rally', 'crash', 'peak', 'bottom', 'rally', 'correction'],
                'economic_terms': ['gdp', 'inflation', 'interest', 'rate', 'economic', 'growth', 'recession', 'expansion', 'monetary', 'fiscal', 'policy', 'unemployment', 'employment', 'consumer', 'confidence', 'sentiment'],
                'financial_metrics': ['roi', 'pe', 'pb', 'debt', 'equity', 'assets', 'liabilities', 'cash', 'flow', 'margin', 'ratio', 'beta', 'alpha', 'sharpe', 'volatility', 'correlation']
            },
            'healthcare': {
                'medical_terms': ['patient', 'treatment', 'diagnosis', 'symptoms', 'medication', 'therapy', 'recovery', 'health', 'medical', 'clinical', 'vital', 'signs', 'blood', 'pressure', 'heart', 'rate', 'temperature', 'oxygen', 'saturation'],
                'physiological': ['respiratory', 'cardiovascular', 'neurological', 'metabolic', 'immune', 'endocrine', 'digestive', 'musculoskeletal', 'circulatory', 'nervous', 'system', 'organ', 'tissue', 'cell'],
                'clinical_metrics': ['bmi', 'bpm', 'spo2', 'bp', 'hr', 'temp', 'glucose', 'cholesterol', 'hemoglobin', 'platelet', 'white', 'blood', 'cell', 'red', 'blood', 'cell']
            },
            'technology': {
                'performance_terms': ['performance', 'speed', 'latency', 'throughput', 'bandwidth', 'cpu', 'memory', 'storage', 'network', 'server', 'database', 'application', 'system', 'load', 'capacity', 'utilization'],
                'tech_metrics': ['response', 'time', 'uptime', 'downtime', 'availability', 'reliability', 'scalability', 'efficiency', 'optimization', 'bottleneck', 'throughput', 'concurrency', 'queue', 'cache', 'buffer'],
                'system_health': ['error', 'rate', 'failure', 'success', 'rate', 'health', 'status', 'monitoring', 'alert', 'warning', 'critical', 'normal', 'degraded', 'offline', 'online']
            },
            'retail': {
                'sales_terms': ['sales', 'revenue', 'profit', 'margin', 'inventory', 'stock', 'demand', 'supply', 'customer', 'purchase', 'order', 'shipping', 'delivery', 'return', 'refund'],
                'consumer_behavior': ['shopping', 'browsing', 'cart', 'checkout', 'payment', 'discount', 'promotion', 'loyalty', 'retention', 'acquisition', 'conversion', 'abandonment', 'engagement'],
                'retail_metrics': ['roi', 'cac', 'ltv', 'arpu', 'churn', 'satisfaction', 'rating', 'review', 'feedback', 'recommendation', 'referral', 'word', 'mouth']
            },
            'weather': {
                'weather_terms': ['temperature', 'humidity', 'pressure', 'wind', 'speed', 'direction', 'precipitation', 'rain', 'snow', 'storm', 'forecast', 'climate', 'seasonal', 'atmospheric'],
                'environmental': ['air', 'quality', 'pollution', 'visibility', 'cloud', 'cover', 'sunshine', 'uv', 'index', 'heat', 'index', 'wind', 'chill', 'dew', 'point'],
                'weather_metrics': ['celsius', 'fahrenheit', 'kelvin', 'mm', 'inches', 'mph', 'kmh', 'hpa', 'mb', 'percent', 'relative', 'absolute']
            },
            'iot': {
                'iot_terms': ['sensor', 'device', 'connected', 'smart', 'automation', 'monitoring', 'control', 'data', 'stream', 'real', 'time', 'edge', 'cloud', 'gateway', 'protocol'],
                'iot_metrics': ['connectivity', 'battery', 'signal', 'strength', 'data', 'rate', 'packet', 'loss', 'jitter', 'latency', 'throughput', 'bandwidth', 'coverage', 'range'],
                'iot_health': ['status', 'health', 'fault', 'error', 'warning', 'normal', 'degraded', 'offline', 'online', 'maintenance', 'update', 'firmware', 'software']
            }
        }
        
        self.models = {}
        self.scalers = {}
    
    def extract_comprehensive_features(self, text, domain):
        """Extract comprehensive features for domain-specific optimization"""
        text_lower = text.lower()
        
        # Basic features
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'unique_words_ratio': len(set(text.split())) / max(len(text.split()), 1),
            'number_count': len(re.findall(r'\d+', text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1)
        }
        
        # Domain-specific pattern matching
        domain_patterns = self.pattern_libraries.get(domain, {})
        domain_confidence = 0
        total_patterns = 0
        
        for category, terms in domain_patterns.items():
            category_matches = sum(1 for term in terms if term in text_lower)
            total_patterns += len(terms)
            domain_confidence += category_matches
        
        features['domain_confidence'] = domain_confidence / max(total_patterns, 1)
        features['pattern_density'] = domain_confidence / max(len(text.split()), 1)
        
        # Complexity score
        features['complexity_score'] = (
            features['unique_words_ratio'] * 0.3 +
            features['pattern_density'] * 0.4 +
            (features['text_length'] / 100) * 0.3
        )
        
        return features
    
    def generate_optimized_series(self, text, length, freq, domain):
        """Generate optimized time series with domain-specific parameters"""
        features = self.extract_comprehensive_features(text, domain)
        params = self.domain_params[domain]
        
        # Base series generation with domain-specific parameters
        base_value = params['base_value']
        volatility = params['volatility']
        mean_reversion = params['mean_reversion']
        value_range = params['value_range']
        trend_strength = params['trend_strength']
        
        # Adjust parameters based on text features
        if features['domain_confidence'] > 0.5:
            volatility *= 0.8  # More confident domain = less volatility
            mean_reversion *= 1.2  # Stronger mean reversion
        
        if features['complexity_score'] > 0.6:
            trend_strength *= 1.3  # Complex text = stronger trends
        
        # Generate base series
        series = np.zeros(length)
        current_value = base_value
        
        for i in range(length):
            # Mean reversion component
            reversion = mean_reversion * (base_value - current_value)
            
            # Trend component
            trend = trend_strength * np.sin(i * 0.1) * base_value * 0.1
            
            # Random walk with volatility
            noise = np.random.normal(0, volatility * base_value)
            
            # Update value
            current_value += reversion + trend + noise
            
            # Apply domain-specific constraints
            if domain == 'finance':
                # Finance: Allow wider ranges, more volatility
                current_value = np.clip(current_value, value_range[0], value_range[1])
            elif domain == 'healthcare':
                # Healthcare: Tighter ranges, physiological constraints
                current_value = np.clip(current_value, value_range[0], value_range[1])
                if i > 0 and abs(current_value - series[i-1]) > base_value * 0.2:
                    current_value = series[i-1] + np.sign(current_value - series[i-1]) * base_value * 0.1
            elif domain == 'technology':
                # Technology: Performance constraints
                current_value = np.clip(current_value, value_range[0], value_range[1])
                if current_value < value_range[0] * 0.8:  # System degradation
                    current_value = value_range[0] * 0.8
            elif domain == 'retail':
                # Retail: Seasonal patterns
                seasonal_factor = 1 + 0.2 * np.sin(i * 2 * np.pi / 24)  # Daily pattern
                current_value *= seasonal_factor
                current_value = np.clip(current_value, value_range[0], value_range[1])
            elif domain == 'weather':
                # Weather: Environmental constraints
                current_value = np.clip(current_value, value_range[0], value_range[1])
                if i > 0 and abs(current_value - series[i-1]) > base_value * 0.15:
                    current_value = series[i-1] + np.sign(current_value - series[i-1]) * base_value * 0.08
            elif domain == 'iot':
                # IoT: Technology monitoring
                current_value = np.clip(current_value, value_range[0], value_range[1])
                if current_value < value_range[0] * 0.9:  # Device issues
                    current_value = value_range[0] * 0.9
            
            series[i] = current_value
        
        # Apply domain-specific smoothing
        window_length = params['window_length']
        smoothing_passes = params['smoothing_passes']
        
        for _ in range(smoothing_passes):
            if len(series) > window_length:
                series = savgol_filter(series, window_length, 3)
        
        # Final calibration
        series = np.clip(series, value_range[0], value_range[1])
        
        # Add small noise for realism
        noise_factor = 0.02
        series += np.random.normal(0, noise_factor * base_value, length)
        
        return series.tolist()
    
    def test_comprehensive_system(self):
        """Test the comprehensive MAE reduction system"""
        print("🚀 Testing Comprehensive MAE Reduction System")
        print("=" * 60)
        
        domain_results = {}
        total_mae = 0
        total_count = 0
        
        for domain in self.domain_targets.keys():
            domain_data = [item for item in self.data if item.get('domain') == domain]
            
            if not domain_data:
                continue
            
            domain_mae = 0
            domain_count = 0
            
            print(f"\n📊 Testing {domain.upper()} domain ({len(domain_data)} samples)")
            
            for item in domain_data:
                text = item['text']
                target_series = item['series']
                length = len(target_series)
                freq = item.get('freq', 'D')
                
                # Generate optimized series
                predicted_series = self.generate_optimized_series(text, length, freq, domain)
                
                # Calculate MAE
                mae = np.mean(np.abs(np.array(predicted_series) - np.array(target_series)))
                domain_mae += mae
                domain_count += 1
            
            if domain_count > 0:
                avg_mae = domain_mae / domain_count
                domain_results[domain] = avg_mae
                total_mae += domain_mae
                total_count += domain_count
                
                status = "✅ TARGET ACHIEVED" if avg_mae < self.domain_targets[domain] else "❌ NEEDS WORK"
                print(f"   MAE: {avg_mae:.2f} (Target: {self.domain_targets[domain]}) {status}")
        
        if total_count > 0:
            overall_mae = total_mae / total_count
            success_count = sum(1 for mae in domain_results.values() if mae < 25.0)
            success_rate = (success_count / len(domain_results)) * 100
            
            print(f"\n" + "=" * 60)
            print(f"🎯 COMPREHENSIVE RESULTS SUMMARY")
            print(f"=" * 60)
            print(f"Overall MAE: {overall_mae:.2f}")
            print(f"Success Rate: {success_rate:.1f}% ({success_count}/{len(domain_results)} domains)")
            
            print(f"\n📈 Domain Performance:")
            for domain, mae in sorted(domain_results.items(), key=lambda x: x[1]):
                status = "✅" if mae < 25.0 else "❌"
                print(f"   {domain.upper()}: {mae:.2f} {status}")
            
            return domain_results, overall_mae, success_rate
        
        return {}, 0, 0

if __name__ == "__main__":
    system = ComprehensiveMAEReduction()
    results, overall_mae, success_rate = system.test_comprehensive_system()
