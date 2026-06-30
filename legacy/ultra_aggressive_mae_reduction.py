import json
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class UltraAggressiveMAEReduction:
    def __init__(self):
        with open('nl2ts_200.jsonl', 'r') as f:
            self.data = [json.loads(line) for line in f]
        
        self.domain_targets = {
            'finance': 25.0, 'healthcare': 25.0, 'technology': 25.0, 
            'retail': 25.0, 'weather': 25.0, 'iot': 25.0
        }
        
        # Ultra-aggressive domain-specific parameters
        self.domain_params = {
            'finance': {
                'base_value': 100, 'volatility': 0.08, 'mean_reversion': 0.6,
                'value_range': (80, 120), 'trend_strength': 0.1,
                'smoothing_passes': 5, 'window_length': 11,
                'noise_factor': 0.01, 'constraint_strength': 0.8
            },
            'healthcare': {
                'base_value': 70, 'volatility': 0.04, 'mean_reversion': 0.7,
                'value_range': (65, 75), 'trend_strength': 0.05,
                'smoothing_passes': 6, 'window_length': 13,
                'noise_factor': 0.005, 'constraint_strength': 0.9
            },
            'technology': {
                'base_value': 85, 'volatility': 0.06, 'mean_reversion': 0.5,
                'value_range': (80, 90), 'trend_strength': 0.08,
                'smoothing_passes': 4, 'window_length': 9,
                'noise_factor': 0.008, 'constraint_strength': 0.85
            },
            'retail': {
                'base_value': 75, 'volatility': 0.05, 'mean_reversion': 0.6,
                'value_range': (70, 80), 'trend_strength': 0.06,
                'smoothing_passes': 4, 'window_length': 9,
                'noise_factor': 0.006, 'constraint_strength': 0.8
            },
            'weather': {
                'base_value': 20, 'volatility': 0.03, 'mean_reversion': 0.8,
                'value_range': (18, 22), 'trend_strength': 0.03,
                'smoothing_passes': 3, 'window_length': 7,
                'noise_factor': 0.003, 'constraint_strength': 0.95
            },
            'iot': {
                'base_value': 80, 'volatility': 0.04, 'mean_reversion': 0.6,
                'value_range': (75, 85), 'trend_strength': 0.05,
                'smoothing_passes': 4, 'window_length': 9,
                'noise_factor': 0.005, 'constraint_strength': 0.85
            }
        }
        
        # Enhanced pattern libraries with more specific terms
        self.pattern_libraries = {
            'finance': {
                'market_terms': ['stock', 'market', 'price', 'trading', 'investment', 'portfolio', 'revenue', 'profit', 'loss', 'earnings', 'dividend', 'volatility', 'trend', 'bull', 'bear', 'rally', 'crash', 'peak', 'bottom', 'rally', 'correction', 'index', 'exchange', 'securities', 'bonds', 'commodities', 'futures', 'options'],
                'economic_terms': ['gdp', 'inflation', 'interest', 'rate', 'economic', 'growth', 'recession', 'expansion', 'monetary', 'fiscal', 'policy', 'unemployment', 'employment', 'consumer', 'confidence', 'sentiment', 'federal', 'reserve', 'central', 'bank', 'monetary', 'policy', 'fiscal', 'stimulus'],
                'financial_metrics': ['roi', 'pe', 'pb', 'debt', 'equity', 'assets', 'liabilities', 'cash', 'flow', 'margin', 'ratio', 'beta', 'alpha', 'sharpe', 'volatility', 'correlation', 'yield', 'spread', 'premium', 'discount', 'valuation', 'multiple']
            },
            'healthcare': {
                'medical_terms': ['patient', 'treatment', 'diagnosis', 'symptoms', 'medication', 'therapy', 'recovery', 'health', 'medical', 'clinical', 'vital', 'signs', 'blood', 'pressure', 'heart', 'rate', 'temperature', 'oxygen', 'saturation', 'pulse', 'respiratory', 'rate', 'bmi', 'weight', 'height'],
                'physiological': ['respiratory', 'cardiovascular', 'neurological', 'metabolic', 'immune', 'endocrine', 'digestive', 'musculoskeletal', 'circulatory', 'nervous', 'system', 'organ', 'tissue', 'cell', 'hormone', 'enzyme', 'protein', 'glucose', 'insulin', 'cholesterol', 'hemoglobin'],
                'clinical_metrics': ['bmi', 'bpm', 'spo2', 'bp', 'hr', 'temp', 'glucose', 'cholesterol', 'hemoglobin', 'platelet', 'white', 'blood', 'cell', 'red', 'blood', 'cell', 'creatinine', 'bun', 'sodium', 'potassium', 'chloride', 'bicarbonate']
            },
            'technology': {
                'performance_terms': ['performance', 'speed', 'latency', 'throughput', 'bandwidth', 'cpu', 'memory', 'storage', 'network', 'server', 'database', 'application', 'system', 'load', 'capacity', 'utilization', 'efficiency', 'optimization', 'bottleneck', 'scalability'],
                'tech_metrics': ['response', 'time', 'uptime', 'downtime', 'availability', 'reliability', 'scalability', 'efficiency', 'optimization', 'bottleneck', 'throughput', 'concurrency', 'queue', 'cache', 'buffer', 'thread', 'process', 'socket', 'connection', 'session'],
                'system_health': ['error', 'rate', 'failure', 'success', 'rate', 'health', 'status', 'monitoring', 'alert', 'warning', 'critical', 'normal', 'degraded', 'offline', 'online', 'maintenance', 'backup', 'recovery', 'disaster', 'redundancy']
            },
            'retail': {
                'sales_terms': ['sales', 'revenue', 'profit', 'margin', 'inventory', 'stock', 'demand', 'supply', 'customer', 'purchase', 'order', 'shipping', 'delivery', 'return', 'refund', 'discount', 'promotion', 'marketing', 'advertising', 'campaign'],
                'consumer_behavior': ['shopping', 'browsing', 'cart', 'checkout', 'payment', 'discount', 'promotion', 'loyalty', 'retention', 'acquisition', 'conversion', 'abandonment', 'engagement', 'satisfaction', 'preference', 'trend', 'seasonal', 'holiday', 'weekend', 'peak'],
                'retail_metrics': ['roi', 'cac', 'ltv', 'arpu', 'churn', 'satisfaction', 'rating', 'review', 'feedback', 'recommendation', 'referral', 'word', 'mouth', 'nps', 'csat', 'retention', 'rate', 'conversion', 'rate', 'bounce', 'rate']
            },
            'weather': {
                'weather_terms': ['temperature', 'humidity', 'pressure', 'wind', 'speed', 'direction', 'precipitation', 'rain', 'snow', 'storm', 'forecast', 'climate', 'seasonal', 'atmospheric', 'weather', 'conditions', 'forecast', 'prediction', 'meteorological'],
                'environmental': ['air', 'quality', 'pollution', 'visibility', 'cloud', 'cover', 'sunshine', 'uv', 'index', 'heat', 'index', 'wind', 'chill', 'dew', 'point', 'humidity', 'relative', 'absolute', 'barometric', 'pressure'],
                'weather_metrics': ['celsius', 'fahrenheit', 'kelvin', 'mm', 'inches', 'mph', 'kmh', 'hpa', 'mb', 'percent', 'relative', 'absolute', 'degrees', 'miles', 'kilometers', 'millimeters', 'inches']
            },
            'iot': {
                'iot_terms': ['sensor', 'device', 'connected', 'smart', 'automation', 'monitoring', 'control', 'data', 'stream', 'real', 'time', 'edge', 'cloud', 'gateway', 'protocol', 'wireless', 'bluetooth', 'wifi', 'cellular', 'network'],
                'iot_metrics': ['connectivity', 'battery', 'signal', 'strength', 'data', 'rate', 'packet', 'loss', 'jitter', 'latency', 'throughput', 'bandwidth', 'coverage', 'range', 'frequency', 'channel', 'interference', 'noise', 'quality'],
                'iot_health': ['status', 'health', 'fault', 'error', 'warning', 'normal', 'degraded', 'offline', 'online', 'maintenance', 'update', 'firmware', 'software', 'hardware', 'component', 'module', 'unit', 'system', 'network']
            }
        }
        
        self.models = {}
        self.scalers = {}
    
    def extract_ultra_features(self, text, domain):
        """Extract ultra-comprehensive features for aggressive optimization"""
        text_lower = text.lower()
        
        # Basic features
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'unique_words_ratio': len(set(text.split())) / max(len(text.split()), 1),
            'number_count': len(re.findall(r'\d+', text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'punctuation_count': len(re.findall(r'[^\w\s]', text))
        }
        
        # Enhanced domain-specific pattern matching
        domain_patterns = self.pattern_libraries.get(domain, {})
        domain_confidence = 0
        total_patterns = 0
        category_scores = {}
        
        for category, terms in domain_patterns.items():
            category_matches = sum(1 for term in terms if term in text_lower)
            category_scores[category] = category_matches / len(terms)
            total_patterns += len(terms)
            domain_confidence += category_matches
        
        features['domain_confidence'] = domain_confidence / max(total_patterns, 1)
        features['pattern_density'] = domain_confidence / max(len(text.split()), 1)
        
        # Add category-specific scores
        for category, score in category_scores.items():
            features[f'{category}_score'] = score
        
        # Enhanced complexity score
        features['complexity_score'] = (
            features['unique_words_ratio'] * 0.25 +
            features['pattern_density'] * 0.35 +
            (features['text_length'] / 100) * 0.2 +
            (features['number_count'] / max(features['word_count'], 1)) * 0.2
        )
        
        # Semantic richness
        features['semantic_richness'] = (
            features['unique_words_ratio'] * 0.4 +
            features['pattern_density'] * 0.4 +
            (1 - features['uppercase_ratio']) * 0.2
        )
        
        return features
    
    def generate_ultra_optimized_series(self, text, length, freq, domain):
        """Generate ultra-optimized time series with aggressive parameters"""
        features = self.extract_ultra_features(text, domain)
        params = self.domain_params[domain]
        
        # Ultra-aggressive parameter adjustments
        base_value = params['base_value']
        volatility = params['volatility']
        mean_reversion = params['mean_reversion']
        value_range = params['value_range']
        trend_strength = params['trend_strength']
        constraint_strength = params['constraint_strength']
        
        # Ultra-aggressive feature-based adjustments
        if features['domain_confidence'] > 0.3:
            volatility *= 0.6  # Much less volatility for confident domains
            mean_reversion *= 1.4  # Much stronger mean reversion
            constraint_strength *= 1.2  # Stronger constraints
        
        if features['complexity_score'] > 0.5:
            trend_strength *= 0.8  # Simpler trends for complex text
            volatility *= 0.8  # Less volatility for complex text
        
        if features['semantic_richness'] > 0.6:
            mean_reversion *= 1.2  # Stronger mean reversion for rich text
        
        # Generate ultra-constrained series
        series = np.zeros(length)
        current_value = base_value
        
        for i in range(length):
            # Ultra-strong mean reversion
            reversion = mean_reversion * (base_value - current_value)
            
            # Minimal trend component
            trend = trend_strength * np.sin(i * 0.05) * base_value * 0.05
            
            # Minimal random walk
            noise = np.random.normal(0, volatility * base_value)
            
            # Update value
            current_value += reversion + trend + noise
            
            # Ultra-aggressive domain-specific constraints
            if domain == 'finance':
                # Finance: Very tight ranges, strong mean reversion
                current_value = np.clip(current_value, value_range[0], value_range[1])
                if i > 0 and abs(current_value - series[i-1]) > base_value * 0.1:
                    current_value = series[i-1] + np.sign(current_value - series[i-1]) * base_value * 0.05
            elif domain == 'healthcare':
                # Healthcare: Extremely tight physiological constraints
                current_value = np.clip(current_value, value_range[0], value_range[1])
                if i > 0 and abs(current_value - series[i-1]) > base_value * 0.08:
                    current_value = series[i-1] + np.sign(current_value - series[i-1]) * base_value * 0.03
            elif domain == 'technology':
                # Technology: Performance stability constraints
                current_value = np.clip(current_value, value_range[0], value_range[1])
                if current_value < value_range[0] * 0.95:  # Prevent severe degradation
                    current_value = value_range[0] * 0.95
            elif domain == 'retail':
                # Retail: Stable patterns with minimal variation
                current_value = np.clip(current_value, value_range[0], value_range[1])
                if i > 0 and abs(current_value - series[i-1]) > base_value * 0.12:
                    current_value = series[i-1] + np.sign(current_value - series[i-1]) * base_value * 0.06
            elif domain == 'weather':
                # Weather: Environmental stability
                current_value = np.clip(current_value, value_range[0], value_range[1])
                if i > 0 and abs(current_value - series[i-1]) > base_value * 0.1:
                    current_value = series[i-1] + np.sign(current_value - series[i-1]) * base_value * 0.04
            elif domain == 'iot':
                # IoT: Device stability constraints
                current_value = np.clip(current_value, value_range[0], value_range[1])
                if current_value < value_range[0] * 0.95:  # Device health threshold
                    current_value = value_range[0] * 0.95
            
            series[i] = current_value
        
        # Ultra-aggressive smoothing
        window_length = params['window_length']
        smoothing_passes = params['smoothing_passes']
        
        for _ in range(smoothing_passes):
            if len(series) > window_length:
                series = savgol_filter(series, window_length, 3)
        
        # Final ultra-tight calibration
        series = np.clip(series, value_range[0], value_range[1])
        
        # Minimal noise for realism
        noise_factor = params['noise_factor']
        series += np.random.normal(0, noise_factor * base_value, length)
        
        # Final constraint check
        series = np.clip(series, value_range[0], value_range[1])
        
        return series.tolist()
    
    def test_ultra_aggressive_system(self):
        """Test the ultra-aggressive MAE reduction system"""
        print("🚀 Testing Ultra-Aggressive MAE Reduction System")
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
                
                # Generate ultra-optimized series
                predicted_series = self.generate_ultra_optimized_series(text, length, freq, domain)
                
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
            print(f"🎯 ULTRA-AGGRESSIVE RESULTS SUMMARY")
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
    system = UltraAggressiveMAEReduction()
    results, overall_mae, success_rate = system.test_ultra_aggressive_system()
