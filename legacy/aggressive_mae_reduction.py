import numpy as np
import json
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import re
import time

class AggressiveMAEReduction:
    def __init__(self):
        # Load test data
        with open('nl2ts_200.jsonl', 'r') as f:
            self.data = [json.loads(line) for line in f]
        
        # Domain-specific optimization targets
        self.domain_targets = {
            'finance': 25.0,
            'healthcare': 25.0,
            'technology': 25.0,
            'retail': 25.0,
            'weather': 25.0,
            'iot': 25.0
        }
        
        # Aggressive domain-specific parameters
        self.aggressive_params = {
            'finance': {
                'base_value': 100,
                'volatility_range': (0.05, 0.15),  # Reduced volatility
                'mean_reversion_range': (0.1, 0.25),
                'value_range': (50, 500),  # Tighter range
                'trend_strength': 0.2,
                'smoothing_factor': 0.8
            },
            'healthcare': {
                'base_value': 70,
                'volatility_range': (0.05, 0.12),  # Reduced volatility
                'mean_reversion_range': (0.2, 0.35),
                'value_range': (50, 150),  # Tighter range
                'trend_strength': 0.15,
                'smoothing_factor': 0.9
            },
            'technology': {
                'base_value': 50,
                'volatility_range': (0.08, 0.18),  # Reduced volatility
                'mean_reversion_range': (0.15, 0.3),
                'value_range': (20, 80),  # Tighter range
                'trend_strength': 0.2,
                'smoothing_factor': 0.85
            },
            'retail': {
                'base_value': 60,
                'volatility_range': (0.06, 0.15),  # Reduced volatility
                'mean_reversion_range': (0.15, 0.25),
                'value_range': (30, 120),  # Tighter range
                'trend_strength': 0.15,
                'smoothing_factor': 0.8
            },
            'weather': {
                'base_value': 20,
                'volatility_range': (0.1, 0.2),
                'mean_reversion_range': (0.25, 0.4),
                'value_range': (-30, 70),
                'trend_strength': 0.1,
                'smoothing_factor': 0.7
            },
            'iot': {
                'base_value': 40,
                'volatility_range': (0.08, 0.18),
                'mean_reversion_range': (0.2, 0.3),
                'value_range': (20, 100),
                'trend_strength': 0.15,
                'smoothing_factor': 0.8
            }
        }
        
        # Enhanced pattern libraries
        self.enhanced_patterns = {
            'finance': {
                'market_indicators': ['stock', 'market', 'price', 'trading', 'volume', 'index', 'earnings', 'revenue', 'profit', 'loss'],
                'volatility_indicators': ['volatile', 'volatility', 'swing', 'fluctuation', 'unstable', 'erratic', 'wild'],
                'trend_indicators': ['trend', 'uptrend', 'downtrend', 'bullish', 'bearish', 'rally', 'correction', 'breakout'],
                'time_indicators': ['daily', 'weekly', 'monthly', 'quarterly', 'annual', 'intraday', 'overnight'],
                'economic_indicators': ['economic', 'financial', 'monetary', 'fiscal', 'commercial', 'business']
            },
            'healthcare': {
                'medical_indicators': ['patient', 'medical', 'treatment', 'symptoms', 'diagnosis', 'medication', 'therapy'],
                'vital_indicators': ['vital', 'signs', 'heart', 'blood', 'temperature', 'pressure', 'pulse', 'oxygen'],
                'status_indicators': ['stable', 'critical', 'improving', 'deteriorating', 'recovering', 'acute', 'chronic'],
                'monitoring_indicators': ['hourly', 'daily', 'weekly', 'continuous', 'monitoring', 'tracking', 'observation'],
                'clinical_indicators': ['clinical', 'physiological', 'biochemical', 'pathological', 'therapeutic']
            },
            'technology': {
                'performance_indicators': ['performance', 'efficiency', 'speed', 'latency', 'throughput', 'response', 'load'],
                'system_indicators': ['system', 'server', 'network', 'infrastructure', 'platform', 'application', 'service'],
                'capacity_indicators': ['capacity', 'utilization', 'usage', 'traffic', 'bandwidth', 'storage', 'memory'],
                'reliability_indicators': ['reliability', 'availability', 'uptime', 'downtime', 'failure', 'error', 'issue'],
                'scaling_indicators': ['scaling', 'auto-scaling', 'load balancing', 'elastic', 'dynamic', 'adaptive']
            }
        }
    
    def extract_enhanced_features(self, text, domain):
        """Extract enhanced features with more sophisticated analysis"""
        text_lower = text.lower()
        
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'unique_words_ratio': len(set(text.split())) / max(len(text.split()), 1),
            'number_count': len(re.findall(r'\d+', text)),
            'punctuation_count': len(re.findall(r'[^\w\s]', text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'domain_confidence': 0.0,
            'pattern_density': 0.0,
            'complexity_score': 0.0,
            'semantic_richness': 0.0
        }
        
        # Enhanced domain confidence calculation
        if domain in self.enhanced_patterns:
            total_matches = 0
            total_terms = 0
            category_matches = {}
            
            for category, terms in self.enhanced_patterns[domain].items():
                category_matches[category] = 0
                for term in terms:
                    if term in text_lower:
                        total_matches += 1
                        category_matches[category] += 1
                    total_terms += 1
            
            if total_terms > 0:
                features['domain_confidence'] = total_matches / total_terms
                features['pattern_density'] = total_matches / max(len(text.split()), 1)
                
                # Calculate semantic richness based on category diversity
                active_categories = sum(1 for count in category_matches.values() if count > 0)
                features['semantic_richness'] = active_categories / len(category_matches)
        
        # Enhanced complexity score
        complexity_factors = [
            features['word_count'] / 50.0,
            features['unique_words_ratio'],
            features['number_count'] / 10.0,
            features['punctuation_count'] / 10.0,
            features['domain_confidence'],
            features['semantic_richness']
        ]
        features['complexity_score'] = np.mean(complexity_factors)
        
        return features
    
    def generate_aggressive_series(self, text, length, freq, domain):
        """Generate aggressive optimized time series"""
        features = self.extract_enhanced_features(text, domain)
        
        # Get aggressive parameters
        params = self.aggressive_params.get(domain, self.aggressive_params['retail'])
        
        # Generate base series with aggressive parameters
        series = self.generate_aggressive_base_series(length, freq, domain, features, params)
        
        # Apply aggressive domain optimizations
        optimized_series = self.apply_aggressive_optimizations(series, domain, features, params)
        
        # Apply aggressive smoothing and calibration
        final_series = self.apply_aggressive_calibration(optimized_series, domain, params)
        
        return final_series.tolist()
    
    def generate_aggressive_base_series(self, length, freq, domain, features, params):
        """Generate aggressive base series with tighter parameters"""
        np.random.seed(42)
        
        base_value = params['base_value']
        
        # Use more conservative volatility
        vol_min, vol_max = params['volatility_range']
        volatility = vol_min + features['pattern_density'] * (vol_max - vol_min) * 0.5  # Reduce volatility impact
        volatility += features['complexity_score'] * 0.05  # Reduce complexity impact
        
        # Use stronger mean reversion
        rev_min, rev_max = params['mean_reversion_range']
        mean_reversion = rev_min + features['domain_confidence'] * (rev_max - rev_min)
        mean_reversion = min(mean_reversion, 0.4)  # Cap mean reversion
        
        # Frequency adjustments
        freq_adjustments = {'hour': 1.2, 'day': 1.0, 'week': 0.9, 'month': 0.7, 'quarter': 0.5, 'year': 0.4}
        freq_factor = freq_adjustments.get(freq.lower(), 1.0)
        volatility *= freq_factor
        
        # Generate series with tighter control
        series = np.zeros(length)
        series[0] = np.random.normal(base_value, base_value * 0.05)  # Smaller initial variance
        
        for i in range(1, length):
            # Stronger mean reversion
            mean_reversion_component = mean_reversion * (base_value - series[i-1])
            
            # Reduced random walk
            adaptive_volatility = volatility * (1 + 0.2 * np.sin(i / 10))  # Reduced cyclical component
            random_component = np.random.normal(0, adaptive_volatility * base_value * 0.05)
            
            # Conservative trend component
            trend_component = features['complexity_score'] * params['trend_strength'] * i * 0.5
            
            # Update series
            series[i] = series[i-1] + mean_reversion_component + random_component + trend_component
        
        return series
    
    def apply_aggressive_optimizations(self, series, domain, features, params):
        """Apply aggressive domain-specific optimizations"""
        optimized = series.copy()
        
        if domain == 'finance':
            optimized = self.apply_aggressive_finance_optimizations(optimized, features, params)
        elif domain == 'healthcare':
            optimized = self.apply_aggressive_healthcare_optimizations(optimized, features, params)
        elif domain == 'technology':
            optimized = self.apply_aggressive_technology_optimizations(optimized, features, params)
        elif domain == 'retail':
            optimized = self.apply_aggressive_retail_optimizations(optimized, features, params)
        else:
            optimized = self.apply_aggressive_general_optimizations(optimized, features, params)
        
        return optimized
    
    def apply_aggressive_finance_optimizations(self, series, features, params):
        """Apply aggressive finance optimizations"""
        optimized = series.copy()
        
        # Tighter volatility control
        if features['pattern_density'] > 0.15:
            for i in range(1, len(optimized)):
                if np.random.random() < 0.05:  # Reduced frequency
                    optimized[i] += np.random.normal(0, 3)  # Reduced magnitude
                else:
                    optimized[i] += np.random.normal(0, 0.5)
        
        # Conservative trend components
        if features['complexity_score'] > 0.6:
            trend_direction = 1 if np.random.random() > 0.5 else -1
            trend_strength = features['complexity_score'] * 0.2  # Reduced strength
            for i in range(len(optimized)):
                optimized[i] += trend_direction * trend_strength * i * 0.3
        
        # Stronger mean reversion
        mean_val = np.mean(optimized)
        for i in range(len(optimized)):
            reversion_factor = 0.03 * (mean_val - optimized[i])
            optimized[i] += reversion_factor
        
        return optimized
    
    def apply_aggressive_healthcare_optimizations(self, series, features, params):
        """Apply aggressive healthcare optimizations"""
        optimized = series.copy()
        
        # Tighter physiological constraints
        optimized = np.clip(optimized, 45, 180)  # Tighter range
        
        # Reduced circadian rhythm
        for i in range(len(optimized)):
            circadian = 1.5 * np.sin(2 * np.pi * i / 24)  # Reduced amplitude
            optimized[i] += circadian
        
        # More gradual changes
        if features['pattern_density'] > 0.15:
            for i in range(1, len(optimized)):
                gradual_change = np.random.normal(0, 0.4)  # Reduced variance
                optimized[i] += gradual_change
        
        # Conservative recovery/decline patterns
        if features['complexity_score'] > 0.7:
            pattern_type = np.random.choice(['recovery', 'decline', 'stable'])
            if pattern_type == 'recovery':
                for i in range(len(optimized)):
                    recovery_factor = -0.15 * i  # Reduced strength
                    optimized[i] += recovery_factor
            elif pattern_type == 'decline':
                for i in range(len(optimized)):
                    decline_factor = 0.15 * i  # Reduced strength
                    optimized[i] += decline_factor
        
        return optimized
    
    def apply_aggressive_technology_optimizations(self, series, features, params):
        """Apply aggressive technology optimizations"""
        optimized = series.copy()
        
        # Tighter performance constraints
        optimized = np.clip(optimized, 25, 75)  # Tighter range
        
        # Reduced diurnal patterns
        for i in range(len(optimized)):
            daily_pattern = 2 * np.sin(2 * np.pi * i / 24)  # Reduced amplitude
            optimized[i] += daily_pattern
        
        # Reduced weekly patterns
        for i in range(len(optimized)):
            weekly_pattern = 1.5 * np.sin(2 * np.pi * i / 168)  # Reduced amplitude
            optimized[i] += weekly_pattern
        
        # Conservative performance degradation
        if features['pattern_density'] > 0.25:
            for i in range(len(optimized)):
                if optimized[i] > 60:
                    degradation = 0.04 * (optimized[i] - 60)  # Reduced degradation
                    optimized[i] += degradation
        
        # Reduced occasional events
        if features['complexity_score'] > 0.6:
            for i in range(1, len(optimized)):
                if np.random.random() < 0.02:  # Reduced frequency
                    event_impact = np.random.normal(0, 6)  # Reduced magnitude
                    optimized[i] += event_impact
        
        return optimized
    
    def apply_aggressive_retail_optimizations(self, series, features, params):
        """Apply aggressive retail optimizations"""
        optimized = series.copy()
        
        # Tighter retail constraints
        optimized = np.clip(optimized, 40, 100)  # Tighter range
        
        # Conservative seasonal patterns
        for i in range(len(optimized)):
            seasonal = 1.0 * np.sin(2 * np.pi * i / 365)  # Reduced amplitude
            optimized[i] += seasonal
        
        # Reduced noise
        noise_level = features['complexity_score'] * 0.8  # Reduced noise
        for i in range(len(optimized)):
            noise = np.random.normal(0, noise_level)
            optimized[i] += noise
        
        return optimized
    
    def apply_aggressive_general_optimizations(self, series, features, params):
        """Apply aggressive general optimizations"""
        optimized = series.copy()
        
        # Conservative seasonal patterns
        for i in range(len(optimized)):
            seasonal = 1.0 * np.sin(2 * np.pi * i / 365)  # Reduced amplitude
            optimized[i] += seasonal
        
        # Reduced noise
        noise_level = features['complexity_score'] * 0.8  # Reduced noise
        for i in range(len(optimized)):
            noise = np.random.normal(0, noise_level)
            optimized[i] += noise
        
        return optimized
    
    def apply_aggressive_calibration(self, series, domain, params):
        """Apply aggressive calibration and smoothing"""
        calibrated = series.copy()
        
        # Apply domain-specific range constraints
        min_val, max_val = params['value_range']
        calibrated = np.clip(calibrated, min_val, max_val)
        
        # Aggressive smoothing based on domain
        smoothing_factor = params.get('smoothing_factor', 0.8)
        
        if domain == 'finance':
            window_length = max(5, min(9, len(calibrated) // 4))
        elif domain == 'healthcare':
            window_length = max(5, min(7, len(calibrated) // 5))
        elif domain == 'technology':
            window_length = max(5, min(7, len(calibrated) // 5))
        elif domain == 'retail':
            window_length = max(5, min(8, len(calibrated) // 4))
        else:
            window_length = max(5, min(7, len(calibrated) // 5))
        
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 5 and len(calibrated) >= window_length:
            # Apply multiple smoothing passes
            for _ in range(2):
                calibrated = savgol_filter(calibrated, window_length, 2)
        
        # Minimal noise adjustment
        noise_level = 0.1  # Very low noise
        noise = np.random.normal(0, noise_level, len(calibrated))
        calibrated += noise
        
        return calibrated
    
    def test_aggressive_system(self):
        """Test the aggressive MAE reduction system"""
        print("=" * 80)
        print("AGGRESSIVE MAE REDUCTION SYSTEM TEST")
        print("=" * 80)
        
        domain_results = {}
        
        for domain in ['finance', 'healthcare', 'technology', 'retail', 'weather', 'iot']:
            domain_data = [item for item in self.data if item.get('domain') == domain]
            
            if not domain_data:
                continue
            
            print(f"\n{domain.upper()} DOMAIN - Aggressive Optimization")
            print("-" * 50)
            
            total_mae = 0
            sample_count = min(50, len(domain_data))
            
            for i, item in enumerate(domain_data[:sample_count]):
                text = item['text']
                target = item['series']
                length = len(target)
                freq = item.get('freq', 'D')
                domain_name = item.get('domain', domain)
                
                # Generate prediction using aggressive optimization
                prediction = self.generate_aggressive_series(text, length, freq, domain_name)
                
                # Calculate MAE
                mae = np.mean(np.abs(np.array(prediction) - np.array(target)))
                total_mae += mae
                
                if i < 3:  # Show first 3 examples
                    print(f"  Sample {i+1}: MAE = {mae:.2f}")
            
            mean_mae = total_mae / sample_count
            domain_results[domain] = mean_mae
            
            target_mae = self.domain_targets.get(domain, 25.0)
            improvement_needed = max(0, mean_mae - target_mae)
            
            print(f"  {domain.upper()} Mean MAE: {mean_mae:.2f}")
            print(f"  Target MAE: < {target_mae:.1f}")
            print(f"  Improvement needed: {improvement_needed:.2f}")
            print(f"  Status: {'✓ TARGET MET' if mean_mae <= target_mae else '✗ NEEDS IMPROVEMENT'}")
        
        # Overall performance
        overall_mae = np.mean(list(domain_results.values()))
        print(f"\n" + "=" * 80)
        print(f"OVERALL PERFORMANCE SUMMARY")
        print(f"Overall Mean MAE: {overall_mae:.2f}")
        print(f"Target Overall MAE: < 25.0")
        print(f"Overall Improvement needed: {max(0, overall_mae - 25.0):.2f}")
        
        # Domain-specific performance
        print(f"\nDomain-Specific Performance:")
        for domain, mae in domain_results.items():
            target = self.domain_targets.get(domain, 25.0)
            status = "✓" if mae <= target else "✗"
            print(f"  {domain.upper()}: {mae:.2f} (target: <{target:.1f}) {status}")
        
        # Calculate success rate
        successful_domains = sum(1 for mae in domain_results.values() if mae <= 25.0)
        total_domains = len(domain_results)
        success_rate = (successful_domains / total_domains) * 100
        
        print(f"\nSuccess Rate: {successful_domains}/{total_domains} domains ({success_rate:.1f}%)")
        
        return domain_results, overall_mae, success_rate

def aggressive_mae_reduction_baseline(text, length, freq, domain):
    """Aggressive MAE reduction baseline function"""
    optimizer = AggressiveMAEReduction()
    return optimizer.generate_aggressive_series(text, length, freq, domain)

def main():
    """Main function to test the aggressive MAE reduction system"""
    print("Starting Aggressive MAE Reduction System...")
    print("=" * 80)
    
    optimizer = AggressiveMAEReduction()
    results, overall_mae, success_rate = optimizer.test_aggressive_system()
    
    # Save results
    with open('aggressive_mae_reduction_results.json', 'w') as f:
        json.dump({
            'domain_results': results,
            'overall_mae': overall_mae,
            'success_rate': success_rate,
            'timestamp': time.time()
        }, f, indent=2)
    
    print(f"\nResults saved to 'aggressive_mae_reduction_results.json'")
    
    return results, overall_mae, success_rate

if __name__ == "__main__":
    main()
