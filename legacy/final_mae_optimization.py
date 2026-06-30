import numpy as np
import json
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import re
import time

class FinalMAEOptimization:
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
        
        # Domain-specific base values and parameters
        self.domain_params = {
            'finance': {
                'base_value': 100,
                'volatility_range': (0.1, 0.3),
                'mean_reversion_range': (0.05, 0.2),
                'value_range': (10, 1000),
                'trend_strength': 0.3
            },
            'healthcare': {
                'base_value': 70,
                'volatility_range': (0.08, 0.2),
                'mean_reversion_range': (0.15, 0.3),
                'value_range': (30, 250),
                'trend_strength': 0.2
            },
            'technology': {
                'base_value': 50,
                'volatility_range': (0.12, 0.25),
                'mean_reversion_range': (0.1, 0.25),
                'value_range': (0, 100),
                'trend_strength': 0.25
            },
            'retail': {
                'base_value': 60,
                'volatility_range': (0.1, 0.25),
                'mean_reversion_range': (0.1, 0.2),
                'value_range': (0, 200),
                'trend_strength': 0.2
            },
            'weather': {
                'base_value': 20,
                'volatility_range': (0.15, 0.3),
                'mean_reversion_range': (0.2, 0.4),
                'value_range': (-50, 100),
                'trend_strength': 0.15
            },
            'iot': {
                'base_value': 40,
                'volatility_range': (0.1, 0.25),
                'mean_reversion_range': (0.15, 0.3),
                'value_range': (0, 150),
                'trend_strength': 0.2
            }
        }
        
        # Pattern keywords for each domain
        self.domain_patterns = {
            'finance': {
                'market_terms': ['stock', 'market', 'price', 'trading', 'volume', 'index', 'earnings', 'revenue'],
                'volatility_terms': ['volatile', 'volatility', 'swing', 'fluctuation', 'unstable'],
                'trend_terms': ['trend', 'uptrend', 'downtrend', 'bullish', 'bearish', 'rally'],
                'time_terms': ['daily', 'weekly', 'monthly', 'quarterly', 'annual']
            },
            'healthcare': {
                'medical_terms': ['patient', 'medical', 'treatment', 'symptoms', 'diagnosis', 'medication'],
                'vital_terms': ['vital', 'signs', 'heart', 'blood', 'temperature', 'pressure'],
                'status_terms': ['stable', 'critical', 'improving', 'deteriorating', 'recovering'],
                'time_terms': ['hourly', 'daily', 'weekly', 'continuous', 'monitoring']
            },
            'technology': {
                'performance_terms': ['performance', 'efficiency', 'speed', 'latency', 'throughput'],
                'system_terms': ['system', 'server', 'network', 'infrastructure', 'platform'],
                'load_terms': ['load', 'traffic', 'usage', 'utilization', 'capacity'],
                'time_terms': ['real-time', 'instant', 'continuous', 'periodic', 'scheduled']
            }
        }
    
    def extract_simple_features(self, text, domain):
        """Extract simple but effective features"""
        text_lower = text.lower()
        
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'unique_words_ratio': len(set(text.split())) / max(len(text.split()), 1),
            'number_count': len(re.findall(r'\d+', text)),
            'domain_confidence': 0.0,
            'pattern_density': 0.0,
            'complexity_score': 0.0
        }
        
        # Domain confidence based on domain-specific terms
        if domain in self.domain_patterns:
            total_matches = 0
            total_terms = 0
            
            for category, terms in self.domain_patterns[domain].items():
                for term in terms:
                    if term in text_lower:
                        total_matches += 1
                    total_terms += 1
            
            if total_terms > 0:
                features['domain_confidence'] = total_matches / total_terms
                features['pattern_density'] = total_matches / max(len(text.split()), 1)
        
        # Complexity score
        complexity_factors = [
            features['word_count'] / 50.0,
            features['unique_words_ratio'],
            features['number_count'] / 10.0,
            features['domain_confidence']
        ]
        features['complexity_score'] = np.mean(complexity_factors)
        
        return features
    
    def generate_optimized_series(self, text, length, freq, domain):
        """Generate optimized time series for the given domain"""
        features = self.extract_simple_features(text, domain)
        
        # Get domain parameters
        params = self.domain_params.get(domain, self.domain_params['retail'])
        
        # Generate base series with domain-specific parameters
        series = self.generate_domain_base_series(length, freq, domain, features, params)
        
        # Apply domain-specific optimizations
        optimized_series = self.apply_domain_optimizations(series, domain, features, params)
        
        # Final calibration
        final_series = self.calibrate_series(optimized_series, domain, params)
        
        return final_series.tolist()
    
    def generate_domain_base_series(self, length, freq, domain, features, params):
        """Generate domain-specific base series"""
        np.random.seed(42)
        
        # Determine parameters based on features and domain
        base_value = params['base_value']
        
        # Adjust volatility based on pattern density and complexity
        vol_min, vol_max = params['volatility_range']
        volatility = vol_min + features['pattern_density'] * (vol_max - vol_min)
        volatility += features['complexity_score'] * 0.1
        
        # Adjust mean reversion based on domain confidence
        rev_min, rev_max = params['mean_reversion_range']
        mean_reversion = rev_min + features['domain_confidence'] * (rev_max - rev_min)
        
        # Frequency adjustments
        freq_adjustments = {'hour': 1.5, 'day': 1.0, 'week': 0.8, 'month': 0.6, 'quarter': 0.4, 'year': 0.3}
        freq_factor = freq_adjustments.get(freq.lower(), 1.0)
        volatility *= freq_factor
        
        # Generate series
        series = np.zeros(length)
        series[0] = np.random.normal(base_value, base_value * 0.1)
        
        for i in range(1, length):
            # Mean reversion component
            mean_reversion_component = mean_reversion * (base_value - series[i-1])
            
            # Random walk component with adaptive volatility
            adaptive_volatility = volatility * (1 + 0.3 * np.sin(i / 8))
            random_component = np.random.normal(0, adaptive_volatility * base_value * 0.1)
            
            # Trend component based on complexity
            trend_component = features['complexity_score'] * params['trend_strength'] * i
            
            # Update series
            series[i] = series[i-1] + mean_reversion_component + random_component + trend_component
        
        return series
    
    def apply_domain_optimizations(self, series, domain, features, params):
        """Apply domain-specific optimizations"""
        optimized = series.copy()
        
        if domain == 'finance':
            optimized = self.apply_finance_optimizations(optimized, features)
        elif domain == 'healthcare':
            optimized = self.apply_healthcare_optimizations(optimized, features)
        elif domain == 'technology':
            optimized = self.apply_technology_optimizations(optimized, features)
        else:
            optimized = self.apply_general_optimizations(optimized, features)
        
        return optimized
    
    def apply_finance_optimizations(self, series, features):
        """Apply finance-specific optimizations"""
        optimized = series.copy()
        
        # Add market-like characteristics
        if features['pattern_density'] > 0.2:
            # Volatility clustering
            for i in range(1, len(optimized)):
                if np.random.random() < 0.1:
                    optimized[i] += np.random.normal(0, 5)
                else:
                    optimized[i] += np.random.normal(0, 1)
        
        # Add trend components
        if features['complexity_score'] > 0.5:
            trend_direction = 1 if np.random.random() > 0.5 else -1
            trend_strength = features['complexity_score'] * 0.3
            for i in range(len(optimized)):
                optimized[i] += trend_direction * trend_strength * i
        
        # Mean reversion
        mean_val = np.mean(optimized)
        for i in range(len(optimized)):
            reversion_factor = 0.02 * (mean_val - optimized[i])
            optimized[i] += reversion_factor
        
        return optimized
    
    def apply_healthcare_optimizations(self, series, features):
        """Apply healthcare-specific optimizations"""
        optimized = series.copy()
        
        # Physiological constraints
        optimized = np.clip(optimized, 35, 220)
        
        # Circadian rhythm
        for i in range(len(optimized)):
            circadian = 2 * np.sin(2 * np.pi * i / 24)
            optimized[i] += circadian
        
        # Gradual changes
        if features['pattern_density'] > 0.2:
            for i in range(1, len(optimized)):
                gradual_change = np.random.normal(0, 0.6)
                optimized[i] += gradual_change
        
        # Recovery/decline patterns
        if features['complexity_score'] > 0.6:
            pattern_type = np.random.choice(['recovery', 'decline', 'stable'])
            if pattern_type == 'recovery':
                for i in range(len(optimized)):
                    recovery_factor = -0.2 * i
                    optimized[i] += recovery_factor
            elif pattern_type == 'decline':
                for i in range(len(optimized)):
                    decline_factor = 0.2 * i
                    optimized[i] += decline_factor
        
        return optimized
    
    def apply_technology_optimizations(self, series, features):
        """Apply technology-specific optimizations"""
        optimized = series.copy()
        
        # Performance constraints
        optimized = np.clip(optimized, 0, 100)
        
        # Diurnal patterns
        for i in range(len(optimized)):
            daily_pattern = 3 * np.sin(2 * np.pi * i / 24)
            optimized[i] += daily_pattern
        
        # Weekly patterns
        for i in range(len(optimized)):
            weekly_pattern = 2 * np.sin(2 * np.pi * i / 168)
            optimized[i] += weekly_pattern
        
        # Performance degradation under load
        if features['pattern_density'] > 0.3:
            for i in range(len(optimized)):
                if optimized[i] > 65:
                    degradation = 0.06 * (optimized[i] - 65)
                    optimized[i] += degradation
        
        # Occasional events
        if features['complexity_score'] > 0.5:
            for i in range(1, len(optimized)):
                if np.random.random() < 0.03:
                    event_impact = np.random.normal(0, 10)
                    optimized[i] += event_impact
        
        return optimized
    
    def apply_general_optimizations(self, series, features):
        """Apply general optimizations for other domains"""
        optimized = series.copy()
        
        # Seasonal patterns
        for i in range(len(optimized)):
            seasonal = 1.5 * np.sin(2 * np.pi * i / 365)
            optimized[i] += seasonal
        
        # Noise based on complexity
        noise_level = features['complexity_score'] * 1.2
        for i in range(len(optimized)):
            noise = np.random.normal(0, noise_level)
            optimized[i] += noise
        
        return optimized
    
    def calibrate_series(self, series, domain, params):
        """Final calibration of the series"""
        calibrated = series.copy()
        
        # Apply domain-specific range constraints
        min_val, max_val = params['value_range']
        calibrated = np.clip(calibrated, min_val, max_val)
        
        # Adaptive smoothing based on domain
        if domain == 'finance':
            window_length = max(3, min(7, len(calibrated) // 6))
        elif domain == 'healthcare':
            window_length = max(3, min(5, len(calibrated) // 8))
        elif domain == 'technology':
            window_length = max(3, min(5, len(calibrated) // 8))
        else:
            window_length = max(3, min(6, len(calibrated) // 7))
        
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 3 and len(calibrated) >= window_length:
            calibrated = savgol_filter(calibrated, window_length, 2)
        
        # Final noise adjustment
        noise_level = 0.2
        noise = np.random.normal(0, noise_level, len(calibrated))
        calibrated += noise
        
        return calibrated
    
    def test_final_optimization(self):
        """Test the final MAE optimization system"""
        print("=" * 80)
        print("FINAL MAE OPTIMIZATION SYSTEM TEST")
        print("=" * 80)
        
        domain_results = {}
        
        for domain in ['finance', 'healthcare', 'technology', 'retail', 'weather', 'iot']:
            domain_data = [item for item in self.data if item.get('domain') == domain]
            
            if not domain_data:
                continue
            
            print(f"\n{domain.upper()} DOMAIN - Final Optimization")
            print("-" * 50)
            
            total_mae = 0
            sample_count = min(50, len(domain_data))
            
            for i, item in enumerate(domain_data[:sample_count]):
                text = item['text']
                target = item['series']
                length = len(target)
                freq = item.get('freq', 'D')
                domain_name = item.get('domain', domain)
                
                # Generate prediction using final optimization
                prediction = self.generate_optimized_series(text, length, freq, domain_name)
                
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

def final_mae_optimization_baseline(text, length, freq, domain):
    """Final MAE optimization baseline function"""
    optimizer = FinalMAEOptimization()
    return optimizer.generate_optimized_series(text, length, freq, domain)

def main():
    """Main function to test the final MAE optimization system"""
    print("Starting Final MAE Optimization System...")
    print("=" * 80)
    
    optimizer = FinalMAEOptimization()
    results, overall_mae, success_rate = optimizer.test_final_optimization()
    
    # Save results
    with open('final_mae_optimization_results.json', 'w') as f:
        json.dump({
            'domain_results': results,
            'overall_mae': overall_mae,
            'success_rate': success_rate,
            'timestamp': time.time()
        }, f, indent=2)
    
    print(f"\nResults saved to 'final_mae_optimization_results.json'")
    
    return results, overall_mae, success_rate

if __name__ == "__main__":
    main()
