import numpy as np
import json
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import re
import time

class AdvancedMAEReductionSystem:
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
        
        # Advanced feature extraction patterns
        self.advanced_patterns = {
            'finance': {
                'price_patterns': ['price', 'value', 'cost', 'worth', 'valuation'],
                'market_patterns': ['market', 'trading', 'exchange', 'stock', 'shares'],
                'volatility_patterns': ['volatile', 'fluctuation', 'swing', 'unstable', 'erratic'],
                'trend_patterns': ['trend', 'direction', 'movement', 'trajectory', 'path'],
                'time_patterns': ['daily', 'weekly', 'monthly', 'quarterly', 'annual'],
                'economic_patterns': ['economic', 'financial', 'monetary', 'fiscal', 'commercial']
            },
            'healthcare': {
                'medical_patterns': ['medical', 'clinical', 'health', 'patient', 'treatment'],
                'vital_patterns': ['vital', 'signs', 'pulse', 'heart', 'blood'],
                'symptom_patterns': ['symptom', 'condition', 'disease', 'illness', 'disorder'],
                'measurement_patterns': ['measurement', 'reading', 'level', 'count', 'rate'],
                'time_patterns': ['hourly', 'daily', 'weekly', 'continuous', 'monitoring'],
                'status_patterns': ['stable', 'improving', 'worsening', 'critical', 'normal']
            },
            'technology': {
                'performance_patterns': ['performance', 'efficiency', 'speed', 'latency', 'throughput'],
                'system_patterns': ['system', 'server', 'network', 'infrastructure', 'platform'],
                'load_patterns': ['load', 'traffic', 'usage', 'utilization', 'capacity'],
                'error_patterns': ['error', 'failure', 'issue', 'problem', 'outage'],
                'time_patterns': ['real-time', 'instant', 'continuous', 'periodic', 'scheduled'],
                'metric_patterns': ['metric', 'kpi', 'indicator', 'measure', 'statistic']
            }
        }
        
        # Initialize ML models for domain-specific optimization
        self.domain_models = {}
        self.scalers = {}
        
    def extract_advanced_features(self, text, domain, length, freq):
        """Extract advanced domain-specific features"""
        text_lower = text.lower()
        
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
            'unique_words_ratio': len(set(text.split())) / max(len(text.split()), 1),
            'punctuation_count': len(re.findall(r'[^\w\s]', text)),
            'number_count': len(re.findall(r'\d+', text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'domain_specific_terms': 0,
            'pattern_density': 0,
            'complexity_score': 0
        }
        
        # Domain-specific feature extraction
        if domain in self.advanced_patterns:
            domain_patterns = self.advanced_patterns[domain]
            total_patterns = 0
            matched_patterns = 0
            
            for pattern_category, pattern_terms in domain_patterns.items():
                for term in pattern_terms:
                    if term in text_lower:
                        matched_patterns += 1
                    total_patterns += 1
            
            features['domain_specific_terms'] = matched_patterns
            features['pattern_density'] = matched_patterns / max(total_patterns, 1)
        
        # Complexity score based on various factors
        complexity_factors = [
            features['word_count'] / 50.0,  # Normalize by expected length
            features['unique_words_ratio'],
            features['punctuation_count'] / 10.0,
            features['pattern_density'],
            features['uppercase_ratio']
        ]
        features['complexity_score'] = np.mean(complexity_factors)
        
        # Frequency encoding
        freq_mapping = {'hour': 1, 'day': 2, 'week': 3, 'month': 4, 'quarter': 5, 'year': 6}
        features['freq_encoded'] = freq_mapping.get(freq.lower(), 2)
        
        # Length encoding
        features['length_normalized'] = length / 100.0
        
        return features
    
    def generate_advanced_series(self, text, length, freq, domain):
        """Generate advanced time series with domain-specific optimization"""
        features = self.extract_advanced_features(text, domain, length, freq)
        
        # Base series generation with advanced techniques
        base_series = self.generate_intelligent_base_series(length, freq, domain, features)
        
        # Apply domain-specific transformations
        domain_series = self.apply_domain_specific_transformations(base_series, domain, features)
        
        # Apply advanced smoothing and optimization
        optimized_series = self.apply_advanced_optimization(domain_series, domain, features)
        
        # Final calibration
        final_series = self.calibrate_series(optimized_series, domain, features)
        
        return final_series.tolist()
    
    def generate_intelligent_base_series(self, length, freq, domain, features):
        """Generate intelligent base series using advanced techniques"""
        np.random.seed(42)  # For reproducibility
        
        # Determine base parameters based on domain and features
        if domain == 'finance':
            base_value = 100
            volatility = 0.15 + features['pattern_density'] * 0.2
            mean_reversion = 0.1 + features['complexity_score'] * 0.1
        elif domain == 'healthcare':
            base_value = 70
            volatility = 0.1 + features['pattern_density'] * 0.15
            mean_reversion = 0.2 + features['complexity_score'] * 0.1
        elif domain == 'technology':
            base_value = 50
            volatility = 0.12 + features['pattern_density'] * 0.18
            mean_reversion = 0.15 + features['complexity_score'] * 0.1
        else:
            base_value = 60
            volatility = 0.13 + features['pattern_density'] * 0.16
            mean_reversion = 0.12 + features['complexity_score'] * 0.1
        
        # Adjust based on frequency
        freq_adjustments = {
            'hour': 1.5, 'day': 1.0, 'week': 0.8, 'month': 0.6, 'quarter': 0.4, 'year': 0.3
        }
        freq_factor = freq_adjustments.get(freq.lower(), 1.0)
        volatility *= freq_factor
        
        # Generate base series with intelligent parameters
        series = np.zeros(length)
        series[0] = np.random.normal(base_value, base_value * 0.1)
        
        for i in range(1, length):
            # Mean reversion component
            mean_reversion_component = mean_reversion * (base_value - series[i-1])
            
            # Random walk component with adaptive volatility
            adaptive_volatility = volatility * (1 + 0.5 * np.sin(i / 10))  # Cyclical volatility
            random_component = np.random.normal(0, adaptive_volatility * base_value * 0.1)
            
            # Trend component based on complexity
            trend_component = features['complexity_score'] * 0.1 * i
            
            # Update series
            series[i] = series[i-1] + mean_reversion_component + random_component + trend_component
        
        return series
    
    def apply_domain_specific_transformations(self, series, domain, features):
        """Apply domain-specific transformations"""
        transformed = series.copy()
        
        if domain == 'finance':
            # Financial-specific transformations
            transformed = self.apply_finance_transformations(transformed, features)
        elif domain == 'healthcare':
            # Healthcare-specific transformations
            transformed = self.apply_healthcare_transformations(transformed, features)
        elif domain == 'technology':
            # Technology-specific transformations
            transformed = self.apply_technology_transformations(transformed, features)
        else:
            # General transformations
            transformed = self.apply_general_transformations(transformed, features)
        
        return transformed
    
    def apply_finance_transformations(self, series, features):
        """Apply finance-specific transformations"""
        transformed = series.copy()
        
        # Add market-like characteristics
        if features['pattern_density'] > 0.3:
            # Add volatility clustering
            for i in range(1, len(transformed)):
                if np.random.random() < 0.15:  # Volatility clustering
                    transformed[i] += np.random.normal(0, 8)
                else:
                    transformed[i] += np.random.normal(0, 2)
        
        # Add trend components based on complexity
        if features['complexity_score'] > 0.5:
            trend_direction = 1 if np.random.random() > 0.5 else -1
            trend_strength = features['complexity_score'] * 0.5
            for i in range(len(transformed)):
                transformed[i] += trend_direction * trend_strength * i
        
        # Add mean reversion for financial data
        mean_val = np.mean(transformed)
        for i in range(len(transformed)):
            reversion_factor = 0.05 * (mean_val - transformed[i])
            transformed[i] += reversion_factor
        
        return transformed
    
    def apply_healthcare_transformations(self, series, features):
        """Apply healthcare-specific transformations"""
        transformed = series.copy()
        
        # Add physiological constraints
        transformed = np.clip(transformed, 40, 200)  # Reasonable vital signs range
        
        # Add circadian rhythm
        for i in range(len(transformed)):
            circadian = 3 * np.sin(2 * np.pi * i / 24)  # 24-hour cycle
            transformed[i] += circadian
        
        # Add gradual changes for healthcare data
        if features['pattern_density'] > 0.3:
            for i in range(1, len(transformed)):
                gradual_change = np.random.normal(0, 1)
                transformed[i] += gradual_change
        
        # Add recovery/decline patterns
        if features['complexity_score'] > 0.6:
            pattern_type = np.random.choice(['recovery', 'decline', 'stable'])
            if pattern_type == 'recovery':
                for i in range(len(transformed)):
                    recovery_factor = -0.3 * i
                    transformed[i] += recovery_factor
            elif pattern_type == 'decline':
                for i in range(len(transformed)):
                    decline_factor = 0.3 * i
                    transformed[i] += decline_factor
        
        return transformed
    
    def apply_technology_transformations(self, series, features):
        """Apply technology-specific transformations"""
        transformed = series.copy()
        
        # Add performance characteristics
        transformed = np.clip(transformed, 0, 100)  # Performance percentage
        
        # Add diurnal patterns for technology systems
        for i in range(len(transformed)):
            daily_pattern = 5 * np.sin(2 * np.pi * i / 24)  # 24-hour cycle
            transformed[i] += daily_pattern
        
        # Add weekly patterns
        for i in range(len(transformed)):
            weekly_pattern = 3 * np.sin(2 * np.pi * i / 168)  # 168-hour week
            transformed[i] += weekly_pattern
        
        # Add performance degradation under load
        if features['pattern_density'] > 0.4:
            for i in range(len(transformed)):
                if transformed[i] > 70:
                    degradation = 0.1 * (transformed[i] - 70)
                    transformed[i] += degradation
        
        # Add occasional spikes/drops
        if features['complexity_score'] > 0.5:
            for i in range(1, len(transformed)):
                if np.random.random() < 0.05:  # 5% chance of significant event
                    event_impact = np.random.normal(0, 15)
                    transformed[i] += event_impact
        
        return transformed
    
    def apply_general_transformations(self, series, features):
        """Apply general transformations for other domains"""
        transformed = series.copy()
        
        # Add seasonal patterns
        for i in range(len(transformed)):
            seasonal = 2 * np.sin(2 * np.pi * i / 365)  # Annual cycle
            transformed[i] += seasonal
        
        # Add noise based on complexity
        noise_level = features['complexity_score'] * 2
        for i in range(len(transformed)):
            noise = np.random.normal(0, noise_level)
            transformed[i] += noise
        
        return transformed
    
    def apply_advanced_optimization(self, series, domain, features):
        """Apply advanced optimization techniques"""
        optimized = series.copy()
        
        # Adaptive smoothing based on domain and features
        if domain == 'finance':
            window_length = max(3, min(11, len(optimized) // 4))
        elif domain == 'healthcare':
            window_length = max(3, min(9, len(optimized) // 5))
        elif domain == 'technology':
            window_length = max(3, min(7, len(optimized) // 6))
        else:
            window_length = max(3, min(9, len(optimized) // 5))
        
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 3 and len(optimized) >= window_length:
            # Apply Savitzky-Golay smoothing
            optimized = savgol_filter(optimized, window_length, 2)
        
        # Add minimal noise to maintain realism
        noise_level = 0.5 + features['complexity_score'] * 0.5
        noise = np.random.normal(0, noise_level, len(optimized))
        optimized += noise
        
        return optimized
    
    def calibrate_series(self, series, domain, features):
        """Final calibration of the series"""
        calibrated = series.copy()
        
        # Domain-specific range calibration
        if domain == 'finance':
            # Ensure reasonable financial ranges
            calibrated = np.clip(calibrated, 10, 1000)
        elif domain == 'healthcare':
            # Ensure reasonable healthcare ranges
            calibrated = np.clip(calibrated, 30, 250)
        elif domain == 'technology':
            # Ensure reasonable technology ranges (0-100%)
            calibrated = np.clip(calibrated, 0, 100)
        else:
            # General range calibration
            calibrated = np.clip(calibrated, 0, 200)
        
        # Normalize based on complexity
        if features['complexity_score'] > 0.7:
            # For complex cases, add more variation
            variation_factor = 1.2
            mean_val = np.mean(calibrated)
            calibrated = mean_val + (calibrated - mean_val) * variation_factor
        
        return calibrated
    
    def train_domain_models(self):
        """Train domain-specific ML models for optimization"""
        print("Training domain-specific models...")
        
        for domain in ['finance', 'healthcare', 'technology']:
            domain_data = [item for item in self.data if item.get('domain') == domain]
            
            if len(domain_data) < 10:
                continue
            
            # Prepare training data
            X = []
            y = []
            
            for item in domain_data[:100]:  # Use first 100 samples for training
                text = item['text']
                target = item['series']
                length = len(target)
                freq = item.get('freq', 'D')
                
                features = self.extract_advanced_features(text, domain, length, freq)
                
                # Use target statistics as labels
                target_mean = np.mean(target)
                target_std = np.std(target)
                target_range = np.max(target) - np.min(target)
                
                X.append(list(features.values()))
                y.append([target_mean, target_std, target_range])
            
            if len(X) < 5:
                continue
            
            X = np.array(X)
            y = np.array(y)
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Store model and scaler
            self.domain_models[domain] = model
            self.scalers[domain] = StandardScaler().fit(X)
            
            print(f"Trained model for {domain} domain with {len(X)} samples")
    
    def test_advanced_system(self):
        """Test the advanced MAE reduction system"""
        print("=" * 80)
        print("ADVANCED MAE REDUCTION SYSTEM TEST")
        print("=" * 80)
        
        # Train domain models
        self.train_domain_models()
        
        domain_results = {}
        
        for domain in ['finance', 'healthcare', 'technology', 'retail', 'weather', 'iot']:
            domain_data = [item for item in self.data if item.get('domain') == domain]
            
            if not domain_data:
                continue
            
            print(f"\n{domain.upper()} DOMAIN - Advanced System")
            print("-" * 50)
            
            total_mae = 0
            sample_count = min(50, len(domain_data))
            
            for i, item in enumerate(domain_data[:sample_count]):
                text = item['text']
                target = item['series']
                length = len(target)
                freq = item.get('freq', 'D')
                domain_name = item.get('domain', domain)
                
                # Generate prediction using advanced system
                prediction = self.generate_advanced_series(text, length, freq, domain_name)
                
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
        
        return domain_results, overall_mae

def advanced_mae_reduction_baseline(text, length, freq, domain):
    """Advanced MAE reduction baseline function"""
    system = AdvancedMAEReductionSystem()
    return system.generate_advanced_series(text, length, freq, domain)

def main():
    """Main function to test the advanced MAE reduction system"""
    print("Starting Advanced MAE Reduction System...")
    print("=" * 80)
    
    system = AdvancedMAEReductionSystem()
    results, overall_mae = system.test_advanced_system()
    
    # Save results
    with open('advanced_mae_reduction_results.json', 'w') as f:
        json.dump({
            'domain_results': results,
            'overall_mae': overall_mae,
            'timestamp': time.time()
        }, f, indent=2)
    
    print(f"\nResults saved to 'advanced_mae_reduction_results.json'")
    
    return results, overall_mae

if __name__ == "__main__":
    main()
