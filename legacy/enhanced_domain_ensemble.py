import numpy as np
import json
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from finance_domain_optimization import FinanceDomainOptimizer
from healthcare_domain_optimization import HealthcareDomainOptimizer
from technology_domain_optimization import TechnologyDomainOptimizer
from advanced_pattern_recognition import AdvancedPatternRecognizer
from baselines import ultimate_ensemble_baseline, super_optimized_baseline_v2

class EnhancedDomainEnsemble:
    def __init__(self):
        self.finance_optimizer = FinanceDomainOptimizer()
        self.healthcare_optimizer = HealthcareDomainOptimizer()
        self.technology_optimizer = TechnologyDomainOptimizer()
        self.pattern_recognizer = AdvancedPatternRecognizer()
        
        # Domain-specific weights based on performance analysis
        self.domain_weights = {
            'finance': 0.25,
            'healthcare': 0.25,
            'technology': 0.25,
            'general': 0.25
        }
        
        # Ensemble weights for different baselines
        self.ensemble_weights = {
            'domain_specific': 0.4,
            'pattern_recognition': 0.3,
            'ultimate_ensemble': 0.2,
            'super_optimized': 0.1
        }
        
        # Robustness enhancement parameters
        self.robustness_params = {
            'paraphrase_weight': 0.8,
            'distractor_weight': 0.7,
            'self_consistency_weight': 0.9,
            'noise_injection_weight': 0.6
        }

    def extract_domain_features(self, text, domain):
        """Extract comprehensive domain-specific features"""
        text_lower = text.lower()
        
        features = {
            'domain_confidence': 0.0,
            'text_complexity': len(text.split()) / 100.0,
            'technical_terms': 0,
            'pattern_density': 0,
            'semantic_richness': 0
        }
        
        # Domain confidence based on domain-specific terms
        domain_terms = {
            'finance': ['stock', 'market', 'price', 'trading', 'volatility', 'earnings', 'revenue'],
            'healthcare': ['patient', 'medical', 'treatment', 'symptoms', 'diagnosis', 'medication'],
            'technology': ['system', 'performance', 'server', 'network', 'data', 'application']
        }
        
        if domain in domain_terms:
            domain_term_count = sum(1 for term in domain_terms[domain] if term in text_lower)
            features['domain_confidence'] = min(domain_term_count / 5.0, 1.0)
        
        # Technical terms count
        technical_terms = ['analysis', 'data', 'trend', 'pattern', 'performance', 'system', 'model']
        features['technical_terms'] = sum(1 for term in technical_terms if term in text_lower) / 10.0
        
        # Pattern density
        pattern_keywords = ['trend', 'pattern', 'cycle', 'seasonal', 'periodic', 'fluctuation']
        features['pattern_density'] = sum(1 for term in pattern_keywords if term in text_lower) / 6.0
        
        # Semantic richness (unique words ratio)
        words = text_lower.split()
        unique_words = len(set(words))
        features['semantic_richness'] = unique_words / max(len(words), 1)
        
        return features

    def select_optimal_ensemble(self, text, length, freq, domain):
        """Select optimal ensemble combination based on text characteristics"""
        features = self.extract_domain_features(text, domain)
        
        # Dynamic weight adjustment based on features
        weights = self.ensemble_weights.copy()
        
        # Adjust weights based on domain confidence
        if features['domain_confidence'] > 0.7:
            weights['domain_specific'] *= 1.5
            weights['ultimate_ensemble'] *= 0.8
        elif features['domain_confidence'] < 0.3:
            weights['domain_specific'] *= 0.7
            weights['ultimate_ensemble'] *= 1.3
        
        # Adjust weights based on pattern density
        if features['pattern_density'] > 0.5:
            weights['pattern_recognition'] *= 1.4
            weights['super_optimized'] *= 1.2
        
        # Adjust weights based on text complexity
        if features['text_complexity'] > 0.8:
            weights['ultimate_ensemble'] *= 1.3
            weights['pattern_recognition'] *= 1.1
        
        # Normalize weights
        total_weight = sum(weights.values())
        for key in weights:
            weights[key] /= total_weight
        
        return weights

    def generate_domain_specific_prediction(self, text, length, freq, domain):
        """Generate domain-specific prediction using appropriate optimizer"""
        if domain == 'finance':
            return self.finance_optimizer.generate_finance_optimized_series(text, length, freq, domain)
        elif domain == 'healthcare':
            return self.healthcare_optimizer.generate_healthcare_optimized_series(text, length, freq, domain)
        elif domain == 'technology':
            return self.technology_optimizer.generate_technology_optimized_series(text, length, freq, domain)
        else:
            # Fallback to pattern recognition for other domains
            return self.pattern_recognizer.generate_pattern_based_series(text, length, freq, domain)

    def apply_robustness_enhancement(self, prediction, text, domain):
        """Apply robustness enhancement techniques"""
        enhanced = np.array(prediction)
        
        # Paraphrase robustness: Add slight variations
        if np.random.random() < 0.3:
            paraphrase_noise = np.random.normal(0, 0.5, len(enhanced))
            enhanced += paraphrase_noise
        
        # Distractor robustness: Add small random perturbations
        if np.random.random() < 0.2:
            distractor_noise = np.random.normal(0, 0.3, len(enhanced))
            enhanced += distractor_noise
        
        # Self-consistency: Ensure smooth transitions
        if len(enhanced) > 3:
            # Apply smoothing to ensure consistency
            window_length = min(5, len(enhanced) // 4)
            if window_length % 2 == 0:
                window_length += 1
            if window_length >= 3:
                enhanced = savgol_filter(enhanced, window_length, 2)
        
        # Noise injection: Add minimal realistic noise
        noise = np.random.normal(0, 0.2, len(enhanced))
        enhanced += noise
        
        return enhanced.tolist()

    def generate_enhanced_ensemble_prediction(self, text, length, freq, domain):
        """Generate enhanced ensemble prediction with domain-specific optimization"""
        # Select optimal ensemble weights
        weights = self.select_optimal_ensemble(text, length, freq, domain)
        
        # Generate predictions from different baselines
        predictions = {}
        
        # Domain-specific prediction
        predictions['domain_specific'] = self.generate_domain_specific_prediction(text, length, freq, domain)
        
        # Pattern recognition prediction
        predictions['pattern_recognition'] = self.pattern_recognizer.generate_pattern_based_series(text, length, freq, domain)
        
        # Ultimate ensemble prediction
        predictions['ultimate_ensemble'] = ultimate_ensemble_baseline(text, length, freq, domain)
        
        # Super optimized prediction
        predictions['super_optimized'] = super_optimized_baseline_v2(text, length, freq, domain)
        
        # Combine predictions with weights
        ensemble_pred = np.zeros(length)
        for baseline, pred in predictions.items():
            if baseline in weights:
                ensemble_pred += np.array(pred) * weights[baseline]
        
        # Apply robustness enhancement
        enhanced_pred = self.apply_robustness_enhancement(ensemble_pred, text, domain)
        
        return enhanced_pred

    def optimize_for_domain(self, text, length, freq, domain):
        """Optimize prediction specifically for the given domain"""
        # Get base ensemble prediction
        base_prediction = self.generate_enhanced_ensemble_prediction(text, length, freq, domain)
        
        # Apply domain-specific post-processing
        if domain == 'finance':
            return self.apply_finance_post_processing(base_prediction, text)
        elif domain == 'healthcare':
            return self.apply_healthcare_post_processing(base_prediction, text)
        elif domain == 'technology':
            return self.apply_technology_post_processing(base_prediction, text)
        else:
            return base_prediction

    def apply_finance_post_processing(self, prediction, text):
        """Apply finance-specific post-processing"""
        pred = np.array(prediction)
        
        # Ensure realistic financial ranges
        pred = np.clip(pred, 0, 1000)  # Reasonable price range
        
        # Add mean reversion for financial data
        mean_val = np.mean(pred)
        for i in range(len(pred)):
            reversion_factor = 0.1 * (mean_val - pred[i])
            pred[i] += reversion_factor
        
        # Add volatility clustering
        if 'volatility' in text.lower() or 'volatile' in text.lower():
            for i in range(1, len(pred)):
                if np.random.random() < 0.15:
                    pred[i] += np.random.normal(0, 5)
        
        return pred.tolist()

    def apply_healthcare_post_processing(self, prediction, text):
        """Apply healthcare-specific post-processing"""
        pred = np.array(prediction)
        
        # Ensure realistic healthcare ranges
        pred = np.clip(pred, 0, 200)  # Reasonable vital signs range
        
        # Add circadian rhythm for healthcare data
        for i in range(len(pred)):
            circadian = 2 * np.sin(2 * np.pi * i / 24)
            pred[i] += circadian
        
        # Add recovery patterns if mentioned
        if 'improving' in text.lower() or 'recovery' in text.lower():
            for i in range(len(pred)):
                recovery_factor = -0.5 * i
                pred[i] += recovery_factor
        
        return pred.tolist()

    def apply_technology_post_processing(self, prediction, text):
        """Apply technology-specific post-processing"""
        pred = np.array(prediction)
        
        # Ensure realistic technology ranges (0-100% utilization)
        pred = np.clip(pred, 0, 100)
        
        # Add diurnal patterns for technology systems
        for i in range(len(pred)):
            daily_pattern = 3 * np.sin(2 * np.pi * i / 24)
            pred[i] += daily_pattern
        
        # Add performance degradation under load
        if 'high load' in text.lower() or 'overload' in text.lower():
            for i in range(len(pred)):
                if pred[i] > 70:
                    degradation = 0.1 * (pred[i] - 70)
                    pred[i] += degradation
        
        return pred.tolist()

def enhanced_domain_ensemble_baseline(text, length, freq, domain):
    """Enhanced domain ensemble baseline function"""
    ensemble = EnhancedDomainEnsemble()
    return ensemble.optimize_for_domain(text, length, freq, domain)

def test_enhanced_domain_ensemble():
    """Test the enhanced domain ensemble"""
    # Load test data
    with open('nl2ts_200.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    print("Testing Enhanced Domain Ensemble")
    print("=" * 60)
    
    # Test on all domains
    domains = ['finance', 'healthcare', 'technology', 'retail', 'weather', 'iot']
    domain_results = {}
    
    ensemble = EnhancedDomainEnsemble()
    
    for domain in domains:
        domain_data = [item for item in data if item.get('domain') == domain]
        
        if not domain_data:
            continue
        
        print(f"\nTesting {domain.upper()} domain ({len(domain_data)} samples)")
        print("-" * 40)
        
        total_mae = 0
        sample_count = min(30, len(domain_data))
        
        for i, item in enumerate(domain_data[:sample_count]):
            text = item['text']
            target = item['series']
            length = len(target)
            freq = item.get('freq', 'D')
            domain_name = item.get('domain', domain)
            
            # Generate prediction
            prediction = ensemble.optimize_for_domain(text, length, freq, domain_name)
            
            # Calculate MAE
            mae = np.mean(np.abs(np.array(prediction) - np.array(target)))
            total_mae += mae
            
            if i < 3:  # Show first 3 examples
                print(f"  Sample {i+1}: MAE = {mae:.2f}")
        
        mean_mae = total_mae / sample_count
        domain_results[domain] = mean_mae
        
        # Set target based on domain
        if domain == 'finance':
            target_mae = 25.0
        elif domain == 'healthcare':
            target_mae = 20.0
        elif domain == 'technology':
            target_mae = 15.0
        else:
            target_mae = 20.0
        
        print(f"  {domain.upper()} Mean MAE: {mean_mae:.2f}")
        print(f"  Target MAE: < {target_mae:.1f}")
        print(f"  Improvement needed: {max(0, mean_mae - target_mae):.2f}")
    
    # Overall performance
    overall_mae = np.mean(list(domain_results.values()))
    print(f"\n" + "=" * 60)
    print(f"OVERALL PERFORMANCE SUMMARY")
    print(f"Overall Mean MAE: {overall_mae:.2f}")
    print(f"Target Overall MAE: < 20.0")
    print(f"Overall Improvement needed: {max(0, overall_mae - 20.0):.2f}")
    
    # Domain-specific performance
    print(f"\nDomain-Specific Performance:")
    for domain, mae in domain_results.items():
        print(f"  {domain.upper()}: {mae:.2f}")
    
    return domain_results, overall_mae

if __name__ == "__main__":
    test_enhanced_domain_ensemble()
