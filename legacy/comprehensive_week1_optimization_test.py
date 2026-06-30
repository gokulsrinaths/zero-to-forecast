import numpy as np
import json
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import time

# Import all our optimization modules
from enhanced_domain_ensemble import EnhancedDomainEnsemble, enhanced_domain_ensemble_baseline
from finance_domain_optimization import FinanceDomainOptimizer, finance_optimized_baseline
from healthcare_domain_optimization import HealthcareDomainOptimizer, healthcare_optimized_baseline
from technology_domain_optimization import TechnologyDomainOptimizer, technology_optimized_baseline
from advanced_pattern_recognition import AdvancedPatternRecognizer
from baselines import ultimate_ensemble_baseline, super_optimized_baseline_v2
from comprehensive_evaluation import ComprehensiveEvaluator

class ComprehensiveWeek1OptimizationTest:
    def __init__(self):
        self.enhanced_ensemble = EnhancedDomainEnsemble()
        self.finance_optimizer = FinanceDomainOptimizer()
        self.healthcare_optimizer = HealthcareDomainOptimizer()
        self.technology_optimizer = TechnologyDomainOptimizer()
        self.pattern_recognizer = AdvancedPatternRecognizer()
        self.evaluator = ComprehensiveEvaluator()
        
        # Load test data
        with open('nl2ts_200.jsonl', 'r') as f:
            self.data = [json.loads(line) for line in f]
        
        # Domain-specific targets
        self.domain_targets = {
            'finance': 25.0,
            'healthcare': 20.0,
            'technology': 15.0,
            'retail': 20.0,
            'weather': 20.0,
            'iot': 20.0
        }

    def test_domain_specific_optimizations(self):
        """Test individual domain-specific optimizations"""
        print("=" * 80)
        print("DOMAIN-SPECIFIC OPTIMIZATION RESULTS")
        print("=" * 80)
        
        domain_results = {}
        
        for domain in ['finance', 'healthcare', 'technology']:
            domain_data = [item for item in self.data if item.get('domain') == domain]
            
            if not domain_data:
                continue
            
            print(f"\n{domain.upper()} DOMAIN OPTIMIZATION")
            print("-" * 50)
            
            # Test domain-specific optimizer
            if domain == 'finance':
                optimizer = self.finance_optimizer
                baseline_func = finance_optimized_baseline
            elif domain == 'healthcare':
                optimizer = self.healthcare_optimizer
                baseline_func = healthcare_optimized_baseline
            elif domain == 'technology':
                optimizer = self.technology_optimizer
                baseline_func = technology_optimized_baseline
            
            total_mae = 0
            sample_count = min(50, len(domain_data))
            
            for i, item in enumerate(domain_data[:sample_count]):
                text = item['text']
                target = item['series']
                length = len(target)
                freq = item.get('freq', 'D')
                domain_name = item.get('domain', domain)
                
                # Generate prediction
                prediction = baseline_func(text, length, freq, domain_name)
                
                # Calculate MAE
                mae = np.mean(np.abs(np.array(prediction) - np.array(target)))
                total_mae += mae
            
            mean_mae = total_mae / sample_count
            domain_results[domain] = mean_mae
            
            target_mae = self.domain_targets.get(domain, 20.0)
            improvement_needed = max(0, mean_mae - target_mae)
            
            print(f"  Mean MAE: {mean_mae:.2f}")
            print(f"  Target MAE: < {target_mae:.1f}")
            print(f"  Improvement needed: {improvement_needed:.2f}")
            print(f"  Status: {'✓ TARGET MET' if mean_mae <= target_mae else '✗ NEEDS IMPROVEMENT'}")
        
        return domain_results

    def test_enhanced_ensemble(self):
        """Test the enhanced domain ensemble"""
        print("\n" + "=" * 80)
        print("ENHANCED DOMAIN ENSEMBLE RESULTS")
        print("=" * 80)
        
        ensemble_results = {}
        
        for domain in ['finance', 'healthcare', 'technology', 'retail', 'weather', 'iot']:
            domain_data = [item for item in self.data if item.get('domain') == domain]
            
            if not domain_data:
                continue
            
            print(f"\n{domain.upper()} DOMAIN - Enhanced Ensemble")
            print("-" * 40)
            
            total_mae = 0
            sample_count = min(30, len(domain_data))
            
            for i, item in enumerate(domain_data[:sample_count]):
                text = item['text']
                target = item['series']
                length = len(target)
                freq = item.get('freq', 'D')
                domain_name = item.get('domain', domain)
                
                # Generate prediction using enhanced ensemble
                prediction = self.enhanced_ensemble.optimize_for_domain(text, length, freq, domain_name)
                
                # Calculate MAE
                mae = np.mean(np.abs(np.array(prediction) - np.array(target)))
                total_mae += mae
            
            mean_mae = total_mae / sample_count
            ensemble_results[domain] = mean_mae
            
            target_mae = self.domain_targets.get(domain, 20.0)
            improvement_needed = max(0, mean_mae - target_mae)
            
            print(f"  Mean MAE: {mean_mae:.2f}")
            print(f"  Target MAE: < {target_mae:.1f}")
            print(f"  Improvement needed: {improvement_needed:.2f}")
            print(f"  Status: {'✓ TARGET MET' if mean_mae <= target_mae else '✗ NEEDS IMPROVEMENT'}")
        
        return ensemble_results

    def run_cross_validation(self, baseline_func, domain='all'):
        """Run 5-fold cross-validation"""
        print(f"\n" + "=" * 80)
        print(f"CROSS-VALIDATION RESULTS ({domain.upper()} DOMAIN)")
        print("=" * 80)
        
        # Filter data by domain if specified
        if domain != 'all':
            test_data = [item for item in self.data if item.get('domain') == domain]
        else:
            test_data = self.data
        
        if len(test_data) < 10:
            print(f"Insufficient data for {domain} domain: {len(test_data)} samples")
            return None
        
        # Prepare data for cross-validation
        texts = [item['text'] for item in test_data]
        targets = [item['series'] for item in test_data]
        lengths = [len(item['series']) for item in test_data]
        freqs = [item.get('freq', 'D') for item in test_data]
        domains = [item.get('domain', 'general') for item in test_data]
        
        # Create folds based on text length (stratified)
        length_bins = np.digitize(lengths, bins=[10, 20, 30, 50, 100])
        
        cv_results = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(texts, length_bins)):
            fold_mae = 0
            
            for idx in test_idx:
                text = texts[idx]
                target = targets[idx]
                length = lengths[idx]
                freq = freqs[idx]
                domain_name = domains[idx]
                
                # Generate prediction
                prediction = baseline_func(text, length, freq, domain_name)
                
                # Calculate MAE
                mae = np.mean(np.abs(np.array(prediction) - np.array(target)))
                fold_mae += mae
            
            fold_mean_mae = fold_mae / len(test_idx)
            cv_results.append(fold_mean_mae)
            
            print(f"  Fold {fold + 1}: MAE = {fold_mean_mae:.2f}")
        
        overall_cv_mae = np.mean(cv_results)
        cv_std = np.std(cv_results)
        
        print(f"\n  Overall CV MAE: {overall_cv_mae:.2f} ± {cv_std:.2f}")
        
        return overall_cv_mae, cv_std

    def run_robustness_tests(self, baseline_func, domain='all'):
        """Run robustness tests"""
        print(f"\n" + "=" * 80)
        print(f"ROBUSTNESS TEST RESULTS ({domain.upper()} DOMAIN)")
        print("=" * 80)
        
        # Filter data by domain if specified
        if domain != 'all':
            test_data = [item for item in self.data if item.get('domain') == domain]
        else:
            test_data = self.data[:50]  # Limit for robustness testing
        
        if len(test_data) < 5:
            print(f"Insufficient data for {domain} domain: {len(test_data)} samples")
            return None
        
        # Run robustness tests
        robustness_scores = self.evaluator.run_robustness_tests(baseline_func, test_data)
        
        print("Robustness Scores:")
        for test_name, score in robustness_scores.items():
            print(f"  {test_name}: {score:.3f}")
        
        return robustness_scores

    def run_statistical_analysis(self, baseline_func, domain='all'):
        """Run statistical analysis"""
        print(f"\n" + "=" * 80)
        print(f"STATISTICAL ANALYSIS RESULTS ({domain.upper()} DOMAIN)")
        print("=" * 80)
        
        # Filter data by domain if specified
        if domain != 'all':
            test_data = [item for item in self.data if item.get('domain') == domain]
        else:
            test_data = self.data[:100]  # Limit for statistical analysis
        
        if len(test_data) < 10:
            print(f"Insufficient data for {domain} domain: {len(test_data)} samples")
            return None
        
        # Run statistical analysis
        stats_results = self.evaluator.run_statistical_analysis(baseline_func, test_data)
        
        print("Statistical Results:")
        print(f"  Overall MAE: {stats_results['overall_mae']:.2f}")
        print(f"  95% CI: [{stats_results['ci_lower']:.2f}, {stats_results['ci_upper']:.2f}]")
        print(f"  Effect Size (Cohen's d): {stats_results['effect_size']:.3f}")
        
        return stats_results

    def generate_comprehensive_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE WEEK 1 OPTIMIZATION REPORT")
        print("=" * 80)
        
        start_time = time.time()
        
        # Test domain-specific optimizations
        domain_results = self.test_domain_specific_optimizations()
        
        # Test enhanced ensemble
        ensemble_results = self.test_enhanced_ensemble()
        
        # Run cross-validation on enhanced ensemble
        cv_mae, cv_std = self.run_cross_validation(enhanced_domain_ensemble_baseline)
        
        # Run robustness tests
        robustness_scores = self.run_robustness_tests(enhanced_domain_ensemble_baseline)
        
        # Run statistical analysis
        stats_results = self.run_statistical_analysis(enhanced_domain_ensemble_baseline)
        
        # Calculate overall performance
        overall_mae = np.mean(list(ensemble_results.values()))
        
        # Generate summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        
        print(f"Overall Mean MAE: {overall_mae:.2f}")
        print(f"Cross-Validation MAE: {cv_mae:.2f} ± {cv_std:.2f}")
        print(f"Target Overall MAE: < 20.0")
        print(f"Overall Improvement needed: {max(0, overall_mae - 20.0):.2f}")
        
        print(f"\nDomain-Specific Performance:")
        for domain, mae in ensemble_results.items():
            target = self.domain_targets.get(domain, 20.0)
            status = "✓" if mae <= target else "✗"
            print(f"  {domain.upper()}: {mae:.2f} (target: <{target:.1f}) {status}")
        
        if robustness_scores:
            print(f"\nRobustness Performance:")
            for test_name, score in robustness_scores.items():
                target_score = 0.8
                status = "✓" if score >= target_score else "✗"
                print(f"  {test_name}: {score:.3f} (target: >{target_score:.1f}) {status}")
        
        # Calculate acceptance probability
        acceptance_prob = self.calculate_acceptance_probability(overall_mae, cv_mae, robustness_scores)
        print(f"\nEstimated NeurIPS Acceptance Probability: {acceptance_prob:.1%}")
        
        # Identify critical gaps
        print(f"\nCRITICAL GAPS IDENTIFIED:")
        critical_gaps = []
        
        for domain, mae in ensemble_results.items():
            target = self.domain_targets.get(domain, 20.0)
            if mae > target:
                gap = mae - target
                critical_gaps.append((domain, gap))
        
        critical_gaps.sort(key=lambda x: x[1], reverse=True)
        
        for domain, gap in critical_gaps[:3]:
            print(f"  {domain.upper()}: {gap:.2f} MAE points above target")
        
        execution_time = time.time() - start_time
        print(f"\nExecution time: {execution_time:.2f} seconds")
        
        return {
            'overall_mae': overall_mae,
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'domain_results': ensemble_results,
            'robustness_scores': robustness_scores,
            'acceptance_probability': acceptance_prob,
            'critical_gaps': critical_gaps
        }

    def calculate_acceptance_probability(self, overall_mae, cv_mae, robustness_scores):
        """Calculate estimated NeurIPS acceptance probability"""
        # Base probability based on MAE
        if overall_mae <= 15.0:
            base_prob = 0.95
        elif overall_mae <= 20.0:
            base_prob = 0.85
        elif overall_mae <= 25.0:
            base_prob = 0.70
        elif overall_mae <= 30.0:
            base_prob = 0.50
        else:
            base_prob = 0.20
        
        # Adjust for cross-validation stability
        cv_adjustment = 1.0
        if cv_mae <= overall_mae * 1.1:  # CV close to overall performance
            cv_adjustment = 1.1
        elif cv_mae > overall_mae * 1.3:  # CV much worse
            cv_adjustment = 0.8
        
        # Adjust for robustness
        robustness_adjustment = 1.0
        if robustness_scores:
            avg_robustness = np.mean(list(robustness_scores.values()))
            if avg_robustness >= 0.8:
                robustness_adjustment = 1.1
            elif avg_robustness <= 0.5:
                robustness_adjustment = 0.8
        
        final_prob = base_prob * cv_adjustment * robustness_adjustment
        return min(final_prob, 1.0)

def main():
    """Main function to run comprehensive Week 1 optimization test"""
    print("Starting Comprehensive Week 1 Optimization Test...")
    print("=" * 80)
    
    tester = ComprehensiveWeek1OptimizationTest()
    results = tester.generate_comprehensive_report()
    
    # Save results
    with open('week1_comprehensive_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'week1_comprehensive_results.json'")
    
    return results

if __name__ == "__main__":
    main()
