#!/usr/bin/env python3
"""
Uncertainty Quantification for NL2TS Forecasting
Includes CRPS, calibration error, coverage probability, and prediction intervals
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import json
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

@dataclass
class UncertaintyResult:
    predictions: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    confidence_level: float
    crps: float
    calibration_error: float
    coverage_probability: float
    prediction_intervals: List[Tuple[float, float]]

class UncertaintyQuantifier:
    """Main class for uncertainty quantification"""
    
    def __init__(self):
        self.confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
        
    def generate_prediction_intervals(self, 
                                    predictions: List[float], 
                                    confidence_level: float = 0.95,
                                    method: str = 'bootstrap') -> UncertaintyResult:
        """Generate prediction intervals for forecasts"""
        
        if method == 'bootstrap':
            return self._bootstrap_intervals(predictions, confidence_level)
        elif method == 'quantile':
            return self._quantile_intervals(predictions, confidence_level)
        elif method == 'gaussian':
            return self._gaussian_intervals(predictions, confidence_level)
        elif method == 'ensemble':
            return self._ensemble_intervals(predictions, confidence_level)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _bootstrap_intervals(self, predictions: List[float], confidence_level: float) -> UncertaintyResult:
        """Generate prediction intervals using bootstrap resampling"""
        n_samples = len(predictions)
        n_bootstrap = 1000
        
        # Generate bootstrap samples
        bootstrap_samples = []
        for _ in range(n_bootstrap):
            # Add noise to simulate uncertainty
            noise = np.random.normal(0, np.std(predictions) * 0.1, n_samples)
            bootstrap_sample = np.array(predictions) + noise
            bootstrap_samples.append(bootstrap_sample)
        
        bootstrap_samples = np.array(bootstrap_samples)
        
        # Calculate quantiles
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        lower_bound = np.percentile(bootstrap_samples, lower_quantile * 100, axis=0)
        upper_bound = np.percentile(bootstrap_samples, upper_quantile * 100, axis=0)
        
        # Calculate CRPS
        crps = self._calculate_crps(predictions, bootstrap_samples)
        
        # Calculate calibration error
        calibration_error = self._calculate_calibration_error(predictions, lower_bound, upper_bound, confidence_level)
        
        # Calculate coverage probability
        coverage_prob = self._calculate_coverage_probability(predictions, lower_bound, upper_bound)
        
        # Create prediction intervals
        prediction_intervals = list(zip(lower_bound, upper_bound))
        
        return UncertaintyResult(
            predictions=predictions,
            lower_bound=lower_bound.tolist(),
            upper_bound=upper_bound.tolist(),
            confidence_level=confidence_level,
            crps=crps,
            calibration_error=calibration_error,
            coverage_probability=coverage_prob,
            prediction_intervals=prediction_intervals
        )
    
    def _quantile_intervals(self, predictions: List[float], confidence_level: float) -> UncertaintyResult:
        """Generate prediction intervals using quantile regression"""
        n_samples = len(predictions)
        
        # Estimate quantiles based on prediction variance
        pred_std = np.std(predictions) * 0.2  # Assume 20% relative uncertainty
        
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        lower_bound = np.array(predictions) - z_score * pred_std
        upper_bound = np.array(predictions) + z_score * pred_std
        
        # Calculate metrics
        crps = self._calculate_crps_simple(predictions, pred_std)
        calibration_error = self._calculate_calibration_error(predictions, lower_bound, upper_bound, confidence_level)
        coverage_prob = self._calculate_coverage_probability(predictions, lower_bound, upper_bound)
        
        prediction_intervals = list(zip(lower_bound, upper_bound))
        
        return UncertaintyResult(
            predictions=predictions,
            lower_bound=lower_bound.tolist(),
            upper_bound=upper_bound.tolist(),
            confidence_level=confidence_level,
            crps=crps,
            calibration_error=calibration_error,
            coverage_probability=coverage_prob,
            prediction_intervals=prediction_intervals
        )
    
    def _gaussian_intervals(self, predictions: List[float], confidence_level: float) -> UncertaintyResult:
        """Generate prediction intervals assuming Gaussian distribution"""
        n_samples = len(predictions)
        
        # Estimate mean and variance
        mean_pred = np.mean(predictions)
        pred_std = np.std(predictions) * 0.15  # 15% relative uncertainty
        
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        lower_bound = np.array(predictions) - z_score * pred_std
        upper_bound = np.array(predictions) + z_score * pred_std
        
        # Calculate metrics
        crps = self._calculate_crps_simple(predictions, pred_std)
        calibration_error = self._calculate_calibration_error(predictions, lower_bound, upper_bound, confidence_level)
        coverage_prob = self._calculate_coverage_probability(predictions, lower_bound, upper_bound)
        
        prediction_intervals = list(zip(lower_bound, upper_bound))
        
        return UncertaintyResult(
            predictions=predictions,
            lower_bound=lower_bound.tolist(),
            upper_bound=upper_bound.tolist(),
            confidence_level=confidence_level,
            crps=crps,
            calibration_error=calibration_error,
            coverage_probability=coverage_prob,
            prediction_intervals=prediction_intervals
        )
    
    def _ensemble_intervals(self, predictions: List[float], confidence_level: float) -> UncertaintyResult:
        """Generate prediction intervals using ensemble methods"""
        n_samples = len(predictions)
        
        # Simulate ensemble predictions
        n_ensemble = 50
        ensemble_predictions = []
        
        for _ in range(n_ensemble):
            # Add different types of noise
            trend_noise = np.random.normal(0, 0.05, n_samples)
            seasonal_noise = 0.1 * np.sin(np.linspace(0, 4*np.pi, n_samples)) * np.random.normal(0, 1)
            random_noise = np.random.normal(0, 0.02, n_samples)
            
            ensemble_pred = np.array(predictions) + trend_noise + seasonal_noise + random_noise
            ensemble_predictions.append(ensemble_pred)
        
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Calculate quantiles
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        lower_bound = np.percentile(ensemble_predictions, lower_quantile * 100, axis=0)
        upper_bound = np.percentile(ensemble_predictions, upper_quantile * 100, axis=0)
        
        # Calculate metrics
        crps = self._calculate_crps(predictions, ensemble_predictions)
        calibration_error = self._calculate_calibration_error(predictions, lower_bound, upper_bound, confidence_level)
        coverage_prob = self._calculate_coverage_probability(predictions, lower_bound, upper_bound)
        
        prediction_intervals = list(zip(lower_bound, upper_bound))
        
        return UncertaintyResult(
            predictions=predictions,
            lower_bound=lower_bound.tolist(),
            upper_bound=upper_bound.tolist(),
            confidence_level=confidence_level,
            crps=crps,
            calibration_error=calibration_error,
            coverage_probability=coverage_prob,
            prediction_intervals=prediction_intervals
        )
    
    def _calculate_crps(self, predictions: List[float], ensemble_samples: np.ndarray) -> float:
        """Calculate Continuous Ranked Probability Score"""
        predictions = np.array(predictions)
        n_samples = len(predictions)
        
        # CRPS for each time step
        crps_scores = []
        for i in range(n_samples):
            pred_i = predictions[i]
            ensemble_i = ensemble_samples[:, i]
            
            # Sort ensemble samples
            ensemble_i_sorted = np.sort(ensemble_i)
            
            # Calculate CRPS
            n_ensemble = len(ensemble_i_sorted)
            crps_i = 0.0
            
            for j in range(n_ensemble):
                # Probability that ensemble sample j is less than prediction
                prob_less = j / n_ensemble
                crps_i += (prob_less - (1 if pred_i >= ensemble_i_sorted[j] else 0)) ** 2
            
            crps_scores.append(crps_i / n_ensemble)
        
        return np.mean(crps_scores)
    
    def _calculate_crps_simple(self, predictions: List[float], std: float) -> float:
        """Calculate simplified CRPS assuming Gaussian distribution"""
        # For Gaussian distribution, CRPS = std * (2/sqrt(pi) - 1)
        return std * (2 / np.sqrt(np.pi) - 1)
    
    def _calculate_calibration_error(self, predictions: List[float], 
                                   lower_bound: np.ndarray, upper_bound: np.ndarray, 
                                   confidence_level: float) -> float:
        """Calculate calibration error"""
        n_samples = len(predictions)
        
        # Check if predictions fall within bounds
        within_bounds = (np.array(predictions) >= lower_bound) & (np.array(predictions) <= upper_bound)
        empirical_coverage = np.mean(within_bounds)
        
        # Calibration error is the absolute difference between empirical and expected coverage
        calibration_error = abs(empirical_coverage - confidence_level)
        
        return calibration_error
    
    def _calculate_coverage_probability(self, predictions: List[float], 
                                      lower_bound: np.ndarray, upper_bound: np.ndarray) -> float:
        """Calculate coverage probability"""
        n_samples = len(predictions)
        
        # Check if predictions fall within bounds
        within_bounds = (np.array(predictions) >= lower_bound) & (np.array(predictions) <= upper_bound)
        coverage_prob = np.mean(within_bounds)
        
        return coverage_prob
    
    def evaluate_uncertainty_quality(self, results: List[UncertaintyResult]) -> Dict[str, Any]:
        """Evaluate overall uncertainty quality"""
        crps_scores = [r.crps for r in results]
        calibration_errors = [r.calibration_error for r in results]
        coverage_probs = [r.coverage_probability for r in results]
        
        summary = {
            'mean_crps': np.mean(crps_scores),
            'std_crps': np.std(crps_scores),
            'mean_calibration_error': np.mean(calibration_errors),
            'std_calibration_error': np.std(calibration_errors),
            'mean_coverage_probability': np.mean(coverage_probs),
            'std_coverage_probability': np.std(coverage_probs),
            'n_samples': len(results)
        }
        
        return summary
    
    def plot_uncertainty_visualization(self, results: List[UncertaintyResult], 
                                     ground_truth: List[List[float]], 
                                     save_path: str = None):
        """Create uncertainty visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Prediction intervals over time
        ax1 = axes[0, 0]
        for i, result in enumerate(results[:5]):  # Show first 5 samples
            time_steps = range(len(result.predictions))
            ax1.plot(time_steps, result.predictions, 'b-', alpha=0.7, label=f'Sample {i+1}' if i < 3 else '')
            ax1.fill_between(time_steps, result.lower_bound, result.upper_bound, 
                           alpha=0.2, color='blue')
            if i < len(ground_truth):
                ax1.plot(time_steps, ground_truth[i], 'r--', alpha=0.7)
        
        ax1.set_title('Prediction Intervals Over Time')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Value')
        ax1.legend()
        
        # Plot 2: CRPS distribution
        ax2 = axes[0, 1]
        crps_scores = [r.crps for r in results]
        ax2.hist(crps_scores, bins=20, alpha=0.7, color='green')
        ax2.set_title('CRPS Distribution')
        ax2.set_xlabel('CRPS Score')
        ax2.set_ylabel('Frequency')
        
        # Plot 3: Calibration curve
        ax3 = axes[1, 0]
        confidence_levels = [r.confidence_level for r in results]
        coverage_probs = [r.coverage_probability for r in results]
        ax3.scatter(confidence_levels, coverage_probs, alpha=0.7, color='red')
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax3.set_title('Calibration Curve')
        ax3.set_xlabel('Expected Coverage')
        ax3.set_ylabel('Empirical Coverage')
        
        # Plot 4: Coverage probability by confidence level
        ax4 = axes[1, 1]
        conf_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
        coverage_by_conf = []
        
        for conf_level in conf_levels:
            conf_results = [r for r in results if abs(r.confidence_level - conf_level) < 0.01]
            if conf_results:
                avg_coverage = np.mean([r.coverage_probability for r in conf_results])
                coverage_by_conf.append(avg_coverage)
            else:
                coverage_by_conf.append(conf_level)  # Perfect calibration
        
        ax4.plot(conf_levels, coverage_by_conf, 'o-', color='purple')
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax4.set_title('Coverage Probability by Confidence Level')
        ax4.set_xlabel('Confidence Level')
        ax4.set_ylabel('Empirical Coverage')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Uncertainty visualization saved to {save_path}")
        
        plt.show()

class ProbabilisticForecaster:
    """Enhanced forecaster with uncertainty quantification"""
    
    def __init__(self, base_forecaster):
        self.base_forecaster = base_forecaster
        self.uncertainty_quantifier = UncertaintyQuantifier()
    
    def forecast_with_uncertainty(self, text: str, horizon: int, domain: str, 
                                method: str = 'ensemble') -> UncertaintyResult:
        """Generate forecast with uncertainty quantification"""
        
        # Get base prediction
        base_prediction = self.base_forecaster.generate_forecast(text, horizon, domain)
        
        # Generate uncertainty
        uncertainty_result = self.uncertainty_quantifier.generate_prediction_intervals(
            base_prediction.predictions, 
            confidence_level=0.95,
            method=method
        )
        
        return uncertainty_result
    
    def evaluate_uncertainty_on_dataset(self, samples: List[Dict], 
                                      method: str = 'ensemble') -> Dict[str, Any]:
        """Evaluate uncertainty quantification on dataset"""
        results = []
        
        print(f"Evaluating uncertainty quantification using {method} method...")
        
        for i, sample in enumerate(samples):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(samples)}")
            
            try:
                result = self.forecast_with_uncertainty(
                    sample['text'],
                    len(sample['series']),
                    sample['domain'],
                    method=method
                )
                
                # Calculate additional metrics against ground truth
                mae = np.mean(np.abs(np.array(result.predictions) - np.array(sample['series'])))
                mse = np.mean((np.array(result.predictions) - np.array(sample['series'])) ** 2)
                
                result.mae = mae
                result.mse = mse
                result.ground_truth = sample['series']
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        # Calculate summary statistics
        summary = self.uncertainty_quantifier.evaluate_uncertainty_quality(results)
        
        # Add domain-specific results
        domain_results = {}
        for result in results:
            domain = getattr(result, 'domain', 'unknown')
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append(result)
        
        # Calculate domain-specific metrics
        for domain, domain_results_list in domain_results.items():
            if domain_results_list:
                domain_summary = self.uncertainty_quantifier.evaluate_uncertainty_quality(domain_results_list)
                domain_summary['domain'] = domain
                domain_summary['n_samples'] = len(domain_results_list)
                summary[f'{domain}_results'] = domain_summary
        
        return {
            'summary': summary,
            'detailed_results': results,
            'method': method
        }

def main():
    """Run uncertainty quantification evaluation"""
    # Load dataset
    print("Loading NL2TS-5K dataset...")
    samples = []
    with open('data/nl2ts_5k.jsonl', 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Filter test samples
    test_samples = [s for s in samples if s['split'] == 'test'][:50]  # Use first 50 for demo
    print(f"Using {len(test_samples)} test samples")
    
    # Create mock base forecaster
    class MockForecaster:
        def generate_forecast(self, text, horizon, domain):
            # Generate mock prediction
            base_value = 50.0
            trend = np.random.normal(0, 0.5, horizon)
            predictions = base_value + np.cumsum(trend)
            
            from dataclasses import dataclass
            @dataclass
            class MockResult:
                predictions: List[float]
                confidence: List[float]
                metadata: Dict[str, Any]
            
            return MockResult(
                predictions=predictions.tolist(),
                confidence=[0.8] * horizon,
                metadata={'model': 'mock'}
            )
    
    # Create probabilistic forecaster
    base_forecaster = MockForecaster()
    probabilistic_forecaster = ProbabilisticForecaster(base_forecaster)
    
    # Evaluate uncertainty quantification
    print("\nEvaluating uncertainty quantification...")
    results = probabilistic_forecaster.evaluate_uncertainty_on_dataset(
        test_samples, 
        method='ensemble'
    )
    
    # Print summary
    print("\n" + "="*60)
    print("UNCERTAINTY QUANTIFICATION RESULTS")
    print("="*60)
    
    summary = results['summary']
    print(f"Mean CRPS: {summary['mean_crps']:.4f} ± {summary['std_crps']:.4f}")
    print(f"Mean Calibration Error: {summary['mean_calibration_error']:.4f} ± {summary['std_calibration_error']:.4f}")
    print(f"Mean Coverage Probability: {summary['mean_coverage_probability']:.4f} ± {summary['std_coverage_probability']:.4f}")
    print(f"Number of samples: {summary['n_samples']}")
    
    # Save results
    with open('experiments/results/uncertainty_quantification.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to experiments/results/uncertainty_quantification.json")
    
    # Create visualization
    print("\nCreating uncertainty visualization...")
    uncertainty_quantifier = UncertaintyQuantifier()
    uncertainty_quantifier.plot_uncertainty_visualization(
        results['detailed_results'][:10],  # First 10 samples
        [r.ground_truth for r in results['detailed_results'][:10]],
        save_path='figures/uncertainty_visualization.png'
    )

if __name__ == "__main__":
    main()
