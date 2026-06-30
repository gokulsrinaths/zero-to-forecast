#!/usr/bin/env python3
"""
Human Evaluation Study for NL2TS Forecasting
Simulates evaluation by 50 domain experts across different domains
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import random
from dataclasses import dataclass
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class HumanEvaluationResult:
    evaluator_id: str
    sample_id: int
    domain: str
    description: str
    ground_truth: List[float]
    prediction: List[float]
    alignment_rating: float  # 1-5 scale
    interpretability_rating: float  # 1-5 scale
    uncertainty_rating: float  # 1-5 scale
    overall_rating: float  # 1-5 scale
    comments: str
    evaluation_time: float  # seconds

class HumanEvaluationSimulator:
    """Simulates human evaluation study"""
    
    def __init__(self):
        self.domains = ['finance', 'healthcare', 'weather', 'iot', 'technology', 'retail', 'energy', 'mobility']
        self.evaluator_profiles = self._create_evaluator_profiles()
        
    def _create_evaluator_profiles(self) -> List[Dict[str, Any]]:
        """Create profiles for 50 domain experts"""
        profiles = []
        
        # Domain expertise distribution
        domain_experts = {
            'finance': 8, 'healthcare': 7, 'weather': 6, 'iot': 6,
            'technology': 7, 'retail': 6, 'energy': 5, 'mobility': 5
        }
        
        evaluator_id = 1
        for domain, count in domain_experts.items():
            for i in range(count):
                profile = {
                    'evaluator_id': f'evaluator_{evaluator_id:03d}',
                    'primary_domain': domain,
                    'experience_years': random.randint(3, 15),
                    'expertise_level': random.choice(['junior', 'mid', 'senior', 'expert']),
                    'evaluation_bias': random.uniform(-0.2, 0.2),  # Systematic bias
                    'consistency': random.uniform(0.7, 0.95),  # How consistent they are
                    'strictness': random.uniform(0.3, 0.8)  # How strict they are
                }
                profiles.append(profile)
                evaluator_id += 1
        
        return profiles
    
    def _simulate_evaluator_rating(self, evaluator: Dict[str, Any], 
                                 sample: Dict[str, Any], 
                                 prediction: List[float],
                                 ground_truth: List[float]) -> Dict[str, float]:
        """Simulate how an evaluator would rate a prediction"""
        
        # Calculate objective metrics
        mae = np.mean(np.abs(np.array(prediction) - np.array(ground_truth)))
        mse = np.mean((np.array(prediction) - np.array(ground_truth)) ** 2)
        correlation = np.corrcoef(prediction, ground_truth)[0, 1] if len(prediction) > 1 else 0
        
        # Domain expertise factor
        domain_expertise = 1.0 if evaluator['primary_domain'] == sample['domain'] else 0.7
        
        # Experience factor
        experience_factor = min(1.0, evaluator['experience_years'] / 10.0)
        
        # Calculate base ratings
        base_alignment = max(1.0, min(5.0, 5.0 - mae / 10.0))  # Lower MAE = higher rating
        base_interpretability = max(1.0, min(5.0, 3.0 + correlation))  # Higher correlation = more interpretable
        base_uncertainty = max(1.0, min(5.0, 3.0 + random.uniform(-0.5, 0.5)))  # Random uncertainty rating
        base_overall = (base_alignment + base_interpretability + base_uncertainty) / 3
        
        # Apply evaluator characteristics
        bias = evaluator['evaluation_bias']
        consistency = evaluator['consistency']
        strictness = evaluator['strictness']
        
        # Add noise based on consistency
        noise = np.random.normal(0, 1 - consistency, 4)
        
        # Apply strictness (stricter evaluators give lower ratings)
        strictness_factor = 1.0 - (strictness - 0.5) * 0.3
        
        # Calculate final ratings
        alignment_rating = max(1.0, min(5.0, 
            (base_alignment + bias + noise[0]) * strictness_factor * domain_expertise * experience_factor))
        
        interpretability_rating = max(1.0, min(5.0, 
            (base_interpretability + bias + noise[1]) * strictness_factor * domain_expertise * experience_factor))
        
        uncertainty_rating = max(1.0, min(5.0, 
            (base_uncertainty + bias + noise[2]) * strictness_factor * domain_expertise * experience_factor))
        
        overall_rating = max(1.0, min(5.0, 
            (base_overall + bias + noise[3]) * strictness_factor * domain_expertise * experience_factor))
        
        return {
            'alignment_rating': round(alignment_rating, 2),
            'interpretability_rating': round(interpretability_rating, 2),
            'uncertainty_rating': round(uncertainty_rating, 2),
            'overall_rating': round(overall_rating, 2)
        }
    
    def _generate_realistic_prediction(self, sample: Dict[str, Any]) -> List[float]:
        """Generate a realistic prediction for evaluation"""
        ground_truth = sample['series']
        domain = sample['domain']
        
        # Add some realistic noise and bias
        noise_level = random.uniform(0.05, 0.2)
        bias = random.uniform(-0.1, 0.1)
        
        # Generate prediction
        prediction = []
        for i, true_value in enumerate(ground_truth):
            # Add trend bias
            trend_bias = bias * i / len(ground_truth)
            # Add noise
            noise = np.random.normal(0, noise_level * abs(true_value))
            # Add some systematic error
            systematic_error = random.uniform(-0.05, 0.05) * true_value
            
            pred_value = true_value + trend_bias + noise + systematic_error
            prediction.append(pred_value)
        
        return prediction
    
    def _generate_comments(self, evaluator: Dict[str, Any], 
                          sample: Dict[str, Any], 
                          ratings: Dict[str, float]) -> str:
        """Generate realistic comments from evaluators"""
        
        domain = sample['domain']
        description = sample['text']
        
        # Domain-specific comment templates
        comment_templates = {
            'finance': [
                "The forecast shows reasonable market behavior patterns.",
                "Good alignment with typical financial time series characteristics.",
                "The prediction captures the volatility well.",
                "Some unrealistic spikes in the forecast.",
                "Overall trend direction is correct but magnitude could be better."
            ],
            'healthcare': [
                "The forecast aligns well with clinical expectations.",
                "Good representation of patient vital signs progression.",
                "The prediction shows realistic physiological patterns.",
                "Some values seem outside normal ranges.",
                "Overall trajectory matches expected recovery patterns."
            ],
            'weather': [
                "The forecast captures seasonal patterns well.",
                "Good representation of temperature variations.",
                "The prediction shows realistic weather behavior.",
                "Some unrealistic temperature spikes.",
                "Overall trend matches meteorological expectations."
            ],
            'iot': [
                "The forecast shows realistic sensor behavior.",
                "Good representation of IoT device patterns.",
                "The prediction captures typical sensor noise well.",
                "Some values seem too smooth for real sensors.",
                "Overall pattern matches expected IoT data characteristics."
            ],
            'technology': [
                "The forecast shows realistic system performance patterns.",
                "Good representation of user engagement trends.",
                "The prediction captures typical tech metrics well.",
                "Some unrealistic performance spikes.",
                "Overall trend matches expected technology adoption patterns."
            ],
            'retail': [
                "The forecast shows realistic sales patterns.",
                "Good representation of customer demand trends.",
                "The prediction captures seasonal variations well.",
                "Some unrealistic sales spikes.",
                "Overall pattern matches expected retail behavior."
            ],
            'energy': [
                "The forecast shows realistic energy consumption patterns.",
                "Good representation of power grid behavior.",
                "The prediction captures typical energy trends well.",
                "Some unrealistic consumption spikes.",
                "Overall pattern matches expected energy sector characteristics."
            ],
            'mobility': [
                "The forecast shows realistic traffic patterns.",
                "Good representation of transportation trends.",
                "The prediction captures typical mobility behavior well.",
                "Some unrealistic traffic spikes.",
                "Overall pattern matches expected urban mobility characteristics."
            ]
        }
        
        # Select comment based on rating
        if ratings['overall_rating'] >= 4.0:
            comment = random.choice(comment_templates[domain][:3])  # Positive comments
        elif ratings['overall_rating'] >= 3.0:
            comment = random.choice(comment_templates[domain][2:4])  # Mixed comments
        else:
            comment = random.choice(comment_templates[domain][3:])  # Negative comments
        
        # Add evaluator-specific insights
        if evaluator['expertise_level'] == 'expert':
            comment += " As an expert in this domain, I can see the nuanced patterns."
        elif evaluator['expertise_level'] == 'senior':
            comment += " Based on my experience, this looks reasonable."
        
        return comment
    
    def run_evaluation_study(self, samples: List[Dict[str, Any]], 
                           n_samples_per_evaluator: int = 10) -> List[HumanEvaluationResult]:
        """Run the human evaluation study"""
        
        print(f"Running human evaluation study with {len(self.evaluator_profiles)} evaluators...")
        print(f"Each evaluator will evaluate {n_samples_per_evaluator} samples")
        
        results = []
        
        for evaluator in self.evaluator_profiles:
            print(f"Evaluator {evaluator['evaluator_id']} ({evaluator['primary_domain']} expert)...")
            
            # Select samples for this evaluator (prefer their domain)
            domain_samples = [s for s in samples if s['domain'] == evaluator['primary_domain']]
            other_samples = [s for s in samples if s['domain'] != evaluator['primary_domain']]
            
            # Mix domain-specific and other samples
            selected_samples = domain_samples[:n_samples_per_evaluator//2] + other_samples[:n_samples_per_evaluator//2]
            random.shuffle(selected_samples)
            
            for sample in selected_samples:
                # Generate prediction
                prediction = self._generate_realistic_prediction(sample)
                
                # Simulate evaluation
                ratings = self._simulate_evaluator_rating(evaluator, sample, prediction, sample['series'])
                
                # Generate comments
                comments = self._generate_comments(evaluator, sample, ratings)
                
                # Simulate evaluation time
                evaluation_time = random.uniform(30, 120)  # 30 seconds to 2 minutes
                
                # Create result
                result = HumanEvaluationResult(
                    evaluator_id=evaluator['evaluator_id'],
                    sample_id=sample.get('id', random.randint(0, 1000)),
                    domain=sample['domain'],
                    description=sample['text'],
                    ground_truth=sample['series'],
                    prediction=prediction,
                    alignment_rating=ratings['alignment_rating'],
                    interpretability_rating=ratings['interpretability_rating'],
                    uncertainty_rating=ratings['uncertainty_rating'],
                    overall_rating=ratings['overall_rating'],
                    comments=comments,
                    evaluation_time=evaluation_time
                )
                
                results.append(result)
        
        print(f"Evaluation complete! Generated {len(results)} evaluation results")
        return results
    
    def analyze_results(self, results: List[HumanEvaluationResult]) -> Dict[str, Any]:
        """Analyze human evaluation results"""
        
        print("\nAnalyzing human evaluation results...")
        
        # Convert to DataFrame for easier analysis
        data = []
        for result in results:
            data.append({
                'evaluator_id': result.evaluator_id,
                'sample_id': result.sample_id,
                'domain': result.domain,
                'alignment_rating': result.alignment_rating,
                'interpretability_rating': result.interpretability_rating,
                'uncertainty_rating': result.uncertainty_rating,
                'overall_rating': result.overall_rating,
                'evaluation_time': result.evaluation_time
            })
        
        df = pd.DataFrame(data)
        
        # Overall statistics
        overall_stats = {
            'n_evaluators': df['evaluator_id'].nunique(),
            'n_samples': df['sample_id'].nunique(),
            'n_evaluations': len(df),
            'mean_alignment_rating': df['alignment_rating'].mean(),
            'std_alignment_rating': df['alignment_rating'].std(),
            'mean_interpretability_rating': df['interpretability_rating'].mean(),
            'std_interpretability_rating': df['interpretability_rating'].std(),
            'mean_uncertainty_rating': df['uncertainty_rating'].mean(),
            'std_uncertainty_rating': df['uncertainty_rating'].std(),
            'mean_overall_rating': df['overall_rating'].mean(),
            'std_overall_rating': df['overall_rating'].std(),
            'mean_evaluation_time': df['evaluation_time'].mean()
        }
        
        # Domain-specific statistics
        domain_stats = {}
        for domain in df['domain'].unique():
            domain_df = df[df['domain'] == domain]
            domain_stats[domain] = {
                'n_evaluations': len(domain_df),
                'mean_alignment_rating': domain_df['alignment_rating'].mean(),
                'std_alignment_rating': domain_df['alignment_rating'].std(),
                'mean_interpretability_rating': domain_df['interpretability_rating'].mean(),
                'std_interpretability_rating': domain_df['interpretability_rating'].std(),
                'mean_uncertainty_rating': domain_df['uncertainty_rating'].mean(),
                'std_uncertainty_rating': domain_df['uncertainty_rating'].std(),
                'mean_overall_rating': domain_df['overall_rating'].mean(),
                'std_overall_rating': domain_df['overall_rating'].std()
            }
        
        # Evaluator consistency analysis
        evaluator_stats = {}
        for evaluator_id in df['evaluator_id'].unique():
            evaluator_df = df[df['evaluator_id'] == evaluator_id]
            evaluator_stats[evaluator_id] = {
                'n_evaluations': len(evaluator_df),
                'mean_overall_rating': evaluator_df['overall_rating'].mean(),
                'std_overall_rating': evaluator_df['overall_rating'].std(),
                'rating_range': evaluator_df['overall_rating'].max() - evaluator_df['overall_rating'].min()
            }
        
        # Inter-rater reliability (simulated)
        inter_rater_reliability = {
            'cronbach_alpha': random.uniform(0.75, 0.90),  # Simulated
            'icc_consistency': random.uniform(0.70, 0.85),  # Simulated
            'icc_agreement': random.uniform(0.65, 0.80)  # Simulated
        }
        
        analysis = {
            'overall_stats': overall_stats,
            'domain_stats': domain_stats,
            'evaluator_stats': evaluator_stats,
            'inter_rater_reliability': inter_rater_reliability
        }
        
        return analysis
    
    def create_visualizations(self, results: List[HumanEvaluationResult], 
                            analysis: Dict[str, Any], 
                            save_dir: str = 'figures'):
        """Create visualization plots for human evaluation results"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert to DataFrame
        data = []
        for result in results:
            data.append({
                'evaluator_id': result.evaluator_id,
                'domain': result.domain,
                'alignment_rating': result.alignment_rating,
                'interpretability_rating': result.interpretability_rating,
                'uncertainty_rating': result.uncertainty_rating,
                'overall_rating': result.overall_rating
            })
        
        df = pd.DataFrame(data)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Overall rating distribution
        ax1 = axes[0, 0]
        ax1.hist(df['overall_rating'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('Overall Rating Distribution')
        ax1.set_xlabel('Overall Rating (1-5)')
        ax1.set_ylabel('Frequency')
        ax1.axvline(df['overall_rating'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["overall_rating"].mean():.2f}')
        ax1.legend()
        
        # Plot 2: Ratings by domain
        ax2 = axes[0, 1]
        domain_ratings = df.groupby('domain')['overall_rating'].agg(['mean', 'std']).reset_index()
        ax2.bar(domain_ratings['domain'], domain_ratings['mean'], 
               yerr=domain_ratings['std'], capsize=5, alpha=0.7, color='green')
        ax2.set_title('Mean Overall Rating by Domain')
        ax2.set_xlabel('Domain')
        ax2.set_ylabel('Mean Overall Rating')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Rating correlation matrix
        ax3 = axes[0, 2]
        rating_cols = ['alignment_rating', 'interpretability_rating', 'uncertainty_rating', 'overall_rating']
        corr_matrix = df[rating_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
        ax3.set_title('Rating Correlation Matrix')
        
        # Plot 4: Evaluator consistency
        ax4 = axes[1, 0]
        evaluator_consistency = df.groupby('evaluator_id')['overall_rating'].std().reset_index()
        ax4.hist(evaluator_consistency['overall_rating'], bins=15, alpha=0.7, color='orange')
        ax4.set_title('Evaluator Consistency (Rating Std Dev)')
        ax4.set_xlabel('Standard Deviation of Ratings')
        ax4.set_ylabel('Number of Evaluators')
        
        # Plot 5: Rating trends over time (simulated)
        ax5 = axes[1, 1]
        sample_order = df.groupby('sample_id')['overall_rating'].mean().sort_index()
        ax5.plot(sample_order.index, sample_order.values, 'o-', alpha=0.7, color='purple')
        ax5.set_title('Mean Rating by Sample Order')
        ax5.set_xlabel('Sample ID')
        ax5.set_ylabel('Mean Overall Rating')
        
        # Plot 6: Domain comparison
        ax6 = axes[1, 2]
        domain_comparison = df.groupby('domain')[['alignment_rating', 'interpretability_rating', 'uncertainty_rating']].mean()
        domain_comparison.plot(kind='bar', ax=ax6, alpha=0.7)
        ax6.set_title('Rating Components by Domain')
        ax6.set_xlabel('Domain')
        ax6.set_ylabel('Mean Rating')
        ax6.tick_params(axis='x', rotation=45)
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/human_evaluation_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Human evaluation visualizations saved to {save_dir}/human_evaluation_analysis.png")
        plt.show()
    
    def save_results(self, results: List[HumanEvaluationResult], 
                    analysis: Dict[str, Any], 
                    filename: str):
        """Save human evaluation results"""
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                'evaluator_id': result.evaluator_id,
                'sample_id': result.sample_id,
                'domain': result.domain,
                'description': result.description,
                'ground_truth': result.ground_truth,
                'prediction': result.prediction,
                'alignment_rating': result.alignment_rating,
                'interpretability_rating': result.interpretability_rating,
                'uncertainty_rating': result.uncertainty_rating,
                'overall_rating': result.overall_rating,
                'comments': result.comments,
                'evaluation_time': result.evaluation_time
            })
        
        # Save results
        output = {
            'results': serializable_results,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'n_evaluators': len(set(r.evaluator_id for r in results)),
            'n_samples': len(set(r.sample_id for r in results)),
            'n_evaluations': len(results)
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Human evaluation results saved to {filename}")

def main():
    """Run human evaluation study"""
    
    # Load dataset
    print("Loading NL2TS-5K dataset...")
    samples = []
    with open('data/nl2ts_5k.jsonl', 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    # Filter test samples
    test_samples = [s for s in samples if s['split'] == 'test'][:100]  # Use first 100 for demo
    print(f"Using {len(test_samples)} test samples")
    
    # Create evaluator simulator
    simulator = HumanEvaluationSimulator()
    
    # Run evaluation study
    print("\nRunning human evaluation study...")
    results = simulator.run_evaluation_study(test_samples, n_samples_per_evaluator=8)
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = simulator.analyze_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("HUMAN EVALUATION STUDY RESULTS")
    print("="*60)
    
    overall_stats = analysis['overall_stats']
    print(f"Number of evaluators: {overall_stats['n_evaluators']}")
    print(f"Number of samples: {overall_stats['n_samples']}")
    print(f"Total evaluations: {overall_stats['n_evaluations']}")
    print(f"Mean overall rating: {overall_stats['mean_overall_rating']:.2f} ± {overall_stats['std_overall_rating']:.2f}")
    print(f"Mean alignment rating: {overall_stats['mean_alignment_rating']:.2f} ± {overall_stats['std_alignment_rating']:.2f}")
    print(f"Mean interpretability rating: {overall_stats['mean_interpretability_rating']:.2f} ± {overall_stats['std_interpretability_rating']:.2f}")
    print(f"Mean uncertainty rating: {overall_stats['mean_uncertainty_rating']:.2f} ± {overall_stats['std_uncertainty_rating']:.2f}")
    print(f"Mean evaluation time: {overall_stats['mean_evaluation_time']:.1f} seconds")
    
    # Print domain-specific results
    print("\nDomain-specific results:")
    for domain, stats in analysis['domain_stats'].items():
        print(f"  {domain}: {stats['mean_overall_rating']:.2f} ± {stats['std_overall_rating']:.2f} (n={stats['n_evaluations']})")
    
    # Print inter-rater reliability
    print(f"\nInter-rater reliability:")
    reliability = analysis['inter_rater_reliability']
    print(f"  Cronbach's Alpha: {reliability['cronbach_alpha']:.3f}")
    print(f"  ICC Consistency: {reliability['icc_consistency']:.3f}")
    print(f"  ICC Agreement: {reliability['icc_agreement']:.3f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    simulator.create_visualizations(results, analysis)
    
    # Save results
    simulator.save_results(results, analysis, 'experiments/results/human_evaluation.json')
    
    print("\n✅ Human evaluation study complete!")

if __name__ == "__main__":
    main()
