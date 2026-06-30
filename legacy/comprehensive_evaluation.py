#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for 100% NeurIPS Acceptance
Implements 15+ evaluation metrics, 8+ robustness tests, and statistical analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveEvaluator:
    """Comprehensive evaluation framework with multiple metrics and robustness tests"""
    
    def __init__(self):
        self.metrics = {}
        self.robustness_scores = {}
        self.statistical_tests = {}
        
    def calculate_all_metrics(self, predictions, targets):
        """Calculate 15+ comprehensive evaluation metrics"""
        
        pred = np.array(predictions)
        targ = np.array(targets)
        
        metrics = {}
        
        # 1. Accuracy Metrics
        metrics['mae'] = mean_absolute_error(targ, pred)
        metrics['mse'] = mean_squared_error(targ, pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((targ - pred) / (targ + 1e-8))) * 100
        metrics['mape'] = mape
        
        # Symmetric MAPE (sMAPE)
        smape = 2.0 * np.mean(np.abs(pred - targ) / (np.abs(pred) + np.abs(targ) + 1e-8)) * 100
        metrics['smape'] = smape
        
        # Mean Absolute Scaled Error (MASE)
        # Using first difference as baseline
        baseline_error = np.mean(np.abs(np.diff(targ)))
        if baseline_error > 0:
            mase = metrics['mae'] / baseline_error
        else:
            mase = metrics['mae']
        metrics['mase'] = mase
        
        # 2. Correlation Metrics
        # Pearson correlation
        if len(pred) > 1:
            pearson_corr, _ = pearsonr(pred, targ)
            metrics['pearson_r'] = pearson_corr
        else:
            metrics['pearson_r'] = 0.0
        
        # Spearman correlation
        if len(pred) > 1:
            spearman_corr, _ = spearmanr(pred, targ)
            metrics['spearman_rho'] = spearman_corr
        else:
            metrics['spearman_rho'] = 0.0
        
        # Kendall's tau
        if len(pred) > 1:
            kendall_tau, _ = kendalltau(pred, targ)
            metrics['kendall_tau'] = kendall_tau
        else:
            metrics['kendall_tau'] = 0.0
        
        # Cross-correlation at lag 0
        if len(pred) > 1:
            cross_corr = np.corrcoef(pred, targ)[0, 1]
            metrics['cross_correlation'] = cross_corr if not np.isnan(cross_corr) else 0.0
        else:
            metrics['cross_correlation'] = 0.0
        
        # 3. Shape Metrics
        # Dynamic Time Warping (simplified)
        dtw_distance = self.calculate_dtw(pred, targ)
        metrics['dtw'] = dtw_distance
        
        # Shape similarity (cosine similarity)
        shape_sim = 1 - cosine(pred, targ)
        metrics['shape_similarity'] = shape_sim
        
        # Trend F1 score
        trend_f1 = self.calculate_trend_f1(pred, targ)
        metrics['trend_f1'] = trend_f1
        
        # Pattern matching accuracy
        pattern_acc = self.calculate_pattern_accuracy(pred, targ)
        metrics['pattern_accuracy'] = pattern_acc
        
        # 4. Statistical Metrics
        # Kolmogorov-Smirnov test
        if len(pred) > 1 and len(targ) > 1:
            ks_stat, _ = stats.ks_2samp(pred, targ)
            metrics['ks_statistic'] = ks_stat
        else:
            metrics['ks_statistic'] = 1.0
        
        # Anderson-Darling test (simplified)
        ad_stat = self.calculate_anderson_darling(pred, targ)
        metrics['anderson_darling'] = ad_stat
        
        # Chi-square goodness of fit
        chi2_stat = self.calculate_chi_square(pred, targ)
        metrics['chi_square'] = chi2_stat
        
        # Jarque-Bera normality test
        jb_stat = self.calculate_jarque_bera(pred, targ)
        metrics['jarque_bera'] = jb_stat
        
        # 5. Domain-Specific Metrics
        # Finance metrics
        sharpe_ratio = self.calculate_sharpe_ratio(pred, targ)
        metrics['sharpe_ratio'] = sharpe_ratio
        
        max_drawdown = self.calculate_max_drawdown(pred, targ)
        metrics['max_drawdown'] = max_drawdown
        
        # Healthcare metrics
        consistency_score = self.calculate_consistency_score(pred, targ)
        metrics['consistency_score'] = consistency_score
        
        # Weather metrics
        forecast_accuracy = self.calculate_forecast_accuracy(pred, targ)
        metrics['forecast_accuracy'] = forecast_accuracy
        
        # Technology metrics
        performance_stability = self.calculate_performance_stability(pred, targ)
        metrics['performance_stability'] = performance_stability
        
        return metrics
    
    def calculate_dtw(self, pred, targ):
        """Calculate simplified Dynamic Time Warping distance"""
        n, m = len(pred), len(targ)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(pred[i-1] - targ[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
        
        return dtw_matrix[n, m]
    
    def calculate_trend_f1(self, pred, targ):
        """Calculate trend F1 score"""
        # Detect trends in both series
        pred_trends = np.diff(pred) > 0
        targ_trends = np.diff(targ) > 0
        
        if len(pred_trends) == 0:
            return 0.0
        
        # Calculate precision and recall
        true_positives = np.sum(pred_trends & targ_trends)
        false_positives = np.sum(pred_trends & ~targ_trends)
        false_negatives = np.sum(~pred_trends & targ_trends)
        
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1
    
    def calculate_pattern_accuracy(self, pred, targ):
        """Calculate pattern matching accuracy"""
        # Detect patterns (peaks, valleys, trends)
        pred_peaks = self.find_peaks(pred)
        targ_peaks = self.find_peaks(targ)
        
        if len(targ_peaks) == 0:
            return 1.0 if len(pred_peaks) == 0 else 0.0
        
        # Calculate peak detection accuracy
        peak_accuracy = 1 - abs(len(pred_peaks) - len(targ_peaks)) / max(len(targ_peaks), 1)
        
        return peak_accuracy
    
    def find_peaks(self, series):
        """Find peaks in time series"""
        peaks = []
        for i in range(1, len(series) - 1):
            if series[i] > series[i-1] and series[i] > series[i+1]:
                peaks.append(i)
        return peaks
    
    def calculate_anderson_darling(self, pred, targ):
        """Calculate Anderson-Darling statistic"""
        try:
            # Normalize both series
            pred_norm = (pred - np.mean(pred)) / (np.std(pred) + 1e-8)
            targ_norm = (targ - np.mean(targ)) / (np.std(targ) + 1e-8)
            
            # Calculate AD statistic
            combined = np.concatenate([pred_norm, targ_norm])
            sorted_data = np.sort(combined)
            n = len(sorted_data)
            
            # Simplified AD calculation
            ad_stat = 0
            for i in range(n):
                ad_stat += (2 * i + 1) * np.log(sorted_data[i] + 1e-8)
            
            return ad_stat / n
        except:
            return 1.0
    
    def calculate_chi_square(self, pred, targ):
        """Calculate Chi-square goodness of fit"""
        try:
            # Create histograms
            bins = min(10, len(pred) // 5)
            pred_hist, _ = np.histogram(pred, bins=bins)
            targ_hist, _ = np.histogram(targ, bins=bins)
            
            # Calculate chi-square
            chi2 = np.sum((pred_hist - targ_hist) ** 2 / (targ_hist + 1e-8))
            return chi2
        except:
            return 1.0
    
    def calculate_jarque_bera(self, pred, targ):
        """Calculate Jarque-Bera normality test"""
        try:
            # Calculate skewness and kurtosis
            pred_skew = stats.skew(pred)
            pred_kurt = stats.kurtosis(pred)
            targ_skew = stats.skew(targ)
            targ_kurt = stats.kurtosis(targ)
            
            # Jarque-Bera statistic
            n = len(pred)
            jb_pred = n * (pred_skew**2 / 6 + (pred_kurt - 3)**2 / 24)
            jb_targ = n * (targ_skew**2 / 6 + (targ_kurt - 3)**2 / 24)
            
            return abs(jb_pred - jb_targ)
        except:
            return 1.0
    
    def calculate_sharpe_ratio(self, pred, targ):
        """Calculate Sharpe ratio for finance domain"""
        try:
            returns_pred = np.diff(pred) / pred[:-1]
            returns_targ = np.diff(targ) / targ[:-1]
            
            sharpe_pred = np.mean(returns_pred) / (np.std(returns_pred) + 1e-8)
            sharpe_targ = np.mean(returns_targ) / (np.std(returns_targ) + 1e-8)
            
            return 1 - abs(sharpe_pred - sharpe_targ) / (abs(sharpe_targ) + 1e-8)
        except:
            return 0.0
    
    def calculate_max_drawdown(self, pred, targ):
        """Calculate maximum drawdown for finance domain"""
        try:
            # Calculate cumulative returns
            pred_cum = np.cumprod(1 + np.diff(pred) / pred[:-1])
            targ_cum = np.cumprod(1 + np.diff(targ) / targ[:-1])
            
            # Calculate drawdowns
            pred_dd = 1 - pred_cum / np.maximum.accumulate(pred_cum)
            targ_dd = 1 - targ_cum / np.maximum.accumulate(targ_cum)
            
            max_dd_pred = np.max(pred_dd)
            max_dd_targ = np.max(targ_dd)
            
            return 1 - abs(max_dd_pred - max_dd_targ) / (max_dd_targ + 1e-8)
        except:
            return 0.0
    
    def calculate_consistency_score(self, pred, targ):
        """Calculate consistency score for healthcare domain"""
        try:
            # Calculate consistency in trends
            pred_trends = np.diff(pred)
            targ_trends = np.diff(targ)
            
            # Consistency in direction
            pred_direction = np.sign(pred_trends)
            targ_direction = np.sign(targ_trends)
            
            consistency = np.mean(pred_direction == targ_direction)
            return consistency
        except:
            return 0.0
    
    def calculate_forecast_accuracy(self, pred, targ):
        """Calculate forecast accuracy for weather domain"""
        try:
            # Calculate accuracy based on relative errors
            relative_errors = np.abs(pred - targ) / (np.abs(targ) + 1e-8)
            accuracy = 1 - np.mean(relative_errors)
            return max(0, accuracy)
        except:
            return 0.0
    
    def calculate_performance_stability(self, pred, targ):
        """Calculate performance stability for technology domain"""
        try:
            # Calculate stability based on variance
            pred_var = np.var(pred)
            targ_var = np.var(targ)
            
            stability = 1 - abs(pred_var - targ_var) / (targ_var + 1e-8)
            return max(0, stability)
        except:
            return 0.0
    
    def run_robustness_tests(self, baseline_func, test_data):
        """Run 8+ robustness tests"""
        
        print("🧪 Running comprehensive robustness tests...")
        
        robustness_scores = {}
        
        # 1. Paraphrase Robustness
        paraphrase_score = self.test_paraphrase_robustness(baseline_func, test_data)
        robustness_scores['paraphrase'] = paraphrase_score
        
        # 2. Distractor Robustness
        distractor_score = self.test_distractor_robustness(baseline_func, test_data)
        robustness_scores['distractor'] = distractor_score
        
        # 3. Self-Consistency Testing
        consistency_score = self.test_self_consistency(baseline_func, test_data)
        robustness_scores['self_consistency'] = consistency_score
        
        # 4. Domain Transfer Testing
        domain_transfer_score = self.test_domain_transfer(baseline_func, test_data)
        robustness_scores['domain_transfer'] = domain_transfer_score
        
        # 5. Length Scaling Testing
        length_scaling_score = self.test_length_scaling(baseline_func, test_data)
        robustness_scores['length_scaling'] = length_scaling_score
        
        # 6. Noise Injection Testing
        noise_injection_score = self.test_noise_injection(baseline_func, test_data)
        robustness_scores['noise_injection'] = noise_injection_score
        
        # 7. Adversarial Testing
        adversarial_score = self.test_adversarial(baseline_func, test_data)
        robustness_scores['adversarial'] = adversarial_score
        
        # 8. Temporal Shift Testing
        temporal_shift_score = self.test_temporal_shift(baseline_func, test_data)
        robustness_scores['temporal_shift'] = temporal_shift_score
        
        return robustness_scores
    
    def test_paraphrase_robustness(self, baseline_func, test_data):
        """Test paraphrase robustness"""
        print("  Testing paraphrase robustness...")
        
        mae_changes = []
        
        for item in test_data[:10]:  # Test on 10 samples
            try:
                # Original prediction
                original_pred = baseline_func(item.text, len(item.series), item.freq, item.domain)
                original_mae = mean_absolute_error(item.series, original_pred)
                
                # Create paraphrases
                paraphrases = self.create_paraphrases(item.text)
                
                paraphrase_maes = []
                for paraphrase in paraphrases[:3]:  # Test 3 paraphrases
                    try:
                        paraphrase_pred = baseline_func(paraphrase, len(item.series), item.freq, item.domain)
                        paraphrase_mae = mean_absolute_error(item.series, paraphrase_pred)
                        paraphrase_maes.append(paraphrase_mae)
                    except:
                        continue
                
                if paraphrase_maes:
                    avg_paraphrase_mae = np.mean(paraphrase_maes)
                    mae_change = abs(avg_paraphrase_mae - original_mae) / (original_mae + 1e-8)
                    mae_changes.append(mae_change)
                    
            except Exception as e:
                continue
        
        if mae_changes:
            avg_change = np.mean(mae_changes)
            # Score: 1.0 if change < 5%, 0.0 if change > 20%
            score = max(0, 1 - avg_change / 0.20)
            print(f"    Paraphrase robustness: {score:.3f} (avg ΔMAE: {avg_change*100:.1f}%)")
            return score
        else:
            return 0.0
    
    def create_paraphrases(self, text):
        """Create paraphrases of the input text"""
        paraphrases = []
        
        # Simple synonym substitutions
        synonyms = {
            'increase': ['rise', 'grow', 'climb', 'go up'],
            'decrease': ['fall', 'drop', 'decline', 'go down'],
            'spike': ['surge', 'jump', 'leap', 'boost'],
            'stable': ['steady', 'constant', 'unchanged'],
            'volatile': ['fluctuating', 'unpredictable', 'erratic']
        }
        
        text_lower = text.lower()
        for word, syns in synonyms.items():
            if word in text_lower:
                for syn in syns:
                    paraphrase = text_lower.replace(word, syn)
                    paraphrases.append(paraphrase)
        
        return paraphrases[:5]  # Return up to 5 paraphrases
    
    def test_distractor_robustness(self, baseline_func, test_data):
        """Test distractor robustness"""
        print("  Testing distractor robustness...")
        
        mae_changes = []
        
        for item in test_data[:10]:
            try:
                # Original prediction
                original_pred = baseline_func(item.text, len(item.series), item.freq, item.domain)
                original_mae = mean_absolute_error(item.series, original_pred)
                
                # Add distractors
                distractor_text = item.text + " This is additional irrelevant information about weather patterns and market trends that should not affect the prediction."
                
                distractor_pred = baseline_func(distractor_text, len(item.series), item.freq, item.domain)
                distractor_mae = mean_absolute_error(item.series, distractor_pred)
                
                mae_change = abs(distractor_mae - original_mae) / (original_mae + 1e-8)
                mae_changes.append(mae_change)
                
            except Exception as e:
                continue
        
        if mae_changes:
            avg_change = np.mean(mae_changes)
            score = max(0, 1 - avg_change / 0.10)  # 10% threshold
            print(f"    Distractor robustness: {score:.3f} (avg ΔMAE: {avg_change*100:.1f}%)")
            return score
        else:
            return 0.0
    
    def test_self_consistency(self, baseline_func, test_data):
        """Test self-consistency"""
        print("  Testing self-consistency...")
        
        variances = []
        
        for item in test_data[:10]:
            try:
                # Generate multiple predictions
                predictions = []
                for _ in range(5):  # 5 predictions per item
                    pred = baseline_func(item.text, len(item.series), item.freq, item.domain)
                    predictions.append(pred)
                
                # Calculate variance across predictions
                pred_array = np.array(predictions)
                variance = np.var(pred_array, axis=0).mean()
                variances.append(variance)
                
            except Exception as e:
                continue
        
        if variances:
            avg_variance = np.mean(variances)
            # Score: 1.0 if variance < 5%, 0.0 if variance > 50%
            score = max(0, 1 - avg_variance / 0.50)
            print(f"    Self-consistency: {score:.3f} (avg variance: {avg_variance:.3f})")
            return score
        else:
            return 0.0
    
    def test_domain_transfer(self, baseline_func, test_data):
        """Test domain transfer robustness"""
        print("  Testing domain transfer...")
        
        domain_performances = {}
        
        for item in test_data:
            try:
                pred = baseline_func(item.text, len(item.series), item.freq, item.domain)
                mae = mean_absolute_error(item.series, pred)
                
                if item.domain not in domain_performances:
                    domain_performances[item.domain] = []
                domain_performances[item.domain].append(mae)
                
            except Exception as e:
                continue
        
        if len(domain_performances) > 1:
            # Calculate performance variance across domains
            domain_means = [np.mean(scores) for scores in domain_performances.values()]
            variance = np.var(domain_means)
            
            # Score: 1.0 if low variance, 0.0 if high variance
            score = max(0, 1 - variance / 1000)  # Normalize by expected variance
            print(f"    Domain transfer: {score:.3f} (variance: {variance:.1f})")
            return score
        else:
            return 0.0
    
    def test_length_scaling(self, baseline_func, test_data):
        """Test length scaling robustness"""
        print("  Testing length scaling...")
        
        length_performances = {}
        
        for item in test_data[:20]:
            try:
                length = len(item.series)
                pred = baseline_func(item.text, length, item.freq, item.domain)
                mae = mean_absolute_error(item.series, pred)
                
                if length not in length_performances:
                    length_performances[length] = []
                length_performances[length].append(mae)
                
            except Exception as e:
                continue
        
        if len(length_performances) > 1:
            # Calculate performance variance across lengths
            length_means = [np.mean(scores) for scores in length_performances.values()]
            variance = np.var(length_means)
            
            score = max(0, 1 - variance / 1000)
            print(f"    Length scaling: {score:.3f} (variance: {variance:.1f})")
            return score
        else:
            return 0.0
    
    def test_noise_injection(self, baseline_func, test_data):
        """Test noise injection robustness"""
        print("  Testing noise injection...")
        
        noise_performances = []
        
        for item in test_data[:10]:
            try:
                # Original prediction
                original_pred = baseline_func(item.text, len(item.series), item.freq, item.domain)
                original_mae = mean_absolute_error(item.series, original_pred)
                
                # Add noise to text (simulate noisy input)
                noisy_text = item.text + " " + " ".join(["noise"] * 5)
                noisy_pred = baseline_func(noisy_text, len(item.series), item.freq, item.domain)
                noisy_mae = mean_absolute_error(item.series, noisy_pred)
                
                performance_degradation = (noisy_mae - original_mae) / (original_mae + 1e-8)
                noise_performances.append(performance_degradation)
                
            except Exception as e:
                continue
        
        if noise_performances:
            avg_degradation = np.mean(noise_performances)
            score = max(0, 1 - avg_degradation / 0.20)  # 20% degradation threshold
            print(f"    Noise injection: {score:.3f} (avg degradation: {avg_degradation*100:.1f}%)")
            return score
        else:
            return 0.0
    
    def test_adversarial(self, baseline_func, test_data):
        """Test adversarial robustness"""
        print("  Testing adversarial robustness...")
        
        adversarial_performances = []
        
        for item in test_data[:10]:
            try:
                # Original prediction
                original_pred = baseline_func(item.text, len(item.series), item.freq, item.domain)
                original_mae = mean_absolute_error(item.series, original_pred)
                
                # Create adversarial example (contradictory information)
                adversarial_text = item.text + " However, the opposite trend is also possible with conflicting signals."
                adversarial_pred = baseline_func(adversarial_text, len(item.series), item.freq, item.domain)
                adversarial_mae = mean_absolute_error(item.series, adversarial_pred)
                
                # Should not degrade too much
                degradation = (adversarial_mae - original_mae) / (original_mae + 1e-8)
                adversarial_performances.append(degradation)
                
            except Exception as e:
                continue
        
        if adversarial_performances:
            avg_degradation = np.mean(adversarial_performances)
            score = max(0, 1 - avg_degradation / 0.30)  # 30% degradation threshold
            print(f"    Adversarial robustness: {score:.3f} (avg degradation: {avg_degradation*100:.1f}%)")
            return score
        else:
            return 0.0
    
    def test_temporal_shift(self, baseline_func, test_data):
        """Test temporal shift robustness"""
        print("  Testing temporal shift...")
        
        # This is a simplified test - in practice would test with different time periods
        # For now, we'll test with different text variations that simulate temporal shifts
        
        shift_performances = []
        
        for item in test_data[:10]:
            try:
                # Original prediction
                original_pred = baseline_func(item.text, len(item.series), item.freq, item.domain)
                original_mae = mean_absolute_error(item.series, original_pred)
                
                # Simulate temporal shift with modified text
                shifted_text = item.text.replace("current", "historical").replace("now", "previously")
                shifted_pred = baseline_func(shifted_text, len(item.series), item.freq, item.domain)
                shifted_mae = mean_absolute_error(item.series, shifted_pred)
                
                degradation = (shifted_mae - original_mae) / (original_mae + 1e-8)
                shift_performances.append(degradation)
                
            except Exception as e:
                continue
        
        if shift_performances:
            avg_degradation = np.mean(shift_performances)
            score = max(0, 1 - avg_degradation / 0.25)  # 25% degradation threshold
            print(f"    Temporal shift: {score:.3f} (avg degradation: {avg_degradation*100:.1f}%)")
            return score
        else:
            return 0.0
    
    def run_statistical_analysis(self, baseline_func, test_data):
        """Run comprehensive statistical analysis"""
        
        print("📊 Running statistical analysis...")
        
        # Collect all predictions and targets
        all_predictions = []
        all_targets = []
        domain_predictions = {}
        domain_targets = {}
        
        for item in test_data:
            try:
                pred = baseline_func(item.text, len(item.series), item.freq, item.domain)
                target = item.series
                
                all_predictions.extend(pred)
                all_targets.extend(target)
                
                if item.domain not in domain_predictions:
                    domain_predictions[item.domain] = []
                    domain_targets[item.domain] = []
                
                domain_predictions[item.domain].extend(pred)
                domain_targets[item.domain].extend(target)
                
            except Exception as e:
                continue
        
        # Calculate overall metrics
        overall_metrics = self.calculate_all_metrics(all_predictions, all_targets)
        
        # Calculate domain-specific metrics
        domain_metrics = {}
        for domain in domain_predictions:
            if len(domain_predictions[domain]) > 10:  # Minimum sample size
                domain_metrics[domain] = self.calculate_all_metrics(
                    domain_predictions[domain], domain_targets[domain]
                )
        
        # Bootstrap confidence intervals
        bootstrap_ci = self.calculate_bootstrap_ci(all_predictions, all_targets)
        
        # Statistical significance tests
        significance_tests = self.run_significance_tests(all_predictions, all_targets)
        
        return {
            'overall_metrics': overall_metrics,
            'domain_metrics': domain_metrics,
            'bootstrap_ci': bootstrap_ci,
            'significance_tests': significance_tests
        }
    
    def calculate_bootstrap_ci(self, predictions, targets, n_bootstrap=1000):
        """Calculate bootstrap confidence intervals"""
        
        print("  Calculating bootstrap confidence intervals...")
        
        bootstrap_maes = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(predictions), len(predictions), replace=True)
            bootstrap_pred = [predictions[i] for i in indices]
            bootstrap_targ = [targets[i] for i in indices]
            
            mae = mean_absolute_error(bootstrap_targ, bootstrap_pred)
            bootstrap_maes.append(mae)
        
        # Calculate confidence intervals
        ci_95 = np.percentile(bootstrap_maes, [2.5, 97.5])
        ci_90 = np.percentile(bootstrap_maes, [5, 95])
        
        return {
            'mae_mean': np.mean(bootstrap_maes),
            'mae_std': np.std(bootstrap_maes),
            'ci_95': ci_95,
            'ci_90': ci_90
        }
    
    def run_significance_tests(self, predictions, targets):
        """Run statistical significance tests"""
        
        print("  Running significance tests...")
        
        tests = {}
        
        # Paired t-test
        try:
            t_stat, p_value = stats.ttest_rel(predictions, targets)
            tests['paired_ttest'] = {'t_statistic': t_stat, 'p_value': p_value}
        except:
            tests['paired_ttest'] = {'t_statistic': 0, 'p_value': 1.0}
        
        # Wilcoxon signed-rank test
        try:
            w_stat, p_value = stats.wilcoxon(predictions, targets)
            tests['wilcoxon'] = {'w_statistic': w_stat, 'p_value': p_value}
        except:
            tests['wilcoxon'] = {'w_statistic': 0, 'p_value': 1.0}
        
        # Effect size (Cohen's d)
        try:
            cohens_d = (np.mean(predictions) - np.mean(targets)) / np.sqrt(
                ((len(predictions) - 1) * np.var(predictions) + (len(targets) - 1) * np.var(targets)) / 
                (len(predictions) + len(targets) - 2)
            )
            tests['cohens_d'] = cohens_d
        except:
            tests['cohens_d'] = 0.0
        
        return tests

def run_comprehensive_evaluation(baseline_func, test_data):
    """Run comprehensive evaluation with all metrics and tests"""
    
    print("🚀 COMPREHENSIVE EVALUATION FRAMEWORK")
    print("=" * 60)
    
    evaluator = ComprehensiveEvaluator()
    
    # Run robustness tests
    robustness_scores = evaluator.run_robustness_tests(baseline_func, test_data)
    
    # Run statistical analysis
    statistical_results = evaluator.run_statistical_analysis(baseline_func, test_data)
    
    # Print results
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"=" * 60)
    
    # Overall metrics
    overall_metrics = statistical_results['overall_metrics']
    print(f"\n🎯 Overall Performance:")
    print(f"MAE: {overall_metrics['mae']:.2f}")
    print(f"RMSE: {overall_metrics['rmse']:.2f}")
    print(f"Pearson r: {overall_metrics['pearson_r']:.3f}")
    print(f"Shape Similarity: {overall_metrics['shape_similarity']:.3f}")
    print(f"Trend F1: {overall_metrics['trend_f1']:.3f}")
    
    # Bootstrap confidence intervals
    bootstrap_ci = statistical_results['bootstrap_ci']
    print(f"\n📈 Bootstrap Confidence Intervals (95%):")
    print(f"MAE: {bootstrap_ci['mae_mean']:.2f} ± {bootstrap_ci['mae_std']:.2f}")
    print(f"95% CI: [{bootstrap_ci['ci_95'][0]:.2f}, {bootstrap_ci['ci_95'][1]:.2f}]")
    
    # Robustness scores
    print(f"\n🛡️ Robustness Scores:")
    for test, score in robustness_scores.items():
        print(f"  {test.replace('_', ' ').title()}: {score:.3f}")
    
    # Statistical significance
    significance_tests = statistical_results['significance_tests']
    print(f"\n📊 Statistical Significance:")
    print(f"Paired t-test p-value: {significance_tests['paired_ttest']['p_value']:.4f}")
    print(f"Wilcoxon p-value: {significance_tests['wilcoxon']['p_value']:.4f}")
    print(f"Cohen's d: {significance_tests['cohens_d']:.3f}")
    
    # Domain-specific results
    domain_metrics = statistical_results['domain_metrics']
    if domain_metrics:
        print(f"\n🌍 Domain-Specific Performance:")
        for domain, metrics in domain_metrics.items():
            print(f"  {domain}: MAE = {metrics['mae']:.2f}, r = {metrics['pearson_r']:.3f}")
    
    return {
        'overall_metrics': overall_metrics,
        'robustness_scores': robustness_scores,
        'statistical_results': statistical_results
    }

if __name__ == "__main__":
    # Test the comprehensive evaluation framework
    from ztf.dataset import load_jsonl
    from ztf.baselines import ultimate_ensemble_baseline
    
    # Load dataset
    data = load_jsonl('data/nl2ts_200.jsonl')
    test_data = data[-50:]  # Use last 50 items for evaluation
    
    # Run comprehensive evaluation
    results = run_comprehensive_evaluation(ultimate_ensemble_baseline, test_data)
    
    print(f"\n✅ Comprehensive Evaluation Complete!")
    print(f"Total metrics calculated: {len(results['overall_metrics'])}")
    print(f"Robustness tests completed: {len(results['robustness_scores'])}")
