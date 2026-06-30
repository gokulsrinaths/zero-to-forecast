import numpy as np
import re
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import json

class TechnologyDomainOptimizer:
    def __init__(self):
        # Technology terminology patterns
        self.tech_keywords = {
            'performance': ['cpu', 'memory', 'disk', 'network', 'latency', 'throughput', 'response time', 'load'],
            'metrics': ['uptime', 'availability', 'reliability', 'efficiency', 'utilization', 'capacity'],
            'monitoring': ['monitoring', 'logging', 'alerting', 'dashboard', 'metrics', 'telemetry'],
            'scaling': ['scaling', 'load balancing', 'auto-scaling', 'horizontal', 'vertical', 'elastic'],
            'infrastructure': ['server', 'cloud', 'database', 'api', 'microservice', 'container'],
            'issues': ['error', 'failure', 'outage', 'bug', 'crash', 'timeout', 'overload'],
            'optimization': ['optimization', 'tuning', 'performance', 'efficiency', 'improvement'],
            'trends': ['growth', 'decline', 'spike', 'drop', 'trend', 'pattern', 'cycle']
        }
        
        # System performance patterns
        self.performance_patterns = {
            'load_patterns': ['high load', 'low load', 'peak', 'valley', 'spike', 'drop'],
            'capacity_patterns': ['overcapacity', 'undercapacity', 'optimal', 'bottleneck', 'headroom'],
            'failure_patterns': ['failure', 'error', 'crash', 'timeout', 'overload', 'resource exhaustion'],
            'recovery_patterns': ['recovery', 'restart', 'restore', 'backup', 'failover', 'redundancy']
        }
        
        # Technology data characteristics
        self.tech_characteristics = {
            'real_time': ['real-time', 'live', 'instant', 'immediate', 'current'],
            'batch': ['batch', 'scheduled', 'periodic', 'daily', 'hourly'],
            'event_driven': ['event', 'trigger', 'webhook', 'notification', 'alert'],
            'predictive': ['prediction', 'forecast', 'trend', 'anomaly', 'baseline']
        }

    def extract_technology_features(self, text, length, freq):
        """Extract technology-specific features from text"""
        text_lower = text.lower()
        
        features = {
            'performance_terms': 0,
            'metrics_terms': 0,
            'monitoring_terms': 0,
            'scaling_terms': 0,
            'infrastructure_terms': 0,
            'issues_terms': 0,
            'optimization_terms': 0,
            'trends_terms': 0,
            'load_patterns': 0,
            'capacity_patterns': 0,
            'failure_patterns': 0,
            'recovery_patterns': 0,
            'real_time': 0,
            'batch': 0,
            'event_driven': 0,
            'predictive': 0
        }
        
        # Count technology terminology
        for category, keywords in self.tech_keywords.items():
            for keyword in keywords:
                features[f'{category}_terms'] += text_lower.count(keyword)
        
        # Count performance patterns
        for pattern, keywords in self.performance_patterns.items():
            for keyword in keywords:
                features[pattern] += text_lower.count(keyword)
        
        # Count technology characteristics
        for characteristic, keywords in self.tech_characteristics.items():
            for keyword in keywords:
                if characteristic in features:
                    features[characteristic] += text_lower.count(keyword)
        
        # Normalize by text length
        text_length = len(text.split())
        for key in features:
            features[key] = features[key] / max(text_length, 1)
        
        return features

    def detect_technology_patterns(self, text, length, freq):
        """Detect specific technology patterns in the text"""
        text_lower = text.lower()
        
        patterns = {
            'is_performance_focus': False,
            'is_metrics_focus': False,
            'is_monitoring_focus': False,
            'is_scaling_focus': False,
            'is_infrastructure_focus': False,
            'is_issues_focus': False,
            'is_optimization_focus': False,
            'is_trends_focus': False,
            'load_level': 'normal',  # low, normal, high, critical
            'system_health': 'healthy',  # healthy, degraded, critical
            'data_frequency': 'continuous',  # continuous, batch, event-driven
            'complexity_score': 0.5
        }
        
        # Performance focus detection
        if any(term in text_lower for term in ['cpu', 'memory', 'disk', 'network', 'latency']):
            patterns['is_performance_focus'] = True
        
        # Metrics focus detection
        if any(term in text_lower for term in ['uptime', 'availability', 'reliability', 'efficiency']):
            patterns['is_metrics_focus'] = True
        
        # Monitoring focus detection
        if any(term in text_lower for term in ['monitoring', 'logging', 'alerting', 'dashboard']):
            patterns['is_monitoring_focus'] = True
        
        # Scaling focus detection
        if any(term in text_lower for term in ['scaling', 'load balancing', 'auto-scaling']):
            patterns['is_scaling_focus'] = True
        
        # Infrastructure focus detection
        if any(term in text_lower for term in ['server', 'cloud', 'database', 'api', 'microservice']):
            patterns['is_infrastructure_focus'] = True
        
        # Issues focus detection
        if any(term in text_lower for term in ['error', 'failure', 'outage', 'bug', 'crash']):
            patterns['is_issues_focus'] = True
        
        # Optimization focus detection
        if any(term in text_lower for term in ['optimization', 'tuning', 'performance', 'efficiency']):
            patterns['is_optimization_focus'] = True
        
        # Trends focus detection
        if any(term in text_lower for term in ['growth', 'decline', 'spike', 'drop', 'trend']):
            patterns['is_trends_focus'] = True
        
        # Load level detection
        high_load_terms = ['high load', 'peak', 'spike', 'overload', 'bottleneck']
        low_load_terms = ['low load', 'valley', 'drop', 'underutilized', 'idle']
        critical_terms = ['critical', 'emergency', 'failure', 'crash', 'outage']
        
        if any(term in text_lower for term in critical_terms):
            patterns['load_level'] = 'critical'
        elif any(term in text_lower for term in high_load_terms):
            patterns['load_level'] = 'high'
        elif any(term in text_lower for term in low_load_terms):
            patterns['load_level'] = 'low'
        
        # System health detection
        healthy_terms = ['healthy', 'normal', 'optimal', 'stable', 'good']
        degraded_terms = ['degraded', 'poor', 'slow', 'unstable', 'problem']
        
        if any(term in text_lower for term in degraded_terms):
            patterns['system_health'] = 'degraded'
        elif any(term in text_lower for term in critical_terms):
            patterns['system_health'] = 'critical'
        elif any(term in text_lower for term in healthy_terms):
            patterns['system_health'] = 'healthy'
        
        # Data frequency detection
        if any(term in text_lower for term in ['real-time', 'live', 'instant', 'continuous']):
            patterns['data_frequency'] = 'continuous'
        elif any(term in text_lower for term in ['batch', 'scheduled', 'periodic']):
            patterns['data_frequency'] = 'batch'
        elif any(term in text_lower for term in ['event', 'trigger', 'webhook']):
            patterns['data_frequency'] = 'event-driven'
        
        # Complexity score based on tech terms
        tech_terms = sum(1 for category in self.tech_keywords.values() 
                        for keyword in category if keyword in text_lower)
        patterns['complexity_score'] = min(tech_terms / 8.0, 1.0)
        
        return patterns

    def generate_technology_optimized_series(self, text, length, freq, domain):
        """Generate technology-optimized time series based on detected patterns"""
        if domain != 'technology':
            return self.generate_fallback_series(length, freq)
        
        # Extract features and patterns
        features = self.extract_technology_features(text, length, freq)
        patterns = self.detect_technology_patterns(text, length, freq)
        
        # Base series generation with technology-specific adjustments
        base_series = self.generate_base_technology_series(length, freq, patterns)
        
        # Apply pattern-specific modifications
        modified_series = self.apply_technology_pattern_modifications(base_series, features, patterns)
        
        # Apply system adjustments
        system_adjusted = self.apply_system_adjustments(modified_series, patterns)
        
        # Apply performance adjustments
        performance_adjusted = self.apply_performance_adjustments(system_adjusted, patterns)
        
        # Final smoothing and normalization
        final_series = self.apply_technology_smoothing(performance_adjusted, patterns)
        
        return final_series.tolist()

    def generate_base_technology_series(self, length, freq, patterns):
        """Generate base technology time series"""
        np.random.seed(42)  # For reproducibility
        
        # Base parameters based on patterns
        if patterns['load_level'] == 'critical':
            volatility = 0.5
            mean_reversion = 0.05
        elif patterns['load_level'] == 'high':
            volatility = 0.3
            mean_reversion = 0.1
        elif patterns['load_level'] == 'low':
            volatility = 0.1
            mean_reversion = 0.4
        else:  # normal
            volatility = 0.2
            mean_reversion = 0.2
        
        # Adjust based on system health
        if patterns['system_health'] == 'critical':
            volatility *= 2.0
            mean_reversion *= 0.3
        elif patterns['system_health'] == 'degraded':
            volatility *= 1.5
            mean_reversion *= 0.7
        
        # Generate base series
        series = np.zeros(length)
        series[0] = np.random.normal(50, 5)  # Start around 50% utilization
        
        for i in range(1, length):
            # Mean reversion component
            mean_reversion_component = mean_reversion * (50 - series[i-1])
            
            # Random walk component
            random_component = np.random.normal(0, volatility * 5)
            
            # Update series
            series[i] = series[i-1] + mean_reversion_component + random_component
        
        return series

    def apply_technology_pattern_modifications(self, series, features, patterns):
        """Apply technology-specific pattern modifications"""
        modified = series.copy()
        
        # Performance focus patterns
        if patterns['is_performance_focus']:
            # Performance metrics often have specific ranges and behaviors
            for i in range(len(modified)):
                # Keep within reasonable performance ranges (0-100%)
                if modified[i] < 0:
                    modified[i] = 0 + np.random.normal(0, 1)
                elif modified[i] > 100:
                    modified[i] = 100 - np.random.normal(0, 1)
        
        # Metrics focus patterns
        if patterns['is_metrics_focus']:
            # Metrics often show trends and patterns
            for i in range(1, len(modified)):
                if np.random.random() < 0.1:  # Occasional metric spikes
                    modified[i] += np.random.normal(0, 8)
        
        # Monitoring focus patterns
        if patterns['is_monitoring_focus']:
            # Monitoring data can be more granular
            for i in range(1, len(modified)):
                # Add monitoring noise
                monitoring_noise = np.random.normal(0, 1)
                modified[i] += monitoring_noise
        
        # Scaling patterns
        if patterns['is_scaling_focus']:
            # Scaling events can cause step changes
            for i in range(1, len(modified)):
                if np.random.random() < 0.05:  # Scaling events
                    scaling_effect = np.random.normal(0, 15)
                    modified[i] += scaling_effect
        
        # Infrastructure patterns
        if patterns['is_infrastructure_focus']:
            # Infrastructure changes can be gradual
            for i in range(1, len(modified)):
                # Gradual infrastructure improvements
                infrastructure_effect = np.random.normal(-0.5, 1)
                modified[i] += infrastructure_effect
        
        # Issues patterns
        if patterns['is_issues_focus']:
            # Issues can cause sudden drops or spikes
            for i in range(1, len(modified)):
                if np.random.random() < 0.08:  # Issue events
                    issue_effect = np.random.normal(0, 20)
                    modified[i] += issue_effect
        
        return modified

    def apply_system_adjustments(self, series, patterns):
        """Apply system-specific adjustments"""
        adjusted = series.copy()
        
        # System health affects overall behavior
        if patterns['system_health'] == 'critical':
            # Critical systems show high variability
            for i in range(1, len(adjusted)):
                if np.random.random() < 0.3:  # Frequent critical events
                    adjusted[i] += np.random.normal(0, 12)
        
        elif patterns['system_health'] == 'degraded':
            # Degraded systems show increased variability
            for i in range(1, len(adjusted)):
                if np.random.random() < 0.15:  # Occasional degradation events
                    adjusted[i] += np.random.normal(0, 8)
        
        # Data frequency affects granularity
        if patterns['data_frequency'] == 'continuous':
            # Continuous data is more granular
            for i in range(1, len(adjusted)):
                adjusted[i] += np.random.normal(0, 0.5)
        
        elif patterns['data_frequency'] == 'batch':
            # Batch data can have larger jumps
            for i in range(1, len(adjusted)):
                if np.random.random() < 0.2:  # Batch processing effects
                    adjusted[i] += np.random.normal(0, 5)
        
        return adjusted

    def apply_performance_adjustments(self, series, patterns):
        """Apply performance-specific adjustments"""
        adjusted = series.copy()
        
        # Load level affects performance characteristics
        if patterns['load_level'] == 'high':
            # High load can cause performance degradation
            for i in range(len(adjusted)):
                # Simulate performance degradation under load
                load_penalty = 0.1 * (adjusted[i] - 50) if adjusted[i] > 50 else 0
                adjusted[i] += load_penalty
        
        elif patterns['load_level'] == 'critical':
            # Critical load can cause severe performance issues
            for i in range(len(adjusted)):
                # Simulate severe performance degradation
                critical_penalty = 0.2 * (adjusted[i] - 30) if adjusted[i] > 30 else 0
                adjusted[i] += critical_penalty
        
        # Add diurnal patterns for technology systems
        for i in range(len(adjusted)):
            # Simulate daily usage patterns
            daily_pattern = 5 * np.sin(2 * np.pi * i / 24)  # 24-hour cycle
            adjusted[i] += daily_pattern
        
        # Add weekly patterns
        for i in range(len(adjusted)):
            # Simulate weekly usage patterns
            weekly_pattern = 3 * np.sin(2 * np.pi * i / 168)  # 168-hour week
            adjusted[i] += weekly_pattern
        
        return adjusted

    def apply_technology_smoothing(self, series, patterns):
        """Apply technology-specific smoothing"""
        # Use appropriate smoothing for technology data
        window_length = min(7, len(series) // 5)
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 3:
            smoothed = savgol_filter(series, window_length, 2)
        else:
            smoothed = series
        
        # Add minimal noise to maintain technology realism
        noise = np.random.normal(0, 0.2, len(smoothed))
        final = smoothed + noise
        
        return final

    def generate_fallback_series(self, length, freq):
        """Generate fallback series for non-technology domains"""
        np.random.seed(42)
        series = np.cumsum(np.random.normal(0, 0.6, length)) + 50
        return series.tolist()

def technology_optimized_baseline(text, length, freq, domain):
    """Technology-optimized baseline function"""
    optimizer = TechnologyDomainOptimizer()
    return optimizer.generate_technology_optimized_series(text, length, freq, domain)

def test_technology_optimization():
    """Test the technology domain optimization"""
    # Load test data
    with open('nl2ts_200.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Filter technology domain data
    technology_data = [item for item in data if item.get('domain') == 'technology']
    
    if not technology_data:
        print("No technology domain data found")
        return
    
    print(f"Testing Technology Domain Optimization on {len(technology_data)} samples")
    print("=" * 60)
    
    total_mae = 0
    optimizer = TechnologyDomainOptimizer()
    
    for i, item in enumerate(technology_data[:50]):  # Test first 50 technology samples
        text = item['text']
        target = item['target']
        length = len(target)
        freq = item.get('freq', 'D')
        domain = item.get('domain', 'technology')
        
        # Generate prediction
        prediction = optimizer.generate_technology_optimized_series(text, length, freq, domain)
        
        # Calculate MAE
        mae = np.mean(np.abs(np.array(prediction) - np.array(target)))
        total_mae += mae
        
        if i < 5:  # Show first 5 examples
            print(f"Sample {i+1}:")
            print(f"  Text: {text[:100]}...")
            print(f"  MAE: {mae:.2f}")
            print(f"  Target range: [{min(target):.2f}, {max(target):.2f}]")
            print(f"  Prediction range: [{min(prediction):.2f}, {max(prediction):.2f}]")
            print()
    
    mean_mae = total_mae / min(50, len(technology_data))
    print(f"Technology Domain Mean MAE: {mean_mae:.2f}")
    print(f"Target MAE: < 15.0")
    print(f"Improvement needed: {max(0, mean_mae - 15.0):.2f}")
    
    return mean_mae

if __name__ == "__main__":
    test_technology_optimization()
