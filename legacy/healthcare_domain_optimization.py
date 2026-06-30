import numpy as np
import re
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import json

class HealthcareDomainOptimizer:
    def __init__(self):
        # Medical terminology patterns
        self.medical_keywords = {
            'vital_signs': ['heart rate', 'blood pressure', 'temperature', 'respiratory rate', 'oxygen saturation', 'pulse'],
            'lab_results': ['glucose', 'cholesterol', 'creatinine', 'hemoglobin', 'white blood cells', 'platelets'],
            'symptoms': ['fever', 'pain', 'fatigue', 'nausea', 'dizziness', 'shortness of breath'],
            'medications': ['antibiotics', 'painkillers', 'insulin', 'blood pressure medication', 'anticoagulants'],
            'procedures': ['surgery', 'examination', 'test', 'scan', 'biopsy', 'treatment'],
            'conditions': ['diabetes', 'hypertension', 'infection', 'inflammation', 'cancer', 'heart disease'],
            'patient_status': ['stable', 'critical', 'improving', 'deteriorating', 'recovering', 'discharged'],
            'monitoring': ['continuous', 'hourly', 'daily', 'weekly', 'monthly', 'follow-up']
        }
        
        # Clinical time series patterns
        self.clinical_patterns = {
            'acute_changes': ['sudden', 'rapid', 'sharp', 'dramatic', 'critical', 'emergency'],
            'gradual_changes': ['slow', 'gradual', 'progressive', 'steady', 'consistent'],
            'cyclical': ['daily', 'weekly', 'monthly', 'seasonal', 'periodic', 'regular'],
            'trending': ['improving', 'worsening', 'stable', 'fluctuating', 'unpredictable']
        }
        
        # Healthcare data characteristics
        self.healthcare_characteristics = {
            'normal_ranges': ['normal', 'within range', 'acceptable', 'healthy', 'baseline'],
            'abnormal_values': ['high', 'low', 'elevated', 'decreased', 'abnormal', 'critical'],
            'variability': ['variable', 'unstable', 'fluctuating', 'erratic', 'consistent'],
            'seasonality': ['seasonal', 'monthly', 'weekly', 'daily', 'circadian']
        }

    def extract_healthcare_features(self, text, length, freq):
        """Extract healthcare-specific features from text"""
        text_lower = text.lower()
        
        features = {
            'vital_signs_terms': 0,
            'lab_results_terms': 0,
            'symptoms_terms': 0,
            'medications_terms': 0,
            'procedures_terms': 0,
            'conditions_terms': 0,
            'patient_status_terms': 0,
            'monitoring_terms': 0,
            'acute_changes': 0,
            'gradual_changes': 0,
            'cyclical': 0,
            'trending': 0,
            'normal_ranges': 0,
            'abnormal_values': 0,
            'variability': 0,
            'seasonality': 0
        }
        
        # Count medical terminology
        for category, keywords in self.medical_keywords.items():
            for keyword in keywords:
                features[f'{category}_terms'] += text_lower.count(keyword)
        
        # Count clinical patterns
        for pattern, keywords in self.clinical_patterns.items():
            for keyword in keywords:
                features[pattern] += text_lower.count(keyword)
        
        # Count healthcare characteristics
        for characteristic, keywords in self.healthcare_characteristics.items():
            for keyword in keywords:
                if characteristic in features:
                    features[characteristic] += text_lower.count(keyword)
        
        # Normalize by text length
        text_length = len(text.split())
        for key in features:
            features[key] = features[key] / max(text_length, 1)
        
        return features

    def detect_healthcare_patterns(self, text, length, freq):
        """Detect specific healthcare patterns in the text"""
        text_lower = text.lower()
        
        patterns = {
            'is_vital_signs': False,
            'is_lab_results': False,
            'is_symptoms': False,
            'is_medications': False,
            'is_procedures': False,
            'is_conditions': False,
            'is_patient_status': False,
            'is_monitoring': False,
            'change_type': 'stable',  # acute, gradual, stable
            'trend_direction': 'stable',  # improving, worsening, stable
            'data_type': 'continuous',  # continuous, discrete, categorical
            'urgency_level': 'routine',  # routine, urgent, critical
            'complexity_score': 0.5
        }
        
        # Vital signs detection
        if any(term in text_lower for term in ['heart rate', 'blood pressure', 'temperature', 'respiratory']):
            patterns['is_vital_signs'] = True
        
        # Lab results detection
        if any(term in text_lower for term in ['glucose', 'cholesterol', 'creatinine', 'hemoglobin']):
            patterns['is_lab_results'] = True
        
        # Symptoms detection
        if any(term in text_lower for term in ['fever', 'pain', 'fatigue', 'nausea', 'dizziness']):
            patterns['is_symptoms'] = True
        
        # Medications detection
        if any(term in text_lower for term in ['antibiotics', 'painkillers', 'insulin', 'medication']):
            patterns['is_medications'] = True
        
        # Procedures detection
        if any(term in text_lower for term in ['surgery', 'examination', 'test', 'scan', 'biopsy']):
            patterns['is_procedures'] = True
        
        # Conditions detection
        if any(term in text_lower for term in ['diabetes', 'hypertension', 'infection', 'cancer']):
            patterns['is_conditions'] = True
        
        # Patient status detection
        if any(term in text_lower for term in ['stable', 'critical', 'improving', 'deteriorating']):
            patterns['is_patient_status'] = True
        
        # Monitoring detection
        if any(term in text_lower for term in ['continuous', 'hourly', 'daily', 'monitoring']):
            patterns['is_monitoring'] = True
        
        # Change type detection
        acute_terms = ['sudden', 'rapid', 'sharp', 'dramatic', 'critical']
        gradual_terms = ['slow', 'gradual', 'progressive', 'steady']
        
        if any(term in text_lower for term in acute_terms):
            patterns['change_type'] = 'acute'
        elif any(term in text_lower for term in gradual_terms):
            patterns['change_type'] = 'gradual'
        
        # Trend direction detection
        improving_terms = ['improving', 'better', 'recovering', 'decreasing', 'lower']
        worsening_terms = ['worsening', 'worse', 'deteriorating', 'increasing', 'higher']
        
        if any(term in text_lower for term in improving_terms):
            patterns['trend_direction'] = 'improving'
        elif any(term in text_lower for term in worsening_terms):
            patterns['trend_direction'] = 'worsening'
        
        # Urgency level detection
        urgent_terms = ['urgent', 'critical', 'emergency', 'immediate']
        if any(term in text_lower for term in urgent_terms):
            patterns['urgency_level'] = 'critical'
        elif any(term in text_lower for term in ['routine', 'regular', 'scheduled']):
            patterns['urgency_level'] = 'routine'
        
        # Complexity score based on medical terms
        medical_terms = sum(1 for category in self.medical_keywords.values() 
                           for keyword in category if keyword in text_lower)
        patterns['complexity_score'] = min(medical_terms / 8.0, 1.0)
        
        return patterns

    def generate_healthcare_optimized_series(self, text, length, freq, domain):
        """Generate healthcare-optimized time series based on detected patterns"""
        if domain != 'healthcare':
            return self.generate_fallback_series(length, freq)
        
        # Extract features and patterns
        features = self.extract_healthcare_features(text, length, freq)
        patterns = self.detect_healthcare_patterns(text, length, freq)
        
        # Base series generation with healthcare-specific adjustments
        base_series = self.generate_base_healthcare_series(length, freq, patterns)
        
        # Apply pattern-specific modifications
        modified_series = self.apply_healthcare_pattern_modifications(base_series, features, patterns)
        
        # Apply clinical adjustments
        clinical_adjusted = self.apply_clinical_adjustments(modified_series, patterns)
        
        # Apply trend adjustments
        trend_adjusted = self.apply_healthcare_trend_adjustments(clinical_adjusted, patterns)
        
        # Final smoothing and normalization
        final_series = self.apply_healthcare_smoothing(trend_adjusted, patterns)
        
        return final_series.tolist()

    def generate_base_healthcare_series(self, length, freq, patterns):
        """Generate base healthcare time series"""
        np.random.seed(42)  # For reproducibility
        
        # Base parameters based on patterns
        if patterns['change_type'] == 'acute':
            volatility = 0.4
            mean_reversion = 0.05
        elif patterns['change_type'] == 'gradual':
            volatility = 0.1
            mean_reversion = 0.4
        else:  # stable
            volatility = 0.2
            mean_reversion = 0.2
        
        # Adjust based on urgency level
        if patterns['urgency_level'] == 'critical':
            volatility *= 1.5
            mean_reversion *= 0.5
        
        # Generate base series
        series = np.zeros(length)
        series[0] = np.random.normal(70, 5)  # Start around normal range
        
        for i in range(1, length):
            # Mean reversion component
            mean_reversion_component = mean_reversion * (70 - series[i-1])
            
            # Random walk component
            random_component = np.random.normal(0, volatility * 5)
            
            # Update series
            series[i] = series[i-1] + mean_reversion_component + random_component
        
        return series

    def apply_healthcare_pattern_modifications(self, series, features, patterns):
        """Apply healthcare-specific pattern modifications"""
        modified = series.copy()
        
        # Vital signs patterns
        if patterns['is_vital_signs']:
            # Vital signs typically have normal ranges and physiological constraints
            for i in range(len(modified)):
                # Keep within reasonable vital sign ranges (e.g., heart rate 40-200)
                if modified[i] < 40:
                    modified[i] = 40 + np.random.normal(0, 2)
                elif modified[i] > 200:
                    modified[i] = 200 - np.random.normal(0, 2)
        
        # Lab results patterns
        if patterns['is_lab_results']:
            # Lab results often have reference ranges
            for i in range(len(modified)):
                # Add some lab-specific characteristics
                if np.random.random() < 0.1:  # Occasional outliers
                    modified[i] += np.random.normal(0, 10)
        
        # Symptoms patterns
        if patterns['is_symptoms']:
            # Symptoms can be more variable and subjective
            for i in range(1, len(modified)):
                if np.random.random() < 0.15:  # Symptom flare-ups
                    modified[i] += np.random.normal(0, 8)
        
        # Medications patterns
        if patterns['is_medications']:
            # Medication effects can be gradual
            for i in range(1, len(modified)):
                # Gradual improvement or side effects
                medication_effect = np.random.normal(-1, 2)  # Slight downward trend
                modified[i] += medication_effect
        
        # Patient status patterns
        if patterns['is_patient_status']:
            # Patient status affects variability
            if patterns['trend_direction'] == 'improving':
                for i in range(len(modified)):
                    modified[i] += -0.5 * i  # Gradual improvement
            elif patterns['trend_direction'] == 'worsening':
                for i in range(len(modified)):
                    modified[i] += 0.5 * i  # Gradual worsening
        
        return modified

    def apply_clinical_adjustments(self, series, patterns):
        """Apply clinical-specific adjustments"""
        adjusted = series.copy()
        
        # Clinical data often has specific characteristics
        if patterns['data_type'] == 'continuous':
            # Continuous monitoring data
            for i in range(1, len(adjusted)):
                # Add some measurement noise
                adjusted[i] += np.random.normal(0, 0.5)
        
        # Add circadian rhythm for certain measurements
        if patterns['is_vital_signs']:
            for i in range(len(adjusted)):
                # Simulate daily variation
                circadian = 2 * np.sin(2 * np.pi * i / 24)  # 24-hour cycle
                adjusted[i] += circadian
        
        # Add seasonal effects for certain conditions
        if patterns['is_conditions']:
            for i in range(len(adjusted)):
                # Simulate seasonal variation
                seasonal = 3 * np.sin(2 * np.pi * i / 365)  # Annual cycle
                adjusted[i] += seasonal
        
        return adjusted

    def apply_healthcare_trend_adjustments(self, series, patterns):
        """Apply healthcare-specific trend adjustments"""
        adjusted = series.copy()
        
        if patterns['trend_direction'] == 'improving':
            # Add improvement trend
            improvement_component = np.linspace(0, -15, len(adjusted))
            adjusted += improvement_component
        
        elif patterns['trend_direction'] == 'worsening':
            # Add worsening trend
            worsening_component = np.linspace(0, 15, len(adjusted))
            adjusted += worsening_component
        
        # Add recovery patterns
        if patterns['is_patient_status'] and patterns['trend_direction'] == 'improving':
            # Recovery often follows a sigmoid pattern
            for i in range(len(adjusted)):
                recovery_factor = 1 / (1 + np.exp(-(i - len(adjusted)/2) / 5))
                adjusted[i] -= recovery_factor * 10
        
        return adjusted

    def apply_healthcare_smoothing(self, series, patterns):
        """Apply healthcare-specific smoothing"""
        # Use appropriate smoothing for clinical data
        window_length = min(9, len(series) // 4)
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 3:
            smoothed = savgol_filter(series, window_length, 2)  # Lower polynomial for clinical data
        else:
            smoothed = series
        
        # Add minimal noise to maintain clinical realism
        noise = np.random.normal(0, 0.3, len(smoothed))
        final = smoothed + noise
        
        return final

    def generate_fallback_series(self, length, freq):
        """Generate fallback series for non-healthcare domains"""
        np.random.seed(42)
        series = np.cumsum(np.random.normal(0, 0.8, length)) + 70
        return series.tolist()

def healthcare_optimized_baseline(text, length, freq, domain):
    """Healthcare-optimized baseline function"""
    optimizer = HealthcareDomainOptimizer()
    return optimizer.generate_healthcare_optimized_series(text, length, freq, domain)

def test_healthcare_optimization():
    """Test the healthcare domain optimization"""
    # Load test data
    with open('nl2ts_200.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Filter healthcare domain data
    healthcare_data = [item for item in data if item.get('domain') == 'healthcare']
    
    if not healthcare_data:
        print("No healthcare domain data found")
        return
    
    print(f"Testing Healthcare Domain Optimization on {len(healthcare_data)} samples")
    print("=" * 60)
    
    total_mae = 0
    optimizer = HealthcareDomainOptimizer()
    
    for i, item in enumerate(healthcare_data[:50]):  # Test first 50 healthcare samples
        text = item['text']
        target = item['target']
        length = len(target)
        freq = item.get('freq', 'D')
        domain = item.get('domain', 'healthcare')
        
        # Generate prediction
        prediction = optimizer.generate_healthcare_optimized_series(text, length, freq, domain)
        
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
    
    mean_mae = total_mae / min(50, len(healthcare_data))
    print(f"Healthcare Domain Mean MAE: {mean_mae:.2f}")
    print(f"Target MAE: < 20.0")
    print(f"Improvement needed: {max(0, mean_mae - 20.0):.2f}")
    
    return mean_mae

if __name__ == "__main__":
    test_healthcare_optimization()
