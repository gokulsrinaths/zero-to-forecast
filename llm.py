"""
LLM inference module for Zero-to-Forecast.

Contains functions for generating time series predictions
using Meta LLaMA models via OpenAI-compatible API.
"""

import json
import os
import numpy as np
import time
from typing import List, Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Cost tracking
COST_PER_1K_TOKENS = {
    'input': 0.0001,  # $0.0001 per 1K input tokens
    'output': 0.0002   # $0.0002 per 1K output tokens
}

class CostTracker:
    """Track API costs and latency."""
    
    def __init__(self):
        self.total_cost = 0.0
        self.total_tokens_input = 0
        self.total_tokens_output = 0
        self.total_latency = 0.0
        self.num_requests = 0
    
    def add_request(self, input_tokens: int, output_tokens: int, latency: float):
        """Add a completed request to the tracker."""
        cost = (input_tokens * COST_PER_1K_TOKENS['input'] / 1000 + 
                output_tokens * COST_PER_1K_TOKENS['output'] / 1000)
        
        self.total_cost += cost
        self.total_tokens_input += input_tokens
        self.total_tokens_output += output_tokens
        self.total_latency += latency
        self.num_requests += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cost and latency statistics."""
        return {
            'total_cost': self.total_cost,
            'total_tokens_input': self.total_tokens_input,
            'total_tokens_output': self.total_tokens_output,
            'total_latency': self.total_latency,
            'num_requests': self.num_requests,
            'avg_latency_per_request': self.total_latency / self.num_requests if self.num_requests > 0 else 0,
            'cost_per_request': self.total_cost / self.num_requests if self.num_requests > 0 else 0,
            'tokens_per_request': (self.total_tokens_input + self.total_tokens_output) / self.num_requests if self.num_requests > 0 else 0
        }

# Global cost tracker
_cost_tracker = CostTracker()

def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    return _cost_tracker

def get_llm_client() -> OpenAI:
    """Initialize OpenAI-compatible client for LLaMA."""
    api_key = os.getenv("LLAMA_API_KEY")
    base_url = os.getenv("LLAMA_BASE_URL", "https://api.llama.com/compat/v1")
    
    if not api_key:
        raise ValueError("LLAMA_API_KEY environment variable is required")
    
    return OpenAI(api_key=api_key, base_url=base_url)

def create_enhanced_prompt(text: str, domain: str, freq: str, length: int) -> Dict[str, str]:
    """Create enhanced prompt with few-shot examples for better LLM performance."""
    
    # Domain-specific context
    domain_contexts = {
        "finance": {
            "range": "30-360",
            "description": "financial metrics like stock prices, revenue, or market indices",
            "examples": [
                {"text": "Stock price shows gradual increase over time", "series": [120.5, 125.2, 128.7, 132.1, 135.6, 138.9, 142.3, 145.1, 148.4, 151.2, 154.8, 157.3]},
                {"text": "Revenue spikes after marketing campaign", "series": [100.0, 102.3, 98.7, 105.2, 180.5, 175.2, 168.9, 162.4, 158.7, 155.2, 152.8, 150.1]}
            ]
        },
        "weather": {
            "range": "5-65",
            "description": "weather data like temperature, humidity, or precipitation",
            "examples": [
                {"text": "Temperature shows seasonal variation", "series": [15.2, 18.7, 22.1, 25.8, 28.4, 30.1, 29.8, 27.3, 24.1, 20.5, 17.2, 14.8]},
                {"text": "Humidity remains stable with slight fluctuations", "series": [65.2, 64.8, 66.1, 65.7, 64.9, 65.3, 66.2, 65.8, 65.1, 64.7, 65.4, 65.9]}
            ]
        },
        "healthcare": {
            "range": "45-325",
            "description": "healthcare metrics like heart rate, blood pressure, or patient vitals",
            "examples": [
                {"text": "Heart rate increases during exercise then recovers", "series": [72.1, 75.3, 78.9, 82.4, 85.7, 88.2, 90.5, 87.3, 84.1, 80.8, 77.5, 74.2]},
                {"text": "Blood pressure shows slight upward trend", "series": [120.5, 122.1, 123.8, 125.2, 126.7, 128.3, 129.8, 131.2, 132.7, 134.1, 135.6, 137.2]}
            ]
        },
        "iot": {
            "range": "20-160",
            "description": "IoT sensor data like temperature, pressure, or energy consumption",
            "examples": [
                {"text": "Sensor temperature fluctuates with noise", "series": [45.2, 46.8, 44.9, 47.1, 45.7, 46.3, 44.8, 47.2, 45.9, 46.5, 44.7, 47.0]},
                {"text": "Energy consumption shows daily pattern", "series": [85.3, 82.1, 78.9, 75.6, 72.4, 70.1, 68.9, 71.2, 74.5, 77.8, 80.2, 83.7]}
            ]
        },
        "technology": {
            "range": "15-220",
            "description": "technology metrics like server load, response time, or user activity",
            "examples": [
                {"text": "Server load increases during peak hours", "series": [45.2, 48.7, 52.1, 55.8, 58.4, 60.1, 59.8, 57.3, 54.1, 50.5, 47.2, 44.8]},
                {"text": "Response time shows occasional spikes", "series": [120.5, 118.7, 122.3, 125.8, 180.2, 175.6, 172.1, 168.9, 165.4, 162.8, 159.3, 156.7]}
            ]
        },
        "retail": {
            "range": "35-270",
            "description": "retail metrics like sales, inventory, or customer traffic",
            "examples": [
                {"text": "Sales show weekly seasonal pattern", "series": [150.2, 145.8, 140.3, 135.7, 130.2, 125.8, 120.3, 180.5, 175.2, 170.8, 165.3, 160.7]},
                {"text": "Inventory decreases gradually over time", "series": [200.5, 195.2, 190.8, 185.3, 180.7, 175.2, 170.8, 165.3, 160.7, 155.2, 150.8, 145.3]}
            ]
        }
    }
    
    context = domain_contexts.get(domain, {
        "range": "50-150",
        "description": "time series data",
        "examples": [
            {"text": "Data shows gradual increase", "series": [100.0, 102.5, 105.2, 107.8, 110.3, 112.9, 115.4, 117.8, 120.2, 122.7, 125.1, 127.6]}
        ]
    })
    
    # Create few-shot examples
    examples_text = ""
    for i, example in enumerate(context["examples"]):
        examples_text += f"""
Example {i+1}:
Text: "{example['text']}"
Series: {example['series']}
"""
    
    system_prompt = f"""You are an expert time series generator for {domain} domain. Generate realistic time series based on natural language descriptions.

DOMAIN: {domain} ({context['description']})
VALUE RANGE: {context['range']}
SERIES LENGTH: {length} values

{examples_text}

CRITICAL REQUIREMENTS:
1. Output ONLY valid JSON format: {{"series": [val1, val2, val3, ...]}}
2. Generate exactly {length} numbers
3. Values must be realistic for {domain} domain
4. NO comments, NO explanations, NO extra text
5. Follow the pattern shown in examples

Generate time series for:"""

    user_prompt = f'"{text}"'

    return {"system": system_prompt, "user": user_prompt}

def validate_series(series: List[float], length: int) -> List[float]:
    """Validate and clean generated series."""
    if not isinstance(series, list):
        raise ValueError("Series must be a list")
    
    if len(series) != length:
        raise ValueError(f"Expected length {length}, got {len(series)}")
    
    # Convert to float and handle invalid values
    validated = []
    for val in series:
        try:
            float_val = float(val)
            validated.append(float_val)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid numeric value: {val}")
    
    # Clip extreme outliers (z-score > 6)
    validated = np.array(validated)
    z_scores = np.abs((validated - np.mean(validated)) / np.std(validated))
    outlier_mask = z_scores > 6
    
    if np.any(outlier_mask):
        logger.warning(f"Clipping {np.sum(outlier_mask)} extreme outliers")
        validated[outlier_mask] = np.mean(validated) + 6 * np.std(validated) * np.sign(validated[outlier_mask] - np.mean(validated))
    
    return validated.tolist()

def parse_llm_response(response_text: str, length: int) -> List[float]:
    """Parse LLM response and extract time series with robust error handling."""
    import re
    
    # Clean response text
    response_text = response_text.strip()
    
    # Remove markdown formatting if present
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    
    response_text = response_text.strip()
    
    # Try to extract JSON from potentially messy response
    try:
        # First attempt: direct JSON parsing
        data = json.loads(response_text)
        if isinstance(data, dict) and "series" in data:
            series = data["series"]
            return validate_series(series, length)
    except json.JSONDecodeError:
        pass
    
    # Second attempt: Extract JSON pattern with regex
    try:
        # Look for {"series": [numbers...]} pattern
        json_pattern = r'\{"series":\s*\[([\d\.,\s-]+)\]\}'
        match = re.search(json_pattern, response_text)
        
        if match:
            numbers_str = match.group(1)
            # Extract numbers using regex
            numbers = re.findall(r'-?\d+\.?\d*', numbers_str)
            series = [float(num) for num in numbers]
            
            if len(series) == length:
                return validate_series(series, length)
            else:
                logger.warning(f"Extracted {len(series)} values, expected {length}")
    except Exception as e:
        logger.warning(f"Regex extraction failed: {e}")
    
    # Third attempt: Extract any array of numbers
    try:
        # Look for any array pattern [numbers...]
        array_pattern = r'\[([\d\.,\s-]+)\]'
        match = re.search(array_pattern, response_text)
        
        if match:
            numbers_str = match.group(1)
            numbers = re.findall(r'-?\d+\.?\d*', numbers_str)
            series = [float(num) for num in numbers]
            
            if len(series) >= length:
                # Take first 'length' numbers if too many
                series = series[:length]
                return validate_series(series, length)
            elif len(series) >= length * 0.8:  # At least 80% of expected length
                # Pad with interpolated values if close enough
                while len(series) < length:
                    # Add interpolated value
                    if len(series) >= 2:
                        last_val = series[-1]
                        second_last = series[-2]
                        next_val = last_val + (last_val - second_last)
                        series.append(next_val)
                    else:
                        series.append(series[-1] if series else 100.0)
                return validate_series(series, length)
    except Exception as e:
        logger.warning(f"Array extraction failed: {e}")
    
    # If all parsing attempts fail, raise error
    logger.error(f"All parsing attempts failed for response: {response_text[:200]}...")
    raise ValueError(f"Could not extract valid series from response")

def generate_enhanced_fallback_series(text: str, length: int, domain: str, freq: str) -> List[float]:
    """Generate intelligent fallback series using text analysis."""
    import re
    
    # Extract patterns from text
    text_lower = text.lower()
    
    # Domain-specific base values and ranges
    domain_configs = {
        "finance": {"base": 120.0, "std": 25.0, "min": 50.0, "max": 200.0},
        "iot": {"base": 45.0, "std": 15.0, "min": 20.0, "max": 100.0},
        "weather": {"base": 22.0, "std": 8.0, "min": 10.0, "max": 35.0},
        "healthcare": {"base": 85.0, "std": 20.0, "min": 60.0, "max": 180.0},
        "technology": {"base": 65.0, "std": 18.0, "min": 30.0, "max": 120.0},
        "retail": {"base": 75.0, "std": 22.0, "min": 40.0, "max": 150.0}
    }
    
    config = domain_configs.get(domain, {"base": 100.0, "std": 20.0, "min": 50.0, "max": 200.0})
    
    # Detect patterns
    has_increase = any(word in text_lower for word in ["increase", "rise", "up", "higher", "surge", "grow"])
    has_decrease = any(word in text_lower for word in ["decrease", "drop", "down", "lower", "decline", "fall"])
    has_spike = any(word in text_lower for word in ["spike", "peak", "surge", "jump"])
    has_seasonal = any(word in text_lower for word in ["seasonal", "cyclical", "periodic", "each", "every"])
    has_shock = any(word in text_lower for word in ["after", "during", "following", "due to", "because of"])
    
    # Generate base series
    if has_seasonal:
        # Seasonal pattern
        period = {"hour": 4, "day": 2, "week": 1, "month": 0.5}.get(freq, 1)
        time_points = np.linspace(0, 2 * np.pi * period, length)
        series = config["base"] + config["std"] * 0.5 * np.sin(time_points)
    elif has_spike:
        # Spike pattern
        series = np.full(length, config["base"])
        spike_pos = length // 2
        if 0 <= spike_pos < length:
            series[spike_pos] = config["base"] * 1.8
    elif has_increase:
        # Increasing trend
        trend = np.linspace(0, config["std"] * 2, length)
        series = config["base"] + trend
    elif has_decrease:
        # Decreasing trend
        trend = np.linspace(0, -config["std"] * 2, length)
        series = config["base"] + trend
    else:
        # Stable with noise
        series = config["base"] + np.random.normal(0, config["std"] * 0.3, length)
    
    # Add shock effects if detected
    if has_shock:
        shock_pos = 2 * length // 3
        if 0 <= shock_pos < length:
            if has_increase:
                series[shock_pos] *= 1.5
                # Recovery
                for i in range(shock_pos + 1, min(length, shock_pos + 4)):
                    recovery = 1 - (i - shock_pos) * 0.1
                    series[i] *= recovery
            elif has_decrease:
                series[shock_pos] *= 0.5
                # Recovery
                for i in range(shock_pos + 1, min(length, shock_pos + 4)):
                    recovery = 1 + (i - shock_pos) * 0.1
                    series[i] *= recovery
    
    # Apply domain constraints
    series = np.clip(series, config["min"], config["max"])
    
    return series.tolist()

def llm_generate_enhanced(text: str, domain: str, freq: str, length: int, model_id: str = "Llama-4-Maverick-17B-128E-Instruct-FP8", temperature: float = 0.1) -> List[float]:
    """Enhanced LLM generation with better prompts and error handling."""
    
    try:
        # Create enhanced prompt
        prompts = create_enhanced_prompt(text, domain, freq, length)
        
        # Make API call
        response = get_llm_client().chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": prompts["user"]}
            ],
            temperature=temperature,
            max_tokens=min(4096, max(100, int(length * 15))),
            top_p=0.95,
            frequency_penalty=0.05,
            presence_penalty=0.0
        )
        
        # Parse response
        generated_text = response.choices[0].message.content.strip()
        series = parse_llm_response(generated_text, length)
        
        # Validate series
        if series and len(series) == length:
            # Apply domain-specific constraints
            domain_ranges = {
                "finance": (30, 360),
                "iot": (20, 160),
                "weather": (5, 65),
                "healthcare": (45, 325),
                "technology": (15, 220),
                "retail": (35, 270)
            }
            
            min_val, max_val = domain_ranges.get(domain, (50, 150))
            series = np.clip(series, min_val, max_val)
            
            return series.tolist()
        
    except Exception as e:
        logger.warning(f"LLM generation failed: {e}")
    
    # Fallback to baseline
    return generate_enhanced_fallback_series(text, length, domain, freq)

def llm_generate_self_consistent(
    text: str,
    domain: str,
    freq: str,
    length: int,
    model: str = "Llama-4-Maverick-17B-128E-Instruct-FP8",
    k_samples: int = 10,
    temperature: float = 0.6,
    seed: int = 0
) -> Dict[str, Any]:
    """Generate multiple samples and select best via self-consistency voting."""
    
    samples = []
    for i in range(k_samples):
        try:
            sample = llm_generate_enhanced(
                text=text,
                domain=domain,
                freq=freq,
                length=length,
                model_id=model,
                temperature=temperature,
                seed=seed + i if seed > 0 else 0
            )
            samples.append(sample)
        except Exception as e:
            logger.warning(f"Sample {i} generation failed: {e}")
            continue
    
    if not samples:
        logger.error("No valid samples generated")
        return {
            "series": generate_enhanced_fallback_series(text, length, domain, freq),
            "variance": float('inf'),
            "num_samples": 0
        }
    
    # Calculate variance across samples
    samples_array = np.array(samples)
    variance = np.var(samples_array, axis=0).mean()
    
    # For now, return the first valid sample
    # In a more sophisticated implementation, you could implement
    # majority voting or clustering-based selection
    selected_series = samples[0]
    
    return {
        "series": selected_series,
        "variance": variance,
        "num_samples": len(samples),
        "all_samples": samples
    }

def test_llm_connection() -> bool:
    """Test LLM API connection."""
    try:
        client = get_llm_client()
        models = client.models.list()
        logger.info(f"Available models: {[m.id for m in models.data[:5]]}")
        return True
    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return False
