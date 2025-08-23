# NL2TS-675: Natural Language to Time Series Dataset

**Paper**: [Zero-to-Forecast: Natural Language to Time Series Prediction via Cross-Modal Ensembles](https://github.com/gokulsrinaths/BERT2S)

## 📊 Dataset Overview

**NL2TS-675** is a comprehensive benchmark dataset for natural language to time series prediction, containing **675 description–series pairs** across diverse domains and temporal patterns.

## 🎯 Dataset Statistics

- **Total Pairs**: 675
- **Domains**: 6 (finance, healthcare, weather, IoT, retail, technology)
- **Temporal Horizons**: 3 (12, 24, 48 time steps)
- **Pattern Types**: 5 (trend, seasonal, spike, plateau, irregular)
- **Data Split**: 70% train (472), 15% dev (101), 15% test (102)

## 🏢 Domain Coverage

| Domain | Train | Dev | Test | Value Range | Description |
|--------|-------|-----|------|-------------|-------------|
| **Finance** | 121 | 27 | 32 | 30-360 | Stock prices, revenue, trading volume |
| **Healthcare** | 70 | 9 | 11 | 45-325 | Vital signs, patient metrics |
| **Weather** | 53 | 20 | 17 | 5-65 | Temperature, humidity, atmospheric data |
| **IoT** | 66 | 13 | 11 | 20-160 | Sensor readings, device metrics |
| **Technology** | 66 | 11 | 13 | 15-220 | System performance, network metrics |
| **Retail** | 96 | 21 | 18 | 35-270 | Sales data, inventory levels |

## 📈 Temporal Patterns

### Pattern Types
1. **Trend**: Monotonic increase/decrease patterns
2. **Seasonal**: Cyclical and periodic variations  
3. **Spike**: Sudden peaks and impulse responses
4. **Plateau**: Stable periods with minimal variation
5. **Irregular**: Complex, non-linear patterns

### Temporal Horizons
- **Short-term**: 12 time steps
- **Medium-term**: 24 time steps
- **Long-term**: 48 time steps

## 📁 Data Format

Each item in the dataset follows this JSON structure:

```json
{
  "uid": "domain_pattern_freq_length_xxx",
  "domain": "finance",
  "text": "Stock price shows gradual increase over time",
  "series": [120.5, 125.2, 128.7, 132.1, 135.6, 138.9, 142.3, 145.1, 148.4, 151.2, 154.8, 157.3],
  "freq": "hour",
  "length": 12,
  "split": "train"
}
```

### Field Descriptions
- **uid**: Unique identifier with domain, pattern, frequency, length, and sequence number
- **domain**: One of 6 domains (finance, healthcare, weather, iot, technology, retail)
- **text**: Natural language description of the time series pattern
- **series**: Numerical time series as a list of floats
- **freq**: Temporal frequency (hour, day, week, month)
- **length**: Number of time steps (12, 24, or 48)
- **split**: Data split (train, dev, test)

## 🔬 Quality Controls

### Generation Protocol
- **Text Descriptions**: Human-authored using procedural templates
- **Time Series**: Deterministic procedural algorithms with domain-specific constraints
- **Leakage Prevention**: Strict train/dev/test splits with no overlap
- **Domain Balance**: Representative distribution across all domains

### Validation
- Domain-appropriate value ranges
- Realistic pattern generation
- No exact duplicates across splits
- Balanced representation across patterns and lengths

## 📊 Example Items

### Finance Domain
```json
{
  "uid": "finance_trend_up_gradual_12_001",
  "domain": "finance",
  "text": "Stock price shows gradual increase over time",
  "series": [120.5, 125.2, 128.7, 132.1, 135.6, 138.9, 142.3, 145.1, 148.4, 151.2, 154.8, 157.3],
  "freq": "hour",
  "length": 12,
  "split": "train"
}
```

### Healthcare Domain
```json
{
  "uid": "healthcare_spike_recovery_24_001",
  "domain": "healthcare",
  "text": "Heart rate increases during exercise then recovers",
  "series": [72.1, 75.3, 78.9, 82.4, 85.7, 88.2, 90.5, 87.3, 84.1, 80.8, 77.5, 74.2, 71.8, 70.2, 69.1, 68.5, 67.9, 67.2, 66.8, 66.5, 66.1, 65.8, 65.5, 65.2],
  "freq": "minute",
  "length": 24,
  "split": "train"
}
```

### Weather Domain
```json
{
  "uid": "weather_seasonal_daily_48_001",
  "domain": "weather",
  "text": "Temperature shows seasonal variation",
  "series": [15.2, 18.7, 22.1, 25.8, 28.4, 30.1, 29.8, 27.3, 24.1, 20.5, 17.2, 14.8, 12.5, 11.2, 10.8, 11.5, 13.2, 15.8, 18.9, 22.4, 25.9, 28.7, 30.5, 29.9, 27.1, 23.8, 20.1, 16.7, 14.2, 12.8, 12.1, 12.9, 14.7, 17.3, 20.8, 24.2, 27.6, 29.8, 30.9, 30.2, 28.1, 25.3, 21.9, 18.4, 15.6, 13.5, 12.7, 13.4],
  "freq": "day",
  "length": 48,
  "split": "train"
}
```

## 🚀 Usage

### Loading the Dataset

```python
import json

def load_nl2ts_675(filepath="nl2ts_675.jsonl"):
    """Load NL2TS-675 dataset."""
    items = []
    with open(filepath, 'r') as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items

# Load dataset
dataset = load_nl2ts_675()

# Filter by split
train_items = [item for item in dataset if item['split'] == 'train']
dev_items = [item for item in dataset if item['split'] == 'dev']
test_items = [item for item in dataset if item['split'] == 'test']

# Filter by domain
finance_items = [item for item in dataset if item['domain'] == 'finance']

# Filter by pattern length
short_series = [item for item in dataset if item['length'] == 12]
```

### Basic Statistics

```python
# Domain distribution
from collections import Counter
domains = Counter([item['domain'] for item in dataset])
print("Domain distribution:", dict(domains))

# Length distribution
lengths = Counter([item['length'] for item in dataset])
print("Length distribution:", dict(lengths))

# Split distribution
splits = Counter([item['split'] for item in dataset])
print("Split distribution:", dict(splits))
```

## 📈 Evaluation Metrics

The dataset is designed for evaluation using multiple metrics:

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **Pearson r**: Linear correlation coefficient
- **Spearman ρ**: Rank correlation coefficient
- **DTW**: Dynamic Time Warping distance
- **Trend F1**: F1 score for trend direction prediction

## 📄 Citation

If you use this dataset, please cite:

```bibtex
@article{zero-to-forecast-2024,
  title={Zero-to-Forecast: Natural Language to Time Series Prediction via Cross-Modal Ensembles},
  author={Zero-to-Forecast Team},
  journal={NeurIPS BERT2S Workshop},
  year={2024}
}
```

## 🔗 Related Resources

- **Main Repository**: [https://github.com/gokulsrinaths/BERT2S](https://github.com/gokulsrinaths/BERT2S)
- **Paper**: Zero-to-Forecast: Natural Language to Time Series Prediction via Cross-Modal Ensembles
- **Interactive Demo**: Available in the main repository

## 📞 Contact

For questions about this dataset, please open an issue on the main repository: [https://github.com/gokulsrinaths/BERT2S](https://github.com/gokulsrinaths/BERT2S)

---

**Dataset License**: MIT License  
**Paper**: NeurIPS BERT2S Workshop 2024
