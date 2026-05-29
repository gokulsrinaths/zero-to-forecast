# NL2TS-675 — Natural Language to Time Series Benchmark Dataset

<div align="center">

**📄 Accepted at NeurIPS 2025 Workshop on Recent Advances in Time Series Foundation Models**
*"Have We Reached the BERT Moment?"*

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025_Workshop-purple)](https://neurips.cc/virtual/2025/loc/san-diego/130445)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/forum?id=lERfNeDzul)
[![Dataset](https://img.shields.io/badge/Dataset-675_pairs-orange)](nl2ts_675.jsonl)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Gokul Srinath Seetha Ram**
California State Polytechnic University, Pomona
`gseetharam@cpp.edu` · `s.gokulsrinath@gmail.com`

</div>

---

## Overview

**NL2TS-675** is the benchmark dataset introduced alongside the paper:

> **Zero-to-Forecast: Natural Language to Time Series Prediction via Cross-Modal Ensembles**
> Gokul Srinath Seetha Ram — *NeurIPS 2025 Workshop: Recent Advances in Time Series Foundation Models (BERT2S)*
> San Diego · December 7, 2025 · 3:45 PM – 4:45 PM PST
> [NeurIPS listing →](https://neurips.cc/virtual/2025/loc/san-diego/130445)

The dataset contains **675 natural language description–time series pairs** spanning six real-world domains and five canonical temporal patterns, designed specifically to benchmark the emerging task of **zero-data NL→TS forecasting** — generating a numeric time series directly from a free-form text description, without any historical data.

---

## The Task

Traditional forecasting requires structured numerical histories. NL2TS-675 formalises a different paradigm:

```
Input:  "Stock price shows a gradual increase over the quarter,
         driven by strong earnings guidance."

Output: [120.5, 125.2, 128.7, 132.1, 135.6, 138.9, 142.3, ...]
```

This "zero-data" setting is valuable when historical series are unavailable, for rapid scenario planning, or for evaluating how well models ground language in quantitative structure.

---

## Dataset Statistics

| Property | Value |
|----------|-------|
| **Total pairs** | 675 |
| **Domains** | 6 |
| **Temporal horizons** | 3 (12, 24, 48 time steps) |
| **Pattern types** | 5 |
| **Train split** | 472 (70%) |
| **Dev split** | 101 (15%) |
| **Test split** | 102 (15%) |

---

## Domain Coverage

| Domain | Train | Dev | Test | Value Range | Content |
|--------|-------|-----|------|-------------|---------|
| **Finance** | 121 | 27 | 32 | 30–360 | Stock prices, revenue, trading volume |
| **Healthcare** | 70 | 9 | 11 | 45–325 | Vital signs, patient metrics |
| **Weather** | 53 | 20 | 17 | 5–65 | Temperature, humidity, atmospheric data |
| **IoT** | 66 | 13 | 11 | 20–160 | Sensor readings, device metrics |
| **Technology** | 66 | 11 | 13 | 15–220 | System performance, network metrics |
| **Retail** | 96 | 21 | 18 | 35–270 | Sales data, inventory levels |

---

## Pattern Types

| Pattern | Description |
|---------|-------------|
| **Trend** | Monotonic increase or decrease |
| **Seasonal** | Cyclical and periodic variations |
| **Spike** | Sudden peaks and impulse responses |
| **Plateau** | Stable periods with minimal variation |
| **Irregular** | Complex, non-linear patterns |

Temporal horizons: **short (12)**, **medium (24)**, **long (48)** time steps.

---

## Evaluation Results (Zero-to-Forecast)

The paper's advanced domain-optimised ensemble achieves the following on the NL2TS-675 test set:

| Metric | Overall | Finance | Healthcare | Weather | IoT | Technology | Retail |
|--------|---------|---------|-----------|---------|-----|-----------|--------|
| **MAE ↓** | **16.06** | 18.29 | 10.64 | 15.04 | 16.21 | 15.11 | 16.91 |

All six domains achieve MAE < 25. Healthcare is the strongest domain (10.64); Finance is most challenging due to high volatility and sparse shock specification.

Additional metrics evaluated: MSE, Dynamic Time Warping (DTW), Pearson r, Spearman ρ, Trend-F1.

---

## Data Format

Each line in `nl2ts_675.jsonl` is a JSON object:

```json
{
  "uid":    "finance_trend_up_gradual_12_001",
  "domain": "finance",
  "text":   "Stock price shows gradual increase over time",
  "series": [120.5, 125.2, 128.7, 132.1, 135.6, 138.9, 142.3, 145.1, 148.4, 151.2, 154.8, 157.3],
  "freq":   "hour",
  "length": 12,
  "split":  "train"
}
```

| Field | Description |
|-------|-------------|
| `uid` | Unique ID — `domain_pattern_freq_length_index` |
| `domain` | One of: `finance`, `healthcare`, `weather`, `iot`, `technology`, `retail` |
| `text` | Natural language description of the time series |
| `series` | Ground-truth numeric time series (list of floats) |
| `freq` | Temporal frequency: `hour`, `day`, `week`, `month` |
| `length` | Sequence length: `12`, `24`, or `48` |
| `split` | `train`, `dev`, or `test` |

---

## Usage

### Load the dataset

```python
import json

def load_nl2ts_675(filepath="nl2ts_675.jsonl"):
    with open(filepath) as f:
        return [json.loads(line) for line in f if line.strip()]

dataset = load_nl2ts_675()

# Filter by split
train = [x for x in dataset if x["split"] == "train"]
test  = [x for x in dataset if x["split"] == "test"]

# Filter by domain
finance = [x for x in dataset if x["domain"] == "finance"]

# Filter by horizon
short_term = [x for x in dataset if x["length"] == 12]
```

### Evaluate MAE

```python
import numpy as np

def mae(predictions: list, targets: list) -> float:
    return np.mean([
        np.mean(np.abs(np.array(p) - np.array(t)))
        for p, t in zip(predictions, targets)
    ])

# predictions: list of lists (one per test item)
test_series = [x["series"] for x in test]
score = mae(your_predictions, test_series)
print(f"MAE: {score:.2f}")
```

### Domain statistics

```python
from collections import Counter

print("Domain distribution:", dict(Counter(x["domain"] for x in dataset)))
print("Length distribution:", dict(Counter(x["length"] for x in dataset)))
print("Pattern distribution:", dict(Counter(x["uid"].split("_")[1] for x in dataset)))
```

---

## Quality Controls

- **Text descriptions** — human-authored using procedural templates with domain-specific language
- **Time series** — generated by deterministic algorithms with domain-appropriate value constraints
- **Leakage prevention** — strict train/dev/test splits with no overlap
- **Deduplication** — no exact duplicates across splits
- **Balance** — representative distribution across domains, patterns, and horizons

---

## Paper Abstract

> We introduce Zero-to-Forecast, a cross-modal AI framework that converts natural language descriptions into numerical time series predictions. Our approach unifies large language model reasoning with domain- and pattern-specific predictors and post-hoc calibration (domain-aware smoothing, monotonic constraints), yielding robust, realistic sequences from free-form text. On the NL2TS-675 benchmark spanning six domains, our advanced domain-optimized ensemble achieves overall MAE 16.06 with all domains < 25 MAE, substantially improving over strong baselines. We release code, artifacts, and a live interactive demo, positioning natural language-driven forecasting as a practical paradigm for zero-data scenario planning.

---

## Citation

If you use NL2TS-675 in your research, please cite:

```bibtex
@inproceedings{ram2025zerotoforecast,
  title     = {Zero-to-Forecast: Natural Language to Time Series Prediction via Cross-Modal Ensembles},
  author    = {Gokul Srinath Seetha Ram},
  booktitle = {Recent Advances in Time Series Foundation Models Have We Reached the 'BERT Moment'?},
  year      = {2025},
  url       = {https://openreview.net/forum?id=lERfNeDzul}
}
```

---

## Related Resources

| Resource | Link |
|----------|------|
| **Paper (OpenReview)** | [openreview.net/forum?id=lERfNeDzul →](https://openreview.net/forum?id=lERfNeDzul) |
| **NeurIPS 2025 listing** | [neurips.cc →](https://neurips.cc/virtual/2025/loc/san-diego/130445) |
| **Main code repository** | [github.com/gokulsrinaths/NL2TS-675](https://github.com/gokulsrinaths/NL2TS-675) |

---

## License

MIT — dataset and code are free to use for research purposes.

---

<div align="center">

*NeurIPS 2025 Workshop — Recent Advances in Time Series Foundation Models*
*San Diego · December 7, 2025*

**Gokul Srinath Seetha Ram**
California State Polytechnic University, Pomona

</div>
