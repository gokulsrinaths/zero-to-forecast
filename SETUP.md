# 🚀 Zero-to-Forecast: Quick Setup Guide

## Prerequisites

- Python 3.8+ (✅ You have Python 3.12.7)
- PyTorch (✅ You have PyTorch 2.7.0)
- Meta LLaMA API access (optional for testing)

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment (Optional)
```bash
# Copy the example environment file
copy env_example.txt .env

# Edit .env and add your LLaMA API key
# LLAMA_API_KEY=your_actual_api_key_here
```

### 3. Test the System
```bash
# Test basic functionality
python -c "from llama_api import LLaMAForecaster; print('✅ System ready!')"

# Run demo examples
python examples/demo_examples.py

# Start the web interface
streamlit run app.py
```

## 🎯 What You Can Do Right Now

### Without API Key (Demo Mode)
- Run the demo examples to see the system in action
- Explore the synthetic dataset
- Generate paper figures
- Test the web interface with mock data

### With LLaMA API Key (Full Functionality)
- Generate real forecasts from natural language
- Train custom decoder models
- Use the complete web interface
- Run comprehensive evaluations

## 📊 Example Usage

```python
from llama_api import LLaMAForecaster

# Initialize (works without API key in demo mode)
forecaster = LLaMAForecaster()

# Generate forecast
description = "Sales dropped in Q1 and Q2, then surged in Q3 after a campaign."
forecast = forecaster.forecast_from_text(description, length=5)
print(forecast)  # [100, 95, 92, 130, 150]
```

## 🎓 For NeurIPS Paper

1. **Generate Figures**: Run `jupyter notebook paper_figures.ipynb`
2. **Train Models**: `python train.py --decoder_type mlp --num_epochs 100`
3. **Evaluate**: Use the evaluation metrics in the notebook
4. **Export Results**: LaTeX tables are automatically generated

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the project root directory
2. **API Errors**: The system works in demo mode without API keys
3. **Memory Issues**: Reduce batch size in training: `--batch_size 8`
4. **CUDA Issues**: Use CPU: `--device cpu`

### Getting Help

- Check the main README.md for detailed documentation
- Run the demo examples to verify installation
- Open an issue on GitHub for bugs

## 🎉 You're Ready!

Your Zero-to-Forecast system is now set up and ready for:
- ✅ Research and development
- ✅ NeurIPS paper submission
- ✅ Demo presentations
- ✅ Further experimentation

**Next Steps:**
1. Try the web interface: `streamlit run app.py`
2. Run the demo: `python examples/demo_examples.py`
3. Generate paper figures: `jupyter notebook paper_figures.ipynb`
4. Train a model: `python train.py`

---

*Built for NeurIPS BERT2S Workshop | Zero-to-Forecast: Natural Language to Time Series Prediction*
