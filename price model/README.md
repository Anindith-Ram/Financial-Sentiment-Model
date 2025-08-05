# 🚀 Professional Financial Modeling Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pipeline Status](https://img.shields.io/badge/pipeline-production--ready-green.svg)](.)

**Professional-grade financial modeling system for stock price prediction and trading signals using advanced neural networks.**

## ✨ Key Features

- 🎯 **Advanced Neural Architecture**: Hybrid GPT-2 + Enhanced CNN
- 📊 **Smart Data Management**: Optimized collection with automatic NaN handling  
- 🛡️ **Production-Ready**: Comprehensive error handling and monitoring
- ⚡ **Performance Optimized**: Mixed precision, gradient accumulation
- 🔧 **Professional Structure**: Clean separation of concerns, no duplication
- 📈 **Trading Signals**: Real-time signal generation
- 🧪 **Research Tools**: Hyperparameter search and analysis

## 🏆 Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| **Model Accuracy** | Training/Validation | 51.5% / 51.6% |
| **Data Quality** | Clean Data Retention | 89% (11.2% NaN removed) |
| **Training Speed** | Throughput | 34 batches/second |
| **Model Size** | Parameters | 130M (6.5M trainable) |

## 🚀 Quick Start

### Installation
```bash
git clone <repository-url>
cd price-model
pip install -r requirements.txt
```

### Basic Usage
```bash
# 1. Collect data
python -m finmodel.core.pipeline data collect --mode smart_update

# 2. Train model (research mode)
python -m finmodel.core.pipeline train --mode research --epochs 10

# 3. Train production model
python -m finmodel.core.pipeline train --mode production --epochs 50

# 4. Generate predictions
python -m finmodel.core.pipeline predict --input data/new_data.csv

# 5. Generate trading signals
python -m finmodel.core.pipeline signals --live

# 6. Check status
python -m finmodel.core.pipeline status
```

## 📁 Professional Architecture

```
finmodel/                          # Main package
├── core/pipeline.py              # Single entry point
├── data/manager.py               # Unified data operations
├── models/hybrid_model.py        # GPT-2 + CNN architecture
├── training/orchestrator.py      # Unified training system
├── inference/engine.py           # Prediction system
└── utils/                        # Error handling, logging
```

**Key Design Principles:**
- ✅ **Single Entry Point**: All operations through `pipeline.py`
- ✅ **No Code Duplication**: Each component has single responsibility
- ✅ **Professional Error Handling**: Comprehensive recovery and logging
- ✅ **Performance Optimized**: Advanced training techniques
- ✅ **Production Ready**: Robust, scalable, maintainable

## 🎯 Model Architecture

### Hybrid Neural Network
- **GPT-2 Temporal Extractor**: 127M parameters for sequence understanding
- **Enhanced CNN**: 788K parameters for pattern recognition
- **Attention Fusion**: 361K parameters for feature combination
- **5-Day Context**: Optimized for swing trading (1-3 day predictions)

### Advanced Training Features
- **Mixed Precision Training**: 2x speedup with minimal memory overhead
- **Component-Specific Learning Rates**: Optimized for each model part
- **Advanced Regularization**: MixUp, CutMix, Label Smoothing
- **Automatic Hyperparameter Search**: Optuna-based optimization

## 📊 Data Features

### Technical Indicators (62 features per 5-day sequence)
- **Trend**: SMA, EMA, trend strength, crossovers
- **Momentum**: RSI, MACD, Stochastic, Williams %R
- **Volatility**: ATR, Bollinger Bands
- **Volume**: Volume ratios, OBV
- **Patterns**: 4 key candlestick patterns
- **Price Action**: Returns, gaps, ratios

### Smart Data Processing
- **Adaptive Indicators**: Periods adjust to data availability
- **Quality Scoring**: Automatic assessment and filtering
- **NaN Elimination**: 11.2% → 0% through intelligent preprocessing
- **Hierarchical Calculation**: Dependencies resolved automatically

## 🔧 Configuration

### Training Modes
```bash
# Research mode - fast iterations
python -m finmodel.core.pipeline train --mode research --epochs 10

# Production mode - full optimization
python -m finmodel.core.pipeline train --mode production --epochs 100

# Hyperparameter search
python -m finmodel.core.pipeline train --mode hyperparameter_search
```

### Python API
```python
from finmodel import FinancialPipeline

with FinancialPipeline() as pipeline:
    # Collect data
    pipeline.collect_data(mode="smart_update")
    
    # Train model
    pipeline.train_model(mode="production", epochs=50)
    
    # Generate predictions
    predictions = pipeline.predict("data/test.csv")
    
    # Trading signals
    signals = pipeline.generate_signals(live=True)
```

## 🛡️ Error Handling & Monitoring

### Professional Error Management
- **Structured Exception Hierarchy**: Specific error types
- **Automatic Recovery**: Retry with exponential backoff
- **Detailed Logging**: JSON-structured logs with context
- **Performance Monitoring**: Real-time metrics
- **User-Friendly Messages**: Clear descriptions and suggestions

### System Monitoring
```bash
# Check system status
python -m finmodel.core.pipeline status

# View errors with suggestions
python -m finmodel.core.pipeline logs --errors
```

## 📈 Results Format

### Predictions
```python
{
  "predictions": [0, 1, 2, 1, 0],
  "confidences": [0.89, 0.76, 0.82, 0.91, 0.73],
  "metadata": {
    "model_version": "v3.0.0",
    "processing_time_ms": 234
  }
}
```

### Trading Signals
```python
{
  "signals": {
    "AAPL": {
      "signal": "BUY",
      "confidence": 0.87,
      "target_price": 185.50,
      "expected_return": 4.07
    }
  }
}
```

## 🚀 Production Features

- **Model Versioning**: Automatic checkpoint management
- **Performance Profiling**: Built-in optimization tools
- **Batch Processing**: Efficient large-scale inference
- **Real-time Serving**: Low-latency prediction API
- **Automated Retraining**: Scheduled model updates

## 🔬 Research Tools

- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Ablation Studies**: Component importance analysis
- **Model Interpretability**: Feature importance and explanations
- **Performance Benchmarking**: Comprehensive model comparisons

## 📄 Support

### Getting Help
1. **System Status**: `python -m finmodel.core.pipeline status`
2. **Logs**: Check `logs/` directory for details
3. **Error Messages**: Include troubleshooting suggestions

### Common Solutions
```bash
# Memory issues
--batch-size 32 --gradient-accumulation-steps 2

# CUDA memory
--mixed-precision --batch-size 16

# Data quality
python -m finmodel.core.pipeline data validate
```

---

**🚀 Professional Financial Modeling Pipeline v3.0**  
*Production-Ready • Performance-Optimized • Professionally-Structured*

Ready to start? Run `python -m finmodel.core.pipeline status`