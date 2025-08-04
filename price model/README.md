# 🚀 Financial Sentiment Model

A professional machine learning pipeline for swing trading using candlestick patterns and technical indicators with integrated data quality improvements.

## 📚 Documentation

**📖 [Complete Documentation](docs/README.md)** - Comprehensive guide with all features, usage examples, and troubleshooting.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Collect data with quality improvements
python main.py --mode data

# Train model
python main.py --mode train

# Get trading signals
python main.py --mode signals --portfolio AAPL
```

## 📁 Project Structure

```
price model/
├── main.py                      # 🎯 Main entry point
├── src/                         # 📦 Source code
│   ├── data/                   # 📊 Data collection
│   ├── models/                 # 🧠 Model definitions
│   ├── training/               # 🧠 Training scripts
│   ├── inference/              # 🔮 Prediction
│   └── utils/                  # 🛠️ Utilities
├── tests/                      # 🧪 Test files
├── config/                     # ⚙️ Configuration
├── data/                       # 📊 Data files
├── models/                     # 🧠 Model files
├── logs/                       # 📝 Training logs
└── docs/                       # 📚 Documentation
    └── README.md              # 📖 Complete documentation
```

## 🎯 Key Features

- **Swing Trading Focus**: 1-3 day holding periods
- **Smart Data Collection**: Incremental updates with quality fixes
- **Advanced AI Model**: CNN with attention mechanism
- **Trading Signals**: Real-time predictions with risk management
- **Data Quality**: Automatic NaN handling, outlier detection, class balancing

## 📄 License

This project is for educational purposes. Use trading signals at your own risk.

---

**📖 [View Complete Documentation](docs/README.md)** 