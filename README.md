# 🚀 Financial Analysis Model Suite

A comprehensive machine learning pipeline combining **sentiment analysis** and **technical analysis** for robust financial decision-making and swing trading.

## 📊 Project Overview

This project combines two complementary models for comprehensive financial analysis:

### 🧠 **Sentiment Model** (`sentiment model/`)
- **Unified investor sentiment signals** from multiple data sources
- **Kaggle Twitter dataset** as core foundation
- **FinancialBERT sentiment analysis** for news headlines
- **Google Trends alignment** for interest tracking
- **Machine learning training** and backtesting capabilities

### 📈 **Price Model** (`price model/`)
- **Candlestick pattern recognition** for technical analysis
- **Optimized for 17.1 GB RAM** with memory management
- **Progressive training** with 4-stage optimization
- **Real-time trading signals** with risk management
- **Swing trading focus** (1-3 day holding periods)

## 🎯 Quick Start

### Option 1: Interactive Launcher (Recommended)
```bash
cd "price model"
python launch.py
```

### Option 2: Direct Commands
```bash
# Build dataset
python main.py --mode data

# Progressive training (optimized)
python main.py --mode progressive --experiment-name my_experiment

# Predict patterns
python main.py --mode predict --ticker AAPL

# Generate trading signals
python main.py --mode signals --portfolio AAPL MSFT GOOGL

# Complete demo
python main.py --mode demo
```

## 📊 Optimized Configuration

| Stage | Batch Size | Memory Usage | Improvement |
|-------|------------|--------------|-------------|
| **Stage 1** | **512** | 0.06 GB | 2x larger |
| **Stage 2** | **256** | 0.05 GB | 4x larger |
| **Stage 3** | **128** | 0.05 GB | 4x larger |
| **Stage 4** | **64** | 0.05 GB | 4x larger |

## 🛠️ Available Modes

### `--mode data`
- Build and prepare dataset
- Memory-optimized data loading
- Automatic feature normalization

### `--mode train`
- Standard single-stage training
- Optimized batch size (512)
- Enhanced logging and monitoring

### `--mode progressive`
- 4-stage progressive training
- Optimized batch sizes for each stage
- Memory management between stages
- Automatic model checkpointing

### `--mode predict`
- Pattern prediction for any ticker
- Automatic model selection
- Confidence scoring

### `--mode signals`
- Generate trading signals for portfolios
- Multi-stock analysis
- Signal strength ranking

### `--mode demo`
- Complete pipeline demonstration
- Data → Training → Prediction → Signals

### `--mode all`
- Run complete pipeline
- End-to-end processing

## 🧠 Sentiment Model Features

### **Data Sources**
- **Kaggle Twitter Dataset**: Pre-existing high-quality Twitter sentiment data (core)
- **NewsAPI**: Real-time news headlines with FinancialBERT sentiment analysis
- **Google Trends**: Interest scores aligned with Twitter dataset dates/tickers
- **Market Data**: Stock prices for target variable creation

### **FinancialBERT Integration**
- Domain-specific sentiment analysis for financial text
- Model: `ahmedrachid/FinancialBERT-Sentiment-Analysis`
- Outputs: sentiment scores (-1 to 1), labels, and confidence scores

### **Dataset Alignment**
- All data sources aligned by date and ticker
- Kaggle dataset serves as the foundation
- Supporting features appended without disrupting original schema

### **Feature Engineering**
- Lagged features (1, 3, 7 days)
- Moving averages (3, 7, 14 days) 
- Momentum indicators
- Volatility measures

## 📈 Price Model Features

### **Swing Trading Focus**
- **1-3 day holding periods** - Perfect for swing trading
- **65 features per timestep** - Comprehensive technical analysis
- **Binary classification** - Buy/Sell signals with confidence scores
- **Risk management** - Stop loss and take profit recommendations

### **Smart Data Collection**
- **Incremental updates** - Only collect recent data for existing tickers
- **Graceful interruption** - Press Ctrl+C to stop and save progress
- **Backup protection** - Automatic backup files during processing
- **S&P 500 coverage** - 500 stocks with quality filtering
- **Integrated quality fixes** - Automatic NaN handling, outlier detection, class balancing

### **Advanced AI Model**
- **CNN architecture** - Deep learning for pattern recognition
- **Attention mechanism** - Focus on important features
- **Residual connections** - Better gradient flow
- **Progressive training** - Two-stage training for better performance

### **Trading Signals**
- **Real-time predictions** - Get BUY/SELL signals for any stock
- **Technical confirmation** - RSI, MACD, moving averages
- **Entry/Exit points** - Optimal prices with stop loss/take profit
- **Portfolio analysis** - Compare multiple stocks at once

## 📁 Project Structure

```
Financial Sentiment Model/
├── README.md                    # 📖 This comprehensive guide
├── sentiment model/             # 🧠 Sentiment analysis pipeline
│   ├── README.md               # 📖 Sentiment model documentation
│   ├── src/
│   │   ├── data/              # 📊 Data ingestion and processing
│   │   ├── features/          # 🔧 Normalization + signal fusion
│   │   ├── models/            # 🧠 ML training + evaluation
│   │   └── backtest/          # 📈 Backtesting strategies
│   ├── data/
│   │   ├── raw/               # 📊 Raw data (news, trends, market)
│   │   └── processed/         # 📊 Structured sentiment scores
│   └── configs/               # ⚙️ Configuration files
├── price model/                # 📈 Technical analysis pipeline
│   ├── main.py                # 🎯 Main pipeline script
│   ├── launch.py              # 🚀 Interactive launcher
│   ├── memory_optimization.py # 💾 Memory analysis tool
│   ├── README_STREAMLINED.md  # 📖 Streamlined documentation
│   ├── src/
│   │   ├── training/
│   │   │   ├── train.py      # 🧠 Streamlined training
│   │   │   └── progressive_trainer.py # 🚀 Optimized progressive training
│   │   ├── models/
│   │   │   └── dataset.py    # 💾 Memory-optimized data loading
│   │   └── ...
│   ├── config/
│   │   └── config.json       # ⚙️ Optimized configuration
│   ├── models/                # 🧠 Saved models
│   ├── logs/                  # 📝 Training logs
│   └── data/                  # 📊 Dataset files
└── docs/                      # 📚 Additional documentation
```

## 🎮 Usage Examples

### Interactive Menu
```bash
cd "price model"
python launch.py
# Choose from menu options
```

### Command Line
```bash
# Quick progressive training
python main.py --mode progressive

# Custom experiment
python main.py --mode progressive --experiment-name my_experiment

# Predict specific stock
python main.py --mode predict --ticker TSLA

# Portfolio analysis
python main.py --mode signals --portfolio AAPL MSFT GOOGL TSLA AMZN
```

### Sentiment Model Pipeline
```bash
cd "sentiment model"

# Complete pipeline
python src/data/backfill_historical_new.py \
    --kaggle_path path/to/kaggle_twitter_sentiment.csv \
    --tickers AAPL MSFT GOOGL TSLA \
    --since 2023-01-01 \
    --until 2023-12-31

# Individual components
python src/data/ingest_kaggle_twitter.py --kaggle_path dataset.csv
python src/data/ingest_news_financialbert.py --tickers AAPL TSLA
python src/data/ingest_trends_aligned.py --tickers AAPL TSLA
```

## ⚡ Performance Benefits

### **Price Model Optimizations**
- **2-4x faster training** with larger batch sizes
- **Better GPU utilization** with optimized memory usage
- **Stable convergence** with more stable gradients
- **Memory efficient** using only 0.6% of available RAM
- **Automatic cleanup** between training stages

### **Sentiment Model Features**
- **No API Rate Limits**: Uses pre-existing Twitter data
- **Domain-Specific Sentiment**: FinancialBERT for financial text
- **Reliable Data Sources**: Focused on News + Trends
- **Modular Design**: Run components independently
- **Feature Rich**: Comprehensive sentiment signals
- **Maintainable**: Clean, documented codebase

## 🔧 Memory Optimization Features

- ✅ **Explicit data types** (float32) to prevent object arrays
- ✅ **Memory management** between training stages
- ✅ **Garbage collection** and CUDA cache clearing
- ✅ **Cached data loaders** to prevent multiple instances
- ✅ **Optimized batch sizes** for 17.1 GB RAM
- ✅ **Robust error handling** and graceful failures

## 📈 Expected Results

### **Price Model**
- **Training Speed**: 2-4x faster with optimized batch sizes
- **Memory Usage**: Only 0.6% of available RAM per stage
- **Model Quality**: Better convergence with larger batches
- **Reliability**: Robust error handling and recovery

### **Sentiment Model**
- **Unified Signals**: Combined sentiment from multiple sources
- **Domain Expertise**: FinancialBERT for financial text analysis
- **Feature Rich**: Comprehensive sentiment indicators
- **Scalable**: Modular design for easy expansion

## 🚨 Important Notes

### **Data Collection**
- **Internet required**: Downloads data from various sources
- **Time intensive**: Processing takes 2-4 hours for full datasets
- **Graceful interruption**: Press Ctrl+C to stop safely
- **Backup protection**: Automatic backup files created
- **Quality improvements**: Applied automatically by default

### **Model Training**
- **GPU recommended**: CUDA support for faster training
- **Memory intensive**: 16GB+ RAM recommended
- **Patience required**: Training takes 1-2 hours
- **Early stopping**: Prevents overfitting

### **Trading Signals**
- **Not financial advice**: Use at your own risk
- **Backtesting recommended**: Test on historical data
- **Risk management**: Always use stop losses
- **Diversification**: Don't put all money in one signal

## 🔍 Troubleshooting

### **Common Issues**

**Data Collection Fails:**
```bash
# Check internet connection
# Verify API keys (if using paid data)
# Try incremental mode
python main.py --mode data --incremental
```

**Training Stuck:**
```bash
# Reduce learning rate
# Increase patience
# Check GPU memory
python main.py --mode train --epochs 30
```

**Low Accuracy:**
```bash
# Try progressive training
# Increase model capacity
# Check data quality
python main.py --mode train --epochs 75
```

### **Performance Optimization**

**For Faster Training:**
- Use GPU with CUDA
- Increase batch size (if memory allows)
- Reduce number of tickers for testing

**For Better Accuracy:**
- Use progressive training
- Increase training epochs
- Collect more recent data
- Use quality-improved dataset

## 📚 References

- **Technical Analysis**: TA-Lib library
- **Data Source**: Yahoo Finance via yfinance
- **Deep Learning**: PyTorch framework
- **S&P 500 Data**: DataHub constituents
- **FinancialBERT**: Domain-specific sentiment analysis
- **Kaggle Dataset**: Twitter sentiment for financial markets

## 🔄 Recent Updates

### **Pipeline Organization (Latest)**
- ✅ **Reorganized Structure**: Moved scripts to appropriate folders
- ✅ **Eliminated Redundancy**: Removed duplicate files and scripts
- ✅ **Documentation Consolidation**: All docs organized properly
- ✅ **Utility Scripts**: Organized utility functions
- ✅ **Quality Integration**: Data quality fixes integrated into main pipeline

### **Memory Optimizations**
- ✅ **Optimized Batch Sizes**: 512, 256, 128, 64 for 17.1 GB RAM
- ✅ **Memory Management**: Between training stages
- ✅ **Data Type Optimization**: float32 to prevent object arrays
- ✅ **Cached Data Loaders**: Prevent multiple instances

### **Streamlined Interface**
- ✅ **Interactive Launcher**: Easy-to-use menu system
- ✅ **Unified Commands**: Single main.py handles all modes
- ✅ **Comprehensive Documentation**: All information in one place
- ✅ **Error Handling**: Robust error recovery

## 📄 License

This project is for educational purposes. Use trading signals at your own risk.

---

## 🚀 Ready to Use

The pipeline is now streamlined and optimized for your system. Simply run:

```bash
cd "price model"
python launch.py
```

Or start with:

```bash
python main.py --mode demo
```

Everything is integrated, optimized, and ready to use! 🎯 