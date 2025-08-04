# 🚀 Financial Sentiment Model - Complete Documentation

A professional machine learning pipeline for swing trading using candlestick patterns and technical indicators with integrated data quality improvements.

## 📊 **Features**

### **🎯 Swing Trading Focus**
- **1-3 day holding periods** - Perfect for swing trading
- **65 features per timestep** - Comprehensive technical analysis
- **Binary classification** - Buy/Sell signals with confidence scores
- **Risk management** - Stop loss and take profit recommendations

### **🔄 Smart Data Collection**
- **Incremental updates** - Only collect recent data for existing tickers
- **Graceful interruption** - Press Ctrl+C to stop and save progress
- **Backup protection** - Automatic backup files during processing
- **S&P 500 coverage** - 500 stocks with quality filtering
- **Integrated quality fixes** - Automatic NaN handling, outlier detection, class balancing

### **🧠 Advanced AI Model**
- **CNN architecture** - Deep learning for pattern recognition
- **Attention mechanism** - Focus on important features
- **Residual connections** - Better gradient flow
- **Progressive training** - Two-stage training for better performance

### **📈 Trading Signals**
- **Real-time predictions** - Get BUY/SELL signals for any stock
- **Technical confirmation** - RSI, MACD, moving averages
- **Entry/Exit points** - Optimal prices with stop loss/take profit
- **Portfolio analysis** - Compare multiple stocks at once

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
cd "price model"
pip install -r requirements.txt
```

### **2. Collect Data (with Quality Improvements)**
```bash
# Standard data collection with quality fixes (recommended)
python main.py --mode data

# Incremental data collection (for large datasets)
python main.py --mode data --incremental --save-interval 100

# Smart update (only recent data)
python main.py --mode data --smart-update --days-back 30
```

### **3. Train Model**
```bash
# Standard training
python main.py --mode train

# Progressive training (recommended)
python main.py --mode train --epochs 50
```

### **4. Get Trading Signals**
```bash
# Single stock signal
python main.py --mode signals --portfolio AAPL

# Portfolio signals
python main.py --mode signals --portfolio AAPL MSFT GOOGL TSLA NVDA
```

## 📁 **Project Structure**

```
price model/
├── main.py                      # 🎯 Main entry point
├── requirements.txt             # 📦 Dependencies
├── config/                      # ⚙️ Configuration
│   ├── config.py
│   ├── config.json
│   └── __init__.py
├── src/                         # 📦 Source code
│   ├── data/
│   │   └── data_collection.py  # 📊 Data collection with quality fixes
│   ├── models/
│   │   ├── cnn_model.py
│   │   └── dataset.py
│   ├── training/
│   │   ├── train.py
│   │   ├── progressive_trainer.py
│   │   └── advanced_utils.py
│   ├── inference/
│   │   ├── predict.py
│   │   └── trading_signals.py
│   └── utils/                   # 🛠️ Utilities
│       ├── switch_dataset.py   # 🔄 Dataset switching
│       ├── pattern_validator.py
│       └── helpers.py
├── tests/                       # 🧪 Test files
├── data/                        # 📊 Data files
├── models/                      # 🧠 Model files
├── logs/                        # 📝 Training logs
├── notebooks/                   # 📓 Jupyter notebooks
└── docs/                        # 📚 Documentation (this directory)
```

## 🎯 **Usage Examples**

### **Data Collection Modes**

**Standard (Full Dataset with Quality Fixes):**
```bash
python main.py --mode data
```

**Incremental (Save Every 100 Tickers):**
```bash
python main.py --mode data --incremental --save-interval 100
```

**Smart Update (Recent Data Only):**
```bash
python main.py --mode data --smart-update --days-back 30
```

**Skip Quality Improvements:**
```bash
python src/data/data_collection.py --skip-quality-fixes
```

### **Training Modes**

**Standard Training:**
```bash
python main.py --mode train --epochs 50
```

**Progressive Training (Recommended):**
```bash
python main.py --mode train --epochs 50
```

### **Trading Signals**

**Single Stock:**
```bash
python main.py --mode signals --portfolio AAPL
```

**Portfolio Analysis:**
```bash
python main.py --mode signals --portfolio AAPL MSFT GOOGL TSLA NVDA
```

**Custom Portfolio:**
```bash
python main.py --mode signals --portfolio SPY QQQ IWM TLT GLD
```

### **Dataset Management**

**Switch to Cleaned Dataset:**
```bash
python src/utils/switch_dataset.py --to cleaned
```

**Switch to Original Dataset:**
```bash
python src/utils/switch_dataset.py --to original
```

**Check Current Dataset:**
```bash
python src/utils/switch_dataset.py --show
```

## 📊 **Data Quality Improvements**

The pipeline includes **automatic data quality improvements** that address common issues:

### **Issues Addressed**
- **NaN values** (4,843,050 values, 0.66%) - Handled with smart interpolation
- **Extreme values** (10,690 values) - Detected and treated with isolation forest
- **Class imbalance** (9.8:1 ratio) - Balanced using SMOTE

### **Quality Settings**

**NaN Handling:**
- `auto` (default): Smart strategy based on data type
- `drop`: Remove rows with NaN values
- `fill_mean`: Fill with mean values
- `fill_median`: Fill with median values
- `interpolate`: Use linear interpolation

**Extreme Value Detection:**
- `isolation_forest` (default): Advanced outlier detection
- `iqr`: Interquartile range method
- `zscore`: Z-score method
- `winsorize`: Winsorization

**Class Balancing:**
- `smote` (default): Synthetic Minority Oversampling
- `undersample`: Reduce majority class
- `oversample`: Increase minority class
- `class_weights`: Use class weights (no data modification)

### **Expected Results**

**Before Quality Fixes:**
- NaN values: 4,843,050 (0.66%)
- Extreme values: 10,690
- Class imbalance: 9.8:1 ratio

**After Quality Fixes:**
- NaN values: Near zero
- Extreme values: Reduced to manageable levels
- Class imbalance: Approximately 1:1 ratio

## 📊 **Model Architecture**

### **CNN Model Features**
- **Input**: 5-day sequence × 65 features = 325 dimensions
- **Convolutional layers**: Pattern recognition
- **Attention mechanism**: Focus on important features
- **Residual connections**: Better gradient flow
- **Output**: Binary classification (Buy/Sell)

### **Technical Indicators**
- **Trend**: SMA 10, 20, 50, EMA 10, 20
- **Momentum**: RSI 14, MACD, Stochastic
- **Volatility**: ATR, Bollinger Bands
- **Volume**: OBV, MFI, Volume ratios
- **Patterns**: 10 candlestick patterns

## 🎯 **Trading Signal Output**

**Example Signal Report:**
```
============================================================
📊 TRADING SIGNAL REPORT - AAPL
============================================================
📅 Date: 2024-01-15
💰 Current Price: $185.50
🎯 Model Prediction: BUY
📈 Model Confidence: 85.2%
🔧 Technical Score: 4/5
💪 Signal Strength: 82.6%
✅ Recommendation: STRONG BUY

✅ Confirmations:
   • RSI oversold - good entry point
   • MACD bullish - trend confirmation
   • Above 20-day SMA - bullish trend
   • High volume - strong signal

📈 Entry Strategy:
   • Entry Price: $185.50
   • Stop Loss: $176.23 (-5%)
   • Take Profit: $213.33 (+15%)
   • Risk/Reward: 1:3
============================================================
```

## 🔧 **Configuration**

### **Data Collection Settings**
```json
{
  "N_TICKERS": 500,
  "START": "2020-01-01",
  "END": "2024-01-01",
  "SEQ_LEN": 5,
  "HORIZON": 1
}
```

### **Training Settings**
```json
{
  "BATCH_SIZE": 256,
  "LEARNING_RATE": 0.0002,
  "EPOCHS": 50,
  "EARLY_STOPPING_PATIENCE": 15
}
```

## 📈 **Performance**

### **Expected Results**
- **Accuracy**: 45-55% (realistic for financial markets)
- **Signal Strength**: 60-80% for strong signals
- **Risk/Reward**: 1:3 ratio recommended
- **Holding Period**: 1-3 days (swing trading)

### **Success Indicators**
- ✅ Validation accuracy > 40%
- ✅ Balanced class predictions
- ✅ Smooth loss convergence
- ✅ Strong technical confirmations

## 🛠️ **Advanced Features**

### **Progressive Training**
- **Stage 1**: 300 tickers, 15 epochs, LR=1e-3
- **Stage 2**: 400 tickers, 20 epochs, LR=5e-4
- **Quality filtering**: High liquidity, volatility requirements

### **Advanced Loss Functions**
- **Focal Loss**: Addresses class imbalance
- **Label Smoothing**: Reduces overconfidence
- **Mixup Augmentation**: Data augmentation

### **Real-time Monitoring**
- **Training logs**: Real-time progress tracking
- **Performance metrics**: Accuracy, loss, learning rate
- **Data quality**: Automatic issue detection

## 🚨 **Important Notes**

### **Data Collection**
- **Internet required**: Downloads S&P 500 tickers from DataHub
- **Time intensive**: Processing 500 stocks takes 2-4 hours
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

## 🔍 **Troubleshooting**

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

**Quality Issues:**
```bash
# Check quality reports in data/ directory
# Use different quality settings
python src/data/data_collection.py --nan-strategy interpolate
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

## 📚 **References**

- **Technical Analysis**: TA-Lib library
- **Data Source**: Yahoo Finance via yfinance
- **Deep Learning**: PyTorch framework
- **S&P 500 Data**: DataHub constituents

## 🔄 **Recent Updates**

### **Pipeline Organization (Latest)**
- ✅ **Reorganized Structure**: Moved scripts to appropriate folders
- ✅ **Eliminated Redundancy**: Removed duplicate files and scripts
- ✅ **Documentation Consolidation**: All docs organized in `docs/` directory
- ✅ **Utility Scripts**: Moved `switch_dataset.py` to `src/utils/`
- ✅ **Quality Integration**: Data quality fixes integrated into main pipeline

### **Data Quality Improvements**
- ✅ **Integrated Quality Fixes**: NaN handling, outlier detection, class balancing
- ✅ **Automatic Quality Reports**: Generated during data collection
- ✅ **Dataset Switching**: Easy switching between original and cleaned datasets

### **Usage After Reorganization**
```bash
# Dataset switching (updated path)
python src/utils/switch_dataset.py --show
python src/utils/switch_dataset.py --to cleaned

# Documentation access
ls docs/
cat docs/README.md
```

## 📄 **License**

This project is for educational purposes. Use trading signals at your own risk.

---

**🎯 Ready to start swing trading with AI? Run the quick start commands above!** 