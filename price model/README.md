# 📈 Candlestick Pattern Price Model

A deep learning model for stock price movement prediction using candlestick pattern recognition and technical analysis. This model uses 1D CNN architecture to analyze sequences of candlestick patterns and predict next-day price movements.

## 🚀 Features

- **Candlestick Pattern Recognition**: Analyzes 15 different candlestick patterns using TA-Lib
- **Deep Learning**: 1D CNN architecture optimized for time series prediction
- **5-Class Classification**: Predicts Strong Buy, Buy, Hold, Sell, or Strong Sell signals
- **S&P 500 Coverage**: Trains on up to 400 high-quality S&P 500 stocks
- **Real-time Predictions**: Make predictions on any stock ticker
- **Batch Processing**: Analyze multiple stocks simultaneously
- **Enhanced Training**: Progressive training with comprehensive logging
- **Professional Pipeline**: Explicit raw/adjusted data architecture

## 📋 Requirements

### System Requirements
- Python 3.8+
- PyTorch 2.0+
- TA-Lib (Technical Analysis Library)

### Installation

1. **Install TA-Lib** (required for candlestick patterns):
   ```bash
   # On Windows (using conda)
   conda install -c conda-forge ta-lib
   
   # On macOS
   brew install ta-lib
   
   # On Ubuntu/Debian
   sudo apt-get install libta-lib-dev
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏗️ Project Structure

```
price model/
├── config/
│   ├── config.py              # Configuration and hyperparameters
│   └── hyperparameters.py     # Centralized hyperparameter management
├── src/
│   ├── data/
│   │   └── data_collection.py # Professional data pipeline
│   ├── models/
│   │   ├── cnn_model.py       # CNN architecture
│   │   └── dataset.py         # PyTorch dataset classes
│   ├── training/
│   │   ├── train.py           # Advanced training pipeline
│   │   ├── progressive_trainer.py # Progressive training
│   │   └── advanced_utils.py  # Advanced training utilities
│   ├── inference/
│   │   └── predict.py         # Prediction and inference
│   └── utils/
│       └── helpers.py         # Utility functions
├── logs/
│   ├── training_logs.py       # Comprehensive logging system
│   ├── training_analysis.md   # Training analysis and solutions
│   └── README.md             # Logs directory guide
├── data/                      # Generated datasets
├── models/                    # Saved model files
├── main.py                    # Main execution script
├── src/training/train.py      # Enhanced training script (integrated)
└── requirements.txt           # Dependencies
```

## 🔧 Quick Start

### ⚡ **Important: Directory Navigation**

The professional price model is located in the `price model` folder. You must navigate there first:

```bash
# From the root "Financial Sentiment Model" directory:
cd "price model"

# Then run commands:
python main.py --mode data
```

### **Option 1: Enhanced Training (Recommended)**
```bash
# Run progressive training with comprehensive logging
python src/training/train.py --mode progressive --enable-logging

# Or run standard enhanced training
python src/training/train.py --mode enhanced --enable-logging

# Or run standard training
python src/training/train.py --mode standard
```

### **Option 2: Automatic Setup**
```bash
python main.py quickstart
```
This will automatically:
1. Download S&P 500 data
2. Build the dataset
3. Train the model
4. Run a demo prediction

### **Option 3: Step-by-Step**

1. **Build Dataset**:
   ```bash
   # Build professional dataset (explicit raw/adjusted columns)
   python main.py --mode data
   ```

2. **Train Model**:
   ```bash
   python main.py --mode train
   ```

3. **Make Predictions**:
   ```bash
   python main.py --mode predict --ticker AAPL
   ```

4. **Run Demo**:
   ```bash
   python main.py --mode demo
   ```

## 🎯 Enhanced Training Features

### **Progressive Training**
- **Stage 1**: 300 tickers, 15 epochs, LR=1e-3
- **Stage 2**: 400 tickers, 20 epochs, LR=5e-4
- Prevents overfitting, better convergence

### **Quality Filtering**
- 400 high-quality tickers instead of 500
- Quality thresholds for volume, volatility
- Removes noisy, illiquid stocks

### **Enhanced Regularization**
- Increased dropout: 0.3 → 0.4
- Increased weight decay: 1e-4 → 2e-4
- Added label smoothing: 0.1 → 0.15
- Added spectral normalization

### **Comprehensive Logging**
- Real-time training monitoring
- Data quality analysis
- Performance diagnostics
- Automatic issue detection

## 📊 Professional Data Pipeline

### **Explicit Raw/Adjusted Column Architecture**

**Before (Confusing):**
```python
# Ambiguous data source
df['Open']  # Is this raw or adjusted? Context-dependent!
df['Close'] # Could be either, manual adjustment factors
```

**After (Crystal Clear):**
```python
# Explicit data lineage
df['Open_raw']   # Definitely raw price data
df['Close_raw']  # Raw close for candlestick patterns
df['Open_adj']   # Definitely adjusted price data  
df['Close_adj']  # Adjusted close for indicators/returns
```

### **Smart Data Source Selection**
```python
# Candlestick patterns: RAW data (traditional approach)
hammer = talib.CDLHAMMER(Open_raw, High_raw, Low_raw, Close_raw)

# Technical indicators: ADJUSTED data (economic accuracy)
rsi = talib.RSI(Close_adj)
macd = talib.MACD(Close_adj)

# Returns: ADJUSTED data (essential)
daily_return = Close_adj.pct_change()
```

## 📈 Expected Performance

| Metric | Before (500 tickers) | After (400 tickers) | Improvement |
|--------|---------------------|-------------------|-------------|
| **Validation Accuracy** | 43% | 47-50% | +4-7% |
| **Training Stability** | Poor | Good | ✅ |
| **Convergence** | 35 epochs | 15-20 epochs | -43% |
| **Class Balance** | Skewed | Balanced | ✅ |
| **Overfitting** | High | Low | ✅ |

## 🔍 Monitoring Training

### **✅ Good Signs:**
- Stage 2 val_loss < Stage 1 val_loss
- Smooth learning rate curves
- Balanced class predictions
- Early convergence (15-20 epochs)

### **⚠️ Warning Signs:**
- Stage 2 val_loss > Stage 1 val_loss
- One class >50% of predictions
- LR < 1e-6 (too small)
- No improvement for >10 epochs

## 📁 Key Files

### **Training System:**
- `src/training/train.py` - Enhanced training script (integrated)
- `src/training/progressive_trainer.py` - Progressive training
- `logs/training_logs.py` - Comprehensive logging
- `logs/training_analysis.md` - Training analysis and solutions

### **Configuration:**
- `config/hyperparameters.py` - Centralized hyperparameters
- `config/config.py` - Legacy configuration (for compatibility)

### **Data Pipeline:**
- `src/data/data_collection.py` - Professional data pipeline
- `src/models/dataset.py` - Dataset handling

## 🎯 Usage Examples

### **Single Stock Prediction**
```bash
python main.py --mode predict --ticker AAPL
```

### **Batch Analysis**
```bash
python main.py --mode demo
```

### **Compare Raw vs Adjusted Data**
```bash
python main.py --compare-adjustment
```

### **Enhanced Training with Logging**
```bash
python src/training/train.py --mode progressive --enable-logging --experiment-name "my_experiment"
```

## 🔧 Troubleshooting

**Problem**: `can't open file 'main.py'`
**Solution**: Make sure you're in the `price model` directory:
```bash
cd "price model"
pwd  # Should show: .../Financial Sentiment Model/price model
```

**Problem**: Training performance issues
**Solution**: Use enhanced training with logging:
```bash
python src/training/train.py --mode progressive --enable-logging
```

**Problem**: Data quality issues
**Solution**: Check logs for data quality analysis:
```bash
ls logs/
cat logs/training_analysis.md
```

## 📊 Model Architecture

### **CNN Architecture**
- **Input**: 5-day sequences of 65 features per day
- **Features**: Technical indicators + candlestick patterns
- **Output**: 5-class classification (Strong Sell to Strong Buy)
- **Architecture**: 1D CNN with adaptive pooling

### **Technical Indicators**
- **Momentum**: RSI, MACD, Stochastic, Williams %R
- **Volatility**: Bollinger Bands, ATR, Historical volatility
- **Volume**: OBV, AD, VWAP, Volume ratios
- **Trend**: Moving averages, Parabolic SAR
- **Patterns**: 15 candlestick patterns (Hammer, Doji, etc.)

## 🎯 Expected Outcome

**45-50% validation accuracy with stable training!** 🎯

The enhanced training system provides:
1. **Quality over Quantity**: 400 high-quality tickers > 500 mixed-quality
2. **Progressive Learning**: Stage 1 learns basics, Stage 2 refines
3. **Better Regularization**: Prevents overfitting on larger datasets
4. **Class Balance**: Handles imbalanced data properly
5. **Adaptive Training**: Learning rate adjusts to data size 