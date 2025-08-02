# Incremental Data Collection

This document describes the new incremental data collection feature that allows you to save data periodically and stop gracefully while preserving collected data.

## Features

✅ **Incremental Saving**: Saves data every 100 tickers (configurable)  
✅ **Graceful Interruption**: Press Ctrl+C to stop and save collected data  
✅ **Data Preservation**: All collected data is saved when interrupted  
✅ **Backup Files**: Creates backup files during processing  
✅ **Progress Tracking**: Shows progress and saves intermediate results  

## Usage

### Method 1: Using the main script with --incremental flag

```bash
# Basic incremental data collection (saves every 100 tickers)
python main.py --mode data --incremental

# Custom save interval (saves every 50 tickers)
python main.py --mode data --incremental --save-interval 50
```

### Method 2: Using the dedicated script

```bash
# Run the dedicated incremental data collection script
python run_incremental_data_collection.py
```

### Method 3: Direct function call

```python
from src.data.data_collection import build_dataset_incremental

# Run incremental data collection
dataset_path = build_dataset_incremental(save_interval=100)
```

## How It Works

1. **Processing**: The script processes tickers one by one, collecting data for each
2. **Incremental Saving**: Every N tickers (default: 100), the script saves all collected data to a backup file
3. **Graceful Interruption**: When you press Ctrl+C, the script:
   - Stops processing new tickers
   - Saves all currently collected data
   - Creates the final dataset file
   - Exits cleanly

## File Outputs

- **Main Dataset**: `data/candles.csv` (final complete dataset)
- **Backup File**: `data/candles_backup.csv` (intermediate saves)
- **Feature Summary**: `data/candles_feature_summary.txt` (detailed feature information)

## Benefits

1. **Risk Mitigation**: If the script crashes or is interrupted, you don't lose all progress
2. **Memory Management**: Periodic saving helps manage memory usage for large datasets
3. **Flexibility**: You can stop the process at any time and still have useful data
4. **Progress Monitoring**: You can see intermediate results and verify data quality

## Example Output

```
🔄 Using incremental data collection mode
💾 Saving data every 100 tickers
⚠️  Press Ctrl+C to stop gracefully and save collected data
Using explicit raw/adjusted column families

Processing ticker 1/500: AAPL
Processing ticker 51/500: AXP
...
Processing ticker 100/500: CCI

💾 Saving intermediate data after 100 tickers (12500 samples)...
  Dataset shape: (12500, 326)
  Label distribution:
  0    6250
  1    6250
  Saved to: data/candles_backup.csv

Processing ticker 101/500: CCL
...
```

## Stopping the Process

To stop the data collection gracefully:

1. **Press Ctrl+C** in the terminal
2. The script will:
   - Print "⚠️ Interrupt signal received. Saving collected data and shutting down gracefully..."
   - Save all collected data
   - Create the final dataset file
   - Exit cleanly

## Configuration

You can customize the save interval by changing the `--save-interval` parameter:

```bash
# Save every 50 tickers
python main.py --mode data --incremental --save-interval 50

# Save every 200 tickers  
python main.py --mode data --incremental --save-interval 200
```

## Troubleshooting

### If the script stops unexpectedly:
1. Check the backup file: `data/candles_backup.csv`
2. Restart with the same command - it will start from the beginning
3. The backup file contains all data collected before the interruption

### If you want to resume from where you left off:
Currently, the script starts from the beginning each time. For true resumption, you would need to modify the script to track progress and skip already processed tickers.

## Performance Notes

- **Memory Usage**: Incremental saving helps manage memory for large datasets
- **Disk Space**: Backup files use additional disk space during processing
- **Processing Time**: Slight overhead from periodic saving, but minimal impact on total time 