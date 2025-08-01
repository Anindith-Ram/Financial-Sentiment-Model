"""
Unified Configuration System
Loads from JSON and provides all configuration utilities
"""
import json
import os
import sys
from typing import Dict, Any

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

CONFIG_FILE = "config/config.json"

def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file"""
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
    
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

# Load all configurations
_config = load_config()

# Create individual config objects for backward compatibility
DATA_CONFIG = _config.get('DATA', {})
MODEL_CONFIG = _config.get('MODEL', {})
TRAINING_CONFIG = _config.get('TRAINING', {})
CLASSIFICATION_CONFIG = _config.get('CLASSIFICATION', {})
TECHNICAL_CONFIG = _config.get('TECHNICAL', {})
EVALUATION_CONFIG = _config.get('EVALUATION', {})
INTEGRATION_CONFIG = _config.get('INTEGRATION', {})
PRODUCTION_CONFIG = _config.get('PRODUCTION', {})
EXPERIMENTAL_CONFIG = _config.get('EXPERIMENTAL', {})

# ===== BACKWARD COMPATIBILITY =====
# Legacy variables for backward compatibility
TICKERS_CSV = DATA_CONFIG['TICKERS_CSV']
N_TICKERS = DATA_CONFIG['N_TICKERS']
START = DATA_CONFIG['START_DATE']
END = DATA_CONFIG['END_DATE']
SEQ_LEN = DATA_CONFIG['SEQ_LEN']
HORIZON = DATA_CONFIG['HORIZON']
BATCH_SIZE = TRAINING_CONFIG['BATCH_SIZE']
EPOCHS = TRAINING_CONFIG['EPOCHS']
LEARNING_RATE = TRAINING_CONFIG['LEARNING_RATE']
PATTERNS = TECHNICAL_CONFIG['PATTERNS']
DEVICE = MODEL_CONFIG['DEVICE']
DATA_OUTPUT_PATH = DATA_CONFIG['DATA_OUTPUT_PATH']
MODEL_OUTPUT_PATH = DATA_CONFIG['MODEL_OUTPUT_PATH']
USE_ADJUSTED_CLOSE = DATA_CONFIG['USE_ADJUSTED_CLOSE']
USE_RAW_COLS = ["Open_raw", "High_raw", "Low_raw", "Close_raw", "Volume_raw"]
USE_ADJ_COLS = ["Open_adj", "High_adj", "Low_adj", "Close_adj", "Volume_adj"]
PATTERN_FLAGS = PATTERNS
PRICE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
ALL_FEATURES = USE_RAW_COLS + USE_ADJ_COLS + PATTERN_FLAGS

STRONG_UP_THRESHOLD = CLASSIFICATION_CONFIG['STRONG_UP_THRESHOLD']
MILD_UP_THRESHOLD = CLASSIFICATION_CONFIG['MILD_UP_THRESHOLD']
STRONG_DOWN_THRESHOLD = CLASSIFICATION_CONFIG['STRONG_DOWN_THRESHOLD']
MILD_DOWN_THRESHOLD = CLASSIFICATION_CONFIG['MILD_DOWN_THRESHOLD']

# ===== ADVANCED TRAINING CONFIGURATION =====
EARLY_STOPPING = TRAINING_CONFIG['EARLY_STOPPING']
PATIENCE = TRAINING_CONFIG['PATIENCE']
MIN_DELTA = TRAINING_CONFIG['MIN_DELTA']
OVERFIT_THRESHOLD = TRAINING_CONFIG['OVERFIT_THRESHOLD']
LR_SCHEDULER = TRAINING_CONFIG['LR_SCHEDULER']
LR_PATIENCE = TRAINING_CONFIG['LR_PATIENCE']
LR_FACTOR = TRAINING_CONFIG['LR_FACTOR']
LR_WARMUP = TRAINING_CONFIG['LR_WARMUP']
WARMUP_EPOCHS = TRAINING_CONFIG['WARMUP_EPOCHS']
WARMUP_START_LR = TRAINING_CONFIG['WARMUP_START_LR']
OPTIMIZER = TRAINING_CONFIG['OPTIMIZER']
WEIGHT_DECAY = TRAINING_CONFIG['WEIGHT_DECAY']
GRADIENT_CLIPPING = TRAINING_CONFIG['GRADIENT_CLIPPING']
MAX_GRAD_NORM = TRAINING_CONFIG['MAX_GRAD_NORM']
LOSS_FUNCTION = TRAINING_CONFIG['LOSS_FUNCTION']
FOCAL_ALPHA = TRAINING_CONFIG['FOCAL_ALPHA']
FOCAL_GAMMA = TRAINING_CONFIG['FOCAL_GAMMA']
LABEL_SMOOTHING = TRAINING_CONFIG['LABEL_SMOOTHING']
MIXED_PRECISION = TRAINING_CONFIG['MIXED_PRECISION']
GRADIENT_ACCUMULATION = TRAINING_CONFIG['GRADIENT_ACCUMULATION']
ACCUMULATION_STEPS = TRAINING_CONFIG['ACCUMULATION_STEPS']
ADVANCED_METRICS = TRAINING_CONFIG['ADVANCED_METRICS']
SAVE_CHECKPOINTS = TRAINING_CONFIG['SAVE_CHECKPOINTS']
CHECKPOINT_EVERY = TRAINING_CONFIG['CHECKPOINT_EVERY']
VISUALIZATION = TRAINING_CONFIG['VISUALIZATION']
DROPOUT_RATE = MODEL_CONFIG['DROPOUT_RATE']
ADD_NOISE = TRAINING_CONFIG['ADD_NOISE']
NOISE_STRENGTH = TRAINING_CONFIG['NOISE_STRENGTH']

# ===== UTILITY FUNCTIONS =====
def get_config(config_name: str) -> Dict[str, Any]:
    """Get a specific configuration section"""
    return _config.get(config_name.upper(), {})

def update_config(config_name: str, key: str, value: Any) -> None:
    """Update a configuration value and save to JSON"""
    config_name = config_name.upper()
    if config_name not in _config:
        raise ValueError(f"Configuration '{config_name}' not found")
    
    _config[config_name][key] = value
    
    # Save to file
    with open(CONFIG_FILE, 'w') as f:
        json.dump(_config, f, indent=2)
    
    print(f"Updated {config_name}.{key} = {value}")

def print_config_summary() -> None:
    """Print a summary of all configurations"""
    print("Hyperparameter Configuration Summary")
    print("=" * 50)
    
    for name, config in _config.items():
        print(f"\n{name} CONFIGURATION:")
        for key, value in config.items():
            if isinstance(value, (int, float, bool, str)):
                print(f"  {key}: {value}")
            elif isinstance(value, list):
                print(f"  {key}: {value[:3]}{'...' if len(value) > 3 else ''}")
            else:
                print(f"  {key}: {type(value).__name__}")

def reload_config() -> None:
    """Reload configuration from file"""
    global _config, DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, CLASSIFICATION_CONFIG
    global TECHNICAL_CONFIG, EVALUATION_CONFIG, INTEGRATION_CONFIG, PRODUCTION_CONFIG, EXPERIMENTAL_CONFIG
    
    _config = load_config()
    
    DATA_CONFIG = _config.get('DATA', {})
    MODEL_CONFIG = _config.get('MODEL', {})
    TRAINING_CONFIG = _config.get('TRAINING', {})
    CLASSIFICATION_CONFIG = _config.get('CLASSIFICATION', {})
    TECHNICAL_CONFIG = _config.get('TECHNICAL', {})
    EVALUATION_CONFIG = _config.get('EVALUATION', {})
    INTEGRATION_CONFIG = _config.get('INTEGRATION', {})
    PRODUCTION_CONFIG = _config.get('PRODUCTION', {})
    EXPERIMENTAL_CONFIG = _config.get('EXPERIMENTAL', {})
    
    print("Configuration reloaded from JSON file")

# ===== COMMAND LINE INTERFACE =====
def main():
    """Command line interface for configuration management"""
    if len(sys.argv) < 2:
        print("Usage: python config.py [command] [section] [key] [value]")
        print("Commands: show, update [section] [key] [value]")
        print("Examples:")
        print("  python config.py show")
        print("  python config.py update TRAINING EPOCHS 100")
        return
    
    command = sys.argv[1].lower()
    
    if command == "show":
        print_config_summary()
    
    elif command == "update":
        if len(sys.argv) != 5:
            print("Need exactly 4 arguments: update section key value")
            print("Example: python config.py update TRAINING EPOCHS 100")
            return
        
        section = sys.argv[2]
        key = sys.argv[3]
        value = sys.argv[4]
        
        # Try to convert value to appropriate type
        try:
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # Keep as string if conversion fails
            pass
        
        try:
            update_config(section, key, value)
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main() 