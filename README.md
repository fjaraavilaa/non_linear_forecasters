# LSTM Cryptocurrency Prediction with Deep Kernel Learning

This is a Deep Kernel Learning project I decided to do after my PhD. I was just annoyed at the fact that I never found GPs used with LSTMs. Might sound a bit stubborn but I thought it could help someone (like me at some points).

## Project Overview

This project implements an end-to-end pipeline for cryptocurrency returns and volatility prediction using hybrid deep learning models. It combines Long Short-Term Memory (LSTM) neural networks with Gaussian Processes (GPyTorch) to leverage the strengths of both approaches: LSTM's temporal feature extraction and GP's uncertainty quantification.

The architecture supports three model variants:
- **LSTM Model**: Standalone LSTM for direct prediction
- **Gaussian Process Model**: Multi-task sparse variational GP with uncertainty estimates
- **Hybrid LSTM-GP Model**: LSTM feature extractor feeding into a GP predictor

*I am not very interested in the crypto usage. Please use things as you please. But do not make crypto claims based on this.*

## Project Structure

*It's still a bit messy. Consider it a to do.*

```
├── main.py                          # Main training pipeline
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── __init__.py
├── src/
│   ├── binance_data_collector.py   # Real-time data fetching
│   ├── data_preprocessor.py        # Feature engineering
│   ├── lstm_model.py               # LSTM architecture
│   ├── gpytorch_model.py           # GP and hybrid model implementations
│   ├── deep_kernel.py              # Deep kernel feature extractors
│   ├── training_functions.py       # Training routines
│   ├── optuna_trial_function.py    # Hyperparameter search for hybrid model
│   ├── optuna_trial_lstm_function.py # Hyperparameter search for LSTM
│   ├── study_start.py              # Initialize Optuna study
│   ├── mean_functions.py           # GP mean function definitions
│   └── help_functions/
│       └── predictor.py            # Model inference utilities
├── tests/                           # Unit tests
├── validation/                      # Validation dataset and configuration
│   ├── create_healthy_validation_dataset.ipynb
│   ├── healthy_dates.json          # Train/test date splits
│   └── healthy_data/               # Pre-processed validation data
└── data/                            # Cryptocurrency data storage
```

*Generated artifacts (created during training) are saved in the root directory: trained models, metrics, optimization results, and visualizations.*

## Setup and Installation

### Prerequisites

- Python 3.10
- PyTorch 2.0+ (with CUDA support optional but recommended)
- GPyTorch 1.11+

### Installation Steps

1. **Create and activate a virtual environment:**
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; import pandas; import gpytorch; print('Dependencies installed successfully!')"
   ```

## Usage

### Quick Start: Training with Best Hyperparameters

To train a model using the pre-optimized hyperparameters:

```bash
python main.py \
  --symbols BTC-USD \
  --input_study_features Returns Volume Count \
  --device cuda
```

### Hyperparameter Optimization

The project uses Optuna for automated hyperparameter tuning. Run optimization in two steps:

1. **Initialize the study:**
   ```bash
   python -m src.study_start
   ```

2. **Run optimization trials** (for hybrid LSTM-GP model):
   ```bash
   python -m src.optuna_trial_function \
     --symbols BTC-USD \
     --input_study_features Returns Volume Count \
     --device cuda
   ```

3. **Alternative: Run LSTM-only optimization:**
   ```bash
   python -m src.optuna_trial_lstm_function \
     --symbols BTC-USD \
     --input_study_features Returns Volume Count \
     --device cuda
   ```

### Parallel Hyperparameter Search (HPC Clusters)

For distributed optimization on HPC systems using SLURM:

```bash
python -m src.study_start

# Submit multiple parallel trials
srun --ntasks=1 --gres=gpu:1 --exact python -m src.optuna_trial_function \
  --symbols BTC-USD --input_study_features Returns Volume Count --device cuda &
srun --ntasks=1 --gres=gpu:1 --exact python -m src.optuna_trial_function \
  --symbols BTC-USD --input_study_features Returns Volume Count --device cuda &
wait
```



## Configuration Options

### General Arguments

- `--symbols`: Cryptocurrency symbols to predict (default: `['BTC-USD']`)
  - Examples: `BTC-USD`, `ETH-USD`, `XRP-USD`
  - Multiple symbols: `--symbols BTC-USD ETH-USD XRP-USD`
- `--input_study_features`: Features to use in model training (default: `['Returns']`)
  - Available: `Returns`, `Volume`, `Count`, technical indicators
- `--device`: Computation device (default: `cuda`)
  - Options: `cuda` (GPU), `cpu`

### Data Collection Parameters

- `--interval`: Binance API interval for data fetching (default: `1m`)
  - Options: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`
- `--binance-api-key`: Optional Binance API key for higher rate limits

### Model Architecture Parameters

- `--sequence-length`: Number of time steps in LSTM input (default: `30`)
- `--hidden-size`: Number of hidden units in LSTM (default: `128`)
- `--num-layers`: Number of stacked LSTM layers (default: `2`)
- `--dropout`: Dropout probability for regularization (default: `0.2`)

### Training Parameters

- `--epochs`: Maximum number of training epochs (default: `100`)
- `--batch-size`: Training batch size (default: `32`)
- `--learning-rate`: Optimizer learning rate (default: `0.001`)
- `--patience`: Early stopping patience in epochs (default: `10`)

### GPyTorch-Specific Parameters

- `--model-type`: Model architecture to use (default: `hybrid`)
  - Options: `lstm` (LSTM only), `gp` (Gaussian Process only), `hybrid` (LSTM + GP)
- `--num-inducing`: Number of inducing points for sparse GP (default: `100`)
- `--gp-epochs`: Training epochs for pure GP model (default: `100`)
- `--gp-learning-rate`: Learning rate for GP optimization (default: `0.01`)
- `--gp-feature-dim`: Feature dimension for GP in hybrid model (default: `32`)
- `--lstm-hidden-size`: LSTM hidden size in hybrid model (default: `64`)
- `--gp-init-samples`: Samples for GP initialization (default: `1000`)

## Model Architecture

### LSTM Model

A standard multi-layer LSTM architecture for time series prediction:

- **Multi-layer LSTM**: Configurable number of stacked LSTM layers with shared hidden size
- **Batch-first input**: Accepts sequences of shape (batch_size, sequence_length, features)
- **Dropout Regularization**: Applied between LSTM layers to prevent overfitting
- **Output Layer**: Fully connected layer mapping LSTM hidden state to prediction
- **Training Features**: Early stopping based on validation loss

**Use case**: Direct time series prediction without uncertainty estimates.

### Gaussian Process Model

Multi-task sparse variational Gaussian process with GPyTorch:

- **Sparse Variational Inference**: Uses inducing points for computational efficiency
- **Kernel**: Technically you can choose the kernel
- **Uncertainty Quantification**: Provides prediction confidence intervals

**Use case**: Uncertainty-aware predictions for risk assessment and confidence intervals.

### Hybrid LSTM-GP Model (Recommended)

Combines LSTM feature extraction with Gaussian Process prediction:

1. **Feature Extraction**: LSTM encodes temporal sequences into fixed-dimensional features
2. **Feature Output**: Last hidden state from LSTM is transformed into GP input features
3. **GP Prediction**: Sparse variational GP makes predictions with uncertainty from LSTM features
4. **Joint Optimization**: End-to-end training optimizes both components simultaneously

**Advantages**:
- Captures complex temporal patterns via LSTM
- Provides uncertainty estimates via GP
- More sample-efficient than pure GP
- Better generalization than LSTM alone

**Use case**: Production predictions combining accuracy and uncertainty quantification.

### Feature Engineering

The data preprocessing pipeline creates:

- **Price Features**: Open, High, Low, Close, Volume
- **Returns**: Price change ratios for stationarity
- **Technical Indicators**: RSI, Moving Averages, Bollinger Bands, Average True Range (ATR) (I AM NOT USING THEM, JUST ADDED THE CODE WITH AI IN CASE I NEEDED THEM)
- **Volatility Measures** (IDEM):
  - Parkinson Volatility: $\sqrt{\frac{\ln(H/L)^2}{4\ln(2)}}$ (High-Low range)
  - Garman-Klass Volatility: Incorporates Open-High-Low-Close data
  - Rogers-Satchell Volatility: Drift-independent estimator
- **Lagged Features**: Previous timestep values for temporal context
- **Normalized Values**: Ratios and standardized features for model stability

## Outputs and Results

After training completes, the following artifacts are saved:

- **Trained Model**: PyTorch model weights (`.pt` file)
- **Training Metrics**: JSON file containing:
  - Training loss history
  - Validation loss history
  - Performance metrics per epoch
  - Final model performance
- **Optimization History**: Optuna study results and visualizations (HTML contour plots)
- **Hyperparameter Configuration**: Best parameters from hyperparameter search (`best_trial_params.json`)
- **Predictions**: Numpy arrays or DataFrames with model predictions and confidence intervals (for GP models)

Results are organized by date/timestamp for easy tracking and reproducibility.

## Testing

Unit tests are provided in the `tests/` directory:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_model.py

# Run with coverage
python -m pytest tests/ --cov=src/
```

Test coverage includes:
- Data collection and preprocessing validation
- Model initialization and forward passes
- Training routine correctness
- Prediction consistency
- Loss computation

## Validation Dataset

The `validation/` directory contains:

- **healthy_dates.json**: Train/test date splits ensuring valid timestamp ranges

These are used to ensure consistent train/test splits across experiments.

## Known Limitations and Future Work

- **Stationarity**: Returns are used instead of raw prices due to unit root issues
- **Data Staleness**: Binance API rate limits may affect high-frequency data collection
- **Change to Multi-Tasks/Multi-Target**: Current implementation is not yet multi-task. but i am working on it. I just want to do it properly with BTC before making it more complex.
- **Hyperparameter Search**: Optuna search space can be extended for deeper exploration

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes with clear messages
4. Add or update tests as needed
5. Submit a pull request with a clear description

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

**Educational Use Only**: This project is for educational and research purposes. Cryptocurrency markets are highly volatile and inherently unpredictable. 

**Trading Risk**: Do not use these predictions for actual financial trading or investment.

The authors assume no responsibility for financial losses resulting from use of this code.

*Also, I do not believe in cryptocurrencies. But their dataset is quite extensive and useful for time series research.*

## References

- **GPyTorch**: Gaussian Processes with PyTorch
  - Gardner et al., "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration" (NEURIPS 2018)
  
- **Optuna**: Hyperparameter Optimization Framework
  - Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework" (KDD 2019)

- **Volatility Estimation**:
  - Parkinson (1980): A simple volatility estimator
  - Garman & Klass (1980): On the estimation of security price volatilities
  - Rogers & Satchell (1991): Estimating variance from high, low and closing prices

---

**Author**: Originally developed as a post-PhD research project exploring the combination of LSTM feature extraction with Gaussian Process uncertainty quantification for time series prediction. This is a fun project for me. Most of this Readme is AI. I did check it beforehand but I don't have the inspiration to write such texts.
