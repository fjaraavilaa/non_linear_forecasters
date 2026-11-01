# Non-Linear Forecasters

An experimental time-series forecasting repository. Its main model combines an LSTM feature extractor with a sparse variational Gaussian process; the repository also includes LSTM-only and TimeFM experiments. The work began with market data because it is readily available at several frequencies, but the code is intended as a starting point for forecasting experiments rather than a trading system.

The idea started as an initial project to use LSTMs and Gaussian Processes. The idea is to, given the embeddings of the LSTM, help with the estimation of the second moment. With these it is meant to have a covariance estimation of the final tasks in order to provide a proper analytical result. 

During the development of this tool, Google also published TimeFM 3.0. Due to it's capacity to provide quantiles it was also implemented in a naïve way, for future benchmarks.

The idea is to also implement other non-linear forecasters and be capable of comparing them with ARIMA-ARFIMA and ARCH methdologies. 

## What is here

- An LSTM + Gaussian-process model, implemented with PyTorch and GPyTorch.
- LSTM-only training and prediction scripts.
- Optuna studies for the LSTM/GP, LSTM-only, price, and TimeFM experiments.
- Binance and Yahoo Finance data collectors.
- Preprocessing for OHLCV data, returns, scaling, and fixed-length sequences.
- Notebooks and tests used while developing the models.

The Gaussian-process model is sparse and variational, so it can work with learned LSTM features without requiring an exact GP over the full training set. Its outputs include a predictive distribution, rather than only a point estimate.

## Repository layout

```
.
├── main.py                         # Train or run the LSTM + GP workflow
├── crypto_screener.py               # Simple crypto screening utility
├── requirements.txt
├── src/
│   ├── binance_data_collector.py    # Binance OHLCV collection
│   ├── stock_data_collector.py      # Yahoo Finance OHLCV collection
│   ├── data_preprocessor.py         # Features, scaling, and sequence creation
│   ├── lstm_model.py                # LSTM models
│   ├── gpytorch_model.py            # Sparse variational GP models
│   ├── deep_kernel.py               # LSTM feature extractors for the GP
│   ├── training_functions.py         # Training and evaluation helpers
│   ├── timefm_functions.py           # TimeFM setup and forecasting helpers
│   ├── optuna_trial_*.py            # Hyperparameter-search entry points
│   └── study_start*.py               # Optuna study setup scripts
├── validation/                      # Date splits used by the experiments
├── tests/                           # Tests and exploratory test notebooks
└── *.ipynb                          # Analysis and plotting notebooks
```

## Setup

Python 3.10 is the intended environment. Create a virtual environment, then install the dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch uses the CPU by default unless a compatible CUDA build is installed. If you are working on a machine without CUDA, pass `--device cpu` to scripts that take a device argument.

## Running the LSTM + GP experiment

`main.py` reads the date split from `validation/healthy_dates.json`, downloads data from Binance, prepares sequences, and trains the deep-kernel GP using the checked-in best-parameter files. You can play around with it

```bash
python main.py --symbols BTC-USD --device cpu
```

The input features default to `Log_Returns`. You can supply a different set when the corresponding columns are available:

```bash
python main.py \
  --symbols BTC-USD \
  --input_study_features Log_Returns Volume Count \
  --device cpu
```

The scripts were written for specific studies and some settings, such as the Binance interval and model parameters, remain in the source. Treat the command-line arguments as overrides, not as a complete configuration system.

This is mainly for test purposes.

## Hyperparameter searches

The Optuna scripts keep their studies in local SQLite files. Start the relevant study before launching trials:

```bash
# LSTM + GP
python -m src.study_start
python -m src.optuna_trial_function --symbols BTC-USD --device cpu

# LSTM only
python -m src.optuna_trial_lstm_function --symbols BTC-USD --device cpu

# TimeFM
python -m src.study_start_timefm
python -m src.optuna_trial_timefm_function --symbols BTC-USD --interval 1h
```

The TimeFM search can optionally receive historical columns as covariates:

```bash
python -m src.optuna_trial_timefm_function \
  --symbols BTC-USD \
  --interval 1h \
  --covariates Volume Count
```

Running several trial processes against the same SQLite study is possible, but it is worth checking the database and hardware setup first. The experiments can download a substantial amount of intraday data and, in the LSTM/GP case, benefit from a GPU.

## Data and outputs

The Binance collector accepts symbols such as `BTC-USD` and converts them to the format used by the exchange. The stock collector uses Yahoo Finance tickers. Both return the common OHLCV-style schema consumed by `DataPreprocessor`.

Training and search runs may create model checkpoints, metrics, Optuna SQLite databases, and plotting artefacts in the repository directory. Keep results you care about outside the working tree or add an experiment-specific output path before running large batches.

## Tests

There are regular Python tests alongside exploratory notebook tests. Run the Python tests with. These are under the ```/tests``` directory. You can run them as 

```bash
python test_lstm
```

## Notes and limitations

This is research code under active development. The scripts and notebooks reflect individual experiments, so naming and configuration are not yet fully uniform. A clean run also depends on data availability from the upstream providers.

Market data is used here as a forecasting dataset. It does not make the models suitable for trading or investment decisions, and no result in this repository should be interpreted as financial advice or a claim of reliable market prediction.

Please be mindful that 1) This is not developed by experts in finance. 2) This is entirely a personal project for intellectual and fun purposes. 3) I don't believe in the entirety of the cryptocurrencies project, but their data is quite extensive.
