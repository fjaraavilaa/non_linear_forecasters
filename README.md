# Non-Linear Forecasters

An experimental time-series forecasting repository. Its main model combines an LSTM feature extractor with a sparse variational Gaussian process; the repository also includes LSTM-only and TimeFM experiments. The work began with market data because it is readily available at several frequencies, but the code is intended as a starting point for forecasting experiments rather than a trading system.

The idea started as an initial project to use LSTMs and Gaussian Processes. The idea is to, given the embeddings of the LSTM, help with the estimation of the second moment. With these it is meant to have a covariance estimation of the final tasks in order to provide a proper analytical result. 

During the development of this tool, Google also published TimeFM 3.0. Due to it's capacity to provide quantiles it was also implemented in a naïve way, for future benchmarks.

The idea is to also implement other non-linear forecasters and be capable of comparing them with ARIMA-ARFIMA and ARCH methdologies. 

## What is here

- An LSTM + Gaussian-process model, implemented with PyTorch and GPyTorch.
- LSTM-only training and prediction scripts.
- Classical forecasting models with a shared interface: ARIMA, SARIMA, ARFIMA,
  SARFIMA, GARCH, APARCH, moving averages, and exponential smoothing.
- Optuna studies for the LSTM/GP, LSTM-only, price, and TimeFM experiments.
- A causal tabular XGBoost workflow with lag, trailing-window, and calendar features.
- Binance and Yahoo Finance data collectors.
- Preprocessing for OHLCV data, returns, scaling, and fixed-length sequences.
- Notebooks and tests used while developing the models.

The Gaussian-process model is sparse and variational, so it can work with learned LSTM features without requiring an exact GP over the full training set. Its outputs include a predictive distribution, rather than only a point estimate.

## Classical forecasting models

The classical models are available from `src.classical_models`. Each model supports `fit(values).forecast(horizon)` and returns a
`ForecastResult` with `mean`, `variance`, and model metadata. The result also
supports `decision_signals()`, which provides standardized expected movement,
uncertainty, direction, and signal strength for a separate decision-making
agent:

```python
from src.classical_models import ARIMA, GARCH

arima_forecast = ARIMA(order=(1, 1, 1)).fit(close_values).forecast(24)
garch_forecast = GARCH(p=1, q=1).fit(close_values).forecast(24)
signals = garch_forecast.decision_signals()
```

The models do not emit buy/sell actions or position sizes. A downstream agent
must apply its own risk policy, transaction costs, exposure limits, and
abstention rules. Every signal frame is marked `policy="signals_only"` so it
cannot be confused with an executed decision.

ARFIMA and SARFIMA apply fractional differencing before fitting the ARMA
component and reconstruct forecasts on the Close-price scale. GARCH and APARCH
fit log returns and convert their forecasts back to Close prices. These models
are model modules currently; a unified multi-model benchmark runner is not yet
part of the command-line workflow.

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
python main.py --symbols BTC-USD --device cpu --artifacts_dir artifacts/run-1
```

The input features default to `Log_Returns`. You can supply a different set when the corresponding columns are available:

```bash
python main.py \
  --symbols BTC-USD \
  --input_study_features Log_Returns Volume Count \
  --device cpu --artifacts_dir artifacts/run-2
```

The scripts were written for specific studies and some settings, such as the Binance interval and model parameters, remain in the source. Treat the command-line arguments as overrides, not as a complete configuration system.

The output directory must be new or empty; existing experiments are never
overwritten. For a repeatable comparison, supply a frozen input snapshot with
`--raw_data artifacts/raw_data.parquet`. The target remains `Close` even when
it is not among the input features. The scaler is fitted on training dates only,
and date-only split ends include the entire final day. Validation-based early
stopping is the default. `--no-work_with_validation_set` explicitly selects
legacy train-plus-validation training and training-loss monitoring.
Hourly windows crossing missing bars are discarded rather than treating a gap
as one hour. Sequence artifacts and prediction frames include symbol identifiers.

Use `--target-mode delta` with `Close` as an input to model the change from the
last observed Close. Predictions are reconstructed as Close prices. This is
an experimental option, not a guarantee of higher accuracy. Use `--config`,
`--dates-config`, `--seed`, `--epochs` and `--patience` to reproduce runs.
`--validation-only` suppresses test predictions during model selection.

Each run saves its scaler, sequence data, metadata, model checkpoint, and
prediction files in the chosen directory. `predictions_*_set.parquet` retains
the scaled output format; `predictions_*_native.parquet` contains Close-price
means and actuals, and variance in squared price units. Variance is that of the
Gaussian predictive distribution, including observation noise; interval
widths use its square root. Load a matching checkpoint with `--model_location`
and the same configuration, input features, pretrained mean and training data.
The scaler and model settings are checked on reload. Older checkpoints lacking
these settings cannot be loaded by this workflow; retain the original workflow
when reproducing such experiments. The stored sequence targets remain scaled
Close levels even when delta training is selected.

Compare native predictions only on aligned timestamps, and choose models using
validation data before evaluating a held-out test:

```bash
python -m src.compare_forecasts \
  artifacts/baseline/predictions_test_native.parquet \
  artifacts/candidate/predictions_test_native.parquet
```

The comparison reports MAE, RMSE, Gaussian negative log likelihood, 80% and 95%
coverage and interval width, and unmatched timestamp counts. It requires unique
timestamps and matching actual values, so compare one symbol at a time.

If predictive variance is miscalibrated, an optional scalar Gaussian variance
calibration can be fitted on an initial, chronological portion of validation
predictions. Reserve the later validation period to choose whether to use it;
never fit it on the test set. For example:

```bash
python -m src.calibrate_forecast fit artifacts/run-1/predictions_validation_native.parquet \
  --end '2025-05-01 00:00:00' --output artifacts/run-1/calibration.json
python -m src.calibrate_forecast apply artifacts/run-1/predictions_validation_native.parquet \
  --calibration artifacts/run-1/calibration.json --after '2025-05-01 00:00:00' \
  --output artifacts/run-1/validation_calibrated.parquet
python -m src.calibrate_forecast apply artifacts/run-1/predictions_test_native.parquet \
  --calibration artifacts/run-1/calibration.json \
  --output artifacts/run-1/test_calibrated.parquet
```

The calibration file records its fitting range and multiplier. Applying it
requires every evaluated target timestamp to be later than that range. It is
an empirical adjustment, not a guarantee of future coverage.

## Running the XGBoost time-series baseline

XGBoost is trained on one row per forecast origin.  Features are constructed
per symbol and all generated lags and rolling statistics use only history;
the configured calendar split is applied to the target (forecast) date.

```bash
python -m src.train_xgboost --symbols BTC-USD --interval 1h --target Close \
  --features Log_Returns Volume Count --horizon 1
```

To train on a fractionally differenced price target, add `--fractional-d 0.4`.
That output is in the fractional-difference domain; reconstruct it to price
space before treating it as a Close-price forecast.

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

### Classical-model searches

Classical models use expanding-window validation rather than shuffled splits.
Candidates are pruned if their required transformed series fails the ADF
stationarity check, their fitted residuals retain serial correlation
(Ljung-Box), or, for GARCH/APARCH, their standardized residuals retain ARCH
effects. Thus the selected configuration passes the core statistical checks in
every training fold.

```bash
python -m src.classical_model_search prices.csv \
  --column Close --model sarima --seasonal-period 24 \
  --horizon 24 --folds 4 --trials 100 \
  --storage sqlite:///classical_sarima.db --study-name btc_hourly_sarima
```

Use a seasonal period consistent with the data frequency (for example, 24 for
hourly data with a daily cycle). ARIMA-family models search AR/MA orders 0–3
and integer differencing 0–2; SARIMA/SARFIMA additionally search seasonal
orders; ARFIMA/SARFIMA search fractional differencing from 0.05 to 0.95; and
GARCH/APARCH search orders 1–3 plus normal, Student-t, and GED innovations.
The compact ranges are intended for a defensible initial search, and can be
widened for a well-understood dataset.

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
