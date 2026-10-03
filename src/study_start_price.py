import os

import optuna



storage_name = "sqlite:///mydb_lstm_crypto_raw_GP_5min_w_lag.db"


study = optuna.create_study(
    directions=["minimize"],
    study_name="lstm_cryptoraw_study_PeriodicMaternKernel",
    storage=storage_name,
    load_if_exists=True,
)

