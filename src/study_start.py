import os

import optuna



storage_name = "sqlite:///mydb_lstm_crypto_GP_5min_w_lag.db"


study = optuna.create_study(
    directions=["minimize"],
    study_name="lstm_crypto_study_PeriodicMaternKernel_changepretrainedfunction",
    storage=storage_name,
    load_if_exists=True,
)

