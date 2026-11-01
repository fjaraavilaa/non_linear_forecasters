"""Create the Optuna study used by the TimesFM hyperparameter search."""

import optuna


STORAGE_NAME = "sqlite:///mydb_timefm_5min.db"
STUDY_NAME = "timefm_forecast_search"


optuna.create_study(
    direction="minimize",
    study_name=STUDY_NAME,
    storage=STORAGE_NAME,
    load_if_exists=True,
)
